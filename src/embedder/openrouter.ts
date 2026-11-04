// Remote embedder backed by the OpenRouter embeddings API.

import { createHash } from 'node:crypto'
import { setTimeout as sleep } from 'node:timers/promises'
import { serialize } from 'node:v8'
import { toError } from '../utils/errors.js'
import { isObjectLike } from '../utils/type-guards.js'
import { type EmbeddingComputationIdentity, EmbeddingError, type EmbeddingRole } from './index.js'
import type { TextEmbedder } from './types.js'

export const OPENROUTER_DEFAULT_BASE_URL = 'https://openrouter.ai/api/v1'

/** Bump when request shape or vector post-processing changes, to invalidate reuse. */
const OPENROUTER_IMPLEMENTATION_VERSION = 1

/** Statuses worth retrying: rate limits and transient provider failures. */
const RETRYABLE_STATUSES = new Set([408, 409, 425, 429, 500, 502, 503, 504, 529])

/**
 * Conservative characters-per-token ratio. OpenRouter exposes no tokenizer,
 * so lengths are estimated. Overestimating only makes chunks smaller, which is
 * safe; underestimating would let input exceed the model's window.
 */
const CHARS_PER_TOKEN = 3

export interface OpenRouterEmbedderConfig {
  /** OpenRouter API key. */
  apiKey: string
  /** OpenRouter model id, e.g. `perplexity/pplx-embed-v1-4b`. */
  model: string
  /** API base URL (default {@link OPENROUTER_DEFAULT_BASE_URL}). */
  baseUrl?: string
  /** Texts per request (default 64). */
  batchSize?: number | undefined
  /** Requests in flight at once (default 4). */
  concurrency?: number | undefined
  /** Input token cap used for chunk planning (default 8192). */
  tokenLimit?: number | undefined
  /** OpenRouter `provider.data_collection` policy (default `deny`). */
  dataCollection?: 'allow' | 'deny'
  /** Per-request timeout in milliseconds (default 60000). */
  timeoutMs?: number
  /** Retries after the first attempt for retryable failures (default 5). */
  maxRetries?: number
  /** Embed document chunks behind a `Title:` line naming their document. */
  titlePrefix?: boolean
  /** Add section paths to chunk embeddings when they fit. */
  headingPrefix?: boolean
  /** Injectable for tests. */
  fetch?: typeof fetch
  /** Injectable for tests; resolves after `ms`. */
  sleep?: (ms: number) => Promise<void>
}

interface ResolvedConfig {
  apiKey: string
  model: string
  baseUrl: string
  batchSize: number
  concurrency: number
  tokenLimit: number
  dataCollection: 'allow' | 'deny'
  timeoutMs: number
  maxRetries: number
  titlePrefix: boolean
  headingPrefix: boolean
  fetch: typeof fetch
  sleep: (ms: number) => Promise<void>
}

/** Error carrying whether the failed request may be retried. */
class RequestError extends Error {
  constructor(
    message: string,
    readonly retryable: boolean,
    readonly retryAfterMs: number | null = null
  ) {
    super(message)
  }
}

function positiveInteger(value: number | undefined, fallback: number, name: string): number {
  if (value === undefined) {
    return fallback
  }
  if (!Number.isInteger(value) || value < 1) {
    throw new EmbeddingError(`OpenRouter embedder: ${name} must be a positive integer`)
  }
  return value
}

/** Seconds or HTTP-date `Retry-After`, as milliseconds; `null` when absent or unparseable. */
function parseRetryAfter(header: string | null): number | null {
  if (header === null) {
    return null
  }
  const seconds = Number(header)
  if (Number.isFinite(seconds) && seconds >= 0) {
    return seconds * 1000
  }
  const date = Date.parse(header)
  return Number.isNaN(date) ? null : Math.max(0, date - Date.now())
}

/** Scale to unit length; the vector store ranks by dot product. */
function normalize(vector: number[]): number[] {
  let sumSquares = 0
  for (const value of vector) {
    sumSquares += value * value
  }
  const norm = Math.sqrt(sumSquares)
  if (!Number.isFinite(norm) || norm === 0) {
    throw new EmbeddingError('OpenRouter returned a zero or non-finite embedding vector')
  }
  return vector.map((value) => value / norm)
}

function isNumberArray(value: unknown): value is number[] {
  return (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every((item) => typeof item === 'number' && Number.isFinite(item))
  )
}

/** Validate one `{index, embedding}` row; `position` stands in for a missing index. */
function parseRow(item: unknown, position: number, expected: number): [number, number[]] {
  const index = isObjectLike(item) && typeof item['index'] === 'number' ? item['index'] : position
  if (!Number.isInteger(index) || index < 0 || index >= expected) {
    throw new RequestError('OpenRouter embeddings response has invalid row indexes', false)
  }
  const embedding = isObjectLike(item) ? item['embedding'] : undefined
  if (!isNumberArray(embedding)) {
    throw new RequestError('OpenRouter embeddings response has a non-numeric vector', false)
  }
  return [index, embedding]
}

/** Extract vectors in input order from an OpenAI-compatible embeddings body. */
function parseEmbeddings(body: unknown, expected: number): number[][] {
  const data = isObjectLike(body) ? body['data'] : undefined
  if (!Array.isArray(data) || data.length !== expected) {
    throw new RequestError(
      `Unexpected OpenRouter embeddings response (expected ${expected} rows)`,
      false
    )
  }
  const rows = new Map(data.map((item, position) => parseRow(item, position, expected)))
  if (rows.size !== expected) {
    throw new RequestError('OpenRouter embeddings response has duplicate row indexes', false)
  }
  return Array.from({ length: expected }, (_, index) => rows.get(index) ?? [])
}

/** Server-provided `Retry-After` (capped at 60 s), else jittered exponential backoff. */
function retryDelayMs(error: RequestError, attempt: number): number {
  if (error.retryAfterMs !== null) {
    return Math.min(60_000, error.retryAfterMs)
  }
  return Math.min(30_000, 500 * 2 ** attempt) + Math.floor(Math.random() * 250)
}

/** OpenAI-compatible `/embeddings` client for OpenRouter. */
export class OpenRouterEmbedder implements TextEmbedder {
  private readonly config: ResolvedConfig
  private dimension: number | null = null
  private disposed: boolean = false

  constructor(config: OpenRouterEmbedderConfig) {
    if (config.apiKey.trim() === '') {
      throw new EmbeddingError('OpenRouter embedder: API key is empty')
    }
    if (config.model.trim() === '') {
      throw new EmbeddingError('OpenRouter embedder: model is empty')
    }
    if (
      config.maxRetries !== undefined &&
      !(Number.isInteger(config.maxRetries) && config.maxRetries >= 0)
    ) {
      throw new EmbeddingError('OpenRouter embedder: maxRetries must be a non-negative integer')
    }
    this.config = {
      apiKey: config.apiKey,
      model: config.model,
      baseUrl: (config.baseUrl ?? OPENROUTER_DEFAULT_BASE_URL).replace(/\/+$/, ''),
      batchSize: positiveInteger(config.batchSize, 64, 'batchSize'),
      concurrency: positiveInteger(config.concurrency, 4, 'concurrency'),
      tokenLimit: positiveInteger(config.tokenLimit, 8192, 'tokenLimit'),
      dataCollection: config.dataCollection ?? 'deny',
      timeoutMs: positiveInteger(config.timeoutMs, 60_000, 'timeoutMs'),
      maxRetries: config.maxRetries ?? 5,
      titlePrefix: config.titlePrefix ?? false,
      headingPrefix: config.headingPrefix ?? false,
      fetch: config.fetch ?? fetch,
      sleep: config.sleep ?? ((ms: number): Promise<void> => sleep(ms)),
    }
  }

  get titlePrefix(): boolean {
    return this.config.titlePrefix
  }

  get headingPrefix(): boolean {
    return this.config.headingPrefix
  }

  readonly modelWarnings: readonly string[] = []

  async dispose(): Promise<void> {
    this.disposed = true
  }

  async getTokenLimit(): Promise<number> {
    return this.config.tokenLimit
  }

  async countTokens(texts: string[], _role?: EmbeddingRole): Promise<number[]> {
    return texts.map((text) => Math.ceil(text.length / CHARS_PER_TOKEN))
  }

  async getDocumentPrompt(): Promise<string> {
    return ''
  }

  /**
   * Fingerprint of the remote embedding function. The dimension is learned
   * from the first response, so this embeds one probe text when nothing has
   * been embedded yet.
   */
  async getComputationIdentity(): Promise<EmbeddingComputationIdentity | null> {
    let dimension: number
    try {
      dimension = this.dimension ?? (await this.embed('dimension probe')).length
    } catch {
      return null
    }
    const fingerprint = createHash('sha256')
      .update(
        serialize({
          provider: 'openrouter',
          implementationVersion: OPENROUTER_IMPLEMENTATION_VERSION,
          baseUrl: this.config.baseUrl,
          model: this.config.model,
          dimension,
          normalize: true,
          titlePrefix: this.config.titlePrefix,
          headingPrefix: this.config.headingPrefix,
        })
      )
      .digest('hex')
    return { fingerprint, dimension }
  }

  async embed(text: string, role?: EmbeddingRole): Promise<number[]> {
    const [embedding] = await this.embedBatch([text], role)
    if (embedding === undefined) {
      throw new EmbeddingError('Missing embedder batch output row')
    }
    return embedding
  }

  async embedBatch(texts: string[], _role?: EmbeddingRole): Promise<number[][]> {
    if (texts.length === 0) {
      return []
    }
    if (texts.some((text) => text.length === 0)) {
      throw new EmbeddingError('Cannot generate embedding for empty text')
    }
    if (this.disposed) {
      throw new EmbeddingError('OpenRouter embedder has been disposed')
    }

    const batches: string[][] = []
    for (let i = 0; i < texts.length; i += this.config.batchSize) {
      batches.push(texts.slice(i, i + this.config.batchSize))
    }
    const results: number[][][] = Array.from({ length: batches.length })
    let next = 0
    const worker = async (): Promise<void> => {
      while (next < batches.length) {
        const position = next++
        const batch = batches[position]
        if (batch !== undefined) {
          results[position] = await this.requestWithRetry(batch)
        }
      }
    }
    const workers = Math.min(this.config.concurrency, batches.length)
    await Promise.all(Array.from({ length: workers }, worker))
    return results.flat()
  }

  private async requestWithRetry(texts: string[]): Promise<number[][]> {
    for (let attempt = 0; ; attempt++) {
      try {
        return await this.request(texts)
      } catch (error) {
        const retryable = error instanceof RequestError && error.retryable
        if (!retryable || attempt >= this.config.maxRetries) {
          const attempts = attempt > 0 ? ` after ${attempt + 1} attempts` : ''
          throw new EmbeddingError(
            `OpenRouter embedding request failed${attempts}: ${toError(error).message}`,
            { cause: toError(error) }
          )
        }
        const wait = retryDelayMs(error, attempt)
        console.error(`OpenRouter embedder: ${error.message}; retry ${attempt + 1} in ${wait}ms`)
        await this.config.sleep(wait)
      }
    }
  }

  private async request(texts: string[]): Promise<number[][]> {
    const body = await this.post(texts)
    const vectors = parseEmbeddings(body, texts.length).map(normalize)
    for (const vector of vectors) {
      this.dimension ??= vector.length
      if (vector.length !== this.dimension) {
        throw new RequestError(
          `embedding dimension changed from ${this.dimension} to ${vector.length}`,
          false
        )
      }
    }
    return vectors
  }

  /** POST one batch; returns the parsed JSON body or throws a classified {@link RequestError}. */
  private async post(texts: string[]): Promise<unknown> {
    let response: Response
    try {
      response = await this.config.fetch(`${this.config.baseUrl}/embeddings`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${this.config.apiKey}`,
          'Content-Type': 'application/json',
          'X-Title': 'mcp-local-rag',
        },
        body: JSON.stringify({
          model: this.config.model,
          input: texts,
          provider: { data_collection: this.config.dataCollection },
        }),
        signal: AbortSignal.timeout(this.config.timeoutMs),
      })
    } catch (error) {
      // Network failures and timeouts are transient.
      throw new RequestError(`network error: ${toError(error).message}`, true)
    }

    if (!response.ok) {
      const text = await response.text().catch(() => '')
      throw new RequestError(
        `HTTP ${response.status}${text ? `: ${text.slice(0, 300)}` : ''}`,
        RETRYABLE_STATUSES.has(response.status),
        parseRetryAfter(response.headers.get('retry-after'))
      )
    }

    let body: unknown
    try {
      body = await response.json()
    } catch (error) {
      throw new RequestError(`invalid JSON response: ${toError(error).message}`, true)
    }
    // OpenRouter can report an upstream failure inside a 200 body.
    if (isObjectLike(body) && isObjectLike(body['error'])) {
      const code = body['error']['code']
      const message = String(body['error']['message'] ?? 'unknown error')
      throw new RequestError(
        `provider error${code === undefined ? '' : ` ${String(code)}`}: ${message}`,
        typeof code === 'number' && RETRYABLE_STATUSES.has(code)
      )
    }
    return body
  }
}
