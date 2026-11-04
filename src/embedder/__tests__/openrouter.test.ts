import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { OpenRouterEmbedder, type OpenRouterEmbedderConfig } from '../openrouter.js'
import { createRemoteEmbedderFromEnv, readApiKey } from '../provider.js'

interface Call {
  url: string
  body: { model: string; input: string[]; provider: { data_collection: string } }
}

type Reply = (call: Call) => Response

const json = (body: unknown, status = 200, headers: Record<string, string> = {}): Response =>
  new Response(JSON.stringify(body), { status, headers })

/** Deterministic vector per text: [length, 1, 0]. */
const embeddingFor = (text: string): number[] => [text.length, 1, 0]

const okReply: Reply = (call) =>
  json({ data: call.body.input.map((text, index) => ({ index, embedding: embeddingFor(text) })) })

function harness(
  replies: Reply[],
  config: Partial<OpenRouterEmbedderConfig> = {}
): { embedder: OpenRouterEmbedder; calls: Call[]; sleeps: number[] } {
  const calls: Call[] = []
  const sleeps: number[] = []
  const fakeFetch: typeof fetch = async (url, init) => {
    const call: Call = { url: String(url), body: JSON.parse(String(init?.body)) }
    calls.push(call)
    const reply = replies[Math.min(calls.length - 1, replies.length - 1)]
    if (reply === undefined) {
      throw new Error('no reply configured')
    }
    return reply(call)
  }
  const embedder = new OpenRouterEmbedder({
    apiKey: 'sk-test',
    model: 'test/model',
    fetch: fakeFetch,
    sleep: async (ms) => {
      sleeps.push(ms)
    },
    ...config,
  })
  return { embedder, calls, sleeps }
}

const norm = (vector: number[]): number => Math.hypot(...vector)

describe('OpenRouterEmbedder', () => {
  it('splits batches, keeps input order across out-of-order rows, and returns unit vectors', async () => {
    const reversed: Reply = (call) =>
      json({
        data: call.body.input
          .map((text, index) => ({ index, embedding: embeddingFor(text) }))
          .reverse(),
      })
    const { embedder, calls } = harness([reversed], { batchSize: 2, concurrency: 2 })
    const texts = ['a', 'bb', 'ccc', 'dddd', 'eeeee']

    const vectors = await embedder.embedBatch(texts)

    expect(calls.map((call) => call.body.input)).toEqual([['a', 'bb'], ['ccc', 'dddd'], ['eeeee']])
    expect(vectors).toHaveLength(5)
    for (const [i, vector] of vectors.entries()) {
      expect(norm(vector)).toBeCloseTo(1, 10)
      // First component encodes the source text length, so order is checkable.
      const length = expectLength(texts[i])
      expect(vector[0] / vector[1]).toBeCloseTo(length, 10)
    }
  })

  it('asks providers not to collect data unless told otherwise', async () => {
    const { embedder, calls } = harness([okReply])
    await embedder.embed('hello')
    expect(calls[0]?.body.provider).toEqual({ data_collection: 'deny' })
    expect(calls[0]?.url).toBe('https://openrouter.ai/api/v1/embeddings')
  })

  it('retries rate limits using Retry-After, then succeeds', async () => {
    const { embedder, calls, sleeps } = harness([
      () => json({ error: { message: 'slow down' } }, 429, { 'retry-after': '2' }),
      () => json({ error: { code: 502, message: 'upstream' } }),
      okReply,
    ])
    const vector = await embedder.embed('hello')
    expect(calls).toHaveLength(3)
    expect(sleeps[0]).toBe(2000)
    expect(sleeps).toHaveLength(2)
    expect(norm(vector)).toBeCloseTo(1, 10)
  })

  it('fails fast on non-retryable errors without leaking the key', async () => {
    const { embedder, calls } = harness([() => json({ error: { message: 'bad key' } }, 401)])
    const error = await embedder.embed('hello').catch((caught: unknown) => caught)
    expect(calls).toHaveLength(1)
    expect(String(error)).toContain('HTTP 401')
    expect(String(error)).not.toContain('sk-test')
  })

  it('gives up after maxRetries on persistent transient failures', async () => {
    const { embedder, calls } = harness([() => json({}, 503)], { maxRetries: 2 })
    await expect(embedder.embed('hello')).rejects.toThrow(/after 3 attempts/)
    expect(calls).toHaveLength(3)
  })

  it('rejects a response whose row count or vector dimension does not match', async () => {
    const short = harness([() => json({ data: [] })])
    await expect(short.embedder.embed('hello')).rejects.toThrow(/expected 1 rows/)

    let size = 3
    const growing = harness([
      (call) =>
        json({
          data: call.body.input.map((_, index) => ({
            index,
            embedding: Array.from({ length: size++ }, () => 1),
          })),
        }),
    ])
    await growing.embedder.embed('first')
    await expect(growing.embedder.embed('second')).rejects.toThrow(/dimension changed/)
  })

  it('identifies the embedding function by model and dimension', async () => {
    const a = harness([okReply])
    const b = harness([okReply])
    const other = harness([okReply], { model: 'test/other' })

    const identityA = await a.embedder.getComputationIdentity()
    expect(identityA?.dimension).toBe(3)
    expect(a.calls).toHaveLength(1)
    expect(await b.embedder.getComputationIdentity()).toEqual(identityA)
    expect((await other.embedder.getComputationIdentity())?.fingerprint).not.toBe(
      identityA?.fingerprint
    )

    const down = harness([() => json({}, 401)])
    expect(await down.embedder.getComputationIdentity()).toBeNull()
  })

  it('estimates token counts high so chunks stay within the model window', async () => {
    const { embedder } = harness([okReply], { tokenLimit: 1000 })
    expect(await embedder.getTokenLimit()).toBe(1000)
    expect(await embedder.countTokens(['abcdef', 'abcdefg'])).toEqual([2, 3])
  })
})

describe('createRemoteEmbedderFromEnv', () => {
  const prefixes = { titlePrefix: false, headingPrefix: false }

  it('keeps the local embedder when no provider is selected', () => {
    expect(createRemoteEmbedderFromEnv({}, prefixes)).toBeNull()
    expect(createRemoteEmbedderFromEnv({ EMBEDDING_PROVIDER: 'local' }, prefixes)).toBeNull()
  })

  it('rejects incomplete or invalid OpenRouter settings', () => {
    const base = { EMBEDDING_PROVIDER: 'openrouter', OPENROUTER_API_KEY: 'k' }
    expect(() => createRemoteEmbedderFromEnv(base, prefixes)).toThrow(/OPENROUTER_EMBEDDING_MODEL/)
    expect(() =>
      createRemoteEmbedderFromEnv(
        { EMBEDDING_PROVIDER: 'openrouter', OPENROUTER_EMBEDDING_MODEL: 'm' },
        prefixes
      )
    ).toThrow(/OPENROUTER_API_KEY/)
    expect(() =>
      createRemoteEmbedderFromEnv(
        { ...base, OPENROUTER_EMBEDDING_MODEL: 'm', OPENROUTER_DATA_COLLECTION: 'maybe' },
        prefixes
      )
    ).toThrow(/OPENROUTER_DATA_COLLECTION/)
    expect(() =>
      createRemoteEmbedderFromEnv(
        { ...base, OPENROUTER_EMBEDDING_MODEL: 'm', OPENROUTER_EMBEDDING_BATCH_SIZE: '0' },
        prefixes
      )
    ).toThrow(/positive integer/)
    expect(() => createRemoteEmbedderFromEnv({ EMBEDDING_PROVIDER: 'cohere' }, prefixes)).toThrow(
      /Unknown EMBEDDING_PROVIDER/
    )
  })

  it('prefers the key file over the inline key, trims it, and reports an unreadable file', () => {
    const dir = mkdtempSync(join(tmpdir(), 'or-key-'))
    const keyFile = join(dir, 'key')
    writeFileSync(keyFile, '  sk-from-file\n')
    expect(readApiKey({ OPENROUTER_API_KEY: 'sk-inline', OPENROUTER_API_KEY_FILE: keyFile })).toBe(
      'sk-from-file'
    )
    expect(readApiKey({ OPENROUTER_API_KEY: ' sk-inline ' })).toBe('sk-inline')
    expect(() => readApiKey({ OPENROUTER_API_KEY_FILE: join(dir, 'missing') })).toThrow(
      /Cannot read OPENROUTER_API_KEY_FILE/
    )
  })
})

function expectLength(text: string | undefined): number {
  if (text === undefined) {
    throw new Error('missing text')
  }
  return text.length
}
