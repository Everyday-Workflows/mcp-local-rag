// Embedding provider selection from the environment.

import { readFileSync } from 'node:fs'
import { toError } from '../utils/errors.js'
import { Embedder, type EmbedderConfig, EmbeddingError } from './index.js'
import { OPENROUTER_DEFAULT_BASE_URL, OpenRouterEmbedder } from './openrouter.js'
import type { TextEmbedder } from './types.js'

/** Key from `OPENROUTER_API_KEY_FILE` (preferred) or `OPENROUTER_API_KEY`, trimmed. */
export function readApiKey(env: NodeJS.ProcessEnv): string {
  const keyFile = env['OPENROUTER_API_KEY_FILE']
  let apiKey = env['OPENROUTER_API_KEY'] ?? ''
  if (keyFile) {
    try {
      apiKey = readFileSync(keyFile, 'utf8')
    } catch (error) {
      throw new EmbeddingError(`Cannot read OPENROUTER_API_KEY_FILE ${keyFile}`, {
        cause: toError(error),
      })
    }
  }
  apiKey = apiKey.trim()
  if (apiKey === '') {
    throw new EmbeddingError(
      'EMBEDDING_PROVIDER=openrouter requires OPENROUTER_API_KEY or OPENROUTER_API_KEY_FILE'
    )
  }
  return apiKey
}

/** Optional positive-integer setting; `undefined` when unset. */
function optionalPositiveInt(env: NodeJS.ProcessEnv, name: string): number | undefined {
  const raw = env[name]
  if (raw === undefined || raw.trim() === '') {
    return undefined
  }
  const value = Number(raw)
  if (!Number.isInteger(value) || value < 1) {
    throw new EmbeddingError(`${name} must be a positive integer, got "${raw}"`)
  }
  return value
}

/**
 * Build the remote embedder when `EMBEDDING_PROVIDER=openrouter`, else return
 * `null` so the caller keeps the local Transformers.js embedder.
 *
 * Environment:
 * - `OPENROUTER_API_KEY` or `OPENROUTER_API_KEY_FILE` (file wins; trimmed)
 * - `OPENROUTER_EMBEDDING_MODEL` (required)
 * - `OPENROUTER_BASE_URL`, `OPENROUTER_EMBEDDING_BATCH_SIZE`,
 *   `OPENROUTER_EMBEDDING_CONCURRENCY`, `OPENROUTER_EMBEDDING_TOKEN_LIMIT`,
 *   `OPENROUTER_DATA_COLLECTION` (`deny` default, or `allow`)
 */
export function createRemoteEmbedderFromEnv(
  env: NodeJS.ProcessEnv,
  prefixes: { titlePrefix: boolean; headingPrefix: boolean }
): OpenRouterEmbedder | null {
  const provider = (env['EMBEDDING_PROVIDER'] ?? 'local').trim().toLowerCase()
  if (provider === 'local' || provider === '') {
    return null
  }
  if (provider !== 'openrouter') {
    throw new EmbeddingError(
      `Unknown EMBEDDING_PROVIDER "${provider}". Use "local" or "openrouter".`
    )
  }
  const model = env['OPENROUTER_EMBEDDING_MODEL']?.trim()
  if (!model) {
    throw new EmbeddingError('EMBEDDING_PROVIDER=openrouter requires OPENROUTER_EMBEDDING_MODEL')
  }
  const dataCollection = (env['OPENROUTER_DATA_COLLECTION'] ?? 'deny').trim().toLowerCase()
  if (dataCollection !== 'deny' && dataCollection !== 'allow') {
    throw new EmbeddingError(
      `OPENROUTER_DATA_COLLECTION must be "deny" or "allow", got "${dataCollection}"`
    )
  }
  return new OpenRouterEmbedder({
    apiKey: readApiKey(env),
    model,
    baseUrl: env['OPENROUTER_BASE_URL']?.trim() || OPENROUTER_DEFAULT_BASE_URL,
    dataCollection,
    titlePrefix: prefixes.titlePrefix,
    headingPrefix: prefixes.headingPrefix,
    batchSize: optionalPositiveInt(env, 'OPENROUTER_EMBEDDING_BATCH_SIZE'),
    concurrency: optionalPositiveInt(env, 'OPENROUTER_EMBEDDING_CONCURRENCY'),
    tokenLimit: optionalPositiveInt(env, 'OPENROUTER_EMBEDDING_TOKEN_LIMIT'),
  })
}

/**
 * The embedder for this process: OpenRouter when `EMBEDDING_PROVIDER` selects
 * it, otherwise the local Transformers.js embedder built from `localConfig`.
 * Shared by the MCP server and the CLI so both pick the same provider.
 */
export function createTextEmbedder(localConfig: EmbedderConfig): TextEmbedder {
  return (
    createRemoteEmbedderFromEnv(process.env, {
      titlePrefix: localConfig.titlePrefix === true,
      headingPrefix: localConfig.headingPrefix === true,
    }) ?? new Embedder(localConfig)
  )
}
