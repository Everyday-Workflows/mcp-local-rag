// MCP Server entry point
import {
  type ParseResult,
  parseBooleanEnv,
  parseGroupingMode,
  parseHybridWeight,
  parseMaxDistance,
  parseMaxFiles,
  resolveDevice,
  resolveDtype,
} from './cli/options.js'
import { RAGServer } from './server/index.js'
import {
  BaseDirsConfigError,
  parseBaseDirsEnv,
  parseIgnorePathsEnv,
  resolveBaseDirs,
} from './utils/base-dirs.js'
import {
  DEFAULT_MAX_FILE_SIZE,
  DEFAULT_RERANK_TIMEOUT_MS,
  MAX_CHUNK_MIN_LENGTH,
  MAX_FILE_SIZE_LIMIT,
  RERANK_TIMEOUT_MAX_MS,
  RERANK_TIMEOUT_MIN_MS,
} from './utils/limits.js'
import { checkSensitivePath } from './utils/sensitive-path.js'

// ============================================
// Environment Variable Parsers
// ============================================

/**
 * Parse chunk minimum length from environment variable
 */
export function parseChunkMinLength(value: string | undefined): ParseResult<number> {
  if (!value) {
    return { value: undefined }
  }
  const parsed = /^\d+$/.test(value) ? Number(value) : Number.NaN
  if (Number.isNaN(parsed) || parsed < 1 || parsed > MAX_CHUNK_MIN_LENGTH) {
    const warning = `Invalid CHUNK_MIN_LENGTH value: "${value.slice(0, 100)}". Expected integer between 1 and ${MAX_CHUNK_MIN_LENGTH}. Ignoring.`
    return { value: undefined, warning }
  }
  return { value: parsed }
}

/** Parse the independent PDF image-storage toggle. */
export function parseStoreImages(value: string | undefined): ReturnType<typeof parseBooleanEnv> {
  return parseBooleanEnv('STORE_IMAGES', value)
}

/**
 * Parse the rerank command. Unset, or set to whitespace only, leaves reranking
 * disabled; the string is otherwise kept verbatim because `src/rerank` splits
 * it into argv itself.
 */
export function parseRerankCmd(value: string | undefined): ParseResult<string> {
  const trimmed = value?.trim() ?? ''
  return trimmed.length === 0 ? { value: undefined } : { value: trimmed }
}

/**
 * Parse the rerank timeout, falling back to the default rather than to "no
 * timeout". `Number` rather than `Number.parseInt`, so a fractional value is
 * rejected instead of silently truncated.
 */
export function parseRerankTimeoutMs(value: string | undefined): ParseResult<number> {
  if (!value) {
    return { value: DEFAULT_RERANK_TIMEOUT_MS }
  }
  const parsed = Number(value)
  if (
    !Number.isInteger(parsed) ||
    parsed < RERANK_TIMEOUT_MIN_MS ||
    parsed > RERANK_TIMEOUT_MAX_MS
  ) {
    return {
      value: DEFAULT_RERANK_TIMEOUT_MS,
      warning: `Invalid RAG_RERANK_TIMEOUT_MS value: "${value.slice(0, 100)}". Expected integer between ${RERANK_TIMEOUT_MIN_MS} and ${RERANK_TIMEOUT_MAX_MS}. Using default (${DEFAULT_RERANK_TIMEOUT_MS}).`,
    }
  }
  return { value: parsed }
}

// ============================================
// Server Startup
// ============================================

/** Resolved server config type, named so helpers can share it. */
type ServerConfig = ConstructorParameters<typeof RAGServer>[0]

/** Checked before realpath, which would turn `/etc` into `/private/etc`. */
function collectRawSensitiveErrors(env: NodeJS.ProcessEnv): string[] {
  const errors: string[] = []
  const baseDirs = env['BASE_DIRS']
  if (baseDirs !== undefined && baseDirs.length > 0) {
    const parsed = parseBaseDirsEnv(baseDirs)
    if (!parsed.ok) {
      return errors
    }
    for (const raw of parsed.value) {
      const sensitive = checkSensitivePath(raw, 'BASE_DIRS')
      if (sensitive) {
        errors.push(sensitive)
      }
    }
    return errors
  }
  const baseDir = env['BASE_DIR']
  if (baseDir !== undefined && baseDir.trim().length > 0) {
    const sensitive = checkSensitivePath(baseDir, 'BASE_DIR')
    if (sensitive) {
      errors.push(sensitive)
    }
  }
  return errors
}

/** Roots the server will serve, plus whatever made them unusable. */
interface ResolvedRoots {
  baseDirs: string[]
  /** Normal-path roots, index-aligned with `baseDirs`, for list_files display. */
  rawBaseDirs: string[]
  configError?: BaseDirsConfigError
  warnings: string[]
}

/** No usable root: every tool that needs one fails closed with `error`. */
function noRoots(error: BaseDirsConfigError): ResolvedRoots {
  return { baseDirs: [], rawBaseDirs: [], configError: error, warnings: [error.message] }
}

async function resolveRoots(env: NodeJS.ProcessEnv, cwd: string): Promise<ResolvedRoots> {
  // Raw sensitive-path matches take precedence over resolver errors.
  const rawSensitiveErrors = collectRawSensitiveErrors(env)
  if (rawSensitiveErrors.length > 0) {
    return noRoots(new BaseDirsConfigError([...new Set(rawSensitiveErrors)].join('; ')))
  }

  const result = await resolveBaseDirs({
    envBaseDirs: env['BASE_DIRS'],
    envBaseDir: env['BASE_DIR'],
    cwd,
  })
  if (!result.ok) {
    return noRoots(result.error)
  }

  const baseDirs = env['BASE_DIRS']
  const sourceFlag = baseDirs !== undefined && baseDirs.length > 0 ? 'BASE_DIRS' : 'BASE_DIR'
  const sensitiveErrors: string[] = []
  for (const root of result.config.baseDirs) {
    const sensitive = checkSensitivePath(root, sourceFlag)
    if (sensitive) {
      sensitiveErrors.push(sensitive)
    }
  }
  if (sensitiveErrors.length > 0) {
    return noRoots(new BaseDirsConfigError([...new Set(sensitiveErrors)].join('; ')))
  }
  return {
    baseDirs: result.config.baseDirs,
    rawBaseDirs: result.config.rawBaseDirs,
    warnings: result.warnings.map((warning) => warning.message),
  }
}

function resolveMaxFileSize(env: NodeJS.ProcessEnv): { value: number; warning?: string } {
  const raw = env['MAX_FILE_SIZE']
  if (!raw) {
    return { value: DEFAULT_MAX_FILE_SIZE }
  }
  const parsed = /^\d+$/.test(raw) ? Number(raw) : Number.NaN
  if (!Number.isInteger(parsed) || parsed < 1 || parsed > MAX_FILE_SIZE_LIMIT) {
    return {
      value: DEFAULT_MAX_FILE_SIZE,
      warning: `Invalid MAX_FILE_SIZE value: "${raw.slice(0, 100)}". Expected integer between 1 and ${MAX_FILE_SIZE_LIMIT}. Using default (${DEFAULT_MAX_FILE_SIZE}).`,
    }
  }
  return { value: parsed }
}

/**
 * Apply the quality-filter settings that are only set when defined, so an unset
 * variable keeps meaning "use the downstream default". Returns their warnings.
 */
function applyOptionalSettings(config: ServerConfig, env: NodeJS.ProcessEnv): string[] {
  const maxDistance = parseMaxDistance(env['RAG_MAX_DISTANCE'])
  const grouping = parseGroupingMode(env['RAG_GROUPING'])
  const maxFiles = parseMaxFiles(env['RAG_MAX_FILES'])
  const hybridWeight = parseHybridWeight(env['RAG_HYBRID_WEIGHT'])
  const chunkMinLength = parseChunkMinLength(env['CHUNK_MIN_LENGTH'])
  const storeImages = parseStoreImages(env['STORE_IMAGES'])
  const titlePrefix = parseBooleanEnv('EMBED_TITLE_PREFIX', env['EMBED_TITLE_PREFIX'])
  const headingPrefix = parseBooleanEnv('EMBED_HEADING_PREFIX', env['EMBED_HEADING_PREFIX'])
  const rerankCommand = parseRerankCmd(env['RAG_RERANK_CMD'])
  const rerankTimeoutMs = parseRerankTimeoutMs(env['RAG_RERANK_TIMEOUT_MS'])
  const readOnly = parseBooleanEnv('RAG_READ_ONLY', env['RAG_READ_ONLY'])

  if (maxDistance.value !== undefined) {
    config.maxDistance = maxDistance.value
  }
  if (grouping.value !== undefined) {
    config.grouping = grouping.value
  }
  if (maxFiles.value !== undefined) {
    config.maxFiles = maxFiles.value
  }
  if (hybridWeight.value !== undefined) {
    config.hybridWeight = hybridWeight.value
  }
  if (chunkMinLength.value !== undefined) {
    config.chunkMinLength = chunkMinLength.value
  }
  if (rerankCommand.value !== undefined) {
    config.rerankCommand = rerankCommand.value
  }
  config.storeImages = storeImages.value
  config.titlePrefix = titlePrefix.value
  config.headingPrefix = headingPrefix.value
  config.rerankTimeoutMs = rerankTimeoutMs.value ?? DEFAULT_RERANK_TIMEOUT_MS
  config.readOnly = readOnly.value

  return [
    maxDistance,
    grouping,
    maxFiles,
    hybridWeight,
    chunkMinLength,
    storeImages,
    titlePrefix,
    headingPrefix,
    rerankCommand,
    rerankTimeoutMs,
    readOnly,
  ]
    .map((parsed) => parsed.warning)
    .filter((warning): warning is string => warning !== undefined)
}

/**
 * Single source of truth for BASE_DIRS / BASE_DIR / cwd precedence. A resolver
 * error never falls back to cwd.
 */
export async function resolveServerConfig(
  env: NodeJS.ProcessEnv,
  cwd: string
): Promise<ServerConfig> {
  const roots = await resolveRoots(env, cwd)
  // Fail closed: starting without the exclusion would index the files it names.
  const ignorePatterns = parseIgnorePathsEnv(env['IGNORE_PATHS'])
  if (!ignorePatterns.ok) {
    throw new Error(ignorePatterns.message)
  }
  const maxFileSize = resolveMaxFileSize(env)
  const configWarnings = [...roots.warnings]
  if (maxFileSize.warning !== undefined) {
    configWarnings.push(maxFileSize.warning)
  }

  const config: ServerConfig = {
    dbPath: env['DB_PATH'] || './lancedb/',
    modelName: env['MODEL_NAME'] || 'Xenova/all-MiniLM-L6-v2',
    cacheDir: env['CACHE_DIR'] || './models/',
    baseDirs: roots.baseDirs,
    rawBaseDirs: roots.rawBaseDirs,
    ignorePatterns: ignorePatterns.value,
    maxFileSize: maxFileSize.value,
    device: resolveDevice(env['RAG_DEVICE']),
    storeImages: false,
  }

  configWarnings.push(...applyOptionalSettings(config, env))

  // Set dtype only when defined, so config.dtype === undefined keeps meaning
  // "RAG_DTYPE unset" (the embedder then applies its fp32 default).
  const dtype = resolveDtype(env['RAG_DTYPE'])
  if (dtype !== undefined) {
    config.dtype = dtype
  }

  if (configWarnings.length > 0) {
    config.configWarnings = configWarnings
  }
  if (roots.configError !== undefined) {
    config.configError = roots.configError
  }

  return config
}

/** Env-only configuration, so a bare `mcp-local-rag` launch suits MCP clients. */
export async function startServer(): Promise<void> {
  try {
    const config = await resolveServerConfig(process.env, process.cwd())

    if (config.configWarnings && config.configWarnings.length > 0) {
      console.error('Configuration warnings:', config.configWarnings.join(' | '))
    }

    console.error('Starting RAG MCP Server...')
    console.error('Configuration:', config)

    // Start RAGServer
    const server = new RAGServer(config)
    await server.initialize()
    await server.run()

    console.error('RAG MCP Server started successfully')
  } catch (error) {
    console.error('Failed to start RAG MCP Server:', error)
    process.exit(1)
  }
}
