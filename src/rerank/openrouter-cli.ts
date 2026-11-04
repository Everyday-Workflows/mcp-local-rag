#!/usr/bin/env node
// OpenRouter reranker for the RAG_RERANK_CMD hook.
//
// Usage: RAG_RERANK_CMD="node /path/dist/rerank/openrouter-cli.js --query {query} --top {top}"
// Reads query results (docs/schema/query-output.schema.json) on stdin, reorders
// them with OpenRouter's /rerank endpoint, and writes the best `top` to stdout.
// Any failure exits non-zero, so the server keeps its original order.

import { readApiKey } from '../embedder/provider.js'
import { toError } from '../utils/errors.js'
import { isRecord } from '../utils/type-guards.js'
import { type RerankResult, validateRerankResponse } from './response.js'

const DEFAULT_BASE_URL = 'https://openrouter.ai/api/v1'

interface Args {
  query: string
  top: number
}

function parseArgs(argv: string[]): Args {
  const query = argv[argv.indexOf('--query') + 1]
  const top = Number(argv[argv.indexOf('--top') + 1])
  if (!argv.includes('--query') || query === undefined || query.trim() === '') {
    throw new Error('missing --query')
  }
  if (!Number.isInteger(top) || top < 1) {
    throw new Error('--top must be a positive integer')
  }
  return { query, top }
}

async function readStdin(): Promise<string> {
  const chunks: Buffer[] = []
  for await (const chunk of process.stdin) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk)))
  }
  return Buffer.concat(chunks).toString('utf8')
}

/** `[index, relevance]` pairs from an OpenRouter/Cohere-style rerank body, best first. */
function parseRanking(body: unknown, count: number): [number, number][] {
  const results = isRecord(body) ? body['results'] : undefined
  if (!Array.isArray(results)) {
    throw new Error('rerank response has no results array')
  }
  return results.map((item): [number, number] => {
    const index = isRecord(item) ? item['index'] : undefined
    const relevance = isRecord(item) ? item['relevance_score'] : undefined
    if (typeof index !== 'number' || !Number.isInteger(index) || index < 0 || index >= count) {
      throw new Error('rerank response has an invalid index')
    }
    if (typeof relevance !== 'number' || !Number.isFinite(relevance)) {
      throw new Error('rerank response has an invalid relevance_score')
    }
    return [index, relevance]
  })
}

async function rerank(candidates: RerankResult[], args: Args): Promise<RerankResult[]> {
  const model = process.env['OPENROUTER_RERANK_MODEL']?.trim()
  if (!model) {
    throw new Error('OPENROUTER_RERANK_MODEL is not set')
  }
  const baseUrl = (process.env['OPENROUTER_BASE_URL']?.trim() || DEFAULT_BASE_URL).replace(
    /\/+$/,
    ''
  )
  const dataCollection = process.env['OPENROUTER_DATA_COLLECTION']?.trim().toLowerCase() || 'deny'
  const response = await fetch(`${baseUrl}/rerank`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${readApiKey(process.env)}`,
      'Content-Type': 'application/json',
      'X-Title': 'mcp-local-rag',
    },
    body: JSON.stringify({
      model,
      query: args.query,
      documents: candidates.map((candidate) => candidate.text),
      top_n: Math.min(args.top, candidates.length),
      provider: { data_collection: dataCollection },
    }),
  })
  if (!response.ok) {
    throw new Error(`OpenRouter rerank HTTP ${response.status}`)
  }
  // Scores keep the schema's "lower is better" meaning: 1 - relevance.
  return parseRanking(await response.json(), candidates.length)
    .slice(0, args.top)
    .flatMap(([index, relevance]) => {
      const candidate = candidates[index]
      return candidate === undefined ? [] : [{ ...candidate, score: 1 - relevance }]
    })
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2))
  const input = await readStdin()
  const parsed = validateRerankResponse(input)
  if (!parsed.ok) {
    throw new Error(`invalid input: ${parsed.reason}`)
  }
  const output = parsed.results.length === 0 ? [] : await rerank(parsed.results, args)
  process.stdout.write(JSON.stringify(output))
}

main().catch((error: unknown) => {
  // Message only: never echo document text or the key.
  console.error(`openrouter-rerank: ${toError(error).message}`)
  process.exitCode = 1
})
