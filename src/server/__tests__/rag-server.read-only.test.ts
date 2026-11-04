// RAG_READ_ONLY: index-changing tools are hidden from clients and refused if called.

import { mkdirSync, rmSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { testModelCacheDir, withTestDevice } from '../../__tests__/test-device.js'
import { RAGServer } from '../index.js'

const ROOT = resolve('./tmp/test-server-read-only')
const WRITE_TOOLS = ['delete_file', 'ingest_data', 'ingest_file', 'sync_start']

async function connect(readOnly: boolean): Promise<Client> {
  const server = new RAGServer(
    withTestDevice({
      dbPath: join(ROOT, readOnly ? 'db-read-only' : 'db-writable'),
      modelName: 'Xenova/all-MiniLM-L6-v2',
      cacheDir: testModelCacheDir(),
      baseDir: join(ROOT, 'docs'),
      maxFileSize: 1024 * 1024,
      readOnly,
    })
  )
  const client = new Client({ name: 'read-only-test', version: '1.0.0' })
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair()
  await Promise.all([client.connect(clientTransport), server.connect(serverTransport)])
  return client
}

describe('RAG_READ_ONLY', () => {
  beforeAll(() => {
    mkdirSync(join(ROOT, 'docs'), { recursive: true })
  })

  afterAll(() => {
    rmSync(ROOT, { recursive: true, force: true })
  })

  it('lists only the tools that cannot change the index', async () => {
    const readOnly = await connect(true)
    const names = (await readOnly.listTools()).tools.map((tool) => tool.name).sort()
    expect(names).toEqual([
      'list_files',
      'query_documents',
      'read_chunk_neighbors',
      'status',
      'sync_status',
    ])

    const writable = await connect(false)
    const all = (await writable.listTools()).tools.map((tool) => tool.name)
    expect(all).toEqual(expect.arrayContaining(WRITE_TOOLS))
  })

  it('refuses index-changing calls even when a client names them directly', async () => {
    const client = await connect(true)
    for (const name of WRITE_TOOLS) {
      const result = await client.callTool({
        name,
        arguments: { filePath: join(ROOT, 'docs', 'x.txt') },
      })
      expect(result.isError).toBe(true)
      expect(JSON.stringify(result.content)).toContain('RAG_READ_ONLY=true')
    }
  })
})
