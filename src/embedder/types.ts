// Shared embedder contract implemented by the local and remote embedders.

import type { EmbeddingComputationIdentity, EmbeddingRole } from './index.js'

/**
 * The embedding surface ingestion, search, and the CLI depend on. Both the
 * local Transformers.js {@link Embedder} and the remote
 * {@link OpenRouterEmbedder} implement it, so callers never branch on the
 * provider.
 */
export interface TextEmbedder {
  /** Whether ingestion embeds chunks behind their document title. */
  readonly titlePrefix: boolean
  /** Whether ingestion adds section paths to the final embedding input. */
  readonly headingPrefix: boolean
  /** Model settings this tool cannot honor. */
  readonly modelWarnings: readonly string[]
  /** Release held resources. */
  dispose(): Promise<void>
  /** Embed one text; the vector is unit-length. */
  embed(text: string, role?: EmbeddingRole): Promise<number[]>
  /** Embed texts in input order; every vector is unit-length. */
  embedBatch(texts: string[], role?: EmbeddingRole): Promise<number[][]>
  /** Identity of the embedding function, or `null` when it cannot be proven. */
  getComputationIdentity(): Promise<EmbeddingComputationIdentity | null>
  /** Effective input token cap, or `null` when none applies. */
  getTokenLimit(): Promise<number | null>
  /** Token length of each text as embedded for `role`. */
  countTokens(texts: string[], role?: EmbeddingRole): Promise<number[]>
  /** The document prompt prepended at ingestion, or `''`. */
  getDocumentPrompt(): Promise<string>
}
