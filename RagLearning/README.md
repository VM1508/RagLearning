# RagLearning — Vectors, RAG and Agent Memory in C#

A .NET 10 console app that implements every piece by hand, with **no NuGet packages**, so you can read and debug each step. It runs **offline** using simple stand-ins for the AI models. It switches to **OpenAI / Azure OpenAI** automatically when the environment variables below are set.

```bash
dotnet run          # all lessons (requires .NET 10 SDK)
dotnet run -- 4     # a single lesson
```

## Study order

| # | Folder | What you learn | Read first |
|---|--------|----------------|------------|
| 1 | `Lesson1_Vectors` | Dot product with SIMD, cosine similarity, normalization, the `IEmbedder` interface, feature hashing, batched calls to OpenAI embeddings | `VectorMath.cs` |
| 2 | `Lesson2_VectorStore` | Flat top-k search with a heap, metadata pre-filtering (tenant isolation), **HNSW** from scratch (graph layers, `efSearch`), BM25, hybrid search with RRF | `InMemoryVectorStore.cs` → `HnswIndex.cs` |
| 3 | `Lesson3_Rag` | Markdown-aware plus recursive chunking with overlap, idempotent ingestion (SHA-256), contextual chunks, two-stage rerank, relevance guard, token-budgeted grounded prompt with citations | `Ingestion.cs` → `RagService.cs` |
| 4 | `Lesson4_Memory` | Semantic, episodic and procedural memory; the **ADD / UPDATE / DELETE / NOOP** write path; recall scoring (relevance + recency decay + importance); pinned memories; conversation summary buffer; SQL-style tool for numbers; the `CoachAgent` that combines them all | `MemoryManager.cs` → `CoachAgent.cs` |
| 5 | `Lesson5_Evaluation` | Recall@k and MRR, and comparing vector vs keyword vs hybrid search | `RetrievalEvaluator.cs` |

## Using real models

```bash
# Azure OpenAI (v1 endpoint; model = deployment name)
export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
export AZURE_OPENAI_CHAT_DEPLOYMENT=<your chat deployment>

# or OpenAI
export OPENAI_API_KEY=...
export OPENAI_CHAT_MODEL=<model name>
```

With real models set, the offline implementations are swapped out in `Program.cs`:

| Offline stand-in | Real implementation |
|---|---|
| `HashingEmbedder` | `OpenAiEmbedder` |
| `OfflineExtractiveChatModel` | `OpenAiChatModel` |
| `TermCoverageReranker` | `LlmReranker` |
| `RuleBasedMemoryExtractor` | `LlmMemoryExtractor` |

The rest of the pipeline doesn't change, which is the point of the interfaces.

## From this project to production

| Here | Production (.NET / Azure) |
|---|---|
| `InMemoryVectorStore` + `HnswIndex` | Azure AI Search (vector + BM25 + RRF + semantic ranker), pgvector, Qdrant, Cosmos DB vector search |
| `Bm25Index` | The keyword side of Azure AI Search or Elasticsearch |
| `IEmbedder` / `IChatModel` | `Microsoft.Extensions.AI` abstractions, or Semantic Kernel (check the current API docs) |
| `ToolRouter` (regex) | Real function calling: send a JSON tool schema; the model chooses the tool |
| `IngestionPipeline` | Background worker triggered by Blob events or Service Bus; re-embed when `embeddingModel` metadata changes |
| `MemoryManager` list | Table with `user_id`, `kind`, `key`, `text`, `vector`, `importance`, timestamps, and a vector index filtered by `user_id` |
| `RetrievalEvaluator` | Run in CI on a golden set; add LLM-as-judge for faithfulness |

pgvector equivalent of `InMemoryVectorStore.Search` with a filter:

```sql
CREATE TABLE chunks (id text PRIMARY KEY, user_id text, content text, embedding vector(1536));
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 100);

SET hnsw.ef_search = 50;
SELECT id, content, 1 - (embedding <=> @query) AS score   -- <=> is cosine distance
FROM chunks
WHERE user_id = @userId
ORDER BY embedding <=> @query
LIMIT 5;
```

## Exercises

1. In `Program.cs`, change `RecursiveChunker(400, 60)` to `(150, 0)`, re-run lesson 5, and explain the change in the scores.
2. Add a `delete` for episodic memories older than 90 days (memory "garbage collection").
3. Add `MaxMarginalRelevance` to `RagService` so the four chunks in the prompt are not near-duplicates.
4. Replace `ToolRouter` with real function calling through `OpenAiChatModel` (`tools` + `tool_calls`).
5. Add query rewriting: before retrieval, ask the LLM to turn "and for bench?" into a standalone question, using the `ConversationBuffer`.
