using System.Security.Cryptography;
using System.Text;
using RagLearning.Lesson1_Vectors;
using RagLearning.Lesson2_VectorStore;

namespace RagLearning.Lesson3_Rag;

/// <summary>
/// LESSON 3b — Ingestion pipeline:  Load → Split → Chunk → Embed (batched) → Store.
///
/// Production concerns shown here:
///   • Idempotency: a SHA-256 content hash skips unchanged documents (re-embedding costs money).
///   • Replace, don't append: old chunks of a changed doc are deleted first, or stale text survives.
///   • Contextual chunks: "Title > Section" is prepended before embedding, so a chunk that says
///     "keep it neutral" still knows it is about the deadlift back position.
///   • Metadata on every chunk: source, section, docId → filtering + citations.
/// In a real system this runs as a background worker (e.g. triggered by a Service Bus message).
/// </summary>
public sealed class IngestionPipeline(
    IEmbedder embedder,
    InMemoryVectorStore vectors,
    Bm25Index keywords,
    RecursiveChunker chunker)
{
    private readonly Dictionary<string, string> _docHashes = new();
    private readonly Dictionary<string, List<string>> _docChunkIds = new();

    public async Task<int> IngestMarkdownAsync(
        string docId, string title, string markdown,
        IReadOnlyDictionary<string, string>? extraMetadata = null, CancellationToken ct = default)
    {
        var hash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(markdown)));
        if (_docHashes.TryGetValue(docId, out var old) && old == hash)
            return 0;                                              // unchanged → skip

        DeleteDocument(docId);

        // 1) split by structure, 2) chunk each section by size
        var pending = new List<(string Id, string Text, string Section)>();
        foreach (var section in MarkdownSectionSplitter.Split(markdown, title))
        {
            foreach (var chunk in chunker.Chunk(section.Body))
            {
                var contextual = $"{title} > {section.Heading}\n{chunk}";
                pending.Add(($"{docId}#{pending.Count}", contextual, section.Heading));
            }
        }

        // 3) one batched embedding call instead of N calls
        var embeddings = await embedder.EmbedAsync(pending.Select(p => p.Text).ToArray(), ct);

        var ids = new List<string>();
        for (int i = 0; i < pending.Count; i++)
        {
            var (id, text, section) = pending[i];
            var metadata = new Dictionary<string, string>(extraMetadata ?? new Dictionary<string, string>())
            {
                ["docId"] = docId,
                ["title"] = title,
                ["section"] = section,
                ["embeddingModel"] = embedder.Name,   // lets you detect vectors from an old model
            };
            vectors.Upsert(new VectorRecord(id, text, embeddings[i], metadata));
            keywords.Upsert(id, text);
            ids.Add(id);
        }

        _docHashes[docId] = hash;
        _docChunkIds[docId] = ids;
        return ids.Count;
    }

    public void DeleteDocument(string docId)
    {
        if (!_docChunkIds.Remove(docId, out var ids)) return;
        foreach (var id in ids)
        {
            vectors.Delete(id);
            keywords.Remove(id);
        }
        _docHashes.Remove(docId);
    }
}
