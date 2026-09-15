using System.Collections.Concurrent;
using RagLearning.Lesson1_Vectors;

namespace RagLearning.Lesson2_VectorStore;

public sealed record VectorRecord(
    string Id,
    string Text,
    float[] Vector,
    IReadOnlyDictionary<string, string> Metadata);

public sealed record SearchHit(VectorRecord Record, double Score);

/// <summary>
/// LESSON 2a — a vector database in ~60 lines.
///
/// "Flat" (brute-force) search: compare the query with EVERY vector.
///   + exact results, trivial to filter, fine up to ~100k vectors in memory
///   − O(n) per query, so at millions of vectors you need an ANN index (see HnswIndex.cs)
///
/// Production equivalents: pgvector, Qdrant, Azure AI Search, Cosmos DB vector search.
/// </summary>
public sealed class InMemoryVectorStore
{
    private readonly ConcurrentDictionary<string, VectorRecord> _records = new();
    private int? _dimensions;

    public int Count => _records.Count;

    public void Upsert(VectorRecord record)
    {
        _dimensions ??= record.Vector.Length;
        if (record.Vector.Length != _dimensions)
            throw new InvalidOperationException(
                $"Store holds {_dimensions}-dim vectors, got {record.Vector.Length}. Did you switch embedding models?");

        // Normalize once on write → query time can use the cheaper dot product.
        _records[record.Id] = record with { Vector = VectorMath.Normalize(record.Vector) };
    }

    public bool Delete(string id) => _records.TryRemove(id, out _);

    public bool TryGet(string id, out VectorRecord? record) => _records.TryGetValue(id, out record);

    public IEnumerable<VectorRecord> All => _records.Values;

    /// <summary>
    /// Top-k search with PRE-filtering (filter first, then rank).
    /// Pre-filtering is what guarantees tenant isolation: another user's rows are never even scored.
    /// </summary>
    public IReadOnlyList<SearchHit> Search(float[] query, int k, Func<VectorRecord, bool>? filter = null)
    {
        var q = VectorMath.Normalize(query);

        // Min-heap of size k: the root is the WORST of the current best k.
        // O(n log k) instead of sorting all n results.
        var heap = new PriorityQueue<VectorRecord, double>(k + 1);

        foreach (var record in _records.Values)
        {
            if (filter is not null && !filter(record)) continue;

            double score = VectorMath.Dot(q, record.Vector);   // == cosine (both normalized)
            if (heap.Count < k)
            {
                heap.Enqueue(record, score);
            }
            else if (heap.TryPeek(out _, out var worst) && score > worst)
            {
                heap.DequeueEnqueue(record, score);
            }
        }

        var hits = new List<SearchHit>(heap.Count);
        while (heap.TryDequeue(out var r, out var s)) hits.Add(new SearchHit(r, s));
        hits.Reverse();                                        // best first
        return hits;
    }
}

/// <summary>Reusable metadata filters (equivalent of a WHERE clause).</summary>
public static class Filters
{
    public static Func<VectorRecord, bool> Eq(string key, string value) =>
        r => r.Metadata.TryGetValue(key, out var v) && v == value;

    public static Func<VectorRecord, bool> And(params Func<VectorRecord, bool>[] filters) =>
        r => filters.All(f => f(r));
}
