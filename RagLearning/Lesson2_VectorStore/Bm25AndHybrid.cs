using RagLearning.Lesson1_Vectors;
using RagLearning.Shared;

namespace RagLearning.Lesson2_VectorStore;

/// <summary>
/// LESSON 2c — BM25 keyword search (what Lucene / Elasticsearch / Azure AI Search use by default).
///
/// score(doc) = Σ over query terms t of
///     IDF(t) · tf·(k1+1) / (tf + k1·(1 − b + b·len/avgLen))
///
///   IDF  → rare words ("RDL", "Epley") matter more than common ones
///   k1   → term-frequency saturation (the 10th "squat" adds little)
///   b    → length normalization (long docs don't win just by being long)
///
/// Vectors are great at meaning but weak at exact tokens: IDs, acronyms, numbers, jargon.
/// BM25 is the opposite. That is why production RAG uses both (hybrid).
/// </summary>
public sealed class Bm25Index(double k1 = 1.2, double b = 0.75)
{
    private readonly Dictionary<string, Dictionary<string, int>> _postings = new(); // term → (docId → tf)
    private readonly Dictionary<string, int> _docLengths = new();
    private readonly Dictionary<string, string[]> _docTerms = new();
    private long _totalLength;

    public void Upsert(string docId, string text)
    {
        Remove(docId);
        var terms = TextUtil.Tokens(text).ToArray();
        _docTerms[docId] = terms;
        _docLengths[docId] = terms.Length;
        _totalLength += terms.Length;

        foreach (var term in terms)
        {
            if (!_postings.TryGetValue(term, out var docs))
                _postings[term] = docs = new Dictionary<string, int>();
            docs[docId] = docs.GetValueOrDefault(docId) + 1;
        }
    }

    public void Remove(string docId)
    {
        if (!_docTerms.Remove(docId, out var terms)) return;
        _totalLength -= _docLengths[docId];
        _docLengths.Remove(docId);
        foreach (var term in terms.Distinct())
        {
            _postings[term].Remove(docId);
            if (_postings[term].Count == 0) _postings.Remove(term);
        }
    }

    public IReadOnlyList<(string DocId, double Score)> Search(string query, int k, Func<string, bool>? docFilter = null)
    {
        int n = _docLengths.Count;
        if (n == 0) return [];
        double avgLen = (double)_totalLength / n;
        var scores = new Dictionary<string, double>();

        foreach (var term in TextUtil.Tokens(query).Distinct())
        {
            if (!_postings.TryGetValue(term, out var docs)) continue;
            double idf = Math.Log(1 + (n - docs.Count + 0.5) / (docs.Count + 0.5));

            foreach (var (docId, tf) in docs)
            {
                if (docFilter is not null && !docFilter(docId)) continue;
                double norm = tf + k1 * (1 - b + b * _docLengths[docId] / avgLen);
                scores[docId] = scores.GetValueOrDefault(docId) + idf * tf * (k1 + 1) / norm;
            }
        }

        return scores.OrderByDescending(kv => kv.Value).Take(k)
                     .Select(kv => (kv.Key, kv.Value)).ToList();
    }
}

/// <summary>
/// LESSON 2d — Hybrid search with Reciprocal Rank Fusion (RRF).
///
/// BM25 scores (0..30) and cosine scores (-1..1) are not comparable, so we don't add them.
/// RRF only uses RANK:  score(d) = Σ 1 / (60 + rank_in_list(d))
/// A doc ranked well by both lists beats a doc ranked #1 by only one list.
/// Azure AI Search's hybrid query does exactly this.
/// </summary>
public sealed class HybridSearcher(IEmbedder embedder, InMemoryVectorStore vectors, Bm25Index keywords)
{
    public enum Mode { Vector, Keyword, Hybrid }

    private const int RrfK = 60;

    public async Task<IReadOnlyList<SearchHit>> SearchAsync(
        string query, int k, Func<VectorRecord, bool>? filter = null,
        Mode mode = Mode.Hybrid, int candidatesPerList = 20, CancellationToken ct = default)
    {
        var fused = new Dictionary<string, double>();

        if (mode is Mode.Vector or Mode.Hybrid)
        {
            var qv = await embedder.EmbedOneAsync(query, ct);
            var hits = vectors.Search(qv, candidatesPerList, filter);
            for (int rank = 0; rank < hits.Count; rank++)
                fused[hits[rank].Record.Id] = fused.GetValueOrDefault(hits[rank].Record.Id) + 1.0 / (RrfK + rank + 1);
        }

        if (mode is Mode.Keyword or Mode.Hybrid)
        {
            // Apply the SAME metadata filter to keyword results — easy to forget, and a classic data leak.
            Func<string, bool>? docFilter = filter is null
                ? null
                : id => vectors.TryGet(id, out var r) && filter(r!);

            var hits = keywords.Search(query, candidatesPerList, docFilter);
            for (int rank = 0; rank < hits.Count; rank++)
                fused[hits[rank].DocId] = fused.GetValueOrDefault(hits[rank].DocId) + 1.0 / (RrfK + rank + 1);
        }

        return fused.OrderByDescending(kv => kv.Value)
                    .Take(k)
                    .Select(kv => vectors.TryGet(kv.Key, out var r) ? new SearchHit(r!, kv.Value) : null)
                    .OfType<SearchHit>()
                    .ToList();
    }
}
