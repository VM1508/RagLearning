using System.Text;
using System.Text.Json;
using RagLearning.Lesson2_VectorStore;
using RagLearning.Shared;

namespace RagLearning.Lesson3_Rag;

/// <summary>
/// LESSON 3c — Reranking (second-stage ranking).
///
/// Stage 1 (bi-encoder / BM25): query and docs scored independently → fast, rough, recall-oriented.
/// Stage 2 (cross-encoder / LLM): query and ONE doc read together → slow, precise.
/// So: retrieve ~20 cheaply, rerank, keep the best 3–5 for the prompt.
/// Real options: Cohere Rerank, bge-reranker (ONNX in .NET), Azure AI Search semantic ranker.
/// </summary>
public interface IReranker
{
    Task<IReadOnlyList<SearchHit>> RerankAsync(string query, IReadOnlyList<SearchHit> hits, int topN, CancellationToken ct = default);
}

/// <summary>
/// Offline approximation: rewards chunks that cover MORE distinct query terms,
/// blended with the first-stage rank. Cheap, but shows the two-stage shape.
/// </summary>
public sealed class TermCoverageReranker : IReranker
{
    public Task<IReadOnlyList<SearchHit>> RerankAsync(string query, IReadOnlyList<SearchHit> hits, int topN, CancellationToken ct = default)
    {
        var qTerms = TextUtil.Tokens(query).ToHashSet();
        if (qTerms.Count == 0 || hits.Count == 0) return Task.FromResult<IReadOnlyList<SearchHit>>(hits.Take(topN).ToList());

        IReadOnlyList<SearchHit> result = hits
            .Select((h, rank) =>
            {
                var docTerms = TextUtil.Tokens(h.Record.Text).ToHashSet();
                double coverage = qTerms.Count(docTerms.Contains) / (double)qTerms.Count;   // 0..1
                double prior = 1.0 / (rank + 1);                                           // keep stage-1 signal
                // Zero coverage = nothing in common with the question → score 0 so the relevance guard can refuse.
                return h with { Score = coverage == 0 ? 0 : 0.7 * coverage + 0.3 * prior };
            })
            .OrderByDescending(h => h.Score)
            .Take(topN)
            .ToList();
        return Task.FromResult(result);
    }
}

/// <summary>LLM-as-reranker: asks the model to grade each passage 0–10 in one call (listwise).</summary>
public sealed class LlmReranker(IChatModel model) : IReranker
{
    public async Task<IReadOnlyList<SearchHit>> RerankAsync(string query, IReadOnlyList<SearchHit> hits, int topN, CancellationToken ct = default)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < hits.Count; i++)
            sb.AppendLine($"[{i}] {hits[i].Record.Text}");

        var prompt = $$"""
            Grade how well each passage answers the query, 0 (irrelevant) to 10 (directly answers).
            Query: {{query}}

            Passages:
            {{sb}}
            Return JSON: {"scores":[{"index":0,"score":7}, ...]}
            """;

        try
        {
            var json = await model.CompleteAsync([ChatMessage.User(prompt)], jsonMode: true, ct);
            using var doc = JsonDocument.Parse(json);
            var scores = doc.RootElement.GetProperty("scores").EnumerateArray()
                .ToDictionary(e => e.GetProperty("index").GetInt32(), e => e.GetProperty("score").GetDouble());

            return hits.Select((h, i) => h with { Score = scores.GetValueOrDefault(i) / 10.0 })
                       .OrderByDescending(h => h.Score)
                       .Take(topN)
                       .ToList();
        }
        catch (Exception ex) when (ex is JsonException or KeyNotFoundException or InvalidOperationException)
        {
            // Never let a flaky reranker break answering: fall back to stage-1 order.
            return hits.Take(topN).ToList();
        }
    }
}
