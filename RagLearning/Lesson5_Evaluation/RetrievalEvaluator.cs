using RagLearning.Lesson2_VectorStore;

namespace RagLearning.Lesson5_Evaluation;

/// <summary>A test question and the document(s) that should be retrieved for it.</summary>
public sealed record EvalCase(string Question, params string[] RelevantDocIds);

public sealed record EvalReport(string Mode, double RecallAtK, double Mrr, int K);

/// <summary>
/// LESSON 5 — Measure retrieval before you tune prompts.
///
/// Recall@k : fraction of questions whose relevant doc appears anywhere in the top k.
///            "Did the answer even reach the LLM?"
/// MRR      : mean of 1/rank of the first relevant hit. "How high was it?"
///
/// Generation quality (faithfulness, answer relevance) is scored separately, usually with an
/// LLM-as-judge (RAGAS / DeepEval / Azure AI Foundry evaluators). Build a golden set of
/// 30–50 real questions and run this on every change to chunking, models or search settings.
/// </summary>
public sealed class RetrievalEvaluator(HybridSearcher searcher)
{
    public async Task<EvalReport> EvaluateAsync(IReadOnlyList<EvalCase> cases, HybridSearcher.Mode mode, int k = 3)
    {
        double hitCount = 0, reciprocalRankSum = 0;

        foreach (var c in cases)
        {
            var hits = await searcher.SearchAsync(c.Question, k, mode: mode);
            var docIds = hits.Select(h => h.Record.Metadata["docId"]).ToList();

            int firstRelevant = docIds.FindIndex(c.RelevantDocIds.Contains);
            if (firstRelevant >= 0)
            {
                hitCount++;
                reciprocalRankSum += 1.0 / (firstRelevant + 1);
            }
        }

        return new EvalReport(mode.ToString(), hitCount / cases.Count, reciprocalRankSum / cases.Count, k);
    }
}
