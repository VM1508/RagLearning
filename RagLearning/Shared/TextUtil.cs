using System.Text.RegularExpressions;

namespace RagLearning.Shared;

/// <summary>
/// Tiny tokenizer shared by the offline embedder and BM25.
/// Real systems use the model's own tokenizer (for token counts) and a proper
/// analyzer (Lucene / Azure AI Search) for keyword search.
/// </summary>
public static partial class TextUtil
{
    private static readonly HashSet<string> StopWords =
    [
        "a", "an", "the", "is", "are", "am", "was", "were", "be", "to", "of", "and", "or",
        "in", "on", "for", "with", "at", "by", "it", "i", "my", "me", "you", "your", "we",
        "how", "what", "do", "does", "should", "can", "when", "which", "this", "that", "so"
    ];

    [GeneratedRegex(@"[a-z0-9]+")]
    private static partial Regex WordRegex();

    public static IEnumerable<string> Tokens(string text, bool removeStopWords = true)
    {
        foreach (Match m in WordRegex().Matches(text.ToLowerInvariant()))
        {
            var w = Stem(m.Value);
            if (removeStopWords && StopWords.Contains(w)) continue;
            yield return w;
        }
    }

    /// <summary>Very small suffix stripper so "squatting", "squats", "squat" match.</summary>
    public static string Stem(string w)
    {
        if (w.Length > 5 && w.EndsWith("ing")) w = UnDouble(w[..^3]);
        else if (w.Length > 4 && w.EndsWith("ed")) w = UnDouble(w[..^2]);
        else if (w.Length > 4 && w.EndsWith("ies")) w = w[..^3] + "y";
        else if (w.Length > 4 && (w.EndsWith("ches") || w.EndsWith("shes") || w.EndsWith("xes"))) w = w[..^2];
        else if (w.Length > 3 && w.EndsWith('s') && !w.EndsWith("ss")) w = w[..^1];
        return w;
    }

    private static string UnDouble(string w) =>
        w.Length > 2 && w[^1] == w[^2] && w[^1] is not ('l' or 's' or 'z') ? w[..^1] : w;

    /// <summary>
    /// Stable 32-bit FNV-1a hash. Never use string.GetHashCode() for anything persisted:
    /// in .NET it is randomized per process, so vectors would change on every run.
    /// </summary>
    public static uint Fnv1a(string s)
    {
        uint hash = 2166136261;
        foreach (char c in s)
        {
            hash ^= c;
            hash *= 16777619;
        }
        return hash;
    }

    /// <summary>Rough token estimate (~4 chars per token for English). Good enough for budgeting.</summary>
    public static int EstimateTokens(string text) => (text.Length + 3) / 4;

    public static IEnumerable<string> Sentences(string text) =>
        Regex.Split(text, @"(?<=[.!?])\s+|\n+")
             .Select(s => s.Trim().TrimStart('-', '*', '#', ' '))
             .Where(s => s.Length > 0);
}
