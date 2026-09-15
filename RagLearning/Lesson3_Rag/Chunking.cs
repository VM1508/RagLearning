namespace RagLearning.Lesson3_Rag;

public sealed record Section(string Heading, string Body);

/// <summary>
/// LESSON 3a — Chunking. Most RAG quality problems start here.
///
/// Too big  → the embedding averages several topics and matches nothing well.
/// Too small → the chunk has no context ("It should be 2–3 sets." — what should?).
/// Rule of thumb: 300–800 tokens, 10–20% overlap, split on natural boundaries.
/// </summary>
public static class MarkdownSectionSplitter
{
    /// <summary>Structure-aware split: one section per markdown heading. The heading becomes metadata.</summary>
    public static IReadOnlyList<Section> Split(string markdown, string title)
    {
        var sections = new List<Section>();
        var heading = title;
        var body = new System.Text.StringBuilder();

        foreach (var line in markdown.Split('\n'))
        {
            if (line.StartsWith('#'))
            {
                if (body.ToString().Trim().Length > 0) sections.Add(new Section(heading, body.ToString().Trim()));
                heading = line.TrimStart('#', ' ').Trim();
                body.Clear();
            }
            else
            {
                body.AppendLine(line);
            }
        }
        if (body.ToString().Trim().Length > 0) sections.Add(new Section(heading, body.ToString().Trim()));
        return sections;
    }
}

/// <summary>
/// Recursive character splitter (same idea as LangChain's RecursiveCharacterTextSplitter):
///   1. Try to split on paragraphs; if a piece is still too big, split it on lines,
///      then sentences, then words, then hard-cut.
///   2. Greedily pack pieces into chunks up to maxChars.
///   3. Start each new chunk with the tail of the previous one (overlap), so a fact that
///      straddles a boundary still appears whole in at least one chunk.
/// </summary>
public sealed class RecursiveChunker(int maxChars = 800, int overlapChars = 120)
{
    private static readonly string[] Separators = ["\n\n", "\n", ". ", " "];

    public IReadOnlyList<string> Chunk(string text)
    {
        var pieces = SplitRecursive(text.Trim(), 0);
        var chunks = new List<string>();
        var current = new List<string>();
        int length = 0;

        foreach (var piece in pieces)
        {
            if (length + piece.Length > maxChars && current.Count > 0)
            {
                chunks.Add(string.Concat(current).Trim());

                // Carry the last few pieces forward as overlap.
                var carry = new List<string>();
                int carried = 0;
                for (int i = current.Count - 1; i >= 0; i--)
                {
                    if (carried + current[i].Length > overlapChars) break;
                    carry.Insert(0, current[i]);
                    carried += current[i].Length;
                }
                current = carry;
                length = carried;
            }
            current.Add(piece);
            length += piece.Length;
        }

        if (current.Count > 0) chunks.Add(string.Concat(current).Trim());
        return chunks.Where(c => c.Length > 0).ToList();
    }

    private List<string> SplitRecursive(string text, int level)
    {
        if (text.Length <= maxChars) return [text];

        if (level >= Separators.Length)                     // nothing left: hard cut
        {
            var hard = new List<string>();
            for (int i = 0; i < text.Length; i += maxChars)
                hard.Add(text.Substring(i, Math.Min(maxChars, text.Length - i)));
            return hard;
        }

        var sep = Separators[level];
        var parts = text.Split(sep);
        var result = new List<string>();
        for (int i = 0; i < parts.Length; i++)
        {
            var part = i < parts.Length - 1 ? parts[i] + sep : parts[i];   // keep the separator
            if (part.Length == 0) continue;
            result.AddRange(SplitRecursive(part, level + 1));
        }
        return result;
    }
}
