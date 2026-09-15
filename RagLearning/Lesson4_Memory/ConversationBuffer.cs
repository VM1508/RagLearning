using RagLearning.Lesson3_Rag;

namespace RagLearning.Lesson4_Memory;

/// <summary>
/// SHORT-TERM memory: the running chat.
/// The context window is finite (and every token costs money), so:
///   • keep the last N messages word-for-word
///   • fold older messages into a running summary
/// Prompt = [system] + [summary of older chat] + [recent messages] + [new question]
/// </summary>
public sealed class ConversationBuffer(IConversationSummarizer summarizer, int keepRecentMessages = 6)
{
    private readonly List<ChatMessage> _recent = [];
    public string Summary { get; private set; } = "";
    public IReadOnlyList<ChatMessage> Recent => _recent;

    public async Task AddAsync(ChatMessage message, CancellationToken ct = default)
    {
        _recent.Add(message);
        if (_recent.Count <= keepRecentMessages) return;

        // Fold the oldest pair (user + assistant) into the summary.
        var overflow = _recent.Take(2).ToList();
        _recent.RemoveRange(0, overflow.Count);
        Summary = await summarizer.SummarizeAsync(Summary, overflow, ct);
    }

    public IEnumerable<ChatMessage> AsPromptMessages()
    {
        if (Summary.Length > 0)
            yield return ChatMessage.System($"Summary of earlier conversation: {Summary}");
        foreach (var m in _recent) yield return m;
    }
}

public interface IConversationSummarizer
{
    Task<string> SummarizeAsync(string previousSummary, IReadOnlyList<ChatMessage> newMessages, CancellationToken ct = default);
}

/// <summary>LLM "incremental summary": old summary + new turns → new summary (bounded length).</summary>
public sealed class LlmSummarizer(IChatModel model) : IConversationSummarizer
{
    public Task<string> SummarizeAsync(string previousSummary, IReadOnlyList<ChatMessage> newMessages, CancellationToken ct = default)
    {
        var turns = string.Join("\n", newMessages.Select(m => $"{m.Role}: {m.Content}"));
        return model.CompleteAsync(
        [
            ChatMessage.System("Update the running summary of a coaching chat. Keep facts, decisions and open questions. Max 80 words."),
            ChatMessage.User($"Current summary: {previousSummary}\n\nNew messages:\n{turns}")
        ], ct: ct);
    }
}

/// <summary>Offline: keeps a truncated list of what the user asked. Crude, but bounded.</summary>
public sealed class TruncatingSummarizer(int maxChars = 300) : IConversationSummarizer
{
    public Task<string> SummarizeAsync(string previousSummary, IReadOnlyList<ChatMessage> newMessages, CancellationToken ct = default)
    {
        var asked = newMessages.Where(m => m.Role == "user")
                               .Select(m => m.Content.Length > 60 ? m.Content[..60] + "…" : m.Content);
        var summary = string.Join(" | ", new[] { previousSummary }.Concat(asked).Where(s => s.Length > 0));
        return Task.FromResult(summary.Length > maxChars ? "…" + summary[^maxChars..] : summary);
    }
}
