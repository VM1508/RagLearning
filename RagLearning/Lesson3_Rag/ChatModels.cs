using System.Text.RegularExpressions;
using RagLearning.Shared;

namespace RagLearning.Lesson3_Rag;

public sealed record ChatMessage(string Role, string Content)
{
    public static ChatMessage System(string c) => new("system", c);
    public static ChatMessage User(string c) => new("user", c);
    public static ChatMessage Assistant(string c) => new("assistant", c);
}

public interface IChatModel
{
    string Name { get; }
    Task<string> CompleteAsync(IReadOnlyList<ChatMessage> messages, bool jsonMode = false, CancellationToken ct = default);
}

/// <summary>POST /chat/completions against OpenAI or Azure OpenAI.</summary>
public sealed class OpenAiChatModel(OpenAiHttp http, OpenAiOptions options) : IChatModel
{
    public string Name => options.ChatModel;

    public async Task<string> CompleteAsync(IReadOnlyList<ChatMessage> messages, bool jsonMode = false, CancellationToken ct = default)
    {
        var body = new Dictionary<string, object>
        {
            ["model"] = options.ChatModel,
            ["temperature"] = 0,          // RAG wants faithful, repeatable answers, not creativity
            ["messages"] = messages.Select(m => new { role = m.Role, content = m.Content }).ToArray(),
        };
        if (jsonMode) body["response_format"] = new { type = "json_object" };

        using var doc = await http.PostAsync("chat/completions", body, ct);
        return doc.RootElement.GetProperty("choices")[0]
                  .GetProperty("message").GetProperty("content").GetString() ?? "";
    }
}

/// <summary>
/// OFFLINE stand-in for an LLM so the whole pipeline runs without keys.
/// It is "extractive": it picks the source sentences that best overlap the question and cites them.
/// A real LLM would synthesize an answer — but the plumbing around it is identical.
/// </summary>
public sealed partial class OfflineExtractiveChatModel : IChatModel
{
    public string Name => "offline-extractive";

    [GeneratedRegex("<source id=\"(\\d+)\"[^>]*>(.*?)</source>", RegexOptions.Singleline)]
    private static partial Regex SourceRegex();

    [GeneratedRegex("<tool_result>(.*?)</tool_result>", RegexOptions.Singleline)]
    private static partial Regex ToolRegex();

    [GeneratedRegex("<user_memory>(.*?)</user_memory>", RegexOptions.Singleline)]
    private static partial Regex MemoryRegex();

    public Task<string> CompleteAsync(IReadOnlyList<ChatMessage> messages, bool jsonMode = false, CancellationToken ct = default)
    {
        var prompt = messages.Last(m => m.Role == "user").Content;
        var questionIndex = prompt.LastIndexOf("Question:", StringComparison.Ordinal);
        var question = questionIndex >= 0 ? prompt[(questionIndex + 9)..] : prompt;
        var qTerms = TextUtil.Tokens(question).ToHashSet();

        var answer = new List<string>();

        if (ToolRegex().Match(prompt) is { Success: true } tool)
            answer.Add("From your training log: " + tool.Groups[1].Value.Trim());

        var memory = MemoryRegex().Match(prompt);
        if (memory.Success && memory.Groups[1].Value.Contains("injur", StringComparison.OrdinalIgnoreCase))
            answer.Add("Keeping your noted injury in mind.");

        var best = SourceRegex().Matches(prompt)
            .SelectMany(m => TextUtil.Sentences(m.Groups[2].Value)
                .Skip(1)                                            // skip "Title > Section" header line
                .Select(s => (Id: m.Groups[1].Value, Sentence: s,
                              Score: TextUtil.Tokens(s).Distinct().Count(qTerms.Contains))))
            .Where(x => x.Score > 0)
            .OrderByDescending(x => x.Score)
            .Take(2)
            .Select(x => $"{x.Sentence} [{x.Id}]");

        answer.AddRange(best);

        return Task.FromResult(answer.Count == 0
            ? "I don't know based on the provided sources."
            : string.Join(" ", answer));
    }
}
