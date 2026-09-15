using System.Text.Json;
using System.Text.RegularExpressions;
using RagLearning.Lesson3_Rag;
using RagLearning.Shared;

namespace RagLearning.Lesson4_Memory;

/// <summary>Step 1 of the write path: turn a raw message into candidate memories.</summary>
public interface IMemoryExtractor
{
    Task<IReadOnlyList<MemoryCandidate>> ExtractAsync(string message, DateOnly today, CancellationToken ct = default);
}

/// <summary>
/// Production approach: let the LLM extract structured facts as JSON.
/// Tips: give it a closed list of keys, ask for importance, and tell it to ignore chit-chat.
/// </summary>
public sealed class LlmMemoryExtractor(IChatModel model) : IMemoryExtractor
{
    private const string Instructions = """
        Extract durable facts about the USER from their message for a fitness coaching app.
        Ignore questions, greetings and anything not about the user.
        kind: "semantic" (lasting fact), "episodic" (dated event), "procedural" (how the coach should behave).
        key: one of body_weight, goal, training_days, injury:<body_part>, reply_style — or null for other facts.
        importance: 0.0–1.0 (injuries/goals high, one-off events low).
        action: "upsert", or "delete" when the user says something is no longer true / asks to forget it.
        Prefix episodic text with the date.
        Return JSON only: {"memories":[{"kind":"...","key":null,"text":"...","importance":0.5,"action":"upsert"}]}
        """;

    public async Task<IReadOnlyList<MemoryCandidate>> ExtractAsync(string message, DateOnly today, CancellationToken ct = default)
    {
        var json = await model.CompleteAsync(
            [ChatMessage.System(Instructions), ChatMessage.User($"Today is {today:yyyy-MM-dd}.\nMessage: {message}")],
            jsonMode: true, ct);

        try
        {
            using var doc = JsonDocument.Parse(json);
            return doc.RootElement.GetProperty("memories").EnumerateArray().Select(m => new MemoryCandidate(
                Kind: Enum.Parse<MemoryKind>(m.GetProperty("kind").GetString()!, ignoreCase: true),
                Key: m.TryGetProperty("key", out var k) && k.ValueKind == JsonValueKind.String ? k.GetString() : null,
                Text: m.GetProperty("text").GetString()!,
                Importance: Math.Clamp(m.GetProperty("importance").GetDouble(), 0, 1),
                Action: m.TryGetProperty("action", out var a) && a.GetString() == "delete"
                    ? MemoryAction.Delete : MemoryAction.Upsert)).ToList();
        }
        catch (Exception ex) when (ex is JsonException or KeyNotFoundException or ArgumentException or InvalidOperationException)
        {
            return [];   // bad JSON must never crash the chat — just remember nothing this turn
        }
    }
}

/// <summary>Offline, rule-based extractor. Brittle by design — it shows WHY people use an LLM here.</summary>
public sealed partial class RuleBasedMemoryExtractor : IMemoryExtractor
{
    [GeneratedRegex(@"\bI (?:now )?weigh (\d+(?:\.\d+)?)\s*kg", RegexOptions.IgnoreCase)]
    private static partial Regex Weight();
    [GeneratedRegex(@"\bmy goal is (?:to )?(.+)", RegexOptions.IgnoreCase)]
    private static partial Regex Goal();
    [GeneratedRegex(@"\b(?:hurt|injured|strained|tweaked) my (\w+)|\bmy (\w+) (?:hurts|is injured|is sore)", RegexOptions.IgnoreCase)]
    private static partial Regex Injury();
    [GeneratedRegex(@"\bmy (\w+) (?:is|has|feels) (?:healed|fine|recovered|better)", RegexOptions.IgnoreCase)]
    private static partial Regex Healed();
    [GeneratedRegex(@"\bI prefer (.+)", RegexOptions.IgnoreCase)]
    private static partial Regex Preference();
    [GeneratedRegex(@"\b(?:keep|make) (?:your |the )?(?:answers|replies|responses) (short|brief|detailed|concise)", RegexOptions.IgnoreCase)]
    private static partial Regex ReplyStyle();
    [GeneratedRegex(@"\b(today|yesterday|this morning|last night),? I (.+)", RegexOptions.IgnoreCase)]
    private static partial Regex Event();
    [GeneratedRegex(@"\bforget (?:that |about )?(.+)", RegexOptions.IgnoreCase)]
    private static partial Regex Forget();

    public Task<IReadOnlyList<MemoryCandidate>> ExtractAsync(string message, DateOnly today, CancellationToken ct = default)
    {
        var result = new List<MemoryCandidate>();

        foreach (var sentence in TextUtil.Sentences(message))
        {
            string Clean(string s) => s.Trim().TrimEnd('.', '!', '?');

            if (Forget().Match(sentence) is { Success: true } f)
            {
                result.Add(new(MemoryKind.Semantic, null, Clean(f.Groups[1].Value), 0, MemoryAction.Delete));
                continue;
            }
            if (Healed().Match(sentence) is { Success: true } h)
                result.Add(new(MemoryKind.Semantic, $"injury:{h.Groups[1].Value.ToLowerInvariant()}", "", 0, MemoryAction.Delete));
            if (Weight().Match(sentence) is { Success: true } w)
                result.Add(new(MemoryKind.Semantic, "body_weight", $"Body weight is {w.Groups[1].Value} kg", 0.6));
            if (Goal().Match(sentence) is { Success: true } g)
                result.Add(new(MemoryKind.Semantic, "goal", $"Goal: {Clean(g.Groups[1].Value)}", 0.9));
            if (Injury().Match(sentence) is { Success: true } i)
            {
                var part = (i.Groups[1].Success ? i.Groups[1].Value : i.Groups[2].Value).ToLowerInvariant();
                result.Add(new(MemoryKind.Semantic, $"injury:{part}", $"Has an injured {part} (pain); avoid aggravating it", 0.95));
            }
            if (Preference().Match(sentence) is { Success: true } p)
                result.Add(new(MemoryKind.Semantic, null, $"Prefers {Clean(p.Groups[1].Value)}", 0.5));
            if (ReplyStyle().Match(sentence) is { Success: true } r)
                result.Add(new(MemoryKind.Procedural, "reply_style", $"Keep replies {r.Groups[1].Value.ToLowerInvariant()}", 0.8));
            if (Event().Match(sentence) is { Success: true } e)
            {
                var date = e.Groups[1].Value.Equals("yesterday", StringComparison.OrdinalIgnoreCase) || e.Groups[1].Value.StartsWith("last", StringComparison.OrdinalIgnoreCase)
                    ? today.AddDays(-1) : today;
                result.Add(new(MemoryKind.Episodic, null, $"{date:yyyy-MM-dd}: {Clean(e.Groups[2].Value)}", 0.3));
            }
        }

        return Task.FromResult<IReadOnlyList<MemoryCandidate>>(result);
    }
}
