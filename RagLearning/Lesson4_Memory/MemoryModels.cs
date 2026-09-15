namespace RagLearning.Lesson4_Memory;

/// <summary>
/// LESSON 4 — Agent memory.
///
/// Semantic   : facts that stay true for a while   ("goal: 180kg deadlift", "trains mornings")
/// Episodic   : things that happened, with a date  ("2026-09-14: skipped legs, felt tired")
/// Procedural : how the agent should behave        ("keep replies short")
/// Short-term memory (the chat itself) lives in ConversationBuffer.
/// </summary>
public enum MemoryKind { Semantic, Episodic, Procedural }

public enum MemoryAction { Upsert, Delete }

/// <summary>What the extractor proposes. The MemoryManager decides what actually happens.</summary>
public sealed record MemoryCandidate(
    MemoryKind Kind,
    string? Key,              // stable "slot" (body_weight, goal, injury:knee). Null = free-form fact.
    string Text,
    double Importance,        // 0..1 — how bad is it to forget this?
    MemoryAction Action = MemoryAction.Upsert);

public sealed class MemoryItem
{
    public required string Id { get; init; }
    public required string UserId { get; init; }
    public required MemoryKind Kind { get; init; }
    public string? Key { get; init; }
    public required string Text { get; set; }
    public required float[] Vector { get; set; }
    public double Importance { get; set; }
    public DateTimeOffset CreatedAt { get; init; }
    public DateTimeOffset UpdatedAt { get; set; }
    public DateTimeOffset LastAccessedAt { get; set; }
    public List<string> History { get; } = [];   // previous values — audit trail for updates

    public override string ToString() => $"[{Kind}{(Key is null ? "" : $":{Key}")}] {Text}";
}

public enum MemoryOpType { Add, Update, Delete, Noop }

public sealed record MemoryOperation(MemoryOpType Type, string Text, string Reason);

public sealed record ScoredMemory(MemoryItem Item, double Score, double Relevance, double Recency);

public sealed record MemoryOptions(
    double DuplicateThreshold = 0.92,   // cosine above this = same fact, don't store twice
    double DeleteMatchThreshold = 0.35, // "forget X": how similar a memory must be to X
    double MinRelevance = 0.15,         // below this a memory is noise for this query...
    double PinImportance = 0.9,         // ...unless it is this important (e.g. injuries)
    double WeightRelevance = 0.6,
    double WeightRecency = 0.2,
    double WeightImportance = 0.2);
