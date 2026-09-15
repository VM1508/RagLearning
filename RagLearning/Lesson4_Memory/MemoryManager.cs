using RagLearning.Lesson1_Vectors;

namespace RagLearning.Lesson4_Memory;

/// <summary>
/// Long-term memory = a vector store you READ AND WRITE, scoped per user.
///
/// WRITE PATH                                   READ PATH
///   message                                      query
///   → extract candidates (LLM / rules)           → embed
///   → for each: find existing by key/similarity  → score = w1·relevance + w2·recency + w3·importance
///   → ADD | UPDATE | DELETE | NOOP               → pin procedural + critical memories
///   → embed + store                              → top-k, mark as accessed
///
/// The ADD/UPDATE/DELETE decision is the part most tutorials skip. Without it you get
/// "weighs 80 kg" and "weighs 76 kg" side by side and the model picks one at random.
/// (Mem0 does the same decision with an LLM call; here it's deterministic so you can follow it.)
/// </summary>
public sealed class MemoryManager(
    IEmbedder embedder,
    IMemoryExtractor extractor,
    TimeProvider clock,
    MemoryOptions? options = null)
{
    private readonly MemoryOptions _o = options ?? new MemoryOptions();
    private readonly List<MemoryItem> _items = [];
    private readonly SemaphoreSlim _lock = new(1, 1);   // writes for one user must be serialized

    public IReadOnlyList<MemoryItem> All(string userId) =>
        _items.Where(m => m.UserId == userId).OrderBy(m => m.Kind).ThenBy(m => m.CreatedAt).ToList();

    // ─────────────────────────────── WRITE PATH ───────────────────────────────

    public async Task<IReadOnlyList<MemoryOperation>> WriteAsync(string userId, string message, CancellationToken ct = default)
    {
        var now = clock.GetUtcNow();
        var candidates = await extractor.ExtractAsync(message, DateOnly.FromDateTime(now.UtcDateTime), ct);
        if (candidates.Count == 0) return [];

        // Embed all candidate texts in one call (delete-by-key candidates may have empty text).
        var vectors = await embedder.EmbedAsync(
            candidates.Select(c => string.IsNullOrWhiteSpace(c.Text) ? c.Key ?? "-" : c.Text).ToArray(), ct);

        var ops = new List<MemoryOperation>();
        await _lock.WaitAsync(ct);
        try
        {
            for (int i = 0; i < candidates.Count; i++)
                ops.Add(Apply(userId, candidates[i], vectors[i], now));
        }
        finally
        {
            _lock.Release();
        }
        return ops;
    }

    private MemoryOperation Apply(string userId, MemoryCandidate c, float[] vector, DateTimeOffset now)
    {
        var mine = _items.Where(m => m.UserId == userId).ToList();   // tenant isolation

        // DELETE ─ by key if we have one, otherwise by meaning
        if (c.Action == MemoryAction.Delete)
        {
            var target = c.Key is not null
                ? mine.FirstOrDefault(m => m.Key == c.Key)
                : Nearest(mine, vector, kind: null, out var sim) is { } n && sim >= _o.DeleteMatchThreshold ? n : null;

            if (target is null) return new(MemoryOpType.Noop, c.Text, "nothing matching to forget");
            _items.Remove(target);
            return new(MemoryOpType.Delete, target.Text, c.Key is not null ? $"key '{c.Key}' no longer true" : "user asked to forget");
        }

        // UPDATE ─ same slot, new value (keep the old one in History)
        if (c.Key is not null && mine.FirstOrDefault(m => m.Key == c.Key) is { } existing)
        {
            if (string.Equals(existing.Text, c.Text, StringComparison.OrdinalIgnoreCase))
            {
                existing.LastAccessedAt = now;
                return new(MemoryOpType.Noop, c.Text, "already known");
            }
            existing.History.Add($"{existing.UpdatedAt:yyyy-MM-dd}: {existing.Text}");
            existing.Text = c.Text;
            existing.Vector = vector;
            existing.Importance = Math.Max(existing.Importance, c.Importance);
            existing.UpdatedAt = existing.LastAccessedAt = now;
            return new(MemoryOpType.Update, c.Text, $"replaced '{existing.History[^1]}'");
        }

        // NOOP ─ semantically a duplicate of something we already have
        if (Nearest(mine, vector, c.Kind, out var similarity) is { } dup && similarity >= _o.DuplicateThreshold)
        {
            dup.LastAccessedAt = now;
            return new(MemoryOpType.Noop, c.Text, $"duplicate of '{dup.Text}' (cos={similarity:F2})");
        }

        // ADD
        _items.Add(new MemoryItem
        {
            Id = Guid.NewGuid().ToString("N"),
            UserId = userId,
            Kind = c.Kind,
            Key = c.Key,
            Text = c.Text,
            Vector = vector,
            Importance = c.Importance,
            CreatedAt = now,
            UpdatedAt = now,
            LastAccessedAt = now,
        });
        return new(MemoryOpType.Add, c.Text, "new fact");
    }

    private static MemoryItem? Nearest(IEnumerable<MemoryItem> items, float[] vector, MemoryKind? kind, out double similarity)
    {
        MemoryItem? best = null;
        similarity = double.MinValue;
        foreach (var m in items)
        {
            if (kind is not null && m.Kind != kind) continue;
            double s = VectorMath.Dot(vector, m.Vector);
            if (s > similarity) { similarity = s; best = m; }
        }
        return best;
    }

    // ─────────────────────────────── READ PATH ────────────────────────────────

    public async Task<IReadOnlyList<ScoredMemory>> RecallAsync(string userId, string query, int k = 5, CancellationToken ct = default)
    {
        var now = clock.GetUtcNow();
        var q = await embedder.EmbedOneAsync(query, ct);

        await _lock.WaitAsync(ct);
        try
        {
            var scored = _items.Where(m => m.UserId == userId).Select(m =>
            {
                double relevance = Math.Max(0, VectorMath.Dot(q, m.Vector));
                double recency = Recency(m, now);
                double score = _o.WeightRelevance * relevance
                             + _o.WeightRecency * recency
                             + _o.WeightImportance * m.Importance;
                return new ScoredMemory(m, score, relevance, recency);
            }).ToList();

            // Procedural memories are rules → always in the prompt.
            var pinned = scored.Where(s => s.Item.Kind == MemoryKind.Procedural).ToList();

            // Everything else must be relevant — unless it is critical (an injury must never be "forgotten").
            var ranked = scored
                .Where(s => s.Item.Kind != MemoryKind.Procedural)
                .Where(s => s.Relevance >= _o.MinRelevance || s.Item.Importance >= _o.PinImportance)
                .OrderByDescending(s => s.Score)
                .Take(k);

            var result = pinned.Concat(ranked).ToList();
            foreach (var r in result) r.Item.LastAccessedAt = now;   // "use it or lose it"
            return result;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>Exponential decay with a half-life that depends on the memory type.</summary>
    private static double Recency(MemoryItem m, DateTimeOffset now)
    {
        double halfLifeHours = m.Kind switch
        {
            MemoryKind.Episodic => 72,          // events fade in days
            MemoryKind.Semantic => 24 * 60,     // facts fade in months
            _ => double.PositiveInfinity,       // rules don't fade
        };
        double ageHours = Math.Max(0, (now - m.LastAccessedAt).TotalHours);
        return double.IsInfinity(halfLifeHours) ? 1.0 : Math.Pow(0.5, ageHours / halfLifeHours);
    }
}
