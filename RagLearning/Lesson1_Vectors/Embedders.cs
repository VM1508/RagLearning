using RagLearning.Shared;

namespace RagLearning.Lesson1_Vectors;

/// <summary>
/// Anything that turns text into vectors. Keep this abstraction in your real app:
/// you WILL change embedding models, and every stored vector must then be re-embedded
/// (vectors from different models live in different spaces).
/// </summary>
public interface IEmbedder
{
    string Name { get; }
    int Dimensions { get; }
    Task<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken ct = default);
}

public static class EmbedderExtensions
{
    public static async Task<float[]> EmbedOneAsync(this IEmbedder e, string text, CancellationToken ct = default)
        => (await e.EmbedAsync([text], ct))[0];
}

/// <summary>
/// OFFLINE teaching embedder ("feature hashing").
/// Each word and each 3-letter chunk of a word is hashed to one of N slots.
/// Texts that share words / word-parts get overlapping vectors → high cosine.
///
/// It captures SPELLING overlap, not real meaning. A trained model (OpenAI, bge, e5)
/// knows that "hurts" ≈ "pain" without being told. To show the idea we add a tiny
/// hand-written concept map below — a real model learns millions of these.
/// </summary>
public sealed class HashingEmbedder(int dimensions = 512) : IEmbedder
{
    private static readonly Dictionary<string, string> Concepts = new()
    {
        ["hurt"] = "pain", ["ache"] = "pain", ["sore"] = "pain", ["injury"] = "pain", ["injur"] = "pain",
        ["workout"] = "train", ["session"] = "train", ["lift"] = "train", ["exercise"] = "train",
        ["heavier"] = "strength", ["stronger"] = "strength", ["progress"] = "strength", ["1rm"] = "strength",
        ["sleep"] = "recovery", ["rest"] = "recovery", ["deload"] = "recovery",
    };

    public string Name => $"offline-hashing-{dimensions}";
    public int Dimensions => dimensions;

    public Task<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken ct = default)
        => Task.FromResult(texts.Select(Embed).ToArray());

    private float[] Embed(string text)
    {
        var v = new float[dimensions];
        foreach (var token in TextUtil.Tokens(text))
        {
            AddFeature(v, "w:" + token, 1.0f);                      // whole word

            var padded = $"#{token}#";                              // char trigrams: "#sq","squ","qua",...
            for (int i = 0; i + 3 <= padded.Length; i++)
                AddFeature(v, "c:" + padded.Substring(i, 3), 0.3f);

            if (Concepts.TryGetValue(token, out var concept))       // fake "semantics"
                AddFeature(v, "k:" + concept, 1.2f);
            if (Concepts.ContainsValue(token))
                AddFeature(v, "k:" + token, 1.2f);
        }
        return VectorMath.Normalize(v);
    }

    private void AddFeature(float[] v, string feature, float weight)
    {
        uint h = TextUtil.Fnv1a(feature);
        int index = (int)(h % (uint)v.Length);
        float sign = (h >> 31) == 0 ? 1f : -1f;   // signed hashing keeps collisions from only adding up
        v[index] += sign * weight;
    }
}

/// <summary>
/// Real embeddings via POST /embeddings (OpenAI or Azure OpenAI).
/// Batches inputs because each HTTP call has overhead and a per-request input limit.
/// </summary>
public sealed class OpenAiEmbedder(OpenAiHttp http, OpenAiOptions options, int batchSize = 64) : IEmbedder
{
    public string Name => options.EmbeddingModel;
    public int Dimensions => options.EmbeddingDimensions;

    public async Task<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken ct = default)
    {
        var result = new float[texts.Count][];

        for (int start = 0; start < texts.Count; start += batchSize)
        {
            var batch = texts.Skip(start).Take(batchSize).ToArray();
            using var doc = await http.PostAsync("embeddings",
                new { model = options.EmbeddingModel, input = batch }, ct);

            foreach (var item in doc.RootElement.GetProperty("data").EnumerateArray())
            {
                int index = item.GetProperty("index").GetInt32();       // results may come back unordered
                var vector = item.GetProperty("embedding").EnumerateArray()
                                 .Select(x => x.GetSingle()).ToArray();
                result[start + index] = VectorMath.Normalize(vector);  // OpenAI already normalizes; harmless
            }
        }
        return result;
    }
}
