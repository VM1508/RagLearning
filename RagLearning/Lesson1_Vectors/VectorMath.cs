using System.Numerics;

namespace RagLearning.Lesson1_Vectors;

/// <summary>
/// LESSON 1 — the maths behind "semantic similarity".
///
/// An embedding is just float[]. Two texts are "similar" when their vectors point
/// in the same direction. Everything a vector database does is built on these functions.
/// </summary>
public static class VectorMath
{
    /// <summary>
    /// Dot product = Σ aᵢ·bᵢ. Uses SIMD (System.Numerics.Vector&lt;T&gt;) so it processes
    /// 4–16 floats per CPU instruction. This loop is the hot path of every vector search.
    /// </summary>
    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException($"Dimension mismatch: {a.Length} vs {b.Length}. " +
                                        "Vectors from different embedding models can never be compared.");

        float sum = 0f;
        int i = 0;
        int width = Vector<float>.Count;

        if (Vector.IsHardwareAccelerated)
        {
            for (; i <= a.Length - width; i += width)
                sum += Vector.Dot(new Vector<float>(a.Slice(i, width)), new Vector<float>(b.Slice(i, width)));
        }

        for (; i < a.Length; i++)          // leftover elements
            sum += a[i] * b[i];

        return sum;
    }

    /// <summary>L2 norm (length) = √(Σ aᵢ²).</summary>
    public static float Norm(ReadOnlySpan<float> a) => MathF.Sqrt(Dot(a, a));

    /// <summary>
    /// Scale to length 1. After this, Dot(a, b) == Cosine(a, b), so we normalize ONCE at
    /// write time and use the cheaper dot product at query time.
    /// </summary>
    public static float[] Normalize(ReadOnlySpan<float> a)
    {
        var result = a.ToArray();
        float norm = Norm(a);
        if (norm < 1e-12f) return result;   // zero vector: leave as-is
        for (int i = 0; i < result.Length; i++) result[i] /= norm;
        return result;
    }

    /// <summary>
    /// Cosine similarity = (a·b) / (|a|·|b|), in [-1, 1].
    /// 1 = same direction (same meaning), 0 = unrelated, -1 = opposite.
    /// </summary>
    public static float Cosine(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float denom = Norm(a) * Norm(b);
        return denom < 1e-12f ? 0f : Dot(a, b) / denom;
    }

    /// <summary>Euclidean (L2) distance — smaller is closer. Some indexes (FAISS IndexFlatL2) use this.</summary>
    public static float Euclidean(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length) throw new ArgumentException("Dimension mismatch");
        float sum = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            float d = a[i] - b[i];
            sum += d * d;
        }
        return MathF.Sqrt(sum);
    }
}
