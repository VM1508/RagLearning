using RagLearning.Lesson1_Vectors;

namespace RagLearning.Lesson2_VectorStore;

/// <summary>
/// LESSON 2b — HNSW (Hierarchical Navigable Small World), the ANN index behind
/// pgvector, Qdrant, Azure AI Search, Elasticsearch, Redis...
///
/// Mental model: a skip list made of graphs.
///   • Every vector is a node linked to its ~M nearest neighbours.
///   • Each node also gets a random "level". Few nodes reach high levels, so upper
///     layers are sparse "highways"; layer 0 contains everyone.
///   • Search: start at the top entry point, greedily hop to whichever neighbour is
///     closer to the query, drop a layer, repeat. At layer 0 run a wider beam search (ef).
///
/// Knobs:  M (links per node)      ↑ recall, ↑ memory
///         efConstruction          ↑ graph quality, ↑ build time
///         efSearch                ↑ recall, ↑ latency   ← the one you tune at query time
///
/// Simplified for learning: no deletes (real engines use tombstones + rebuilds) and no
/// neighbour-diversity heuristic. Vectors must be normalized (similarity = dot product).
/// </summary>
public sealed class HnswIndex
{
    private readonly int _m;
    private readonly int _efConstruction;
    private readonly double _levelMultiplier;
    private readonly Random _rng;

    private readonly List<float[]> _vectors = [];
    private readonly List<List<int>[]> _links = [];   // _links[node][level] = neighbour ids
    private int _entryPoint = -1;
    private int _topLevel = -1;

    public HnswIndex(int m = 16, int efConstruction = 100, int seed = 42)
    {
        _m = m;
        _efConstruction = efConstruction;
        _levelMultiplier = 1.0 / Math.Log(m);
        _rng = new Random(seed);
    }

    public int Count => _vectors.Count;

    public int Add(float[] vector)
    {
        int id = _vectors.Count;
        _vectors.Add(vector);

        // Exponentially decaying level distribution: P(level ≥ L) = M^-L
        int level = (int)Math.Floor(-Math.Log(1.0 - _rng.NextDouble()) * _levelMultiplier);
        var links = new List<int>[level + 1];
        for (int l = 0; l <= level; l++) links[l] = [];
        _links.Add(links);

        if (_entryPoint < 0)
        {
            _entryPoint = id;
            _topLevel = level;
            return id;
        }

        // 1) Zoom down through the layers above this node's level (greedy, beam = 1).
        int ep = _entryPoint;
        for (int l = _topLevel; l > level; l--)
            ep = GreedyClosest(vector, ep, l);

        // 2) On each layer the node lives in: find neighbours and link both ways.
        for (int l = Math.Min(level, _topLevel); l >= 0; l--)
        {
            var candidates = SearchLayer(vector, ep, _efConstruction, l);
            int maxLinks = l == 0 ? _m * 2 : _m;           // layer 0 is denser

            foreach (var (neighbour, _) in candidates.Take(_m))
            {
                links[l].Add(neighbour);
                var reverse = _links[neighbour][l];
                reverse.Add(id);
                if (reverse.Count > maxLinks) Prune(neighbour, l, maxLinks);
            }
            ep = candidates[0].Id;
        }

        if (level > _topLevel)
        {
            _topLevel = level;
            _entryPoint = id;
        }
        return id;
    }

    public IReadOnlyList<(int Id, float Score)> Search(float[] query, int k, int efSearch = 50)
    {
        if (_entryPoint < 0) return [];

        int ep = _entryPoint;
        for (int l = _topLevel; l > 0; l--)
            ep = GreedyClosest(query, ep, l);

        return SearchLayer(query, ep, Math.Max(efSearch, k), 0).Take(k).ToList();
    }

    private float Sim(int node, float[] query) => VectorMath.Dot(_vectors[node], query);

    private int GreedyClosest(float[] query, int ep, int level)
    {
        float best = Sim(ep, query);
        bool improved = true;
        while (improved)
        {
            improved = false;
            foreach (int n in _links[ep][level])
            {
                float s = Sim(n, query);
                if (s > best)
                {
                    best = s;
                    ep = n;
                    improved = true;
                    break;              // restart from the new, closer node
                }
            }
        }
        return ep;
    }

    /// <summary>Beam search on one layer. Returns up to ef nodes, best first.</summary>
    private List<(int Id, float Score)> SearchLayer(float[] query, int entry, int ef, int level)
    {
        var visited = new HashSet<int> { entry };
        float entryScore = Sim(entry, query);

        var toExplore = new PriorityQueue<int, float>();   // best first (negated priority)
        var best = new PriorityQueue<int, float>();        // worst on top, capped at ef
        toExplore.Enqueue(entry, -entryScore);
        best.Enqueue(entry, entryScore);

        while (toExplore.TryDequeue(out int current, out float negScore))
        {
            best.TryPeek(out _, out float worstKept);
            if (-negScore < worstKept && best.Count >= ef) break;   // can't improve any more

            foreach (int n in _links[current][level])
            {
                if (!visited.Add(n)) continue;
                float s = Sim(n, query);
                best.TryPeek(out _, out worstKept);
                if (best.Count < ef || s > worstKept)
                {
                    toExplore.Enqueue(n, -s);
                    best.Enqueue(n, s);
                    if (best.Count > ef) best.Dequeue();
                }
            }
        }

        var result = new List<(int, float)>(best.Count);
        while (best.TryDequeue(out int id, out float s)) result.Add((id, s));
        result.Reverse();
        return result;
    }

    private void Prune(int node, int level, int maxLinks)
    {
        var v = _vectors[node];
        _links[node][level] = _links[node][level]
            .Distinct()
            .OrderByDescending(n => VectorMath.Dot(_vectors[n], v))
            .Take(maxLinks)
            .ToList();
    }
}
