using System.Diagnostics;
using RagLearning.Lesson1_Vectors;
using RagLearning.Lesson2_VectorStore;
using RagLearning.Lesson3_Rag;
using RagLearning.Lesson4_Memory;
using RagLearning.Lesson5_Evaluation;
using RagLearning.Shared;

// Usage:  dotnet run            → all lessons
//         dotnet run -- 3       → only lesson 3
var lesson = args.FirstOrDefault() ?? "all";

// ── Composition root: real models if configured, offline teaching models otherwise ──
var options = OpenAiOptions.FromEnvironment();
IEmbedder embedder;
IChatModel chatModel;
IReranker reranker;
IMemoryExtractor extractor;
IConversationSummarizer summarizer;

if (options is not null)
{
    var http = new OpenAiHttp(options);
    embedder = new OpenAiEmbedder(http, options);
    chatModel = new OpenAiChatModel(http, options);
    reranker = new LlmReranker(chatModel);
    extractor = new LlmMemoryExtractor(chatModel);
    summarizer = new LlmSummarizer(chatModel);
    Console.WriteLine($"Using {(options.IsAzure ? "Azure OpenAI" : "OpenAI")}: {options.EmbeddingModel} + {options.ChatModel}");
}
else
{
    embedder = new HashingEmbedder();
    chatModel = new OfflineExtractiveChatModel();
    reranker = new TermCoverageReranker();
    extractor = new RuleBasedMemoryExtractor();
    summarizer = new TruncatingSummarizer();
    Console.WriteLine("No API key found → running OFFLINE with teaching implementations.");
}

if (lesson is "1" or "all") await Lesson1();
if (lesson is "2" or "all") await Lesson2();

// Lessons 3–5 share one knowledge base.
var vectors = new InMemoryVectorStore();
var bm25 = new Bm25Index();
var searcher = new HybridSearcher(embedder, vectors, bm25);
if (lesson is "3" or "4" or "5" or "all") await IngestKnowledgeBase();
var rag = new RagService(searcher, reranker, chatModel);

if (lesson is "3" or "all") await Lesson3();
if (lesson is "4" or "all") await Lesson4();
if (lesson is "5" or "all") await Lesson5();

// ════════════════════════════════════════════════════════════════════════════════

async Task Lesson1()
{
    Header("LESSON 1 — Vectors & similarity");

    float[] a = [1, 2, 3], b = [2, 4, 6], c = [-1, 0, 1];
    Console.WriteLine($"cos(a, 2a) = {VectorMath.Cosine(a, b):F3}   ← same direction, length ignored");
    Console.WriteLine($"cos(a, c)  = {VectorMath.Cosine(a, c):F3}");
    Console.WriteLine($"dot(a, 2a) = {VectorMath.Dot(a, b):F1}  vs dot(norm a, norm 2a) = {VectorMath.Dot(VectorMath.Normalize(a), VectorMath.Normalize(b)):F3}");
    Console.WriteLine($"L2(a, 2a)  = {VectorMath.Euclidean(a, b):F3}");

    string[] sentences =
    [
        "my knee hurts when I squat",
        "knee pain during squats",
        "how much should I sleep to recover",
        "best way to deload after a hard training block",
    ];
    var vecs = await embedder.EmbedAsync(sentences);
    Console.WriteLine($"\nEmbedder: {embedder.Name}, dims = {vecs[0].Length}");
    Console.WriteLine("Cosine similarity matrix:");
    for (int i = 0; i < sentences.Length; i++)
    {
        Console.Write($"  {sentences[i],-48}");
        for (int j = 0; j < sentences.Length; j++) Console.Write($"{VectorMath.Dot(vecs[i], vecs[j]),6:F2}");
        Console.WriteLine();
    }
    Console.WriteLine("→ Rows 1 & 2 score high although 'hurts' ≠ 'pain' as strings. Rows 3 & 4 share no words at all,");
    Console.WriteLine("  yet are related (both about recovery). A trained model learns such links; here a tiny concept map fakes it.");
}

async Task Lesson2()
{
    Header("LESSON 2 — Vector store, HNSW, BM25, hybrid");

    // 2a: filtered search in the flat store
    var store = new InMemoryVectorStore();
    string[] notes = ["user A: knee pain on squats", "user A: prefers morning sessions", "user B: knee pain on squats"];
    var noteVecs = await embedder.EmbedAsync(notes);
    for (int i = 0; i < notes.Length; i++)
        store.Upsert(new VectorRecord($"n{i}", notes[i], noteVecs[i],
            new Dictionary<string, string> { ["user"] = notes[i][5..6] }));

    var q = await embedder.EmbedOneAsync("knee hurts");
    Console.WriteLine("Search 'knee hurts' with filter user == A:");
    foreach (var h in store.Search(q, 3, Filters.Eq("user", "A")))
        Console.WriteLine($"  {h.Score:F3}  {h.Record.Text}");

    // 2b: HNSW vs brute force on random vectors
    const int n = 5000, dims = 64, queries = 200, k = 10;
    var rng = new Random(7);
    float[] RandomUnit() => VectorMath.Normalize(Enumerable.Range(0, dims).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray());
    var data = Enumerable.Range(0, n).Select(_ => RandomUnit()).ToArray();
    var qs = Enumerable.Range(0, queries).Select(_ => RandomUnit()).ToArray();

    var sw = Stopwatch.StartNew();
    var hnsw = new HnswIndex(m: 16, efConstruction: 100);
    foreach (var v in data) hnsw.Add(v);
    Console.WriteLine($"\nBuilt HNSW over {n} × {dims}-dim vectors in {sw.ElapsedMilliseconds} ms");

    var flat = new InMemoryVectorStore();
    for (int i = 0; i < n; i++) flat.Upsert(new VectorRecord(i.ToString(), "", data[i], new Dictionary<string, string>()));

    sw.Restart();
    var exact = qs.Select(x => flat.Search(x, k).Select(h => int.Parse(h.Record.Id)).ToHashSet()).ToArray();
    var flatMs = sw.Elapsed.TotalMilliseconds / queries;

    foreach (var ef in new[] { 10, 50, 200 })
    {
        sw.Restart();
        var approx = qs.Select(x => hnsw.Search(x, k, ef).Select(r => r.Id).ToArray()).ToArray();
        var ms = sw.Elapsed.TotalMilliseconds / queries;
        double recall = approx.Select((ids, i) => ids.Count(exact[i].Contains) / (double)k).Average();
        Console.WriteLine($"  efSearch={ef,3}: recall@{k} = {recall:P1}, {ms:F3} ms/query   (flat: 100%, {flatMs:F3} ms/query)");
    }
    Console.WriteLine("→ Higher efSearch = better recall, slower. The gap in speed grows with n.");

    // 2c/d: keyword vs vector vs hybrid
    var kv = new InMemoryVectorStore();
    var kb = new Bm25Index();
    string[] docs =
    [
        "The RDL (Romanian deadlift) targets hamstrings.",
        "Muscle ache the day after training is normal.",
        "Lower back pain can come from poor deadlift form.",
    ];
    var dv = await embedder.EmbedAsync(docs);
    for (int i = 0; i < docs.Length; i++)
    {
        kv.Upsert(new VectorRecord($"d{i}", docs[i], dv[i], new Dictionary<string, string> { ["docId"] = $"d{i}" }));
        kb.Upsert($"d{i}", docs[i]);
    }
    var hs = new HybridSearcher(embedder, kv, kb);
    Console.WriteLine();
    foreach (var query in new[] { "RDL", "sore legs" })
        foreach (var mode in Enum.GetValues<HybridSearcher.Mode>())
        {
            var top = await hs.SearchAsync(query, 1, mode: mode);
            Console.WriteLine($"  {mode,-8} top hit for '{query}': {top.FirstOrDefault()?.Record.Text ?? "(no match)"}");
        }
    Console.WriteLine("→ 'sore legs' shares no word with 'Muscle ache…' → BM25 finds nothing; vectors do.");
    Console.WriteLine("  With real models the reverse also happens: rare codes/acronyms embed poorly, BM25 nails them.");
    Console.WriteLine("  Hybrid gets both.");
}

async Task IngestKnowledgeBase()
{
    var ingestion = new IngestionPipeline(embedder, vectors, bm25, new RecursiveChunker(maxChars: 400, overlapChars: 60));
    var dir = Path.Combine(AppContext.BaseDirectory, "Data", "knowledge");
    int total = 0;
    foreach (var file in Directory.GetFiles(dir, "*.md").Order())
    {
        var text = await File.ReadAllTextAsync(file);
        var title = text.Split('\n')[0].TrimStart('#', ' ').Trim();
        total += await ingestion.IngestMarkdownAsync(Path.GetFileNameWithoutExtension(file), title, text,
            new Dictionary<string, string> { ["source"] = Path.GetFileName(file) });
    }
    // Idempotency check: second run embeds nothing.
    var again = await ingestion.IngestMarkdownAsync("squat", "Back Squat", await File.ReadAllTextAsync(Path.Combine(dir, "squat.md")));
    if (lesson is "3" or "all")
        Console.WriteLine($"\n[ingestion] {total} chunks indexed; re-ingesting unchanged squat.md added {again} chunks.");
}

async Task Lesson3()
{
    Header("LESSON 3 — RAG pipeline");

    string[] questions =
    [
        "My knee hurts when I squat, what should I do?",
        "How often should I deadlift?",
        "What is the capital of France?",          // not in the knowledge base → should refuse
    ];

    foreach (var question in questions)
    {
        var result = await rag.AskAsync(question);
        Console.WriteLine($"\nQ: {question}");
        Console.WriteLine($"A: {result.Answer}");
        foreach (var c in result.Citations)
            Console.WriteLine($"   [{c.Number}] {c.Title} › {c.Section}  ({c.ChunkId})");
    }
}

async Task Lesson4()
{
    Header("LESSON 4 — Memory + tools + RAG = coach agent");

    var clock = new ManualClock(new DateTimeOffset(2026, 9, 1, 7, 0, 0, TimeSpan.Zero));
    var memory = new MemoryManager(embedder, extractor, clock);
    var repo = new WorkoutRepository();

    // Seed six weeks of squat data for user "vikas" and some for another user (must never leak).
    var start = new DateOnly(2026, 7, 27);
    for (int w = 0; w < 6; w++)
    {
        repo.Add(new WorkoutSet("vikas", start.AddDays(7 * w), "squat", 100 + 2.5 * w, 5));
        repo.Add(new WorkoutSet("someone-else", start.AddDays(7 * w), "squat", 200, 5));
    }

    var agent = new CoachAgent(memory, new ConversationBuffer(summarizer, keepRecentMessages: 4),
        new ToolRouter(repo, clock), rag, chatModel);

    (string Message, int AdvanceHours)[] script =
    [
        ("I weigh 80 kg. My goal is to squat 140 kg", 0),
        ("Keep answers short please.", 1),
        ("I prefer training in the morning.", 1),
        ("I prefer training in the morning.", 1),             // duplicate → NOOP
        ("Yesterday I hurt my knee during squats.", 24),
        ("Am I progressing on squat?", 1),
        ("How should I train legs this week?", 1),
        ("I weigh 82 kg now.", 24 * 7),                       // UPDATE, keeps history
        ("My knee is fine now.", 24 * 3),                     // DELETE injury
        ("Forget that I prefer training in the morning.", 1), // DELETE by meaning
    ];

    foreach (var (message, advance) in script)
    {
        clock.Advance(TimeSpan.FromHours(advance));
        var turn = await agent.AskAsync("vikas", message);

        Console.WriteLine($"\n[{clock.GetUtcNow():MM-dd HH:mm}] USER: {message}");
        foreach (var op in turn.MemoryOps)
            Console.WriteLine($"   memory {op.Type,-6} {op.Text}  ({op.Reason})");
        if (turn.ToolResult is not null)
            Console.WriteLine($"   tool   {turn.ToolResult}");
        if (!turn.Answer.StartsWith("Got it"))
            Console.WriteLine($"   recalled: {string.Join("; ", turn.RecalledMemories.Select(m => $"{m.Item.Text} ({m.Score:F2})"))}");
        Console.WriteLine($"   COACH: {turn.Answer}");
    }

    Console.WriteLine("\nFinal long-term memory for 'vikas':");
    foreach (var m in memory.All("vikas"))
    {
        Console.WriteLine($"  {m}");
        foreach (var h in m.History) Console.WriteLine($"      previously {h}");
    }
}

async Task Lesson5()
{
    Header("LESSON 5 — Evaluating retrieval");

    EvalCase[] golden =
    [
        new("my knee hurts on squats", "squat"),
        new("how often to pull heavy from the floor", "deadlift"),
        new("RDL for hamstrings", "deadlift"),
        new("shoulder pain when pressing", "bench"),
        new("my bench is stuck", "bench"),
        new("how many hours of sleep", "recovery"),
        new("when to take a deload week", "recovery"),
        new("Epley formula", "progression"),
        new("am I getting stronger", "progression"),
        new("sore muscles after workout", "recovery"),
        // paraphrases with little word overlap — the hard cases
        new("my legs ache a lot", "recovery"),
        new("how much rest do I need at night", "recovery"),
        new("lifting heavier every week", "progression"),
        new("my lower back is sore after pulling", "deadlift"),
    ];

    var evaluator = new RetrievalEvaluator(searcher);
    foreach (var k in new[] { 1, 3 })
    {
        Console.WriteLine($"\n{"Mode",-8} {"Recall@" + k,9} {"MRR",6}");
        foreach (var mode in Enum.GetValues<HybridSearcher.Mode>())
        {
            var r = await evaluator.EvaluateAsync(golden, mode, k);
            Console.WriteLine($"{r.Mode,-8} {r.RecallAtK,9:P0} {r.Mrr,6:F2}");
        }
    }
    Console.WriteLine("→ Change chunk size, embedder or k and re-run: this table is how you know it helped.");
}

static void Header(string title)
{
    Console.WriteLine();
    Console.WriteLine(new string('═', 78));
    Console.WriteLine(title);
    Console.WriteLine(new string('═', 78));
}

/// <summary>Controllable clock so recency decay can be demonstrated (and unit-tested).</summary>
sealed class ManualClock(DateTimeOffset start) : TimeProvider
{
    private DateTimeOffset _now = start;
    public override DateTimeOffset GetUtcNow() => _now;
    public void Advance(TimeSpan by) => _now += by;
}
