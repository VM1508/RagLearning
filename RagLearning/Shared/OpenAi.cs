using System.Net;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;

namespace RagLearning.Shared;

/// <summary>
/// Settings for any OpenAI-compatible endpoint (OpenAI, Azure OpenAI v1 endpoint, Ollama, LM Studio...).
///
/// OpenAI:        OPENAI_API_KEY  [OPENAI_BASE_URL] [OPENAI_EMBEDDING_MODEL] [OPENAI_CHAT_MODEL]
/// Azure OpenAI:  AZURE_OPENAI_ENDPOINT  AZURE_OPENAI_API_KEY
///                AZURE_OPENAI_EMBEDDING_DEPLOYMENT  AZURE_OPENAI_CHAT_DEPLOYMENT
/// Nothing set  → the app runs fully offline with the teaching implementations.
/// </summary>
public sealed record OpenAiOptions(
    string BaseUrl, string ApiKey, bool IsAzure,
    string EmbeddingModel, string ChatModel, int EmbeddingDimensions)
{
    public static OpenAiOptions? FromEnvironment()
    {
        string? Env(string name) => Environment.GetEnvironmentVariable(name);

        var azureEndpoint = Env("AZURE_OPENAI_ENDPOINT");
        if (!string.IsNullOrWhiteSpace(azureEndpoint) && Env("AZURE_OPENAI_API_KEY") is { } azureKey)
        {
            return new OpenAiOptions(
                BaseUrl: azureEndpoint.TrimEnd('/') + "/openai/v1/",
                ApiKey: azureKey,
                IsAzure: true,
                EmbeddingModel: Env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") ?? "text-embedding-3-small",
                ChatModel: Env("AZURE_OPENAI_CHAT_DEPLOYMENT") ?? "gpt-4o-mini",
                EmbeddingDimensions: 1536);
        }

        if (Env("OPENAI_API_KEY") is { Length: > 0 } key)
        {
            return new OpenAiOptions(
                BaseUrl: Env("OPENAI_BASE_URL") ?? "https://api.openai.com/v1/",
                ApiKey: key,
                IsAzure: false,
                EmbeddingModel: Env("OPENAI_EMBEDDING_MODEL") ?? "text-embedding-3-small",
                ChatModel: Env("OPENAI_CHAT_MODEL") ?? "gpt-4o-mini",
                EmbeddingDimensions: 1536);
        }

        return null;
    }
}

/// <summary>Thin HTTP wrapper with retry on 429 / 5xx (you will hit rate limits during ingestion).</summary>
public sealed class OpenAiHttp
{
    private readonly HttpClient _http;

    public OpenAiHttp(OpenAiOptions options)
    {
        _http = new HttpClient
        {
            BaseAddress = new Uri(options.BaseUrl.TrimEnd('/') + "/"),
            Timeout = TimeSpan.FromSeconds(100)
        };
        if (options.IsAzure)
            _http.DefaultRequestHeaders.Add("api-key", options.ApiKey);
        else
            _http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", options.ApiKey);
    }

    public async Task<JsonDocument> PostAsync(string path, object body, CancellationToken ct)
    {
        const int maxAttempts = 4;
        for (int attempt = 1; ; attempt++)
        {
            using var response = await _http.PostAsJsonAsync(path, body, ct);

            bool transient = response.StatusCode == HttpStatusCode.TooManyRequests
                             || (int)response.StatusCode >= 500;
            if (transient && attempt < maxAttempts)
            {
                var delay = response.Headers.RetryAfter?.Delta
                            ?? TimeSpan.FromSeconds(Math.Pow(2, attempt));
                await Task.Delay(delay, ct);
                continue;
            }

            if (!response.IsSuccessStatusCode)
            {
                var error = await response.Content.ReadAsStringAsync(ct);
                throw new HttpRequestException($"{(int)response.StatusCode} from {path}: {error}");
            }

            await using var stream = await response.Content.ReadAsStreamAsync(ct);
            return await JsonDocument.ParseAsync(stream, cancellationToken: ct);
        }
    }
}
