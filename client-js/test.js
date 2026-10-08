const assert = require("assert");
const { AdaptiveRAGClient } = require("./index.js");

async function runMockTests() {
  console.log("Testing AdaptiveRAGClient constructor and mock requests...");

  // Mock fetch function to test without network dependency
  const mockFetch = async (url, options = {}) => {
    if (url.endsWith("/health")) {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          status: "ok",
          corpus: "/mock/docs",
          retriever: "BM25Retriever",
          cache_info: { exact_hits: 5, misses: 2 },
        }),
      };
    }

    if (url.endsWith("/search")) {
      const body = JSON.parse(options.body || "{}");
      return {
        ok: true,
        status: 200,
        json: async () => ({
          query: body.query,
          top_k: body.top_k,
          results: [
            {
              id: "doc-1",
              document_id: "policy.txt",
              text: "Refunds are processed in 30 days.",
              score: 1.5,
              metadata: { category: "billing" },
            },
          ],
        }),
      };
    }

    return { ok: false, status: 404, statusText: "Not Found" };
  };

  const client = new AdaptiveRAGClient({
    baseUrl: "http://localhost:8000",
    fetchFn: mockFetch,
  });

  // 1. Test health
  const health = await client.health();
  assert.strictEqual(health.status, "ok");
  assert.strictEqual(health.corpus, "/mock/docs");
  console.log("  ✓ health() passed");

  // 2. Test search
  const results = await client.search("refund policy", 3);
  assert.strictEqual(results.length, 1);
  assert.strictEqual(results[0].id, "doc-1");
  assert.strictEqual(results[0].text, "Refunds are processed in 30 days.");
  console.log("  ✓ search() passed");

  // 3. Test searchRaw
  const raw = await client.searchRaw("billing question", 2);
  assert.strictEqual(raw.query, "billing question");
  assert.strictEqual(raw.results.length, 1);
  console.log("  ✓ searchRaw() passed");

  console.log("[OK] All AdaptiveRAGClient JS tests passed successfully!");
}

runMockTests().catch((err) => {
  console.error("Test failed:", err);
  process.exit(1);
});
