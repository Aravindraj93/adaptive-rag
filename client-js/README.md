# adaptive-rag-client

Lightweight, zero-dependency JavaScript and TypeScript client for the [`adaptive-rag`](https://github.com/Aravindraj93/adaptive-rag) CPU-first retrieval engine.

Connect your Next.js, Node.js, Bun, Deno, or frontend apps directly to your local or private `adaptive-rag` server with no embedding API bills.

---

## ⚡ Quickstart

### 1. Start your `adaptive-rag` server:

```bash
adaptive-rag serve --corpus ./my_docs/ --port 8000
```

### 2. Query from JavaScript or TypeScript:

```typescript
import { AdaptiveRAGClient } from "./client-js"; // or from npm package

const client = new AdaptiveRAGClient({ baseUrl: "http://127.0.0.1:8000" });

// 1. Health check
const status = await client.health();
console.log("Server status:", status);

// 2. Query retrieval
const results = await client.search("What is the refund policy?", 3);
for (const item of results) {
  console.log(`[Score: ${item.score}] ${item.text}`);
}
```

---

## 📚 API Reference

### `new AdaptiveRAGClient(options)`
- `baseUrl` *(optional)*: Server URL (default: `http://127.0.0.1:8000`).
- `fetchFn` *(optional)*: Custom `fetch` implementation.

### `client.search(query, topK = 5)`
Returns `Promise<SearchResult[]>`:
- `id`: Unique chunk identifier
- `document_id`: Parent document name or path
- `text`: Passage content
- `score`: Ranking score (BM25 or fused score)
- `metadata`: Custom document metadata

### `client.health()`
Returns `Promise<HealthResponse>` including server status, active corpus, and cache statistics.

---

## 📄 License

Apache 2.0
