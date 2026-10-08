/**
 * Lightweight JavaScript & TypeScript client for adaptive-rag REST server.
 * Uses standard native fetch (available in Node 18+, Bun, Deno, and all modern browsers).
 */

class AdaptiveRAGClient {
  /**
   * @param {Object} [options]
   * @param {string} [options.baseUrl='http://127.0.0.1:8000'] - Base URL of the adaptive-rag server
   * @param {Function} [options.fetchFn] - Custom fetch function (defaults to global fetch)
   */
  constructor(options = {}) {
    this.baseUrl = (options.baseUrl || "http://127.0.0.1:8000").replace(/\/+$/, "");
    this.fetchFn = options.fetchFn || (typeof fetch !== "undefined" ? fetch : null);

    if (!this.fetchFn) {
      throw new Error(
        "No fetch implementation found. If running on Node < 18, provide a custom fetchFn option or install node-fetch."
      );
    }
  }

  /**
   * Check server health and cache telemetry.
   * @returns {Promise<Object>}
   */
  async health() {
    const res = await this.fetchFn(`${this.baseUrl}/health`);
    if (!res.ok) {
      throw new Error(`Health check failed with status ${res.status}`);
    }
    return res.json();
  }

  /**
   * Query the corpus and return an array of SearchResult objects.
   * @param {string} query - Natural language search query
   * @param {number} [topK=5] - Number of results to return
   * @returns {Promise<Array<Object>>}
   */
  async search(query, topK = 5) {
    const data = await this.searchRaw(query, topK);
    return data.results || [];
  }

  /**
   * Query the corpus and return the full response object with query and top_k metadata.
   * @param {string} query - Natural language search query
   * @param {number} [topK=5] - Number of results to return
   * @returns {Promise<Object>}
   */
  async searchRaw(query, topK = 5) {
    const res = await this.fetchFn(`${this.baseUrl}/search`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query, top_k: topK }),
    });

    if (!res.ok) {
      const errBody = await res.text().catch(() => "");
      throw new Error(`Search request failed (${res.status}): ${errBody || res.statusText}`);
    }

    return res.json();
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { AdaptiveRAGClient };
}

if (typeof window !== "undefined") {
  window.AdaptiveRAGClient = AdaptiveRAGClient;
}
