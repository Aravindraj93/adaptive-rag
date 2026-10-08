export interface SearchResult {
  id: string;
  document_id: string;
  text: string;
  score: number | null;
  rank?: number;
  metadata: Record<string, any>;
}

export interface SearchResponse {
  query: string;
  top_k: number;
  results: SearchResult[];
}

export interface HealthResponse {
  status: string;
  corpus: string;
  retriever: string;
  cache_info?: Record<string, any>;
}

export interface ClientOptions {
  baseUrl?: string;
  fetchFn?: typeof fetch;
}

export class AdaptiveRAGClient {
  private baseUrl: string;
  private fetchFn: typeof fetch;

  constructor(options?: ClientOptions);
  health(): Promise<HealthResponse>;
  search(query: string, topK?: number): Promise<SearchResult[]>;
  searchRaw(query: string, topK?: number): Promise<SearchResponse>;
}
