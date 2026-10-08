"""Zero-dependency HTTP REST API server for adaptive-rag.

Serves standard search, health check, and diagnostic endpoints via stdlib http.server.
Enables any frontend, TypeScript, curl, or microservice to query adaptive-rag over HTTP.

Usage:
    adaptive-rag serve --corpus ./docs --port 8000
    adaptive-rag serve --help
"""

from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .loaders import DirectoryLoader
from .models import Chunk
from .retrievers.bm25 import BM25Retriever
from .semantic_cache import SemanticCachedRetriever


def create_server_handler(retriever: Any, corpus_path: str):
    class AdaptiveRAGRequestHandler(BaseHTTPRequestHandler):
        server_version = "adaptive-rag-server/0.17.0"

        def _send_json(self, status: int, data: dict[str, Any]) -> None:
            payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(payload)

        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:
            def _serialize_info(obj: Any) -> Any:
                import dataclasses
                if dataclasses.is_dataclass(obj):
                    return dataclasses.asdict(obj)
                if hasattr(obj, "_asdict") and callable(obj._asdict):
                    return obj._asdict()
                if isinstance(obj, dict):
                    return {k: _serialize_info(v) for k, v in obj.items()}
                return str(obj)

            if self.path in ("/", "/health"):
                info = getattr(retriever, "info", lambda: {})()
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "corpus": corpus_path,
                        "retriever": type(retriever).__name__,
                        "cache_info": _serialize_info(info),
                    },
                )
            elif self.path == "/info":
                info = getattr(retriever, "info", lambda: {})()
                self._send_json(HTTPStatus.OK, {"info": _serialize_info(info)})
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not Found"})

        def do_POST(self) -> None:
            if self.path in ("/search", "/query", "/v1/search"):
                content_len = int(self.headers.get("Content-Length", 0))
                if content_len == 0:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Empty body"})
                    return

                try:
                    body = json.loads(self.rfile.read(content_len).decode("utf-8"))
                except Exception as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid JSON: {exc}"})
                    return

                query_text = body.get("query") or body.get("prompt") or body.get("text", "")
                if not query_text:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "'query' field is required"})
                    return

                top_k = int(body.get("top_k", 5))
                results = retriever.search(query_text, top_k=top_k)

                response_items = []
                for r in results:
                    chunk = r.chunk
                    response_items.append(
                        {
                            "id": chunk.id,
                            "document_id": chunk.document_id,
                            "text": chunk.text,
                            "score": float(r.score) if r.score is not None else None,
                            "rank": getattr(r, "rank", None),
                            "metadata": dict(chunk.metadata or {}),
                        }
                    )

                self._send_json(
                    HTTPStatus.OK,
                    {
                        "query": query_text,
                        "top_k": top_k,
                        "results": response_items,
                    },
                )
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint '{self.path}'"})

        def log_message(self, format: str, *args: Any) -> None:
            # Clean single-line logging
            sys.stdout.write(f"[{self.log_date_time_string()}] {args[0]} {args[1]} -> {args[2]}\n")

    return AdaptiveRAGRequestHandler


def serve_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="adaptive-rag serve",
        description="Run a zero-dependency HTTP search server on CPU.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to directory of documents to index and serve",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to listen on (default: 8000)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in words (default: 512)",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="http-server",
        help="Cache scope identifier",
    )

    args = parser.parse_args(argv)

    corpus_path: Path = args.corpus
    if not corpus_path.exists() or not corpus_path.is_dir():
        print(f"Error: corpus directory '{corpus_path}' does not exist.", file=sys.stderr)
        return 1

    print(f"\n[adaptive-rag] Scanning and indexing corpus from {corpus_path}...")
    loader = DirectoryLoader(corpus_path)
    docs = loader.load()

    chunks: list[Chunk] = []
    for doc_idx, doc in enumerate(docs):
        words = doc.text.split()
        if not words:
            continue
        for chunk_idx, start in enumerate(range(0, len(words), args.chunk_size)):
            chunk_words = words[start : start + args.chunk_size]
            chunks.append(
                Chunk(
                    id=f"{doc.id}-chunk-{chunk_idx}",
                    document_id=doc.id,
                    text=" ".join(chunk_words),
                    metadata=dict(doc.metadata or {}),
                )
            )

    print(f"[adaptive-rag] Loaded {len(docs)} documents into {len(chunks)} chunks.")
    backend = BM25Retriever()
    backend.add(chunks)

    cached_retriever = SemanticCachedRetriever(
        backend,
        scope=args.scope,
        revision=lambda: 1,
    )

    handler_cls = create_server_handler(cached_retriever, str(corpus_path))
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)

    print(f"[adaptive-rag] Server listening on http://{args.host}:{args.port}")
    print(f"  - Health Check: GET  http://{args.host}:{args.port}/health")
    print(f"  - Search API:   POST http://{args.host}:{args.port}/search")
    print("Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[adaptive-rag] Shutting down server...")
        server.server_close()

    return 0


def main() -> None:
    sys.exit(serve_command())


if __name__ == "__main__":
    main()
