import os
import tempfile
import unittest

from adaptive_rag import (
    CodeDocsRAG,
    DiagnosticResult,
    EducationRAG,
    FinanceRAG,
    HealthcareRAG,
    LegalRAG,
    doctor_command,
)
from adaptive_rag.integrations.langchain import AdaptiveRetriever as LangChainRetriever
from adaptive_rag.integrations.llamaindex import AdaptiveBaseRetriever, AdaptiveQueryEngine


class TestIntegrationsAndPacks(unittest.TestCase):
    def test_langchain_adapter_from_texts(self):
        retriever = LangChainRetriever.from_texts(
            ["Refunds are accepted within 30 days.", "Support email is info@test.com."],
            scope="lc-test",
            use_cache=True,
        )
        docs = retriever.get_relevant_documents("refund policy")
        self.assertGreaterEqual(len(docs), 1)
        self.assertIn("Refunds", docs[0].page_content)
        self.assertEqual(docs[0].metadata["document_id"], "text")

        lc_shim = retriever.as_langchain_retriever()
        res = lc_shim("refund")
        self.assertGreaterEqual(len(res), 1)

    def test_llamaindex_adapter_from_texts(self):
        engine = AdaptiveQueryEngine.from_texts(
            ["Leave policy grants 25 days annual leave.", "Working hours are flexible."],
            scope="llama-test",
            use_cache=True,
        )
        res = engine.query("leave policy")
        self.assertIn("Leave policy", res.response)
        self.assertGreaterEqual(len(res.source_nodes), 1)

    def test_haystack_adapter_from_texts(self):
        from adaptive_rag.integrations.haystack import AdaptiveHaystackRetriever

        retriever = AdaptiveHaystackRetriever.from_texts(
            ["Refunds are processed within 14 days.", "Contact billing@test.com."],
            scope="haystack-test",
            top_k=2,
        )
        res = retriever.run(query="refunds")
        self.assertIn("documents", res)
        self.assertGreaterEqual(len(res["documents"]), 1)
        self.assertIn("Refunds", res["documents"][0].content)

    def test_dspy_adapter_from_texts(self):
        from adaptive_rag.integrations.dspy import AdaptiveDSPyRetriever

        retriever = AdaptiveDSPyRetriever.from_texts(
            ["Security keys provide hardware-based MFA.", "Passwords should be 16 characters."],
            k=2,
            scope="dspy-test",
        )
        passages = retriever("hardware-based MFA")
        self.assertGreaterEqual(len(passages), 1)
        self.assertIn("Security keys", passages[0].long_text)

        # Multi-query test
        multi_passages = retriever(["Security keys", "Passwords"])
        self.assertEqual(len(multi_passages), 2)
        self.assertGreaterEqual(len(multi_passages[0]), 1)

    def test_domain_packs_healthcare(self):
        rag = HealthcareRAG()
        rag.add_texts(["Patient has HTN and stage 2 hypertension."])
        res_htn = rag.search("HTN")
        self.assertGreaterEqual(len(res_htn), 1)
        res_expanded = rag.search("hypertension")
        self.assertGreaterEqual(len(res_expanded), 1)
        info = rag.info()
        self.assertEqual(info["pack"], "HealthcareRAG")

    def test_domain_packs_legal(self):
        rag = LegalRAG()
        rag.add_texts(["Section 10. Indemnification shall apply to all claims."])
        res = rag.search("indemnity liability")
        self.assertGreaterEqual(len(res), 1)

    def test_domain_packs_education_and_code(self):
        edu = EducationRAG()
        edu.add_texts(["Gradient descent is an optimization algorithm."])
        res_edu = edu.search("what is gradient descent")
        self.assertGreaterEqual(len(res_edu), 1)

        code = CodeDocsRAG()
        code.add_texts(["SemanticCachedRetriever caches by revision identifier."])
        res_code = code.search("SemanticCachedRetriever")
        self.assertGreaterEqual(len(res_code), 1)

    def test_domain_pack_finance(self):
        fin = FinanceRAG()
        fin.add_texts(["Company reported EBITDA of $100M and 20% YoY growth."])
        res = fin.search("earnings before interest taxes")
        self.assertGreaterEqual(len(res), 1)

    def test_doctor_command_json_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = os.path.join(tmpdir, "policy.txt")
            with open(sample_file, "w", encoding="utf-8") as f:
                f.write("Company policies require annual training for security.\n" * 5)

            exit_code = doctor_command(["--corpus", tmpdir, "--json"])
            self.assertEqual(exit_code, 0)

    def test_server_api(self):
        import json
        import threading
        import urllib.request
        from http.server import ThreadingHTTPServer
        from adaptive_rag import BM25Retriever, Chunk, SemanticCachedRetriever
        from adaptive_rag.server import create_server_handler

        backend = BM25Retriever()
        backend.add([
            Chunk("doc-1", "docs", "Server API supports instant retrieval."),
            Chunk("doc-2", "docs", "CORS and JSON responses enabled by default."),
        ])
        cached = SemanticCachedRetriever(backend, scope="test-server", revision=lambda: 1)
        handler_cls = create_server_handler(cached, "test_corpus")

        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        port = server.server_address[1]

        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            # 1. Health check
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data["status"], "ok")

            # 2. Search query POST
            req_data = json.dumps({"query": "Server API", "top_k": 2}).encode("utf-8")
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/search",
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                search_data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(search_data["query"], "Server API")
                self.assertGreaterEqual(len(search_data["results"]), 1)
                self.assertIn("Server API", search_data["results"][0]["text"])

        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
