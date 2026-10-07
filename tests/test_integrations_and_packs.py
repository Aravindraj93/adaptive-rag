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


if __name__ == "__main__":
    unittest.main()
