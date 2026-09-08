import unittest

from adaptive_rag import (
    AdaptiveRetriever,
    BM25Retriever,
    Chunk,
    HardwareProfile,
    NormalizedTokenizer,
)


def hardware(*, cpus: int, gib: int) -> HardwareProfile:
    return HardwareProfile(cpus, gib * 1024**3, "test", "test", "3.11", "test")


class TokenizationTests(unittest.TestCase):
    def test_light_normalization_matches_common_inflections(self) -> None:
        tokenizer = NormalizedTokenizer()
        self.assertEqual(
            tokenizer("Products returned policies apples buses status"),
            ["product", "return", "policy", "apple", "bus", "status"],
        )

    def test_accent_folding_is_optional(self) -> None:
        self.assertEqual(NormalizedTokenizer(fold_accents=True)("Café"), ["cafe"])
        self.assertEqual(NormalizedTokenizer(fold_accents=False)("Café"), ["café"])


class PlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        baseline = BM25Retriever()
        baseline.add([
            Chunk("a", "d", "Fresh apples grow in a hillside orchard."),
            Chunk("b", "d", "Ocean currents influence coastal weather."),
        ])
        self.baseline = baseline

    def test_high_confidence_results_are_returned(self) -> None:
        adaptive = AdaptiveRetriever(self.baseline, hardware=hardware(cpus=8, gib=16))
        results = adaptive.search("fresh apple orchard")
        self.assertEqual(results[0].chunk.id, "a")
        self.assertTrue(adaptive.last_decision.accepted)

    def test_weak_match_can_be_rejected(self) -> None:
        adaptive = AdaptiveRetriever(
            self.baseline,
            hardware=hardware(cpus=8, gib=16),
            confidence_threshold=0.9,
        )
        self.assertEqual(adaptive.search("unknown terms plus apple"), [])
        self.assertEqual(adaptive.last_decision.reason, "confidence below threshold")

    def test_constrained_hardware_caps_result_budget(self) -> None:
        adaptive = AdaptiveRetriever(self.baseline, hardware=hardware(cpus=2, gib=2))
        plan = adaptive.plan(top_k=50)
        self.assertEqual(plan.capacity_tier, "constrained")
        self.assertEqual(plan.effective_top_k, 10)


if __name__ == "__main__":
    unittest.main()

