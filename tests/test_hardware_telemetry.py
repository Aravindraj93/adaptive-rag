import unittest

from adaptive_rag import TelemetryCollector, profile_hardware


class ProfilingTests(unittest.TestCase):
    def test_hardware_profile_is_serializable(self) -> None:
        profile = profile_hardware()
        self.assertGreaterEqual(profile.logical_cpus, 1)
        self.assertIn("python_version", profile.to_dict())

    def test_measure_records_even_when_body_raises(self) -> None:
        collector = TelemetryCollector()
        with self.assertRaises(RuntimeError):
            with collector.measure("failing", stage="test"):
                raise RuntimeError("expected")
        metric = collector.snapshot()[0]
        self.assertEqual(metric.name, "failing")
        self.assertEqual(metric.attributes["stage"], "test")


if __name__ == "__main__":
    unittest.main()

