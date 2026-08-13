"""Tests for dependency-free load-report aggregation."""
import unittest

from scripts.load_test_api import summarize


class LoadSummaryTests(unittest.TestCase):
    def test_separates_admitted_and_shed_latency(self):
        rows = [
            {"status": 200, "latency_ms": 10.0, "correlation_ok": True},
            {"status": 200, "latency_ms": 30.0, "correlation_ok": True},
            {"status": 429, "latency_ms": 1.0, "correlation_ok": True},
            {"status": 429, "latency_ms": 2.0, "correlation_ok": True},
        ]
        summary = summarize(rows, wall_s=2.0)
        self.assertEqual(summary["throughput_rps"], 2.0)
        self.assertEqual(summary["success_throughput_rps"], 1.0)
        self.assertEqual(summary["latency_ms"]["samples"], 4)
        self.assertEqual(
            summary["latency_ms_by_status"]["200"]["samples"], 2)
        self.assertEqual(
            summary["latency_ms_by_status"]["200"]["p50"], 20.0)
        self.assertEqual(
            summary["latency_ms_by_status"]["429"]["p50"], 1.5)

    def test_timeout_status_is_reported_separately(self):
        rows = [{
            "status": 0,
            "latency_ms": 1000.0,
            "correlation_ok": False,
            "error": "TimeoutError: timed out",
        }]
        summary = summarize(rows, wall_s=1.0)
        self.assertEqual(summary["success_throughput_rps"], 0.0)
        self.assertEqual(summary["exception_counts"], {"TimeoutError": 1})
        self.assertEqual(summary["latency_ms_by_status"]["0"]["max"], 1000.0)


if __name__ == "__main__":
    unittest.main()
