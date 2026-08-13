"""Tests for single-host multi-worker metrics aggregation."""
import json
import tempfile
import time
import unittest
from pathlib import Path

from metrics_registry import (
    FileMetricsRegistry,
    aggregate_snapshots,
    render_prometheus,
)
from observability import RuntimeMetrics, ServiceMetrics


def worker_snapshot(pid, status, latency, rss):
    metrics = ServiceMetrics()
    metrics.begin_request()
    metrics.finish_request("/recommend", status, latency)
    runtime = RuntimeMetrics()
    runtime.observe(latency / 10, 25.0, rss)
    snapshot = metrics.snapshot(
        model_loaded=True,
        admission={
            "max_in_flight": 8, "in_flight": 0, "peak_in_flight": 1,
            "admitted_total": 1, "shed_total": int(status == 429),
        },
        rate_limit={
            "enabled": True, "requests_per_second": 100, "burst": 8,
            "admitted_total": int(status != 429),
            "rejected_total": int(status == 429),
        },
        runtime=runtime.snapshot(include_samples=True),
        include_samples=True,
    )
    snapshot["process"]["pid"] = pid
    return snapshot


class AggregateTests(unittest.TestCase):
    def test_merges_counters_samples_and_runtime(self):
        aggregate = aggregate_snapshots([
            worker_snapshot(101, 200, 10, 1000),
            worker_snapshot(202, 429, 30, 2000),
        ])
        self.assertEqual(aggregate["worker_count"], 2)
        self.assertEqual(aggregate["requests"]["total"], 2)
        self.assertEqual(
            aggregate["requests"]["status_counts"], {"200": 1, "429": 1})
        self.assertEqual(
            aggregate["routes"]["/recommend"]["latency_ms"]["p50"], 20.0)
        self.assertEqual(
            aggregate["routes"]["/recommend"]
            ["latency_ms_by_status"]["200"]["p99"], 10.0)
        self.assertEqual(aggregate["model"]["loaded_workers"], 2)
        self.assertEqual(aggregate["admission"]["max_in_flight"], 16)
        self.assertEqual(aggregate["rate_limit"]["rejected_total"], 1)
        self.assertEqual(aggregate["runtime"]["rss_bytes_sum"], 3000)

    def test_prometheus_contains_core_operational_metrics(self):
        aggregate = aggregate_snapshots([
            worker_snapshot(101, 200, 10, 1000)])
        text = render_prometheus(aggregate)
        self.assertIn('recsys_requests_total{status="200"} 1', text)
        self.assertIn("recsys_model_loaded_workers 1", text)
        self.assertIn("recsys_process_resident_memory_bytes 1000", text)
        self.assertIn(
            'route="/recommend",status="200",quantile="0.99"', text)
        self.assertIn(
            'recsys_event_loop_lag_ms{quantile="0.99"} 1.0', text)


class FileRegistryTests(unittest.TestCase):
    def test_atomic_publish_collect_remove_and_stale_filter(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry = FileMetricsRegistry(
                Path(temporary), stale_after_s=10)
            registry.publish(worker_snapshot(101, 200, 10, 1000))
            self.assertEqual(
                [row["process"]["pid"] for row in registry.collect()], [101])

            stale = worker_snapshot(202, 200, 10, 1000)
            stale["_registry_published_epoch"] = time.time() - 100
            stale_path = Path(temporary) / "worker-202.json"
            stale_path.write_text(
                json.dumps(stale), encoding="utf-8")
            self.assertEqual(
                [row["process"]["pid"] for row in registry.collect()], [101])
            self.assertFalse(stale_path.exists())

            temporary_path = Path(temporary) / ".worker-101-test.tmp"
            temporary_path.write_text("partial", encoding="utf-8")
            registry.remove(101)
            self.assertEqual(registry.collect(), [])
            self.assertFalse(temporary_path.exists())


if __name__ == "__main__":
    unittest.main()
