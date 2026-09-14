"""Tests for dependency-free load-report aggregation."""
import unittest
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from scripts.load_test_api import run_load, summarize


class PostLoadTests(unittest.TestCase):
    def test_post_load_sends_json_auth_and_correlates_responses(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                received.append((self.path, json.loads(body),
                                 self.headers.get("X-API-Key"),
                                 self.headers.get("X-Actor-ID")))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "2")
                self.send_header("X-Request-ID", self.headers["X-Request-ID"])
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with patch.dict("os.environ", {"RECSYS_API_KEY": "test-only-key"}):
                    rows, _ = run_load(
                        f"http://127.0.0.1:{server.server_port}",
                        "/v1/recommendations", 4, 2, 2,
                        method="POST", json_body={"user_id": 1, "k": 5})
                self.assertEqual(len(rows), 4)
                self.assertTrue(all(row["status"] == 200 and row["correlation_ok"]
                                    for row in rows))
                self.assertEqual(received, [
                    ("/v1/recommendations", {"user_id": 1, "k": 5},
                     "test-only-key", "load-benchmark")] * 4)
            finally:
                server.shutdown()
                thread.join(timeout=3)


class LoadSummaryTests(unittest.TestCase):
    def test_separates_admitted_and_shed_latency(self):
        rows = [
            {"status": 200, "latency_ms": 10.0, "correlation_ok": True},
            {"status": 200, "latency_ms": 30.0, "correlation_ok": True},
            {"status": 429, "latency_ms": 1.0, "correlation_ok": True,
             "error_class": "rate_limited"},
            {"status": 429, "latency_ms": 2.0, "correlation_ok": True,
             "error_class": "overloaded"},
        ]
        summary = summarize(rows, wall_s=2.0)
        self.assertEqual(summary["throughput_rps"], 2.0)
        self.assertEqual(summary["success_throughput_rps"], 1.0)
        self.assertEqual(summary["error_class_counts"],
                         {"overloaded": 1, "rate_limited": 1})
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
