"""CPU-only tests for the recommendation API observability layer."""
import asyncio
import logging
import unittest

from observability import (
    AdmissionControlMiddleware,
    ObservabilityMiddleware,
    RateLimitMiddleware,
    RuntimeMetrics,
    ServiceMetrics,
    classify_status,
    normalize_route,
    percentile,
)


class PercentileTests(unittest.TestCase):
    def test_empty_and_interpolated_percentiles(self):
        self.assertIsNone(percentile([], 0.99))
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2.5)
        self.assertAlmostEqual(percentile([10, 20], 0.95), 19.5)

    def test_invalid_quantile_is_rejected(self):
        with self.assertRaises(ValueError):
            percentile([1], 1.1)


class CardinalityTests(unittest.TestCase):
    def test_dynamic_user_ids_share_one_route_label(self):
        self.assertEqual(normalize_route("/users/1"), "/users/{uid}")
        self.assertEqual(normalize_route("/users/999999"), "/users/{uid}")
        self.assertEqual(normalize_route("/users/{uid}"), "/users/{uid}")
        self.assertEqual(normalize_route("/unknown/anything"), "<unmatched>")

    def test_status_taxonomy_is_bounded(self):
        self.assertIsNone(classify_status(200))
        self.assertEqual(classify_status(400), "invalid_request")
        self.assertEqual(classify_status(404), "not_found")
        self.assertEqual(classify_status(429), "overloaded")
        self.assertEqual(classify_status(503), "model_unavailable")
        self.assertEqual(classify_status(500), "server_error")


class ServiceMetricsTests(unittest.TestCase):
    def test_request_and_model_snapshots(self):
        metrics = ServiceMetrics(latency_samples=3)
        for status, latency in [(200, 10), (200, 20), (404, 30), (200, 40)]:
            metrics.begin_request()
            metrics.finish_request("/recommend", status, latency)
        metrics.model_load_started()
        metrics.model_load_finished(True, 123.4567)

        snapshot = metrics.snapshot(
            model_loaded=True, n_users=12, n_items=34)
        self.assertEqual(snapshot["requests"]["total"], 4)
        self.assertEqual(snapshot["requests"]["in_flight"], 0)
        self.assertEqual(snapshot["requests"]["status_counts"],
                         {"200": 3, "404": 1})
        route = snapshot["routes"]["/recommend"]
        self.assertEqual(route["count"], 4)
        self.assertEqual(route["errors"], 1)
        self.assertEqual(route["latency_ms"]["samples"], 3)
        self.assertEqual(route["latency_ms"]["min"], 20.0)
        self.assertEqual(route["latency_ms"]["p50"], 30.0)
        self.assertEqual(route["latency_ms"]["max"], 40.0)
        self.assertEqual(snapshot["model"]["load_attempts"], 1)
        self.assertEqual(snapshot["model"]["load_successes"], 1)
        self.assertEqual(snapshot["model"]["last_load_ms"], 123.457)
        self.assertTrue(snapshot["model"]["loaded"])
        self.assertGreater(snapshot["requests"]["rps_10s"], 0)

    def test_internal_samples_are_opt_in(self):
        metrics = ServiceMetrics()
        metrics.begin_request()
        metrics.finish_request("/recommend", 200, 12.5)
        public = metrics.snapshot()
        internal = metrics.snapshot(include_samples=True)
        self.assertNotIn(
            "latency_samples_ms", public["routes"]["/recommend"])
        self.assertEqual(
            internal["routes"]["/recommend"]["latency_samples_ms"], [12.5])


class RuntimeMetricsTests(unittest.TestCase):
    def test_runtime_snapshot_contains_bounded_lag_samples(self):
        runtime = RuntimeMetrics(samples=2)
        runtime.observe(1.0, 20.0, 100)
        runtime.observe(2.0, 30.0, 200)
        runtime.observe(3.0, 40.0, 300)
        snapshot = runtime.snapshot(include_samples=True)
        self.assertEqual(snapshot["event_loop_lag_samples_ms"], [2.0, 3.0])
        self.assertEqual(snapshot["event_loop_lag_ms"]["p50"], 2.5)
        self.assertEqual(snapshot["cpu_percent"], 40.0)
        self.assertEqual(snapshot["rss_bytes"], 300)


class MiddlewareTests(unittest.TestCase):
    @staticmethod
    def _logger():
        logger = logging.getLogger("test.recsys.observability")
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        return logger

    def test_adds_correlation_headers_and_records_request(self):
        async def app(scope, receive, send):
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            })
            await send({"type": "http.response.body", "body": b"{}"})

        metrics = ServiceMetrics()
        middleware = ObservabilityMiddleware(
            app, metrics, logger=self._logger())
        sent = []
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/users/42",
            "headers": [(b"x-request-id", b"test-request-42")],
        }

        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(message):
            sent.append(message)

        asyncio.run(middleware(scope, receive, send))
        headers = dict(sent[0]["headers"])
        self.assertEqual(headers[b"x-request-id"], b"test-request-42")
        self.assertTrue(headers[b"server-timing"].startswith(b"app;dur="))
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["requests"]["total"], 1)
        self.assertEqual(snapshot["routes"]["/users/{uid}"]["count"], 1)

    def test_unhandled_exception_is_counted_and_inflight_balanced(self):
        async def broken(scope, receive, send):
            raise RuntimeError("boom")

        metrics = ServiceMetrics()
        middleware = ObservabilityMiddleware(
            broken, metrics, logger=self._logger())
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/recommend",
            "headers": [],
        }

        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(message):
            return None

        with self.assertRaises(RuntimeError):
            asyncio.run(middleware(scope, receive, send))
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["requests"]["in_flight"], 0)
        self.assertEqual(
            snapshot["requests"]["error_counts"]["unhandled_exception"], 1)


class AdmissionControlTests(unittest.TestCase):
    @staticmethod
    def _scope(path="/recommend"):
        return {"type": "http", "method": "GET", "path": path, "headers": []}

    @staticmethod
    async def _receive():
        return {"type": "http.request", "body": b""}

    def test_sheds_beyond_capacity_with_retry_after(self):
        async def scenario():
            gate = asyncio.Event()
            entered = asyncio.Event()

            async def slow_app(scope, receive, send):
                entered.set()
                await gate.wait()
                await send({"type": "http.response.start", "status": 200,
                            "headers": []})
                await send({"type": "http.response.body", "body": b"{}"})

            mw = AdmissionControlMiddleware(slow_app, max_in_flight=1)
            sent1, sent2 = [], []

            async def send1(message):
                sent1.append(message)

            async def send2(message):
                sent2.append(message)

            # First request occupies the only slot and blocks on the gate.
            task1 = asyncio.create_task(
                mw(self._scope(), self._receive, send1))
            await entered.wait()

            # Second request arrives while the slot is busy → shed with 429.
            await mw(self._scope(), self._receive, send2)

            gate.set()
            await task1
            return sent1, sent2, mw.stats()

        sent1, sent2, stats = asyncio.run(scenario())
        self.assertEqual(sent1[0]["status"], 200)
        self.assertEqual(sent2[0]["status"], 429)
        headers = dict(sent2[0]["headers"])
        self.assertIn(b"retry-after", headers)
        self.assertEqual(stats["admitted_total"], 1)
        self.assertEqual(stats["shed_total"], 1)
        self.assertEqual(stats["in_flight"], 0)     # released in finally
        self.assertEqual(stats["peak_in_flight"], 1)

    def test_exempt_paths_never_count_against_the_gate(self):
        async def scenario():
            async def app(scope, receive, send):
                await send({"type": "http.response.start", "status": 200,
                            "headers": []})
                await send({"type": "http.response.body", "body": b"{}"})

            mw = AdmissionControlMiddleware(
                app, max_in_flight=1, exempt_paths=("/metrics",))
            sent = []

            async def send(message):
                sent.append(message)

            for _ in range(5):
                await mw(self._scope("/metrics"), self._receive, send)
            return mw.stats()

        stats = asyncio.run(scenario())
        self.assertEqual(stats["admitted_total"], 0)
        self.assertEqual(stats["shed_total"], 0)

    def test_rejects_nonpositive_capacity(self):
        with self.assertRaises(ValueError):
            AdmissionControlMiddleware(lambda *a: None, max_in_flight=0)


class RateLimitTests(unittest.TestCase):
    def test_token_bucket_rejects_then_refills(self):
        async def scenario():
            current = [0.0]

            async def app(scope, receive, send):
                await send({"type": "http.response.start", "status": 200,
                            "headers": []})
                await send({"type": "http.response.body", "body": b"{}"})

            limiter = RateLimitMiddleware(
                app, requests_per_second=2, burst=1,
                clock=lambda: current[0])
            statuses = []
            headers = []

            async def receive():
                return {"type": "http.request", "body": b""}

            async def send(message):
                if message["type"] == "http.response.start":
                    statuses.append(message["status"])
                    headers.append(dict(message["headers"]))

            scope = {"type": "http", "method": "GET",
                     "path": "/recommend", "headers": []}
            await limiter(scope, receive, send)
            await limiter(scope, receive, send)
            current[0] = 0.5
            await limiter(scope, receive, send)
            return statuses, headers, limiter.stats()

        statuses, headers, stats = asyncio.run(scenario())
        self.assertEqual(statuses, [200, 429, 200])
        self.assertEqual(headers[1][b"x-error-class"], b"rate_limited")
        self.assertEqual(headers[1][b"retry-after"], b"1")
        self.assertEqual(stats["admitted_total"], 2)
        self.assertEqual(stats["rejected_total"], 1)

    def test_probe_path_is_exempt(self):
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200,
                        "headers": []})
            await send({"type": "http.response.body", "body": b"{}"})

        limiter = RateLimitMiddleware(app, 1, 1)
        sent = []

        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "method": "GET",
                 "path": "/metrics/aggregate", "headers": []}
        asyncio.run(limiter(scope, receive, send))
        self.assertEqual(sent[0]["status"], 200)
        self.assertEqual(limiter.stats()["admitted_total"], 0)


if __name__ == "__main__":
    unittest.main()
