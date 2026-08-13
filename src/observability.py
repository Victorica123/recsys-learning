"""Lightweight production observability for the recommendation API.

The module intentionally uses only the Python standard library.  It provides:

* a thread-safe in-process metrics registry with bounded latency samples;
* stable, low-cardinality route names;
* JSON access logs without query strings or user identifiers;
* request correlation and ``Server-Timing`` response headers.

The registry is process-local.  For a multi-worker deployment, scrape each
worker separately or replace it with a shared metrics backend.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
import uuid
from collections import Counter, deque
from datetime import datetime, timezone
from typing import Iterable


ACCESS_LOG = logging.getLogger("recsys.access")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_KNOWN_ROUTES = {
    "/health", "/live", "/ready", "/metrics", "/metrics/aggregate",
    "/metrics/prometheus", "/recommend", "/users", "/users/{uid}",
    "/events/impression", "/events/feedback", "/events/stats",
}


def percentile(values: Iterable[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile, or ``None`` when empty."""
    items = sorted(float(value) for value in values)
    if not items:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    position = (len(items) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return items[lower]
    weight = position - lower
    return items[lower] * (1.0 - weight) + items[upper] * weight


def normalize_route(path: str) -> str:
    """Map request paths to bounded-cardinality route labels."""
    if path in _KNOWN_ROUTES:
        return path
    if path.startswith("/users/") and path.count("/") == 2:
        return "/users/{uid}"
    return "<unmatched>"


def classify_status(status: int) -> str | None:
    """Convert HTTP status codes to a small operational error taxonomy."""
    if status < 400:
        return None
    if status == 400:
        return "invalid_request"
    if status == 404:
        return "not_found"
    if status == 429:
        return "overloaded"
    if status == 503:
        return "model_unavailable"
    if status >= 500:
        return "server_error"
    return "http_client_error"


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


class ServiceMetrics:
    """Thread-safe, bounded in-process metrics for one API worker."""

    def __init__(self, service: str = "recsys-api", latency_samples: int = 2048):
        if latency_samples <= 0:
            raise ValueError("latency_samples must be positive")
        self.service = service
        self.latency_samples = latency_samples
        self._started_wall = time.time()
        self._started_perf = time.perf_counter()
        self._lock = threading.Lock()
        self._requests_total = 0
        self._in_flight = 0
        # Bounded even under abusive traffic; 100k timestamps cover >1.6k RPS
        # across the full 60-second rolling window.
        self._recent_requests: deque[float] = deque(maxlen=100_000)
        self._status_counts: Counter[str] = Counter()
        self._error_counts: Counter[str] = Counter()
        self._routes: dict[str, dict] = {}
        self._model = {
            "load_attempts": 0,
            "load_successes": 0,
            "load_failures": 0,
            "last_load_ms": None,
            "last_loaded_at": None,
            "last_error": None,
        }

    def begin_request(self) -> None:
        with self._lock:
            self._in_flight += 1

    def finish_request(self, route: str, status: int, latency_ms: float,
                       error_class: str | None = None) -> None:
        route = normalize_route(route) if route.startswith("/") else route
        latency_ms = max(0.0, float(latency_ms))
        error_class = error_class or classify_status(status)
        now = time.perf_counter()
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._requests_total += 1
            self._status_counts[str(status)] += 1
            self._recent_requests.append(now)
            cutoff = now - 60.0
            while (self._recent_requests
                   and self._recent_requests[0] < cutoff):
                self._recent_requests.popleft()
            if error_class:
                self._error_counts[error_class] += 1
            stats = self._routes.setdefault(route, {
                "count": 0,
                "errors": 0,
                "status_counts": Counter(),
                "error_counts": Counter(),
                "latencies": deque(maxlen=self.latency_samples),
                "latencies_by_status": {},
            })
            stats["count"] += 1
            stats["status_counts"][str(status)] += 1
            stats["latencies"].append(latency_ms)
            stats["latencies_by_status"].setdefault(
                str(status), deque(maxlen=self.latency_samples)).append(
                    latency_ms)
            if error_class:
                stats["errors"] += 1
                stats["error_counts"][error_class] += 1

    def model_load_started(self) -> None:
        with self._lock:
            self._model["load_attempts"] += 1

    def model_load_finished(self, success: bool, duration_ms: float,
                            error: str | None = None) -> None:
        with self._lock:
            self._model["last_load_ms"] = round(max(0.0, duration_ms), 3)
            if success:
                self._model["load_successes"] += 1
                self._model["last_loaded_at"] = datetime.now(
                    timezone.utc).isoformat()
                self._model["last_error"] = None
            else:
                self._model["load_failures"] += 1
                self._model["last_error"] = error

    def snapshot(self, *, model_loaded: bool = False,
                 load_error: str | None = None,
                 n_users: int = 0, n_items: int = 0,
                 admission: dict | None = None,
                 rate_limit: dict | None = None,
                 runtime: dict | None = None,
                 include_samples: bool = False) -> dict:
        """Return a JSON-serializable point-in-time metrics snapshot."""
        with self._lock:
            now = time.perf_counter()
            uptime = max(now - self._started_perf, 0.001)
            cutoff = now - 60.0
            while (self._recent_requests
                   and self._recent_requests[0] < cutoff):
                self._recent_requests.popleft()
            recent = list(self._recent_requests)
            routes = {}
            for route, stats in sorted(self._routes.items()):
                latencies = list(stats["latencies"])
                latencies_by_status = {
                    status: list(values)
                    for status, values in stats["latencies_by_status"].items()
                }
                route_snapshot = {
                    "count": stats["count"],
                    "errors": stats["errors"],
                    "status_counts": dict(sorted(stats["status_counts"].items())),
                    "error_counts": dict(sorted(stats["error_counts"].items())),
                    "latency_ms": {
                        "samples": len(latencies),
                        "min": _round(min(latencies) if latencies else None),
                        "mean": _round(
                            sum(latencies) / len(latencies) if latencies else None),
                        "p50": _round(percentile(latencies, 0.50)),
                        "p95": _round(percentile(latencies, 0.95)),
                        "p99": _round(percentile(latencies, 0.99)),
                        "max": _round(max(latencies) if latencies else None),
                    },
                    "latency_ms_by_status": {
                        status: {
                            "samples": len(values),
                            "min": _round(min(values) if values else None),
                            "mean": _round(
                                sum(values) / len(values)
                                if values else None),
                            "p50": _round(percentile(values, 0.50)),
                            "p95": _round(percentile(values, 0.95)),
                            "p99": _round(percentile(values, 0.99)),
                            "max": _round(max(values) if values else None),
                        }
                        for status, values
                        in sorted(latencies_by_status.items())
                    },
                }
                if include_samples:
                    route_snapshot["latency_samples_ms"] = latencies
                    route_snapshot["latency_samples_ms_by_status"] = \
                        latencies_by_status
                routes[route] = route_snapshot
            model = dict(self._model)
            model.update({
                "loaded": bool(model_loaded),
                "load_error": load_error,
                "n_users": int(n_users),
                "n_items": int(n_items),
            })
            return {
                "service": self.service,
                "process": {"pid": os.getpid()},
                "started_at": datetime.fromtimestamp(
                    self._started_wall, timezone.utc).isoformat(),
                "uptime_s": round(uptime, 3),
                "requests": {
                    "total": self._requests_total,
                    "in_flight": self._in_flight,
                    "rps_10s": round(
                        sum(timestamp >= now - 10.0 for timestamp in recent)
                        / min(uptime, 10.0), 3),
                    "rps_60s": round(
                        len(recent) / min(uptime, 60.0), 3),
                    "status_counts": dict(sorted(self._status_counts.items())),
                    "error_counts": dict(sorted(self._error_counts.items())),
                },
                "routes": routes,
                "model": model,
                "admission": dict(admission) if admission else None,
                "rate_limit": dict(rate_limit) if rate_limit else None,
                "runtime": dict(runtime) if runtime else None,
            }


def process_rss_bytes() -> int | None:
    """Return current resident memory using only the standard library."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            get_current_process = ctypes.windll.kernel32.GetCurrentProcess
            get_current_process.restype = wintypes.HANDLE
            get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
            get_memory.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            get_memory.restype = wintypes.BOOL
            handle = get_current_process()
            ok = get_memory(
                handle, ctypes.byref(counters), counters.cb)
            return int(counters.WorkingSetSize) if ok else None
        except (AttributeError, OSError):
            return None
    try:  # pragma: no cover - Windows is the primary project platform
        page_size = os.sysconf("SC_PAGE_SIZE")
        with open("/proc/self/statm", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * page_size
    except (AttributeError, OSError, ValueError, IndexError):
        return None


class RuntimeMetrics:
    """Thread-safe event-loop lag, process CPU, and RSS samples."""

    def __init__(self, samples: int = 2048):
        if samples <= 0:
            raise ValueError("samples must be positive")
        self._lock = threading.Lock()
        self._lag_ms: deque[float] = deque(maxlen=samples)
        self._cpu_percent = 0.0
        self._rss_bytes: int | None = None
        self._sampled_at: str | None = None

    def observe(self, lag_ms: float, cpu_percent: float,
                rss_bytes: int | None) -> None:
        with self._lock:
            self._lag_ms.append(max(0.0, float(lag_ms)))
            self._cpu_percent = max(0.0, float(cpu_percent))
            self._rss_bytes = rss_bytes
            self._sampled_at = datetime.now(timezone.utc).isoformat()

    def snapshot(self, include_samples: bool = False) -> dict:
        with self._lock:
            values = list(self._lag_ms)
            result = {
                "sampled_at": self._sampled_at,
                "cpu_percent": round(self._cpu_percent, 3),
                "rss_bytes": self._rss_bytes,
                "event_loop_lag_ms": {
                    "samples": len(values),
                    "latest": _round(values[-1] if values else None),
                    "p50": _round(percentile(values, 0.50)),
                    "p95": _round(percentile(values, 0.95)),
                    "p99": _round(percentile(values, 0.99)),
                    "max": _round(max(values) if values else None),
                },
            }
            if include_samples:
                result["event_loop_lag_samples_ms"] = values
            return result


def _request_id(scope: dict) -> str:
    for key, value in scope.get("headers", []):
        if key.lower() == b"x-request-id":
            candidate = value.decode("ascii", errors="ignore")
            if _REQUEST_ID_RE.fullmatch(candidate):
                return candidate
    return uuid.uuid4().hex


class ObservabilityMiddleware:
    """Pure-ASGI request metrics, correlation headers, and JSON access logs."""

    def __init__(self, app, metrics: ServiceMetrics,
                 excluded_paths: Iterable[str] = (
                     "/health", "/live", "/ready", "/metrics",
                     "/metrics/aggregate", "/metrics/prometheus"),
                 logger: logging.Logger = ACCESS_LOG):
        self.app = app
        self.metrics = metrics
        self.excluded_paths = frozenset(excluded_paths)
        self.logger = logger

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in self.excluded_paths:
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET")
        route = normalize_route(path)
        request_id = _request_id(scope)
        scope["recsys.request_id"] = request_id
        started = time.perf_counter()
        status = 500
        response_error_class = None
        self.metrics.begin_request()

        async def send_observed(message):
            nonlocal status, response_error_class
            if message.get("type") == "http.response.start":
                status = int(message.get("status", 500))
                headers = list(message.get("headers", []))
                for key, value in headers:
                    if key.lower() == b"x-error-class":
                        response_error_class = value.decode(
                            "ascii", errors="ignore") or None
                header_names = {key.lower() for key, _ in headers}
                if b"x-request-id" not in header_names:
                    headers.append((b"x-request-id", request_id.encode("ascii")))
                if b"server-timing" not in header_names:
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    headers.append((
                        b"server-timing",
                        f"app;dur={elapsed_ms:.3f}".encode("ascii"),
                    ))
                message = dict(message)
                message["headers"] = headers
            await send(message)

        exception_name = None
        try:
            await self.app(scope, receive, send_observed)
        except Exception as exc:
            exception_name = type(exc).__name__
            raise
        finally:
            latency_ms = (time.perf_counter() - started) * 1000
            error_class = (
                "unhandled_exception" if exception_name
                else response_error_class
            )
            self.metrics.finish_request(route, status, latency_ms, error_class)
            payload = {
                "event": "http_request",
                "request_id": request_id,
                "method": method,
                "route": route,
                "status": status,
                "latency_ms": round(latency_ms, 3),
            }
            if exception_name:
                payload["exception"] = exception_name
            self.logger.info(json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")))


class AdmissionControlMiddleware:
    """Pure-ASGI concurrency limiter that sheds load to bound tail latency.

    One API worker runs a single event loop.  When the heavy handlers are
    offloaded to a thread pool, several requests are genuinely in flight at
    once; past a point extra concurrency only deepens the queue and inflates
    the tail latency of *every* admitted request.  This middleware caps the
    number of concurrently admitted requests: once ``max_in_flight`` is
    reached, further requests are rejected immediately with HTTP 429 and a
    ``Retry-After`` header instead of being queued.  Health and metrics probes
    are exempt so an overloaded worker stays observable.

    Place this *inside* :class:`ObservabilityMiddleware` so shed responses are
    still counted (status 429 → ``overloaded`` in the error taxonomy) and get
    correlation/timing headers.
    """

    def __init__(self, app, max_in_flight: int, *,
                 exempt_paths: Iterable[str] = (
                     "/health", "/live", "/ready", "/metrics",
                     "/metrics/aggregate", "/metrics/prometheus"),
                 retry_after: int = 1):
        if max_in_flight <= 0:
            raise ValueError("max_in_flight must be positive")
        self.app = app
        self.max_in_flight = int(max_in_flight)
        self.exempt_paths = frozenset(exempt_paths)
        self.retry_after = max(0, int(retry_after))
        self._lock = threading.Lock()
        self._in_flight = 0
        self._admitted_total = 0
        self._shed_total = 0
        self._peak_in_flight = 0

    def stats(self) -> dict:
        """Point-in-time admission gauges for the metrics snapshot."""
        with self._lock:
            return {
                "max_in_flight": self.max_in_flight,
                "in_flight": self._in_flight,
                "peak_in_flight": self._peak_in_flight,
                "admitted_total": self._admitted_total,
                "shed_total": self._shed_total,
            }

    async def __call__(self, scope, receive, send):
        if (scope.get("type") != "http"
                or scope.get("path") in self.exempt_paths):
            await self.app(scope, receive, send)
            return

        with self._lock:
            if self._in_flight >= self.max_in_flight:
                self._shed_total += 1
                admitted = False
            else:
                self._in_flight += 1
                self._admitted_total += 1
                self._peak_in_flight = max(
                    self._peak_in_flight, self._in_flight)
                admitted = True

        if not admitted:
            await self._send_shed(send)
            return

        try:
            await self.app(scope, receive, send)
        finally:
            with self._lock:
                self._in_flight = max(0, self._in_flight - 1)

    async def _send_shed(self, send) -> None:
        body = (b'{"error":"server overloaded, retry later",'
                b'"code":"overloaded"}')
        await send({
            "type": "http.response.start",
            "status": 429,
            "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"retry-after", str(self.retry_after).encode("ascii")),
                (b"x-error-class", b"overloaded"),
            ],
        })
        await send({"type": "http.response.body", "body": body})


class RateLimitMiddleware:
    """Per-worker token-bucket rate limiter with an explicit burst budget."""

    def __init__(self, app, requests_per_second: float, burst: int, *,
                 exempt_paths: Iterable[str] = (
                     "/health", "/live", "/ready", "/metrics",
                     "/metrics/aggregate", "/metrics/prometheus"),
                 clock=time.perf_counter):
        if requests_per_second < 0:
            raise ValueError("requests_per_second cannot be negative")
        if burst <= 0:
            raise ValueError("burst must be positive")
        self.app = app
        self.requests_per_second = float(requests_per_second)
        self.burst = int(burst)
        self.exempt_paths = frozenset(exempt_paths)
        self.clock = clock
        self._lock = threading.Lock()
        self._tokens = float(burst)
        self._last_refill = clock()
        self._admitted_total = 0
        self._rejected_total = 0

    @property
    def enabled(self) -> bool:
        return self.requests_per_second > 0

    def stats(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "requests_per_second": self.requests_per_second,
                "burst": self.burst,
                "tokens": round(self._tokens, 3),
                "admitted_total": self._admitted_total,
                "rejected_total": self._rejected_total,
            }

    def _allow(self) -> tuple[bool, int]:
        if not self.enabled:
            with self._lock:
                self._admitted_total += 1
            return True, 0
        with self._lock:
            now = self.clock()
            elapsed = max(0.0, now - self._last_refill)
            self._tokens = min(
                float(self.burst),
                self._tokens + elapsed * self.requests_per_second)
            self._last_refill = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                self._admitted_total += 1
                return True, 0
            self._rejected_total += 1
            retry_after = max(
                1, math.ceil((1.0 - self._tokens)
                             / self.requests_per_second))
            return False, retry_after

    async def __call__(self, scope, receive, send):
        if (scope.get("type") != "http"
                or scope.get("path") in self.exempt_paths):
            await self.app(scope, receive, send)
            return
        allowed, retry_after = self._allow()
        if allowed:
            await self.app(scope, receive, send)
            return
        body = (b'{"error":"rate limit exceeded, retry later",'
                b'"code":"rate_limited"}')
        await send({
            "type": "http.response.start",
            "status": 429,
            "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"retry-after", str(retry_after).encode("ascii")),
                (b"x-error-class", b"rate_limited"),
            ],
        })
        await send({"type": "http.response.body", "body": body})
