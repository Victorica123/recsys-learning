"""Single-host, multi-worker metrics aggregation with no external dependency.

Each API worker atomically publishes a bounded JSON snapshot into an
instance-specific runtime directory. Any worker can then read the fresh files
and expose an aggregate JSON or Prometheus-compatible text view. This solves
single-host Uvicorn aggregation; cross-host production deployments should
still use a real metrics collector.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from observability import percentile


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 3)


def _latency(values: list[float]) -> dict:
    return {
        "samples": len(values),
        "min": _round(min(values) if values else None),
        "mean": _round(sum(values) / len(values) if values else None),
        "p50": _round(percentile(values, 0.50)),
        "p95": _round(percentile(values, 0.95)),
        "p99": _round(percentile(values, 0.99)),
        "max": _round(max(values) if values else None),
    }


def _add_counts(target: Counter, values: dict | None) -> None:
    for key, value in (values or {}).items():
        target[str(key)] += int(value)


def aggregate_snapshots(snapshots: list[dict]) -> dict:
    """Merge worker snapshots, including exact bounded latency samples."""
    request_status: Counter[str] = Counter()
    request_errors: Counter[str] = Counter()
    routes: dict[str, dict] = {}
    runtime_lags: list[float] = []
    workers = []
    admission = Counter()
    rate_limit = Counter()
    total_requests = 0
    in_flight = 0
    rps_10s = 0.0
    rps_60s = 0.0
    loaded_workers = 0
    model_attempts = model_successes = model_failures = 0

    for snapshot in snapshots:
        requests = snapshot.get("requests") or {}
        total_requests += int(requests.get("total", 0))
        in_flight += int(requests.get("in_flight", 0))
        rps_10s += float(requests.get("rps_10s", 0))
        rps_60s += float(requests.get("rps_60s", 0))
        _add_counts(request_status, requests.get("status_counts"))
        _add_counts(request_errors, requests.get("error_counts"))

        for route, stats in (snapshot.get("routes") or {}).items():
            merged = routes.setdefault(route, {
                "count": 0,
                "errors": 0,
                "status_counts": Counter(),
                "error_counts": Counter(),
                "latencies": [],
                "latencies_by_status": {},
            })
            merged["count"] += int(stats.get("count", 0))
            merged["errors"] += int(stats.get("errors", 0))
            _add_counts(merged["status_counts"], stats.get("status_counts"))
            _add_counts(merged["error_counts"], stats.get("error_counts"))
            merged["latencies"].extend(
                float(value)
                for value in stats.get("latency_samples_ms", []))
            for status, values in (
                    stats.get("latency_samples_ms_by_status") or {}).items():
                merged["latencies_by_status"].setdefault(
                    str(status), []).extend(float(value) for value in values)

        model = snapshot.get("model") or {}
        loaded_workers += int(bool(model.get("loaded")))
        model_attempts += int(model.get("load_attempts", 0))
        model_successes += int(model.get("load_successes", 0))
        model_failures += int(model.get("load_failures", 0))

        adm = snapshot.get("admission") or {}
        for key in ("max_in_flight", "in_flight", "admitted_total",
                    "shed_total"):
            admission[key] += int(adm.get(key, 0))
        admission["peak_in_flight_sum"] += int(
            adm.get("peak_in_flight", 0))

        limiter = snapshot.get("rate_limit") or {}
        rate_limit["enabled_workers"] += int(
            bool(limiter.get("enabled")))
        rate_limit["configured_rps"] += float(
            limiter.get("requests_per_second", 0))
        rate_limit["burst"] += int(limiter.get("burst", 0))
        rate_limit["admitted_total"] += int(
            limiter.get("admitted_total", 0))
        rate_limit["rejected_total"] += int(
            limiter.get("rejected_total", 0))

        runtime = snapshot.get("runtime") or {}
        runtime_lags.extend(
            float(value)
            for value in runtime.get("event_loop_lag_samples_ms", []))
        workers.append({
            "pid": (snapshot.get("process") or {}).get("pid"),
            "uptime_s": snapshot.get("uptime_s"),
            "model_loaded": bool(model.get("loaded")),
            "cpu_percent": runtime.get("cpu_percent"),
            "rss_bytes": runtime.get("rss_bytes"),
            "event_loop_lag_ms": runtime.get("event_loop_lag_ms"),
            "published_at": snapshot.get("_registry_published_at"),
        })

    route_output = {}
    for route, stats in sorted(routes.items()):
        route_output[route] = {
            "count": stats["count"],
            "errors": stats["errors"],
            "status_counts": dict(sorted(stats["status_counts"].items())),
            "error_counts": dict(sorted(stats["error_counts"].items())),
            "latency_ms": _latency(stats["latencies"]),
            "latency_ms_by_status": {
                status: _latency(values)
                for status, values
                in sorted(stats["latencies_by_status"].items())
            },
        }

    rss_values = [
        int(worker["rss_bytes"]) for worker in workers
        if worker["rss_bytes"] is not None
    ]
    cpu_values = [
        float(worker["cpu_percent"]) for worker in workers
        if worker["cpu_percent"] is not None
    ]
    return {
        "service": "recsys-api",
        "scope": "aggregate",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "worker_count": len(workers),
        "workers": sorted(workers, key=lambda row: row["pid"] or 0),
        "requests": {
            "total": total_requests,
            "in_flight": in_flight,
            "rps_10s": round(rps_10s, 3),
            "rps_60s": round(rps_60s, 3),
            "status_counts": dict(sorted(request_status.items())),
            "error_counts": dict(sorted(request_errors.items())),
        },
        "routes": route_output,
        "model": {
            "loaded_workers": loaded_workers,
            "load_attempts": model_attempts,
            "load_successes": model_successes,
            "load_failures": model_failures,
        },
        "admission": dict(admission),
        "rate_limit": dict(rate_limit),
        "runtime": {
            "cpu_percent_sum": round(sum(cpu_values), 3),
            "rss_bytes_sum": sum(rss_values),
            "event_loop_lag_ms": _latency(runtime_lags),
        },
    }


class FileMetricsRegistry:
    """Atomically publish and aggregate fresh per-worker JSON snapshots."""

    def __init__(self, directory: Path, stale_after_s: float = 5.0):
        if stale_after_s <= 0:
            raise ValueError("stale_after_s must be positive")
        self.directory = Path(directory)
        self.stale_after_s = float(stale_after_s)

    def _path(self, pid: int) -> Path:
        return self.directory / f"worker-{int(pid)}.json"

    def publish(self, snapshot: dict) -> None:
        pid = int((snapshot.get("process") or {})["pid"])
        now = time.time()
        payload = dict(snapshot)
        payload["_registry_published_epoch"] = now
        payload["_registry_published_at"] = datetime.fromtimestamp(
            now, timezone.utc).isoformat()
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self._path(pid)
        temporary = self.directory / (
            f".worker-{pid}-{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8")
        os.replace(temporary, target)

    def remove(self, pid: int) -> None:
        try:
            self._path(pid).unlink()
        except FileNotFoundError:
            pass
        if self.directory.exists():
            for temporary in self.directory.glob(f".worker-{int(pid)}-*.tmp"):
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

    def collect(self) -> list[dict]:
        if not self.directory.exists():
            return []
        cutoff = time.time() - self.stale_after_s
        snapshots = []
        for path in self.directory.glob("worker-*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if float(payload.get(
                        "_registry_published_epoch", 0)) >= cutoff:
                    snapshots.append(payload)
                else:
                    path.unlink(missing_ok=True)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        for temporary in self.directory.glob(".worker-*.tmp"):
            try:
                if temporary.stat().st_mtime < cutoff:
                    temporary.unlink()
            except OSError:
                continue
        return snapshots

    def aggregate(self) -> dict:
        return aggregate_snapshots(self.collect())


def _label(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace(
        "\n", "\\n")


def render_prometheus(snapshot: dict) -> str:
    """Render the aggregate schema in Prometheus text exposition format."""
    lines = [
        "# HELP recsys_workers Number of fresh API worker snapshots.",
        "# TYPE recsys_workers gauge",
        f"recsys_workers {snapshot.get('worker_count', 0)}",
        "# HELP recsys_requests_total HTTP requests by status.",
        "# TYPE recsys_requests_total counter",
    ]
    requests = snapshot.get("requests") or {}
    for status, value in (requests.get("status_counts") or {}).items():
        lines.append(
            f'recsys_requests_total{{status="{_label(status)}"}} {value}')
    lines.extend([
        "# HELP recsys_requests_in_flight Current requests across workers.",
        "# TYPE recsys_requests_in_flight gauge",
        f"recsys_requests_in_flight {requests.get('in_flight', 0)}",
        "# HELP recsys_request_rate Rolling requests per second.",
        "# TYPE recsys_request_rate gauge",
        f'recsys_request_rate{{window="10s"}} {requests.get("rps_10s", 0)}',
        f'recsys_request_rate{{window="60s"}} {requests.get("rps_60s", 0)}',
        "# HELP recsys_request_errors_total HTTP errors by operational class.",
        "# TYPE recsys_request_errors_total counter",
    ])
    for error_class, value in (requests.get("error_counts") or {}).items():
        lines.append(
            "recsys_request_errors_total"
            f'{{class="{_label(error_class)}"}} {value}')
    lines.extend([
        "# HELP recsys_request_latency_ms Bounded request latency quantiles.",
        "# TYPE recsys_request_latency_ms gauge",
    ])
    for route, stats in (snapshot.get("routes") or {}).items():
        by_status = stats.get("latency_ms_by_status") or {}
        for status, latency in by_status.items():
            for key, quantile in (("p50", "0.5"), ("p95", "0.95"),
                                  ("p99", "0.99")):
                value = latency.get(key)
                if value is not None:
                    lines.append(
                        "recsys_request_latency_ms"
                        f'{{route="{_label(route)}",status="{_label(status)}",'
                        f'quantile="{quantile}"}} {value}')
    model = snapshot.get("model") or {}
    admission = snapshot.get("admission") or {}
    rate_limit = snapshot.get("rate_limit") or {}
    runtime = snapshot.get("runtime") or {}
    lines.extend([
        "# HELP recsys_model_loaded_workers Workers with a loaded model.",
        "# TYPE recsys_model_loaded_workers gauge",
        f"recsys_model_loaded_workers {model.get('loaded_workers', 0)}",
        "# HELP recsys_model_load_failures_total Failed model loads.",
        "# TYPE recsys_model_load_failures_total counter",
        f"recsys_model_load_failures_total "
        f"{model.get('load_failures', 0)}",
        "# HELP recsys_admission_capacity Total in-flight capacity.",
        "# TYPE recsys_admission_capacity gauge",
        f"recsys_admission_capacity "
        f"{admission.get('max_in_flight', 0)}",
        "# HELP recsys_admission_in_flight Admitted work across workers.",
        "# TYPE recsys_admission_in_flight gauge",
        f"recsys_admission_in_flight {admission.get('in_flight', 0)}",
        "# HELP recsys_admission_shed_total Requests shed by concurrency gates.",
        "# TYPE recsys_admission_shed_total counter",
        f"recsys_admission_shed_total {admission.get('shed_total', 0)}",
        "# HELP recsys_rate_limit_rejected_total Token-bucket rejections.",
        "# TYPE recsys_rate_limit_rejected_total counter",
        f"recsys_rate_limit_rejected_total "
        f"{rate_limit.get('rejected_total', 0)}",
        "# HELP recsys_process_cpu_percent Summed process CPU percentage.",
        "# TYPE recsys_process_cpu_percent gauge",
        f"recsys_process_cpu_percent {runtime.get('cpu_percent_sum', 0)}",
        "# HELP recsys_process_resident_memory_bytes Summed worker RSS.",
        "# TYPE recsys_process_resident_memory_bytes gauge",
        f"recsys_process_resident_memory_bytes "
        f"{runtime.get('rss_bytes_sum', 0)}",
        "# HELP recsys_event_loop_lag_ms Event-loop lag quantiles.",
        "# TYPE recsys_event_loop_lag_ms gauge",
    ])
    lag = runtime.get("event_loop_lag_ms") or {}
    for key, quantile in (("p50", "0.5"), ("p95", "0.95"),
                          ("p99", "0.99")):
        value = lag.get(key)
        if value is not None:
            lines.append(
                f'recsys_event_loop_lag_ms{{quantile="{quantile}"}} {value}')
    return "\n".join(lines) + "\n"
