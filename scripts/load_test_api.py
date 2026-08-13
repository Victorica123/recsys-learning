"""Run a repeatable, dependency-free HTTP load test against the recommendation API.

The tool is deliberately small and CPU-friendly: each worker owns one
keep-alive connection, latency includes the full response body, and every run
is saved as a unique JSON artifact under ``experiments/``.
"""
from __future__ import annotations

import argparse
import http.client
import json
import math
import re
import statistics
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = "/recommend?user_id=1&k=10"


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    items = sorted(values)
    position = (len(items) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return items[lower]
    weight = position - lower
    return items[lower] * (1.0 - weight) + items[upper] * weight


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def latency_summary(latencies: list[float]) -> dict:
    """Summarize one latency population with stable percentile fields."""
    return {
        "samples": len(latencies),
        "min": _round(min(latencies) if latencies else None),
        "mean": _round(statistics.fmean(latencies) if latencies else None),
        "p50": _round(percentile(latencies, 0.50)),
        "p95": _round(percentile(latencies, 0.95)),
        "p99": _round(percentile(latencies, 0.99)),
        "max": _round(max(latencies) if latencies else None),
    }


def _connection(parts, timeout: float):
    port = parts.port or (443 if parts.scheme == "https" else 80)
    cls = (http.client.HTTPSConnection
           if parts.scheme == "https" else http.client.HTTPConnection)
    return cls(parts.hostname, port, timeout=timeout)


def _target(parts, path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    prefix = parts.path.rstrip("/")
    return prefix + path


def fetch_json(base_url: str, path: str, timeout: float) -> dict | None:
    parts = urlsplit(base_url)
    conn = _connection(parts, timeout)
    try:
        conn.request("GET", _target(parts, path),
                     headers={"Accept": "application/json"})
        response = conn.getresponse()
        body = response.read()
        if response.status != 200:
            return None
        return json.loads(body.decode("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    finally:
        conn.close()


def run_warmup(base_url: str, path: str, count: int, timeout: float) -> None:
    if count <= 0:
        return
    parts = urlsplit(base_url)
    conn = _connection(parts, timeout)
    try:
        for index in range(count):
            request_id = f"warmup-{index}-{uuid.uuid4().hex[:12]}"
            try:
                conn.request("GET", _target(parts, path), headers={
                    "Accept": "application/json",
                    "Connection": "keep-alive",
                    "X-Request-ID": request_id,
                })
                conn.getresponse().read()
            except OSError:
                conn.close()
                conn = _connection(parts, timeout)
    finally:
        conn.close()


def run_load(base_url: str, path: str, requests: int, concurrency: int,
             timeout: float) -> tuple[list[dict], float]:
    parts = urlsplit(base_url)
    target = _target(parts, path)
    workers = min(concurrency, requests)
    counts = [requests // workers] * workers
    for index in range(requests % workers):
        counts[index] += 1
    ready = threading.Barrier(workers + 1)

    def worker(worker_id: int, count: int) -> list[dict]:
        conn = _connection(parts, timeout)
        rows = []
        ready.wait()
        for index in range(count):
            request_id = f"load-{worker_id}-{index}-{uuid.uuid4().hex[:12]}"
            started = time.perf_counter()
            try:
                conn.request("GET", target, headers={
                    "Accept": "application/json",
                    "Connection": "keep-alive",
                    "X-Request-ID": request_id,
                })
                response = conn.getresponse()
                response.read()
                rows.append({
                    "status": response.status,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "correlation_ok": response.getheader("X-Request-ID")
                    == request_id,
                })
            except (OSError, http.client.HTTPException) as exc:
                rows.append({
                    "status": 0,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "correlation_ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                conn.close()
                conn = _connection(parts, timeout)
        conn.close()
        return rows

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(worker, worker_id, count)
            for worker_id, count in enumerate(counts)
        ]
        ready.wait()
        started = time.perf_counter()
        rows = [row for future in futures for row in future.result()]
        wall_s = time.perf_counter() - started
    return rows, wall_s


def summarize(rows: list[dict], wall_s: float) -> dict:
    latencies = [row["latency_ms"] for row in rows]
    successes = sum(1 for row in rows if 200 <= row["status"] < 300)
    errors = len(rows) - successes
    status_counts = Counter(str(row["status"]) for row in rows)
    latencies_by_status = {
        status: [
            row["latency_ms"] for row in rows if str(row["status"]) == status
        ]
        for status in sorted(status_counts)
    }
    exception_counts = Counter(
        row["error"].split(":", 1)[0] for row in rows if "error" in row)
    return {
        "requests": len(rows),
        "successes": successes,
        "errors": errors,
        "error_rate": round(errors / len(rows), 6) if rows else 0.0,
        "wall_s": round(wall_s, 3),
        "throughput_rps": round(len(rows) / wall_s, 3) if wall_s else None,
        "success_throughput_rps": (
            round(successes / wall_s, 3) if wall_s else None),
        "status_counts": dict(sorted(status_counts.items())),
        "exception_counts": dict(sorted(exception_counts.items())),
        "correlation_failures": sum(
            1 for row in rows if not row["correlation_ok"]),
        "latency_ms": latency_summary(latencies),
        "latency_ms_by_status": {
            status: latency_summary(values)
            for status, values in latencies_by_status.items()
        },
    }


def metrics_delta(before: dict | None, after: dict | None) -> dict | None:
    if before is None or after is None:
        return None
    before_requests = before.get("requests", {})
    after_requests = after.get("requests", {})
    statuses = set(before_requests.get("status_counts", {}))
    statuses.update(after_requests.get("status_counts", {}))
    return {
        "requests_total": (
            after_requests.get("total", 0) - before_requests.get("total", 0)
        ),
        "status_counts": {
            status: (
                after_requests.get("status_counts", {}).get(status, 0)
                - before_requests.get("status_counts", {}).get(status, 0)
            )
            for status in sorted(statuses)
        },
    }


def safe_tag(value: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    if not tag:
        raise ValueError("tag must contain at least one safe filename character")
    return tag[:80]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--max-error-rate", type=float, default=0.0)
    parser.add_argument(
        "--max-p99-ms", type=float, default=None,
        help="optional p99 latency budget; disabled when omitted")
    args = parser.parse_args()

    if args.requests <= 0 or args.concurrency <= 0 or args.warmup < 0:
        parser.error("requests/concurrency must be positive; warmup cannot be negative")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if not 0.0 <= args.max_error_rate <= 1.0:
        parser.error("max-error-rate must be between 0 and 1")
    parts = urlsplit(args.base_url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        parser.error("base-url must be an absolute http(s) URL")

    run_tag = safe_tag(args.tag or datetime.now(
        timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output = ROOT / "experiments" / f"serve_load_{run_tag}.json"
    if output.exists():
        parser.error(f"refusing to overwrite existing artifact: {output}")

    run_warmup(args.base_url, args.path, args.warmup, args.timeout)
    metrics_before = fetch_json(args.base_url, "/metrics", args.timeout)
    rows, wall_s = run_load(
        args.base_url, args.path, args.requests, args.concurrency, args.timeout)
    metrics_after = fetch_json(args.base_url, "/metrics", args.timeout)
    summary = summarize(rows, wall_s)
    report = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "base_url": args.base_url,
            "path": args.path,
            "requests": args.requests,
            "concurrency": args.concurrency,
            "warmup": args.warmup,
            "timeout_s": args.timeout,
        },
        "summary": summary,
        "metrics": {
            "before": metrics_before,
            "after": metrics_after,
            "delta": metrics_delta(metrics_before, metrics_after),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report: {output}")
    breached = summary["error_rate"] > args.max_error_rate
    p99 = summary["latency_ms"]["p99"]
    if args.max_p99_ms is not None and p99 is not None:
        breached = breached or p99 > args.max_p99_ms
    return 2 if breached else 0


if __name__ == "__main__":
    raise SystemExit(main())
