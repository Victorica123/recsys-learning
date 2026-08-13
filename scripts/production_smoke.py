"""Exercise the complete production serving surface with real checkpoints.

Starts a two-worker API instance, verifies live/ready probes, cross-worker JSON
and Prometheus metrics, runtime CPU/RSS/event-loop signals, a paced success
phase, a rate-limited burst phase, and graceful registry cleanup. The report
is written under ``experiments/`` with a unique tag.
"""
from __future__ import annotations

import argparse
import http.client
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent

from load_test_api import run_load, safe_tag, summarize  # noqa: E402
from scaling_experiment import (  # noqa: E402
    fetch_json, start_server, stop_server, wait_for_workers)

import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from metrics_registry import FileMetricsRegistry  # noqa: E402
from feedback import FeedbackStore  # noqa: E402


def fetch_text(base_url: str, path: str, timeout: float) -> tuple[int, str]:
    parts = urlsplit(base_url)
    connection = http.client.HTTPConnection(
        parts.hostname, parts.port or 80, timeout=timeout)
    try:
        connection.request("GET", path, headers={"Connection": "close"})
        response = connection.getresponse()
        return response.status, response.read().decode("utf-8")
    finally:
        connection.close()


def paced_load(base_url: str, path: str, requests: int, interval_s: float,
               timeout: float) -> tuple[list[dict], float]:
    """Send a below-limit keep-alive stream for the clean success phase."""
    parts = urlsplit(base_url)
    connection = http.client.HTTPConnection(
        parts.hostname, parts.port or 80, timeout=timeout)
    rows = []
    wall_started = time.perf_counter()
    try:
        for index in range(requests):
            iteration_started = time.perf_counter()
            request_id = f"paced-{index}-{uuid.uuid4().hex[:12]}"
            try:
                connection.request("GET", path, headers={
                    "Accept": "application/json",
                    "Connection": "keep-alive",
                    "X-Request-ID": request_id,
                })
                response = connection.getresponse()
                response.read()
                rows.append({
                    "status": response.status,
                    "latency_ms": (
                        time.perf_counter() - iteration_started) * 1000,
                    "correlation_ok": (
                        response.getheader("X-Request-ID") == request_id),
                })
            except (OSError, http.client.HTTPException) as exc:
                rows.append({
                    "status": 0,
                    "latency_ms": (
                        time.perf_counter() - iteration_started) * 1000,
                    "correlation_ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                connection.close()
                connection = http.client.HTTPConnection(
                    parts.hostname, parts.port or 80, timeout=timeout)
            remaining = interval_s - (
                time.perf_counter() - iteration_started)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        connection.close()
    return rows, time.perf_counter() - wall_started


def wait_for_aggregate(base_url: str, minimum_requests: int,
                       workers: int, timeout: float) -> dict:
    deadline = time.perf_counter() + timeout
    last = None
    while time.perf_counter() < deadline:
        last = fetch_json(base_url, "/metrics/aggregate", timeout=2)
        if (last and last.get("worker_count") == workers
                and last.get("requests", {}).get("total", 0)
                >= minimum_requests):
            return last
        time.sleep(0.1)
    raise TimeoutError(
        f"aggregate did not reach workers={workers}, requests="
        f"{minimum_requests}; last={last}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8820)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-in-flight", type=int, default=8)
    parser.add_argument("--rate-limit", type=float, default=80)
    parser.add_argument("--rate-burst", type=int, default=8)
    parser.add_argument("--paced-requests", type=int, default=40)
    parser.add_argument("--paced-interval", type=float, default=0.025)
    parser.add_argument("--burst-requests", type=int, default=300)
    parser.add_argument("--burst-concurrency", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    run_tag = safe_tag(args.tag or datetime.now(
        timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output = ROOT / "experiments" / f"serve_production_smoke_{run_tag}.json"
    if output.exists():
        parser.error(f"refusing to overwrite existing artifact: {output}")
    metrics_dir = ROOT / "runtime" / "metrics" / f"production-{run_tag}"
    feedback_db = (
        ROOT / "runtime" / "feedback" / f"production-{run_tag}.sqlite3")
    if feedback_db.exists():
        parser.error(
            f"refusing to append to existing feedback database: {feedback_db}")
    base_url = f"http://127.0.0.1:{args.port}"
    path = "/recommend?user_id=1&k=10"
    registry = FileMetricsRegistry(metrics_dir, stale_after_s=5)
    report = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "config": vars(args),
        "checks": {},
    }
    proc = None
    failure = None
    try:
        proc = start_server(
            args.port, args.workers, args.max_in_flight,
            rate_limit=args.rate_limit, rate_burst=args.rate_burst,
            metrics_interval=0.25, metrics_dir=metrics_dir,
            feedback_db=feedback_db)
        worker_pids = wait_for_workers(base_url, args.workers)
        report["worker_pids"] = worker_pids

        live = fetch_json(base_url, "/live", args.timeout)
        ready = fetch_json(base_url, "/ready", args.timeout)
        baseline = wait_for_aggregate(
            base_url, minimum_requests=0, workers=args.workers,
            timeout=args.timeout)

        paced_rows, paced_wall = paced_load(
            base_url, path, args.paced_requests, args.paced_interval,
            args.timeout)
        paced = summarize(paced_rows, paced_wall)
        after_paced = wait_for_aggregate(
            base_url, args.paced_requests, args.workers, args.timeout)

        burst_rows, burst_wall = run_load(
            base_url, path, args.burst_requests,
            args.burst_concurrency, args.timeout)
        burst = summarize(burst_rows, burst_wall)
        total_expected = args.paced_requests + args.burst_requests
        after_burst = wait_for_aggregate(
            base_url, total_expected, args.workers, args.timeout)
        prometheus_status, prometheus = fetch_text(
            base_url, "/metrics/prometheus", args.timeout)
        worker_snapshots = registry.collect()
        feedback = FeedbackStore(feedback_db).stats()

        report.update({
            "probes": {"live": live, "ready": ready},
            "paced": paced,
            "burst": burst,
            "aggregate": {
                "baseline": baseline,
                "after_paced": after_paced,
                "after_burst": after_burst,
            },
            "per_worker_request_totals": {
                str((snapshot.get("process") or {}).get("pid")):
                (snapshot.get("requests") or {}).get("total", 0)
                for snapshot in worker_snapshots
            },
            "prometheus": {
                "status": prometheus_status,
                "line_count": len(prometheus.splitlines()),
                "required_lines_present": all(token in prometheus for token in (
                    "recsys_workers 2",
                    "recsys_requests_total",
                    "recsys_event_loop_lag_ms",
                    "recsys_process_resident_memory_bytes",
                    "recsys_rate_limit_rejected_total",
                )),
            },
            "feedback": feedback,
        })
        runtime = after_burst.get("runtime") or {}
        error_counts = (after_burst.get("requests") or {}).get(
            "error_counts", {})
        report["checks"] = {
            "live_200": bool(live and live.get("status") == "alive"),
            "ready_200": bool(ready and ready.get("status") == "ready"),
            "all_workers_registered": (
                after_burst.get("worker_count") == args.workers),
            "all_models_loaded": (
                after_burst.get("model", {}).get("loaded_workers")
                == args.workers),
            "paced_all_success": paced["successes"] == args.paced_requests,
            "burst_rate_limited": (
                burst["status_counts"].get("429", 0) > 0
                and error_counts.get("rate_limited", 0) > 0),
            "aggregate_request_count_exact": (
                after_burst.get("requests", {}).get("total")
                == total_expected),
            "every_worker_served_business_traffic": (
                len(worker_snapshots) == args.workers
                and all(
                    snapshot.get("requests", {}).get("total", 0) > 0
                    for snapshot in worker_snapshots)),
            "runtime_cpu_present": runtime.get("cpu_percent_sum") is not None,
            "runtime_rss_present": (runtime.get("rss_bytes_sum") or 0) > 0,
            "event_loop_lag_sampled": (
                runtime.get("event_loop_lag_ms", {}).get("samples", 0) > 0),
            "prometheus_valid": (
                prometheus_status == 200
                and report["prometheus"]["required_lines_present"]),
            "successful_recommendations_logged": (
                feedback["recommendations"]
                == paced["successes"] + burst["successes"]),
        }
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        report["failure"] = failure
    finally:
        if proc is not None:
            stop_server(proc)
        # The test harness deliberately force-kills the isolated process tree
        # so it cannot signal the parent terminal on Windows. Verify stale-file
        # crash recovery separately; graceful lifespan cleanup has a unit test.
        time.sleep(1.1)
        FileMetricsRegistry(
            metrics_dir, stale_after_s=1).collect()
        remaining = [path for path in metrics_dir.iterdir() if path.is_file()] \
            if metrics_dir.exists() else []
        report["checks"]["crash_registry_cleanup"] = not remaining
        report["registry_files_after_shutdown"] = [
            path.name for path in remaining]
        report["passed"] = (
            failure is None and all(report["checks"].values()))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")

    print(json.dumps({
        "passed": report["passed"],
        "checks": report["checks"],
        "paced": report.get("paced"),
        "burst": report.get("burst"),
        "report": str(output),
    }, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
