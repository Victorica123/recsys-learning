"""Exercise the complete production serving surface with real checkpoints.

Starts a two-worker API instance (or uses --base-url for an existing container),
verifies live/ready probes, cross-worker JSON
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

from load_test_api import (  # noqa: E402
    run_load, safe_tag, summarize, request_headers, request_json)
from scaling_experiment import (  # noqa: E402
    fetch_json, start_server, stop_server, wait_for_workers)

import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from metrics_registry import FileMetricsRegistry  # noqa: E402
from feedback import FeedbackStore  # noqa: E402


def fetch_text(base_url: str, path: str, timeout: float) -> tuple[int, str]:
    parts = urlsplit(base_url)
    connection_type = (http.client.HTTPSConnection if parts.scheme == "https"
                       else http.client.HTTPConnection)
    connection = connection_type(
        parts.hostname, parts.port, timeout=timeout)
    try:
        connection.request("GET", parts.path.rstrip("/") + path,
                           headers=request_headers({"Connection": "close"}))
        response = connection.getresponse()
        return response.status, response.read().decode("utf-8")
    finally:
        connection.close()


def paced_load(base_url: str, path: str, requests: int, interval_s: float,
               timeout: float, *, method: str = "GET",
               json_body: dict | None = None) -> tuple[list[dict], float]:
    """Send a below-limit keep-alive stream for the clean success phase."""
    parts = urlsplit(base_url)
    connection_type = (http.client.HTTPSConnection if parts.scheme == "https"
                       else http.client.HTTPConnection)
    connection = connection_type(parts.hostname, parts.port, timeout=timeout)
    body = None if json_body is None else json.dumps(json_body).encode()
    target = parts.path.rstrip("/") + path
    rows = []
    wall_started = time.perf_counter()
    try:
        for index in range(requests):
            iteration_started = time.perf_counter()
            request_id = f"paced-{index}-{uuid.uuid4().hex[:12]}"
            try:
                connection.request(method, target, body=body, headers=request_headers({
                    "Accept": "application/json",
                    "Connection": "keep-alive",
                    "X-Actor-ID": "demo-production-smoke",
                    "X-Request-ID": request_id,
                }))
                response = connection.getresponse()
                response.read()
                rows.append({
                    "status": response.status,
                    "latency_ms": (
                        time.perf_counter() - iteration_started) * 1000,
                    "correlation_ok": (
                        response.getheader("X-Request-ID") == request_id),
                    "error_class": response.getheader("X-Error-Class"),
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
                connection = connection_type(
                    parts.hostname, parts.port, timeout=timeout)
            remaining = interval_s - (
                time.perf_counter() - iteration_started)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        connection.close()
    return rows, time.perf_counter() - wall_started


def wait_for_aggregate(base_url: str, minimum_requests: int,
                       workers: int, timeout: float, *,
                       route: str | None = None, fresh: bool = False) -> dict:
    deadline = time.perf_counter() + timeout
    last = None
    initial_publications = None
    while time.perf_counter() < deadline:
        last = fetch_json(base_url, "/metrics/aggregate", timeout=2)
        if last and last.get("worker_count") == workers:
            publications = {row["pid"]: row.get("published_at")
                            for row in last.get("workers", [])}
            count = (last.get("routes", {}).get(route, {}).get("count", 0)
                     if route else last.get("requests", {}).get("total", 0))
            if fresh and initial_publications is None:
                initial_publications = publications
            elif (count >= minimum_requests and (
                    not fresh or all(value != initial_publications.get(pid)
                                     for pid, value in publications.items()))):
                return last
        time.sleep(0.1)
    raise TimeoutError(
        f"aggregate did not reach workers={workers}, requests="
        f"{minimum_requests}; last={last}")


def remote_feedback_stats(base_url: str, timeout: float) -> dict:
    for _ in range(4):
        status, payload, headers = request_json(
            base_url, "/events/stats", timeout=timeout)
        if status == 200:
            return payload
        if status != 429:
            raise RuntimeError(f"feedback stats returned HTTP {status}")
        delay = float(next((value for key, value in headers.items()
                            if key.lower() == "retry-after"), "1"))
        time.sleep(min(max(delay, 0.1), 2))
    raise RuntimeError("feedback stats still rate limited after bounded retries")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=None,
                        help="verify an existing service; never starts/stops it")
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
    external = args.base_url is not None
    if feedback_db.exists() and not external:
        parser.error(
            f"refusing to append to existing feedback database: {feedback_db}")
    base_url = args.base_url or f"http://127.0.0.1:{args.port}"
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        parser.error("base-url must be an absolute http(s) URL")
    path = "/v1/recommendations"
    request_options = {"method": "POST", "json_body": {"user_id": 1, "k": 10}}
    registry = FileMetricsRegistry(metrics_dir, stale_after_s=5)
    report = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "config": vars(args),
        "traffic_kind": "scripted_smoke",
        "service_ownership": "external" if external else "managed_local",
        "request": {"method": "POST", "path": path, "body": {"user_id": 1, "k": 10}},
        "checks": {},
    }
    proc = None
    failure = None
    try:
        if not external:
            proc = start_server(
                args.port, args.workers, args.max_in_flight,
                rate_limit=args.rate_limit, rate_burst=args.rate_burst,
                metrics_interval=0.25, metrics_dir=metrics_dir,
                feedback_db=feedback_db)
        worker_pids = wait_for_workers(base_url, args.workers)
        report["worker_pids"] = worker_pids

        live = fetch_json(base_url, "/live", args.timeout)
        ready = fetch_json(base_url, "/ready", args.timeout)
        feedback_before = (remote_feedback_stats(base_url, args.timeout) if external
                           else FeedbackStore(feedback_db).stats())
        baseline = wait_for_aggregate(
            base_url, minimum_requests=0, workers=args.workers,
            timeout=args.timeout, fresh=True)
        before_count = baseline.get("routes", {}).get(path, {}).get("count", 0)

        paced_rows, paced_wall = paced_load(
            base_url, path, args.paced_requests, args.paced_interval,
            args.timeout, **request_options)
        paced = summarize(paced_rows, paced_wall)
        after_paced = wait_for_aggregate(
            base_url, before_count + args.paced_requests, args.workers,
            args.timeout, route=path)

        burst_rows, burst_wall = run_load(
            base_url, path, args.burst_requests,
            args.burst_concurrency, args.timeout, **request_options)
        burst = summarize(burst_rows, burst_wall)
        total_expected = args.paced_requests + args.burst_requests
        after_burst = wait_for_aggregate(
            base_url, before_count + total_expected, args.workers, args.timeout,
            route=path)
        prometheus_status, prometheus = fetch_text(
            base_url, "/metrics/prometheus", args.timeout)
        per_worker_totals = {
            str(worker["pid"]): worker.get("route_counts", {}).get(path, 0)
            for worker in after_burst.get("workers", [])
        }
        per_worker_before = {
            str(worker["pid"]): worker.get("route_counts", {}).get(path, 0)
            for worker in baseline.get("workers", [])
        }
        feedback = (remote_feedback_stats(base_url, args.timeout) if external
                    else FeedbackStore(feedback_db).stats())

        report.update({
            "probes": {"live": live, "ready": ready},
            "paced": paced,
            "burst": burst,
            "aggregate": {
                "baseline": baseline,
                "after_paced": after_paced,
                "after_burst": after_burst,
            },
            "per_worker_recommendation_totals": per_worker_totals,
            "prometheus": {
                "status": prometheus_status,
                "line_count": len(prometheus.splitlines()),
                "required_lines_present": all(token in prometheus for token in (
                    f"recsys_workers {args.workers}",
                    "recsys_requests_total",
                    "recsys_event_loop_lag_ms",
                    "recsys_process_resident_memory_bytes",
                    "recsys_rate_limit_rejected_total",
                )),
            },
            "feedback": feedback,
            "feedback_before": feedback_before,
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
                after_burst.get("routes", {}).get(path, {}).get("count", 0)
                - before_count == total_expected),
            "every_worker_served_business_traffic": (
                len(per_worker_totals) == args.workers
                and all(value > per_worker_before.get(pid, 0)
                        for pid, value in per_worker_totals.items())),
            "runtime_cpu_present": runtime.get("cpu_percent_sum") is not None,
            "runtime_rss_present": (runtime.get("rss_bytes_sum") or 0) > 0,
            "event_loop_lag_sampled": (
                runtime.get("event_loop_lag_ms", {}).get("samples", 0) > 0),
            "prometheus_valid": (
                prometheus_status == 200
                and report["prometheus"]["required_lines_present"]),
            "successful_recommendations_logged": (
                feedback["recommendations"]
                - feedback_before["recommendations"]
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
        if not external:
            time.sleep(1.1)
            FileMetricsRegistry(metrics_dir, stale_after_s=1).collect()
            remaining = [entry for entry in metrics_dir.iterdir() if entry.is_file()] \
                if metrics_dir.exists() else []
            report["checks"]["crash_registry_cleanup"] = not remaining
            report["registry_files_after_shutdown"] = [entry.name for entry in remaining]
        else:
            report["lifecycle_note"] = "External service left running; shutdown cleanup not tested."
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
