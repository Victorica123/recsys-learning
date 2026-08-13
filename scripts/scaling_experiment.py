"""Measure worker scaling and overload protection for the recommendation API.

This orchestrator starts ``serve.py`` at 1/2/4 uvicorn workers, drives the
existing dependency-free load generator (`load_test_api`) across a sweep of
client concurrencies, and records the throughput / tail-latency scaling curve.
It then runs a focused overload experiment that contrasts a *small* admission
gate (load shedding on) against a *wide* gate (effectively off) at high
concurrency, to show that shedding bounds the tail latency of admitted
requests instead of letting the queue explode.

Everything is CPU-only and uses one keep-alive connection per client worker.
Results are written as a single JSON artifact under ``experiments/`` and a
figure under ``figures/``.  Nothing is retrained.

Example:
    .venv/Scripts/python.exe scripts/scaling_experiment.py --tag 20260726
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from load_test_api import (  # noqa: E402
    fetch_json, metrics_delta, run_load, run_warmup, safe_tag, summarize)

PY = sys.executable


# --------------------------------------------------------------------- server
def _health_ok(base_url: str, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(base_url + "/health", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def start_server(port: int, workers: int, max_in_flight: int,
                 boot_timeout: float = 120.0, *,
                 rate_limit: float = 0.0, rate_burst: int = 16,
                 metrics_interval: float = 1.0,
                 metrics_dir: Path | None = None,
                 feedback_db: Path | None = None) -> subprocess.Popen:
    """Launch ``serve.py`` and block until /health reports the model loaded."""
    command = [
        PY, str(ROOT / "serve.py"), "--host", "127.0.0.1", "--port",
        str(port), "--workers", str(workers), "--max-in-flight",
        str(max_in_flight), "--rate-limit", str(rate_limit), "--rate-burst",
        str(rate_burst), "--metrics-interval", str(metrics_interval),
        "--preload",
    ]
    if metrics_dir is not None:
        command.extend(["--metrics-dir", str(metrics_dir)])
    if feedback_db is not None:
        command.extend(["--feedback-db", str(feedback_db)])
    proc = subprocess.Popen(
        command, cwd=str(ROOT), stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=(
            subprocess.CREATE_NO_WINDOW
            if sys.platform.startswith("win") else 0))
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.perf_counter() + boot_timeout
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early (code {proc.returncode})")
        if _health_ok(base_url, timeout=2.0):
            return proc
        time.sleep(0.5)
    stop_server(proc)
    raise TimeoutError(f"server on :{port} did not become healthy in "
                       f"{boot_timeout:.0f}s")


def wait_for_workers(base_url: str, expected: int,
                     timeout: float = 120.0) -> list[int]:
    """Wait until a fresh metrics connection has observed every loaded worker."""
    seen: set[int] = set()
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        snapshot = fetch_json(base_url, "/metrics", timeout=2.0)
        process = (snapshot or {}).get("process") or {}
        model = (snapshot or {}).get("model") or {}
        pid = process.get("pid")
        if model.get("loaded") and isinstance(pid, int):
            seen.add(pid)
        if len(seen) >= expected:
            return sorted(seen)
        time.sleep(0.1)
    raise TimeoutError(
        f"observed {len(seen)}/{expected} loaded workers on {base_url}: "
        f"{sorted(seen)}")


def sample_process_memory(pids: list[int]) -> dict:
    """Return RSS for exact server/worker PIDs without adding a dependency."""
    unique_pids = sorted(set(int(pid) for pid in pids))
    if not unique_pids:
        return {"tracked_rss_mb": None, "processes": [],
                "error": "no process ids supplied"}
    try:
        if sys.platform.startswith("win"):
            ids = ",".join(str(pid) for pid in unique_pids)
            command = (
                f"$ids=@({ids}); "
                "Get-Process -Id $ids -ErrorAction SilentlyContinue | "
                "Select-Object Id,ProcessName,WorkingSet64 | "
                "ConvertTo-Json -Compress"
            )
            completed = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive",
                 "-Command", command],
                capture_output=True, text=True, check=True, timeout=15)
            payload = json.loads(completed.stdout)
            rows = payload if isinstance(payload, list) else [payload]
            processes = [{
                "pid": int(row["Id"]),
                "name": row["ProcessName"],
                "rss_mb": round(int(row["WorkingSet64"]) / 1024 ** 2, 3),
            } for row in rows]
        else:  # pragma: no cover - project is normally run on Windows
            completed = subprocess.run(
                ["ps", "-o", "pid=,rss=,comm=", "-p",
                 ",".join(str(pid) for pid in unique_pids)],
                capture_output=True, text=True, check=True, timeout=15)
            processes = []
            for line in completed.stdout.splitlines():
                pid, rss_kb, name = line.split(maxsplit=2)
                processes.append({
                    "pid": int(pid),
                    "name": name,
                    "rss_mb": round(int(rss_kb) / 1024, 3),
                })
        return {
            "tracked_rss_mb": round(
                sum(row["rss_mb"] for row in processes), 3),
            "processes": sorted(processes, key=lambda row: row["pid"]),
        }
    except Exception as exc:
        return {
            "tracked_rss_mb": None,
            "processes": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def stop_server(proc: subprocess.Popen) -> None:
    """Terminate the uvicorn parent and every worker subprocess (Windows tree)."""
    if proc.poll() is not None:
        return
    if sys.platform.startswith("win"):
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    else:  # pragma: no cover - project runs on Windows
        proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:  # pragma: no cover
        proc.kill()


# ----------------------------------------------------------------- one measure
def measure(base_url: str, path: str, requests: int, concurrency: int,
            warmup: int, timeout: float,
            trust_server_metrics: bool = True) -> dict:
    """Warm up, run the load, and (single-worker only) fold in the shed delta.

    ``/metrics`` is a per-process, in-memory counter.  With more than one
    uvicorn worker a single scrape lands on one of N workers at random, so the
    before/after deltas are not aggregatable — they can even go negative or
    exceed the request count.  When ``trust_server_metrics`` is False we skip
    the scrapes entirely and record ``None`` for the server-side deltas; the
    client-measured latency/throughput below is then the authoritative signal.
    """
    run_warmup(base_url, path, warmup, timeout)
    before = fetch_json(base_url, "/metrics", timeout) \
        if trust_server_metrics else None
    rows, wall_s = run_load(base_url, path, requests, concurrency, timeout)
    after = fetch_json(base_url, "/metrics", timeout) \
        if trust_server_metrics else None
    summary = summarize(rows, wall_s)
    if trust_server_metrics:
        adm_before = (before or {}).get("admission") or {}
        adm_after = (after or {}).get("admission") or {}
        summary["admission_delta"] = {
            "admitted": adm_after.get("admitted_total", 0)
            - adm_before.get("admitted_total", 0),
            "shed": adm_after.get("shed_total", 0)
            - adm_before.get("shed_total", 0),
            "peak_in_flight": adm_after.get("peak_in_flight", 0),
        }
        summary["metrics_delta"] = metrics_delta(before, after)
    else:
        summary["admission_delta"] = None
        summary["metrics_delta"] = None
    summary["server_side_metrics_reliable"] = trust_server_metrics
    summary["concurrency"] = concurrency
    return summary


def _transient_timeouts(summary: dict) -> int:
    """Count client-side hangs (TimeoutError / dropped connection) in a run.

    These are transient and corrupt a point's throughput/tail — unlike genuine
    overload, which shows up as 429s or high-but-finite latency, never a
    30-second hang.  Used to decide whether a point is worth re-measuring.
    """
    return (summary.get("exception_counts", {}).get("TimeoutError", 0)
            + summary.get("status_counts", {}).get("0", 0))


def measure_stable(base_url: str, path: str, requests: int, concurrency: int,
                   warmup: int, timeout: float, trust_server_metrics: bool,
                   retries: int = 1) -> dict:
    """Measure a point, re-measuring once if it hit a transient client hang.

    Only client-side timeouts trigger a retry; 429s and elevated-but-finite
    latency (real capacity signals) are kept as measured.  Returns the sample
    with the fewest errors so a lingering transient can't silently win.
    """
    best = None
    for attempt in range(retries + 1):
        s = measure(base_url, path, requests, concurrency, warmup, timeout,
                    trust_server_metrics=trust_server_metrics)
        if best is None or s["errors"] < best["errors"]:
            best = s
        if _transient_timeouts(s) == 0:
            return s
        if attempt < retries:
            print(f"    [retry] c={concurrency}: {_transient_timeouts(s)} "
                  f"transient timeout(s); re-measuring ...")
            time.sleep(1.0)
    return best


def _row(workers: int, s: dict) -> str:
    lat = s["latency_ms"]
    adm = s.get("admission_delta")
    shed = adm["shed"] if adm else "n/a"
    return (f"  w={workers} c={s['concurrency']:<3} "
            f"rps={s['throughput_rps']:>8.1f}  "
            f"ok_rps={s['success_throughput_rps']:>8.1f}  "
            f"p50={lat['p50']:>7.1f}  p95={lat['p95']:>7.1f}  "
            f"p99={lat['p99']:>8.1f}  err={s['errors']:<4} "
            f"shed={shed}")


# ------------------------------------------------------------------------ main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--path", default="/recommend?user_id=1&k=10")
    parser.add_argument("--workers", default="1,2,4",
                        help="comma-separated uvicorn worker counts")
    parser.add_argument("--concurrency", default="1,4,16,32",
                        help="comma-separated client concurrencies (scaling)")
    parser.add_argument("--requests", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--scale-max-in-flight", type=int, default=256,
                        help="wide gate for the pure scaling sweep (no shedding)")
    parser.add_argument("--overload-concurrency", type=int, default=64)
    parser.add_argument("--overload-gate", type=int, default=8,
                        help="single small gate (kept for compatibility)")
    parser.add_argument(
        "--overload-gates", default=None,
        help="optional comma-separated gate sweep, e.g. 1,2,4,8,16")
    parser.add_argument(
        "--skip-scaling", action="store_true",
        help="run only the overload gate sweep")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    worker_counts = [int(w) for w in args.workers.split(",") if w.strip()]
    concurrencies = [int(c) for c in args.concurrency.split(",") if c.strip()]
    if (not args.skip_scaling) and (not worker_counts or not concurrencies):
        parser.error("workers and concurrency must each list at least one value")
    overload_gates = (
        [int(g) for g in args.overload_gates.split(",") if g.strip()]
        if args.overload_gates else [args.overload_gate]
    )
    if any(g <= 0 for g in overload_gates):
        parser.error("overload gates must be positive")

    run_tag = safe_tag(args.tag or datetime.now(
        timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output = ROOT / "experiments" / f"serve_scaling_{run_tag}.json"
    if output.exists():
        parser.error(f"refusing to overwrite existing artifact: {output}")
    existing_feedback = list(
        (ROOT / "runtime" / "feedback").glob(
            f"scaling-{run_tag}-*.sqlite3"))
    if existing_feedback:
        parser.error(
            "refusing to append to existing synthetic feedback database: "
            f"{existing_feedback[0]}")

    base_url = f"http://127.0.0.1:{args.port}"

    # 1) Scaling sweep: wide gate so throughput reflects capacity, not shedding.
    scaling: dict[str, list[dict]] = {}
    resources: dict[str, dict] = {}
    for workers in ([] if args.skip_scaling else worker_counts):
        print(f"[scaling] starting {workers}-worker server on :{args.port} ...")
        feedback_db = (
            ROOT / "runtime" / "feedback"
            / f"scaling-{run_tag}-workers-{workers}.sqlite3"
        )
        proc = start_server(
            args.port, workers, args.scale_max_in_flight,
            feedback_db=feedback_db)
        try:
            # Metrics exposes the serving PID. Repeated fresh connections prove
            # every worker completed preload before any measured request.
            worker_pids = wait_for_workers(base_url, workers)
            resources[str(workers)] = {
                "manager_pid": proc.pid,
                "worker_pids": worker_pids,
                "memory": sample_process_memory([proc.pid, *worker_pids]),
            }
            memory = resources[str(workers)]["memory"]["tracked_rss_mb"]
            print(f"  ready pids={worker_pids}; tracked_rss_mb={memory}")
            runs = []
            for c in concurrencies:
                # Only a single-worker server has aggregatable /metrics counters;
                # for >1 worker the client-side numbers are authoritative.
                s = measure_stable(base_url, args.path, args.requests, c,
                                   args.warmup, args.timeout,
                                   trust_server_metrics=(workers == 1))
                runs.append(s)
                print(_row(workers, s))
            scaling[str(workers)] = runs
        finally:
            stop_server(proc)
            time.sleep(1.0)

    # 2) Overload demo: small gate (shedding) vs wide gate (queue explodes),
    #    single worker, same high concurrency.
    print(f"[overload] concurrency={args.overload_concurrency} "
          f"gates={overload_gates} vs wide gate ...")
    overload = {}
    gates = list(dict.fromkeys([*overload_gates, args.scale_max_in_flight]))
    for gate in gates:
        label = ("gate_wide" if gate == args.scale_max_in_flight
                 else f"gate_{gate}")
        feedback_db = (
            ROOT / "runtime" / "feedback"
            / f"scaling-{run_tag}-{label}.sqlite3"
        )
        proc = start_server(
            args.port, 1, gate, feedback_db=feedback_db)
        try:
            run_load(base_url, args.path, 40, 8, args.timeout)
            s = measure(base_url, args.path, args.requests,
                        args.overload_concurrency, args.warmup, args.timeout)
            s["max_in_flight"] = gate
            overload[label] = s
            print(_row(1, s) + f"  gate={gate}")
        finally:
            stop_server(proc)
            time.sleep(1.0)

    report = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "notes": {
            "server_side_metrics": (
                "Per-worker in-process /metrics counters are not aggregatable "
                "across a single scrape; for worker_count > 1 the client-measured "
                "latency/throughput is authoritative and admission_delta/"
                "metrics_delta are recorded as null (server_side_metrics_reliable "
                "= false). Single-worker and overload runs keep valid deltas."
            ),
            "memory": (
                "RSS tracks the uvicorn manager PID plus worker PIDs observed "
                "from /metrics. It excludes unrelated processes and is a "
                "point-in-time localhost measurement after model preload."
            ),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
        },
        "config": {
            "path": args.path,
            "requests": args.requests,
            "warmup": args.warmup,
            "worker_counts": worker_counts,
            "concurrencies": concurrencies,
            "scale_max_in_flight": args.scale_max_in_flight,
            "overload_concurrency": args.overload_concurrency,
            "overload_gates": overload_gates,
            "host": "127.0.0.1",
            "port": args.port,
        },
        "resources": resources,
        "scaling": scaling,
        "overload": overload,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(f"report: {output}")

    if not args.no_figure:
        try:
            if scaling:
                figure = plot_scaling(report, run_tag)
                print(f"figure: {figure}")
            overload_figure = plot_overload(report, run_tag)
            print(f"overload figure: {overload_figure}")
        except Exception as exc:  # pragma: no cover - plotting is best-effort
            print(f"figure skipped: {type(exc).__name__}: {exc}")
    return 0


def plot_scaling(report: dict, run_tag: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scaling = report["scaling"]
    worker_counts = sorted(int(w) for w in scaling)
    fig, (ax_rps, ax_p99, ax_mem) = plt.subplots(1, 3, figsize=(15, 4.2))

    for w in worker_counts:
        runs = sorted(scaling[str(w)], key=lambda s: s["concurrency"])
        xs = [s["concurrency"] for s in runs]
        ax_rps.plot(xs, [s["throughput_rps"] for s in runs],
                    marker="o", label=f"{w} worker" + ("s" if w > 1 else ""))
        ax_p99.plot(xs, [s["latency_ms"]["p99"] for s in runs], marker="o",
                    label=f"{w} worker" + ("s" if w > 1 else ""))

    ax_rps.set_title("Throughput vs client concurrency")
    ax_rps.set_xlabel("client concurrency")
    ax_rps.set_ylabel("throughput (RPS)")
    ax_rps.set_xscale("log", base=2)
    ax_rps.grid(True, alpha=0.3)
    ax_rps.legend()

    ax_p99.set_title("Tail latency (p99) vs client concurrency")
    ax_p99.set_xlabel("client concurrency")
    ax_p99.set_ylabel("p99 latency (ms)")
    ax_p99.set_xscale("log", base=2)
    ax_p99.grid(True, alpha=0.3)
    ax_p99.legend()

    resources = report.get("resources", {})
    memory = [
        (resources.get(str(w), {}).get("memory", {})
         .get("tracked_rss_mb") or 0)
        for w in worker_counts
    ]
    ax_mem.bar([str(w) for w in worker_counts], memory)
    ax_mem.set_title("Loaded server memory")
    ax_mem.set_xlabel("worker count")
    ax_mem.set_ylabel("tracked RSS (MiB)")
    ax_mem.grid(True, axis="y", alpha=0.3)
    for index, value in enumerate(memory):
        ax_mem.text(index, value, f"{value:.0f}", ha="center", va="bottom")

    fig.suptitle("Recommendation API worker scaling (ML-1M, CPU)")
    fig.tight_layout()
    figure = ROOT / "figures" / f"serve_worker_scaling_{run_tag}.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=130)
    plt.close(fig)
    return figure


def plot_overload(report: dict, run_tag: str) -> Path:
    """Plot the latency/capacity cost of each admission-control threshold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = sorted(report["overload"].values(),
                  key=lambda summary: summary["max_in_flight"])
    gates = [summary["max_in_flight"] for summary in runs]
    accepted_p99 = [
        summary["latency_ms_by_status"]["200"]["p99"] for summary in runs
    ]
    success_rps = [summary["success_throughput_rps"] for summary in runs]
    shed_pct = [
        100 * summary["status_counts"].get("429", 0) / summary["requests"]
        for summary in runs
    ]

    fig, (ax_latency, ax_capacity) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax_latency.plot(gates, accepted_p99, marker="o", color="tab:red")
    ax_latency.set_xscale("log", base=2)
    ax_latency.set_title("Accepted-request tail latency")
    ax_latency.set_xlabel("max in-flight per worker")
    ax_latency.set_ylabel("HTTP 200 p99 (ms)")
    ax_latency.grid(True, alpha=0.3)

    ax_capacity.plot(gates, success_rps, marker="o", color="tab:blue",
                     label="successful RPS")
    ax_capacity.set_xscale("log", base=2)
    ax_capacity.set_title("Capacity vs load shedding")
    ax_capacity.set_xlabel("max in-flight per worker")
    ax_capacity.set_ylabel("successful RPS", color="tab:blue")
    ax_capacity.tick_params(axis="y", labelcolor="tab:blue")
    ax_capacity.grid(True, alpha=0.3)
    ax_shed = ax_capacity.twinx()
    ax_shed.plot(gates, shed_pct, marker="s", color="tab:orange",
                 label="shed %")
    ax_shed.set_ylabel("HTTP 429 share (%)", color="tab:orange")
    ax_shed.tick_params(axis="y", labelcolor="tab:orange")
    ax_shed.set_ylim(0, 100)

    fig.suptitle("Admission-control trade-off (concurrency 64, ML-1M, CPU)")
    fig.tight_layout()
    figure = ROOT / "figures" / f"serve_overload_gates_{run_tag}.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=130)
    plt.close(fig)
    return figure


if __name__ == "__main__":
    raise SystemExit(main())
