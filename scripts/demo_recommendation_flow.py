"""Reproduce one recommendation -> impression -> click -> retry scenario.

All writes are scripted demo events, tagged with a demo actor. They are not
real-user feedback or evidence of recommendation lift.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from load_test_api import request_json, safe_tag

ROOT = Path(__file__).resolve().parent.parent


def run_flow(base_url: str, tag: str, *, user_id: int = 1,
             k: int = 5, timeout: float = 30) -> dict:
    report = {
        "schema_version": 1, "recorded_at": datetime.now(timezone.utc).isoformat(),
        "traffic_kind": "scripted_demo", "real_user_feedback": False,
        "actor_id": f"demo-{tag}", "user_id": user_id, "checks": {}, "steps": [],
    }

    def call(name, path, payload=None, expected=200):
        for _ in range(4):
            status, body, headers = request_json(
                base_url, path, method="GET" if payload is None else "POST",
                json_body=payload, timeout=timeout,
                headers={"X-Actor-ID": report["actor_id"]})
            if status != 429:
                break
            delay = float(next((value for key, value in headers.items()
                                if key.lower() == "retry-after"), "1"))
            time.sleep(min(max(delay, 0.1), 2))
        report["steps"].append({"name": name, "path": path, "status": status,
                                "response": body})
        if status != expected:
            raise RuntimeError(f"{name}: expected HTTP {expected}, got {status}: {body}")
        return body

    try:
        before = call("counts_before", "/events/stats")
        slate = call("recommend", "/v1/recommendations", {"user_id": user_id, "k": k})
        items = slate["recommendations"]
        if not items or not slate["recommendation_id"]:
            raise RuntimeError("chosen user has no available recommendations")
        report["recommendation"] = slate
        rid = slate["recommendation_id"]
        ids = [item["movie_id"] for item in items]
        event = {
            "recommendation_id": rid, "movie_id": ids[0], "event_type": "click",
            "event_id": f"demo-{tag}-click",
            "occurred_at": datetime.now(timezone.utc).isoformat(),
        }
        call("feedback_without_impression_rejected", "/v1/events/feedback", event, 400)
        exposure = {"recommendation_id": rid, "movie_ids": ids}
        call("impression", "/v1/events/impression", exposure, 201)
        repeated_exposure = call("impression_retry", "/v1/events/impression", exposure)
        call("click", "/v1/events/feedback", event, 201)
        repeated_click = call("click_retry", "/v1/events/feedback", event)
        call("conflicting_event_rejected", "/v1/events/feedback", {**event, "value": 2}, 400)
        after = call("counts_after", "/events/stats")
        keys = ("recommendations", "recommendation_items", "impressions", "feedback_events")
        delta = {key: after[key] - before[key] for key in keys}
        report.update(counts_before=before, counts_after=after, count_delta=delta,
                      feedback_event_id=event["event_id"])
        report["checks"] = {
            "default_retrieval_policy": slate["ranking_policy"] == "retrieval",
            "one_recommendation_registered": delta["recommendations"] == 1,
            "candidates_registered": delta["recommendation_items"] >= len(ids),
            "one_impression_per_displayed_item": delta["impressions"] == len(ids),
            "impression_retry_is_noop": repeated_exposure["inserted"] == 0,
            "click_retry_is_noop": repeated_click["duplicate"] is True,
            "one_feedback_event_after_retries": delta["feedback_events"] == 1,
        }
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
    report["passed"] = "failure" not in report and all(report["checks"].values())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    if args.user_id <= 0 or not 1 <= args.k <= 100 or args.timeout <= 0:
        parser.error("user-id/timeout must be positive; k must be between 1 and 100")
    tag = safe_tag(args.tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output = ROOT / "experiments" / f"demo_flow_{tag}.json"
    if output.exists():
        parser.error(f"refusing to overwrite existing report: {output}")
    report = run_flow(args.base_url, tag, user_id=args.user_id, k=args.k, timeout=args.timeout)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "passed": report["passed"], "traffic_kind": "scripted_demo",
        "user_id": args.user_id, "counts": report.get("count_delta"),
        "checks": report["checks"], "failure": report.get("failure"),
        "report": str(output),
    }, ensure_ascii=True, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
