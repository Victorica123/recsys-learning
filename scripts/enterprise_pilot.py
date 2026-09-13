"""Run the enterprise synthetic seed through validation, replay and paired ranking.

Example:
    .venv/Scripts/python.exe scripts/enterprise_pilot.py --tag enterprise-seed-v1
    .venv/Scripts/python.exe scripts/enterprise_pilot.py --dataset <seed-data-dir> --tag imported-v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from enterprise_dataset import EnterpriseDataset, validate_request
from enterprise_evaluation import evaluate_dataset, history_popularity, render_report
from enterprise_feedback import KnowledgeFeedbackStore
from enterprise_recommendation import KnowledgeRanker, canonical_json, policy_bundle
from enterprise_seed import generate_dataset


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def run(*, tag: str, dataset_dir: Path | None = None, seed: int = 20260913,
        requests: int = 640, output_root: Path | None = None) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", tag):
        raise ValueError("tag must be a simple filename component")
    output = (output_root or ROOT / "experiments") / f"enterprise_pilot_{tag}"
    if dataset_dir is not None:
        # Validate external input before creating any local output or database.
        dataset = EnterpriseDataset.load(dataset_dir)
    output.mkdir(parents=True, exist_ok=False)
    if dataset_dir is None:
        dataset_dir = output / "seed"
        generate_dataset(dataset_dir, seed=seed, requests=requests)
        dataset = EnterpriseDataset.load(dataset_dir)
    ranker = KnowledgeRanker()
    report, slates = evaluate_dataset(dataset, ranker)
    report["run"] = {
        "tag": tag, "created_at": datetime.now(timezone.utc).isoformat(),
        "code_sha256": {
            name: hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()
            for name in ("enterprise_seed.py", "enterprise_recommendation.py",
                         "enterprise_dataset.py", "enterprise_feedback.py", "enterprise_evaluation.py")},
    }
    with KnowledgeFeedbackStore(output / "feedback.sqlite3") as store:
        first = store.import_dataset(dataset)
        repeat = store.import_dataset(dataset)
        if repeat != {"requests_added": 0, "events_added": 0}:
            raise RuntimeError("idempotent replay failed")
        with (output / "replay.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for tenant in dataset.tenants.values():
                for row in store.replay(tenant["tenant_id"], tenant["workspace_id"]):
                    handle.write(canonical_json(row) + "\n")
        report["replay_store"] = {"first_import": first, "repeat_import": repeat, "counts": store.counts()}
    with (output / "paired_slates.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in slates:
            handle.write(canonical_json(row) + "\n")
    write_json(output / "policy.json", policy_bundle())
    request = next((row for row in dataset.requests if row["split"] == "test"), dataset.requests[-1])
    principal = dataset.principal(request)
    recommendation = ranker.recommend(
        principal, request, dataset.items, popularity=history_popularity(dataset, request))
    recommendation.update({key: request[key] for key in ("split", "split_start", "split_end")})
    validate_request(recommendation)
    write_json(output / "agent_service_example.json", {
        "synthetic": True, "principal_resolved_by_server": vars(principal),
        "request": {key: request[key] for key in (
            "recommendation_id", "tenant_id", "workspace_id", "workspace_type", "actor_id",
            "requested_at", "context_type", "context_id", "context", "synthetic")},
        "response": recommendation,
        "note": "Example ranks the full authorized catalog; paired benchmark uses judged pools only.",
    })
    shutil.copyfile(ROOT / "src" / "enterprise_recommendation.py",
                    output / "enterprise_recommendation.py")
    write_json(output / "report.json", report)
    (output / "report.md").write_text(render_report(report), encoding="utf-8", newline="\n")
    write_json(output / "manifest.json", {
        "schema_version": 1, "synthetic": True, "tag": tag,
        "source_sha256": dataset.source_hashes,
        "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(output.iterdir()) if path.is_file()},
    })
    return output


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="unique run name; existing runs are never overwritten")
    parser.add_argument("--dataset", type=Path, help="read an original seed data/ directory")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--requests", type=int, default=640)
    args = parser.parse_args()
    try:
        output = run(tag=args.tag, dataset_dir=args.dataset, seed=args.seed, requests=args.requests)
    except (ValueError, OSError, KeyError) as exc:
        print(f"enterprise pilot failed: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: enterprise synthetic pilot -> {output}")
    print("Decision: not_promoted_synthetic_only; see report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
