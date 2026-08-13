"""Export immutable feedback logs and run a minimal item-level OPE report."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feedback import (FeedbackStore, compare_logged_policies,  # noqa: E402
                      evaluate_top_k_policy, write_jsonl)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", default="runtime/feedback/events.sqlite3",
        help="SQLite event database, relative to the project root")
    parser.add_argument("--tag", required=True, help="unique artifact tag")
    parser.add_argument("--target-k", type=int, default=10)
    parser.add_argument("--diversity-weight", type=float, default=0.15)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--min-recommendations", type=int, default=200)
    parser.add_argument("--min-actors", type=int, default=5)
    parser.add_argument("--min-ess", type=float, default=100.0)
    parser.add_argument("--min-impression-coverage", type=float, default=0.95)
    args = parser.parse_args()
    if args.target_k <= 0:
        parser.error("--target-k must be positive")
    if args.diversity_weight < 0:
        parser.error("--diversity-weight cannot be negative")
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must be between 0 and 1")
    if (args.min_recommendations <= 0 or args.min_actors <= 0
            or args.min_ess <= 0):
        parser.error("minimum sample gates must be positive")
    if not 0.0 <= args.min_impression_coverage <= 1.0:
        parser.error("--min-impression-coverage must be between 0 and 1")

    database = Path(args.db)
    if not database.is_absolute():
        database = ROOT / database
    dataset_path = ROOT / "experiments" / f"feedback_{args.tag}.jsonl"
    report_path = ROOT / "experiments" / f"feedback_{args.tag}_ope.json"
    if dataset_path.exists() or report_path.exists():
        parser.error(
            f"tag {args.tag!r} already exists; refusing to overwrite artifacts")

    store = FeedbackStore(database)
    rows = store.export_rows()
    if not rows:
        parser.error("feedback database contains no recommendation rows")
    write_jsonl(rows, dataset_path)
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    report = {
        "protocol": {
            "unit": "candidate item within a recommendation",
            "reward": (
                "click +1, like +2, dislike -1, skip 0, "
                "dwell_seconds capped at +1"
            ),
            "target_k": args.target_k,
            "dataset_sha256": digest,
        },
        "source": {
            "database": str(database.resolve()),
            "dataset": str(dataset_path.resolve()),
        },
        "store_stats": store.stats(),
        "evaluation_v1": evaluate_top_k_policy(rows, args.target_k),
        "formal_comparison": compare_logged_policies(
            rows,
            target_k=args.target_k,
            diversity_weight=args.diversity_weight,
            bootstrap_samples=args.bootstrap_samples,
            confidence=args.confidence,
            min_recommendations=args.min_recommendations,
            min_actors=args.min_actors,
            min_effective_sample_size=args.min_ess,
            min_impression_coverage=args.min_impression_coverage,
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"dataset: {dataset_path}")
    print(f"report:  {report_path}")


if __name__ == "__main__":
    main()
