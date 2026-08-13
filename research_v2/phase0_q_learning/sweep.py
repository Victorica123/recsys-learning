"""Multi-seed sensitivity sweep for Phase 0 tabular Q-learning."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

from .q_learning import evaluate_policy, train_q_learning


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "experiments"
ALPHAS = (0.0, 0.01, 0.05, 0.1, 0.5)
GAMMAS = (0.0, 0.5, 0.8, 0.95, 0.99)
EPISODE_COUNTS = (20, 100, 500, 2_000)
RAW_FIELDS = (
    "alpha",
    "gamma",
    "episodes",
    "seed",
    "success_rate",
    "avg_return",
    "avg_steps",
    "elapsed_seconds",
)


def load_completed(path: Path) -> set[tuple[float, float, int, int]]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            (
                float(row["alpha"]),
                float(row["gamma"]),
                int(row["episodes"]),
                int(row["seed"]),
            )
            for row in csv.DictReader(handle)
        }


def summarize(raw_path: Path, summary_path: Path) -> None:
    groups: dict[tuple[float, float, int], list[dict[str, float]]] = defaultdict(list)
    with raw_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (float(row["alpha"]), float(row["gamma"]), int(row["episodes"]))
            groups[key].append(
                {
                    "success_rate": float(row["success_rate"]),
                    "avg_return": float(row["avg_return"]),
                    "avg_steps": float(row["avg_steps"]),
                }
            )

    fields = (
        "alpha",
        "gamma",
        "episodes",
        "seeds",
        "success_mean",
        "success_std",
        "success_ci95",
        "reliable_seed_rate",
        "return_mean",
        "steps_mean",
    )
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (alpha, gamma, episodes), rows in sorted(groups.items()):
            successes = [row["success_rate"] for row in rows]
            success_std = statistics.stdev(successes) if len(successes) > 1 else 0.0
            writer.writerow(
                {
                    "alpha": alpha,
                    "gamma": gamma,
                    "episodes": episodes,
                    "seeds": len(rows),
                    "success_mean": round(statistics.mean(successes), 6),
                    "success_std": round(success_std, 6),
                    "success_ci95": round(1.96 * success_std / math.sqrt(len(rows)), 6),
                    "reliable_seed_rate": round(
                        sum(value >= 0.95 for value in successes) / len(successes), 6
                    ),
                    "return_mean": round(
                        statistics.mean(row["avg_return"] for row in rows), 6
                    ),
                    "steps_mean": round(
                        statistics.mean(row["avg_steps"] for row in rows), 6
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--tag", default="phase0_q_sweep")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.seeds < 1 or args.eval_episodes < 1:
        parser.error("--seeds and --eval-episodes must be positive")

    configurations = list(product(ALPHAS, GAMMAS, EPISODE_COUNTS, range(args.seeds)))
    print(
        f"configs={len(ALPHAS) * len(GAMMAS) * len(EPISODE_COUNTS)}, "
        f"seeds={args.seeds}, runs={len(configurations)}",
        flush=True,
    )
    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / f"{args.tag}_raw.csv"
    summary_path = args.output_dir / f"{args.tag}_summary.csv"
    if raw_path.exists() and not args.resume:
        raise FileExistsError(f"{raw_path} exists; choose another --tag or use --resume")

    completed = load_completed(raw_path) if args.resume else set()
    mode = "a" if args.resume and raw_path.exists() else "w"
    start = time.time()
    new_runs = 0
    with raw_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        if mode == "w":
            writer.writeheader()
        for index, (alpha, gamma, episodes, seed) in enumerate(configurations, start=1):
            key = (alpha, gamma, episodes, seed)
            if key in completed:
                continue
            run_start = time.time()
            q_table, _ = train_q_learning(
                episodes=episodes,
                alpha=alpha,
                gamma=gamma,
                seed=seed,
                eval_every=0,
            )
            metrics = evaluate_policy(
                q_table,
                episodes=args.eval_episodes,
                seed=100_000 + seed,
            )
            writer.writerow(
                {
                    "alpha": alpha,
                    "gamma": gamma,
                    "episodes": episodes,
                    "seed": seed,
                    **{name: round(value, 6) for name, value in metrics.items()},
                    "elapsed_seconds": round(time.time() - run_start, 6),
                }
            )
            handle.flush()
            new_runs += 1
            if new_runs % 250 == 0 or index == len(configurations):
                elapsed = time.time() - start
                rate = new_runs / max(elapsed, 1e-9)
                remaining = (len(configurations) - index) / max(rate, 1e-9)
                print(
                    f"progress={index}/{len(configurations)} "
                    f"elapsed={elapsed / 60:.1f}min eta={remaining / 60:.1f}min",
                    flush=True,
                )

    summarize(raw_path, summary_path)
    print(f"raw={raw_path}", flush=True)
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
