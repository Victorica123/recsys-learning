# -*- coding: utf-8 -*-
"""Generate a synthetic exposure dataset with oracle values and validate OPE.

This is an estimator test bench, not evidence about real users. The behavior
policy uses logged epsilon-slate propensities; counterfactual reward means are
known, so IPS/SNIPS/DR errors can be measured rather than guessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from feedback import (  # noqa: E402
    REPLAY_SCHEMA_VERSION,
    compare_logged_policies,
    diversity_rerank,
    select_slate,
    validate_replay_rows,
    write_jsonl,
)


GENRES = ("Drama", "Comedy", "Action", "Sci-Fi", "Romance")


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def generate_synthetic_exposures(
    *,
    recommendations: int,
    pool_size: int,
    slate_k: int,
    exploration_rate: float,
    actors: int,
    seed: int,
) -> list[dict]:
    if recommendations <= 0 or pool_size <= 0 or slate_k <= 0:
        raise ValueError("recommendations, pool_size and slate_k must be positive")
    if slate_k > pool_size:
        raise ValueError("slate_k cannot exceed pool_size")
    if actors <= 0:
        raise ValueError("actors must be positive")
    rng = random.Random(seed)
    rows = []
    for rec_index in range(recommendations):
        actor_index = rec_index % actors
        actor_taste = actor_index % len(GENRES)
        candidates = []
        for rank in range(1, pool_size + 1):
            # Top scorer positions deliberately repeat Drama so the diversity
            # target policy differs from scorer Top-k and has a non-zero oracle delta.
            genre_index = (
                0 if rank <= 3 else 1 + ((rank - 4) % (len(GENRES) - 1)))
            score = 1.0 - 0.055 * rank + 0.015 * math.sin(rec_index + rank)
            # The oracle intentionally contains information not perfectly
            # captured by scorer rank, making outcome modeling useful but imperfect.
            affinity = 0.65 if genre_index == actor_taste else 0.0
            diversity_quality = 0.35 if genre_index in (1, 2) else 0.0
            reward_mean = sigmoid(-1.25 + 1.35 * score + affinity + diversity_quality)
            candidates.append({
                "movie_id": rank,
                "title": f"synthetic-{rank}",
                "genres": GENRES[genre_index],
                "score": score,
                "oracle_reward_mean": reward_mean,
            })
        _served, logged = select_slate(
            candidates, slate_k, exploration_rate, rng=rng)
        recommendation_id = f"synthetic-{rec_index:06d}"
        for row in logged:
            impressed = row["served_rank"] is not None
            reward = (
                float(rng.random() < row["oracle_reward_mean"])
                if impressed else None)
            rows.append({
                "schema_version": REPLAY_SCHEMA_VERSION,
                "recommendation_id": recommendation_id,
                "actor_id": f"synthetic-actor-{actor_index:03d}",
                "user_id": actor_index,
                "model_version": "synthetic-oracle-v1",
                "policy_name": "epsilon_slate_synthetic_v1",
                "exploration_rate": exploration_rate,
                "requested_k": slate_k,
                "candidate_count": pool_size,
                "created_at": f"synthetic-step-{rec_index:06d}",
                **row,
                "impressed": impressed,
                "feedback": [],
                "reward": reward,
            })
    return rows


def oracle_policy_values(
    rows: list[dict], *, target_k: int, diversity_weight: float
) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["recommendation_id"]), []).append(row)
    baseline, candidate = [], []
    for candidates in grouped.values():
        candidates.sort(key=lambda row: int(row["candidate_rank"]))
        baseline_rows = candidates[:target_k]
        candidate_rows = diversity_rerank(
            candidates, target_k, diversity_weight=diversity_weight)
        baseline.append(float(np.mean(
            [row["oracle_reward_mean"] for row in baseline_rows])))
        candidate.append(float(np.mean(
            [row["oracle_reward_mean"] for row in candidate_rows])))
    return {
        "baseline_reward_per_item": float(np.mean(baseline)),
        "candidate_reward_per_item": float(np.mean(candidate)),
        "delta_reward_per_item": float(np.mean(np.asarray(candidate) - baseline)),
    }


def reserve(tag: str) -> tuple[Path, Path]:
    dataset = ROOT / "experiments" / f"ope_synthetic_{tag}.jsonl"
    report = ROOT / "experiments" / f"ope_synthetic_{tag}_report.json"
    existing = [str(path) for path in (dataset, report) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite synthetic OPE artifacts:\n" + "\n".join(existing))
    return dataset, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--recommendations", type=int, default=5000)
    parser.add_argument("--pool-size", type=int, default=10)
    parser.add_argument("--slate-k", type=int, default=3)
    parser.add_argument("--exploration-rate", type=float, default=0.30)
    parser.add_argument("--actors", type=int, default=20)
    parser.add_argument("--diversity-weight", type=float, default=0.15)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dataset_path, report_path = reserve(args.tag)

    rows = generate_synthetic_exposures(
        recommendations=args.recommendations,
        pool_size=args.pool_size,
        slate_k=args.slate_k,
        exploration_rate=args.exploration_rate,
        actors=args.actors,
        seed=args.seed)
    profile = validate_replay_rows(rows)
    truth = oracle_policy_values(
        rows, target_k=args.slate_k,
        diversity_weight=args.diversity_weight)
    ope = compare_logged_policies(
        rows, target_k=args.slate_k,
        diversity_weight=args.diversity_weight,
        bootstrap_samples=args.bootstrap_samples,
        min_recommendations=min(200, args.recommendations),
        min_actors=min(5, args.actors),
        min_effective_sample_size=50,
        min_impression_coverage=0.99,
        outcome_folds=5,
        outcome_ridge=1.0,
        seed=args.seed)

    ips = {
        "baseline": ope["baseline"]["ips_reward_per_item"],
        "candidate": ope["candidate"]["ips_reward_per_item"],
        "delta": ope["paired_delta"]["ips_reward_per_item"],
    }
    snips = {
        "baseline": ope["baseline"]["snips_reward_per_item"],
        "candidate": ope["candidate"]["snips_reward_per_item"],
        # SNIPS values are normalized separately; their difference is still a
        # useful robustness diagnostic but not the pre-registered paired IPS.
        "delta": (
            ope["candidate"]["snips_reward_per_item"]
            - ope["baseline"]["snips_reward_per_item"]),
    }
    dr_block = ope["doubly_robust_cross_check"]
    dr = {
        "baseline": dr_block["baseline"]["dr_reward_per_item"],
        "candidate": dr_block["candidate"]["dr_reward_per_item"],
        "delta": dr_block["paired_delta"]["dr_reward_per_item"],
    }
    truth_flat = {
        "baseline": truth["baseline_reward_per_item"],
        "candidate": truth["candidate_reward_per_item"],
        "delta": truth["delta_reward_per_item"],
    }
    errors = {
        estimator: {
            key: abs(values[key] - truth_flat[key])
            for key in truth_flat
        }
        for estimator, values in (("ips", ips), ("snips", snips), ("dr", dr))
    }

    write_jsonl(rows, dataset_path)
    report = {
        "schema_version": 2,
        "experiment": "synthetic logged-exposure OPE calibration",
        "warning": "synthetic estimator validation; not evidence about real users",
        "config": vars(args),
        "dataset": {
            **profile,
            "path": str(dataset_path.relative_to(ROOT)),
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        },
        "oracle": truth_flat,
        "estimates": {"ips": ips, "snips": snips, "dr": dr},
        "absolute_error": errors,
        "ope_report": ope,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "oracle": truth_flat, "estimates": report["estimates"],
        "absolute_error": errors,
        "outcome_model": dr_block["outcome_model"],
    }, ensure_ascii=False, indent=2))
    print(f"dataset: {dataset_path.relative_to(ROOT)}")
    print(f"report:  {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
