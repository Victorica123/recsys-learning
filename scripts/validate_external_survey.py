# -*- coding: utf-8 -*-
"""Validate diversity-list satisfaction on the GroupLens Personality 2018 survey.

This is an external transportability check, not an evaluation of the exact
DeepFM reranker. The survey's diversity treatment and this project's
``diversity_weight=0.15`` are related concepts but are not identical policies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feedback import FeedbackStore, compare_logged_policies  # noqa: E402


DATASET = (
    ROOT / "data" / "external" / "personality-isf2018" / "raw"
    / "personality-isf2018" / "personality-data.csv"
)
ARCHIVE = (
    ROOT / "data" / "external" / "personality-isf2018"
    / "personality-isf2018.zip"
)
EXPECTED_SHA256 = (
    "910f44919d8483143e1704cff880133f3fee227008792043e7dd6747cd1b6a01"
)
DEFAULT_FEEDBACK_DB = ROOT / "runtime" / "feedback" / "events.sqlite3"
OUTCOMES = ("is_personalized", "enjoy_watching")
GROUPS = {
    "default": ("all", "default"),
    "diversity_low": ("diversity", "low"),
    "diversity_medium": ("diversity", "medium"),
    "diversity_high": ("diversity", "high"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_survey(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle, skipinitialspace=True):
            row = {
                str(key).strip(): value.strip()
                for key, value in raw.items()
            }
            rows.append(row)
    required = {
        "userid", "openness", "assigned metric", "assigned condition",
        *OUTCOMES,
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("unexpected Personality 2018 schema")
    return rows


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sample")
    return sum(values) / len(values)


def bootstrap_difference(
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    rng = random.Random(seed)
    treatment_size = len(treatment)
    control_size = len(control)
    draws = []
    for _ in range(samples):
        treatment_mean = sum(
            treatment[rng.randrange(treatment_size)]
            for _ in range(treatment_size)
        ) / treatment_size
        control_mean = sum(
            control[rng.randrange(control_size)]
            for _ in range(control_size)
        ) / control_size
        draws.append(treatment_mean - control_mean)
    draws.sort()
    return [
        draws[int(0.025 * (samples - 1))],
        draws[int(0.975 * (samples - 1))],
    ]


def permutation_p_value(
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    treatment_size = len(treatment)
    combined = list(treatment) + list(control)
    observed = abs(mean(treatment) - mean(control))
    extreme = 0
    for _ in range(samples):
        shuffled = combined.copy()
        rng.shuffle(shuffled)
        difference = abs(
            mean(shuffled[:treatment_size])
            - mean(shuffled[treatment_size:])
        )
        if difference >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (samples + 1)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    count = len(ordered)
    adjusted = {}
    running_max = 0.0
    for rank, name in enumerate(ordered):
        corrected = min(1.0, (count - rank) * p_values[name])
        running_max = max(running_max, corrected)
        adjusted[name] = running_max
    return adjusted


def survey_analysis(
    rows: Sequence[dict],
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict:
    values: dict[str, dict[str, list[float]]] = {
        group: {outcome: [] for outcome in OUTCOMES}
        for group in GROUPS
    }
    openness: dict[str, list[float]] = {
        group: [] for group in GROUPS}
    for row in rows:
        for group, assignment in GROUPS.items():
            if (
                row["assigned metric"],
                row["assigned condition"],
            ) == assignment:
                for outcome in OUTCOMES:
                    values[group][outcome].append(float(row[outcome]))
                openness[group].append(float(row["openness"]))
                break

    summaries = {}
    for group in GROUPS:
        summaries[group] = {
            "users": len(openness[group]),
            "openness_mean": mean(openness[group]),
            **{
                f"{outcome}_mean": mean(values[group][outcome])
                for outcome in OUTCOMES
            },
        }

    contrasts = {}
    raw_p_values = {}
    contrast_index = 0
    for group in ("diversity_low", "diversity_medium", "diversity_high"):
        for outcome in OUTCOMES:
            contrast_index += 1
            treatment = values[group][outcome]
            control = values["default"][outcome]
            name = f"{group}_minus_default:{outcome}"
            difference = mean(treatment) - mean(control)
            p_value = permutation_p_value(
                treatment,
                control,
                samples=permutation_samples,
                seed=seed + 100 * contrast_index,
            )
            contrasts[name] = {
                "treatment_users": len(treatment),
                "control_users": len(control),
                "treatment_mean": mean(treatment),
                "control_mean": mean(control),
                "mean_difference": difference,
                "bootstrap_95_interval": bootstrap_difference(
                    treatment,
                    control,
                    samples=bootstrap_samples,
                    seed=seed + contrast_index,
                ),
                "two_sided_permutation_p": p_value,
            }
            raw_p_values[name] = p_value

    adjusted = holm_adjust(raw_p_values)
    for name, adjusted_p in adjusted.items():
        contrasts[name]["holm_adjusted_p"] = adjusted_p

    significant = [
        name for name, result in contrasts.items()
        if result["holm_adjusted_p"] < 0.05
    ]
    return {
        "dataset_users": len(rows),
        "analyzed_users": sum(
            summary["users"] for summary in summaries.values()),
        "group_summaries": summaries,
        "contrasts": contrasts,
        "multiple_testing": {
            "method": "Holm family-wise error correction",
            "family_size": len(contrasts),
            "significant_at_0.05": significant,
        },
    }


def completed_personal_pilot_rows(
    rows: Sequence[dict],
    actor_id: str,
) -> list[dict]:
    pilot_rows = [
        row for row in rows
        if row.get("policy_name") == "epsilon_slate_personal_pilot_v1"
        and str(row.get("actor_id")) == actor_id
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in pilot_rows:
        grouped[str(row["recommendation_id"])].append(row)
    complete_ids = {
        recommendation_id
        for recommendation_id, candidates in grouped.items()
        if (
            (served := [
                row for row in candidates
                if row.get("served_rank") is not None
            ])
            and all(row.get("feedback") for row in served)
        )
    }
    return [
        row for row in pilot_rows
        if str(row["recommendation_id"]) in complete_ids
    ]


def local_pilot_analysis(database: Path, actor_id: str) -> dict | None:
    store = FeedbackStore(database)
    rows = completed_personal_pilot_rows(store.export_rows(), actor_id)
    if not rows:
        return None
    return compare_logged_policies(
        rows,
        target_k=5,
        bootstrap_samples=1000,
        confidence=0.90,
        min_recommendations=20,
        min_actors=1,
        min_effective_sample_size=5.0,
        min_impression_coverage=1.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--feedback-db", type=Path, default=DEFAULT_FEEDBACK_DB)
    parser.add_argument("--actor-id", default="local-evaluator")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--permutation-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = ROOT / "experiments" / f"external_survey_{args.tag}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if args.bootstrap_samples <= 0 or args.permutation_samples <= 0:
        raise ValueError("sample counts must be positive")
    archive_hash = sha256_file(args.archive)
    if archive_hash != EXPECTED_SHA256:
        raise ValueError(
            f"unexpected archive SHA-256: {archive_hash}")

    rows = load_survey(args.dataset)
    survey = survey_analysis(
        rows,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        seed=args.seed,
    )
    local = local_pilot_analysis(args.feedback_db, args.actor_id)
    report = {
        "protocol": {
            "purpose": (
                "external transportability check for diversity-list "
                "satisfaction"
            ),
            "source": "GroupLens Personality 2018",
            "source_url": (
                "https://grouplens.org/datasets/personality-2018/"
            ),
            "archive_sha256": archive_hash,
            "bootstrap_samples": args.bootstrap_samples,
            "permutation_samples": args.permutation_samples,
            "seed": args.seed,
            "limitations": [
                "Survey diversity conditions are not the exact DeepFM "
                "diversity_weight=0.15 policy.",
                "Likert outcomes measure perceived personalization and "
                "expected enjoyment, not observed clicks or retention.",
                "This external analysis cannot replace a randomized test on "
                "the target product population.",
            ],
        },
        "external_survey": survey,
        "local_personal_pilot": local,
        "decision": {
            "global_rollout_supported": False,
            "personal_candidate_supported": bool(
                local
                and local["gate"]["formal_ready"]
                and local["gate"]["decision"] == "promote_candidate"
            ),
            "summary": (
                "The local pilot supports diversity reranking for the current "
                "evaluator, while the external survey does not establish a "
                "universal satisfaction benefit. Keep the candidate personal "
                "or experimental rather than globally promoting it."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "output": str(output),
        "dataset_users": survey["dataset_users"],
        "analyzed_users": survey["analyzed_users"],
        "decision": report["decision"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

