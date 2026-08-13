"""Durable recommendation exposure and feedback logging.

The store deliberately uses SQLite from the Python standard library.  Each
operation opens a short-lived connection so multiple Uvicorn workers can share
one WAL database.  The logged candidate pool includes unserved items: this is
necessary to detect whether an offline target policy has support under the
behavior policy instead of silently reporting a biased estimate.
"""
from __future__ import annotations

import json
import math
import random
import sqlite3
import threading
import uuid
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


EVENT_TYPES = frozenset({
    "click", "like", "dislike", "skip", "dwell_seconds",
})
_DEFAULT_EVENT_VALUES = {
    "click": 1.0,
    "like": 1.0,
    "dislike": 1.0,
    "skip": 1.0,
    "dwell_seconds": 0.0,
}


class FeedbackValidationError(ValueError):
    """Raised when an event cannot be attached to a logged impression."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def select_slate(
    candidates: Sequence[dict],
    k: int,
    exploration_rate: float,
    rng: random.Random | None = None,
) -> tuple[list[dict], list[dict]]:
    """Select a slate and return ``(served, all_candidates_for_logging)``.

    The behavior policy is a mixture:

    * with probability ``1-epsilon`` serve the scorer's top-k;
    * with probability ``epsilon`` sample k candidates uniformly.

    ``behavior_propensity`` is the exact marginal inclusion probability for
    each item.  It is suitable for item-level Horvitz-Thompson/IPS estimates;
    it is not a joint slate probability.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not 0.0 <= exploration_rate <= 1.0:
        raise ValueError("exploration_rate must be between 0 and 1")

    pool = [dict(item) for item in candidates]
    n = len(pool)
    take = min(k, n)
    if take == 0:
        return [], []

    rng = rng or random.SystemRandom()
    explore = exploration_rate > 0.0 and n > take \
        and rng.random() < exploration_rate
    selected_indices = (
        sorted(rng.sample(range(n), take)) if explore else list(range(take))
    )
    selected = set(selected_indices)

    logged = []
    served = []
    for index, item in enumerate(pool):
        in_exploit = index < take
        if n == take:
            propensity = 1.0
        else:
            propensity = (
                (1.0 - exploration_rate) * float(in_exploit)
                + exploration_rate * take / n
            )
        row = dict(item)
        row["candidate_rank"] = index + 1
        row["served_rank"] = (
            selected_indices.index(index) + 1 if index in selected else None
        )
        row["behavior_propensity"] = float(propensity)
        logged.append(row)
        if index in selected:
            output = dict(item)
            output["rank"] = row["served_rank"]
            output["candidate_rank"] = row["candidate_rank"]
            output["behavior_propensity"] = row["behavior_propensity"]
            served.append(output)
    return served, logged


class FeedbackStore:
    """Multi-process-safe SQLite event store with idempotent event writes."""

    def __init__(self, path: str | Path, timeout_s: float = 5.0):
        self.path = Path(path)
        self.timeout_s = float(timeout_s)
        self._initialized = False
        self._initialize_lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.path, timeout=self.timeout_s, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute(
            f"PRAGMA busy_timeout = {max(1, int(self.timeout_s * 1000))}")
        return connection

    def initialize(self) -> None:
        if self._initialized:
            return
        with self._initialize_lock:
            if self._initialized:
                return
            with closing(self._connect()) as db:
                db.execute("PRAGMA journal_mode = WAL")
                db.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS recommendations (
                        recommendation_id TEXT PRIMARY KEY,
                        request_id TEXT,
                        actor_id TEXT,
                        user_id INTEGER NOT NULL,
                        model_version TEXT NOT NULL,
                        policy_name TEXT NOT NULL,
                        exploration_rate REAL NOT NULL,
                        requested_k INTEGER NOT NULL,
                        candidate_count INTEGER NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS recommendation_items (
                        recommendation_id TEXT NOT NULL,
                        movie_id INTEGER NOT NULL,
                        candidate_rank INTEGER NOT NULL,
                        served_rank INTEGER,
                        score REAL NOT NULL,
                        behavior_propensity REAL NOT NULL,
                        title TEXT,
                        genres TEXT,
                        PRIMARY KEY (recommendation_id, movie_id),
                        FOREIGN KEY (recommendation_id)
                            REFERENCES recommendations(recommendation_id)
                            ON DELETE CASCADE
                    );
                    CREATE TABLE IF NOT EXISTS impressions (
                        recommendation_id TEXT NOT NULL,
                        movie_id INTEGER NOT NULL,
                        impressed_at TEXT NOT NULL,
                        PRIMARY KEY (recommendation_id, movie_id),
                        FOREIGN KEY (recommendation_id, movie_id)
                            REFERENCES recommendation_items(
                                recommendation_id, movie_id)
                            ON DELETE CASCADE
                    );
                    CREATE TABLE IF NOT EXISTS feedback_events (
                        event_id TEXT PRIMARY KEY,
                        recommendation_id TEXT NOT NULL,
                        movie_id INTEGER NOT NULL,
                        event_type TEXT NOT NULL,
                        value REAL NOT NULL,
                        occurred_at TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (recommendation_id, movie_id)
                            REFERENCES impressions(recommendation_id, movie_id)
                            ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS idx_recommendations_created
                        ON recommendations(created_at);
                    CREATE INDEX IF NOT EXISTS idx_feedback_rec_item
                        ON feedback_events(recommendation_id, movie_id);
                    """
                )
                item_columns = {
                    row["name"] for row in db.execute(
                        "PRAGMA table_info(recommendation_items)")
                }
                for name in ("title", "genres"):
                    if name not in item_columns:
                        db.execute(
                            f"ALTER TABLE recommendation_items "
                            f"ADD COLUMN {name} TEXT")
                recommendation_columns = {
                    row["name"] for row in db.execute(
                        "PRAGMA table_info(recommendations)")
                }
                if "actor_id" not in recommendation_columns:
                    db.execute(
                        "ALTER TABLE recommendations ADD COLUMN actor_id TEXT")
            self._initialized = True

    def record_recommendation(
        self,
        *,
        recommendation_id: str,
        request_id: str | None,
        user_id: int,
        model_version: str,
        policy_name: str,
        exploration_rate: float,
        requested_k: int,
        candidates: Sequence[dict],
        created_at: str | None = None,
        actor_id: str | None = None,
    ) -> None:
        if not candidates:
            raise FeedbackValidationError("cannot log an empty candidate pool")
        if actor_id is not None and (
            not isinstance(actor_id, str) or not 1 <= len(actor_id) <= 128
        ):
            raise FeedbackValidationError(
                "actor_id must contain 1 to 128 characters")
        ids = [int(item["movie_id"]) for item in candidates]
        if len(ids) != len(set(ids)):
            raise FeedbackValidationError("candidate movie_ids must be unique")
        for item in candidates:
            propensity = float(item["behavior_propensity"])
            if not 0.0 <= propensity <= 1.0:
                raise FeedbackValidationError(
                    "behavior_propensity must be between 0 and 1")

        self.initialize()
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute(
                    """
                    INSERT INTO recommendations (
                        recommendation_id, request_id, actor_id, user_id,
                        model_version, policy_name, exploration_rate,
                        requested_k, candidate_count, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        recommendation_id, request_id, actor_id, int(user_id),
                        model_version, policy_name, float(exploration_rate),
                        int(requested_k), len(candidates),
                        created_at or utc_now(),
                    ),
                )
                db.executemany(
                    """
                    INSERT INTO recommendation_items (
                        recommendation_id, movie_id, candidate_rank,
                        served_rank, score, behavior_propensity, title, genres
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            recommendation_id, int(item["movie_id"]),
                            int(item["candidate_rank"]),
                            (None if item.get("served_rank") is None
                             else int(item["served_rank"])),
                            float(item["score"]),
                            float(item["behavior_propensity"]),
                            item.get("title"),
                            item.get("genres"),
                        )
                        for item in candidates
                    ],
                )
            except Exception:
                db.rollback()
                raise
            else:
                db.commit()

    def record_impressions(
        self,
        recommendation_id: str,
        movie_ids: Iterable[int] | None = None,
        impressed_at: str | None = None,
    ) -> dict:
        self.initialize()
        with closing(self._connect()) as db:
            served = db.execute(
                """
                SELECT movie_id FROM recommendation_items
                WHERE recommendation_id = ? AND served_rank IS NOT NULL
                ORDER BY served_rank
                """,
                (recommendation_id,),
            ).fetchall()
            if not served:
                raise FeedbackValidationError(
                    f"unknown recommendation_id {recommendation_id}")
            served_ids = {int(row["movie_id"]) for row in served}
            requested = (
                sorted(served_ids) if movie_ids is None
                else [int(movie_id) for movie_id in movie_ids]
            )
            if not requested:
                raise FeedbackValidationError("movie_ids cannot be empty")
            invalid = sorted(set(requested) - served_ids)
            if invalid:
                raise FeedbackValidationError(
                    f"movie_ids were not served: {invalid}")
            before = db.total_changes
            db.executemany(
                """
                INSERT OR IGNORE INTO impressions (
                    recommendation_id, movie_id, impressed_at
                ) VALUES (?, ?, ?)
                """,
                [
                    (recommendation_id, movie_id, impressed_at or utc_now())
                    for movie_id in dict.fromkeys(requested)
                ],
            )
            inserted = db.total_changes - before
            return {
                "recommendation_id": recommendation_id,
                "requested": len(set(requested)),
                "inserted": inserted,
                "duplicates": len(set(requested)) - inserted,
            }

    def record_feedback(
        self,
        *,
        recommendation_id: str,
        movie_id: int,
        event_type: str,
        value: float | None = None,
        event_id: str | None = None,
        occurred_at: str | None = None,
    ) -> dict:
        if event_type not in EVENT_TYPES:
            raise FeedbackValidationError(
                f"event_type must be one of {sorted(EVENT_TYPES)}")
        numeric_value = (
            _DEFAULT_EVENT_VALUES[event_type] if value is None else float(value)
        )
        if not math.isfinite(numeric_value) or numeric_value < 0:
            raise FeedbackValidationError(
                "feedback value must be a finite non-negative number")
        identifier = event_id or uuid.uuid4().hex
        self.initialize()
        try:
            with closing(self._connect()) as db:
                before = db.total_changes
                db.execute(
                    """
                    INSERT OR IGNORE INTO feedback_events (
                        event_id, recommendation_id, movie_id, event_type,
                        value, occurred_at, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identifier, recommendation_id, int(movie_id),
                        event_type, numeric_value, occurred_at or utc_now(),
                        utc_now(),
                    ),
                )
                inserted = db.total_changes - before
                if not inserted:
                    existing = db.execute(
                        """
                        SELECT recommendation_id, movie_id, event_type, value
                        FROM feedback_events WHERE event_id = ?
                        """,
                        (identifier,),
                    ).fetchone()
                    same_event = (
                        existing is not None
                        and existing["recommendation_id"] == recommendation_id
                        and int(existing["movie_id"]) == int(movie_id)
                        and existing["event_type"] == event_type
                        and float(existing["value"]) == numeric_value
                    )
                    if not same_event:
                        raise FeedbackValidationError(
                            "event_id already belongs to a different event")
        except sqlite3.IntegrityError as exc:
            raise FeedbackValidationError(
                "feedback requires a matching logged impression") from exc
        return {
            "event_id": identifier,
            "inserted": bool(inserted),
            "duplicate": not bool(inserted),
        }

    def stats(self) -> dict:
        self.initialize()
        with closing(self._connect()) as db:
            counts = {
                table: int(db.execute(
                    f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in (
                    "recommendations", "recommendation_items",
                    "impressions", "feedback_events",
                )
            }
            served = int(db.execute(
                "SELECT COUNT(*) FROM recommendation_items "
                "WHERE served_rank IS NOT NULL").fetchone()[0])
            clicked = int(db.execute(
                "SELECT COUNT(DISTINCT recommendation_id || ':' || movie_id) "
                "FROM feedback_events WHERE event_type = 'click'"
            ).fetchone()[0])
        return {
            **counts,
            "served_items": served,
            "impression_coverage": (
                round(counts["impressions"] / served, 6) if served else None
            ),
            "observed_ctr": (
                round(clicked / counts["impressions"], 6)
                if counts["impressions"] else None
            ),
        }

    def export_rows(self) -> list[dict]:
        """Return candidate-level replay rows, including unobserved candidates."""
        self.initialize()
        with closing(self._connect()) as db:
            candidates = db.execute(
                """
                SELECT r.recommendation_id, r.actor_id, r.user_id,
                       r.model_version,
                       r.policy_name, r.exploration_rate, r.requested_k,
                       r.candidate_count, r.created_at,
                       i.movie_id, i.candidate_rank, i.served_rank, i.score,
                       i.behavior_propensity, i.title, i.genres,
                       x.impressed_at
                FROM recommendations r
                JOIN recommendation_items i USING (recommendation_id)
                LEFT JOIN impressions x USING (recommendation_id, movie_id)
                ORDER BY r.created_at, r.recommendation_id, i.candidate_rank
                """
            ).fetchall()
            feedback = db.execute(
                """
                SELECT recommendation_id, movie_id, event_type, value
                FROM feedback_events
                ORDER BY occurred_at, event_id
                """
            ).fetchall()

        grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for event in feedback:
            grouped[(event["recommendation_id"], int(event["movie_id"]))].append(
                {"event_type": event["event_type"], "value": event["value"]})

        rows = []
        for candidate in candidates:
            row = dict(candidate)
            events = grouped[(row["recommendation_id"], int(row["movie_id"]))]
            row["impressed"] = row.pop("impressed_at") is not None
            row["feedback"] = events
            row["reward"] = feedback_reward(events) if row["impressed"] else None
            rows.append(row)
        return rows


def feedback_reward(events: Iterable[dict]) -> float:
    """Map raw events to an explicit, small item-level reward."""
    reward = 0.0
    for event in events:
        kind = event["event_type"]
        value = float(event["value"])
        if kind == "click":
            reward += value
        elif kind == "like":
            reward += 2.0 * value
        elif kind == "dislike":
            reward -= value
        elif kind == "dwell_seconds":
            reward += min(value, 300.0) / 300.0
        elif kind != "skip":
            raise FeedbackValidationError(f"unknown event type {kind}")
    return reward


def evaluate_top_k_policy(rows: Sequence[dict], target_k: int) -> dict:
    """Evaluate deterministic scorer Top-k with IPS and self-normalized IPS."""
    if target_k <= 0:
        raise ValueError("target_k must be positive")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["recommendation_id"])].append(row)
    if not grouped:
        raise ValueError("dataset is empty")

    ips_by_recommendation = []
    weighted_reward = 0.0
    weight_sum = 0.0
    weight_sq_sum = 0.0
    unsupported = 0
    target_items = 0
    observed_target_items = 0
    observed_rewards = []

    for candidates in grouped.values():
        available = len(candidates)
        take = min(target_k, available)
        contribution = 0.0
        for row in candidates:
            if row.get("impressed"):
                observed_rewards.append(float(row.get("reward") or 0.0))
            if int(row["candidate_rank"]) > take:
                continue
            target_items += 1
            propensity = float(row["behavior_propensity"])
            if propensity <= 0.0:
                unsupported += 1
                continue
            if row.get("impressed"):
                observed_target_items += 1
                weight = 1.0 / propensity
                reward = float(row.get("reward") or 0.0)
                contribution += weight * reward
                weighted_reward += weight * reward
                weight_sum += weight
                weight_sq_sum += weight * weight
        ips_by_recommendation.append(contribution / take)

    support_ok = unsupported == 0
    ips = sum(ips_by_recommendation) / len(ips_by_recommendation)
    snips = weighted_reward / weight_sum if weight_sum else None
    ess = weight_sum * weight_sum / weight_sq_sum if weight_sq_sum else 0.0
    return {
        "estimator": "item_inclusion_ips_v1",
        "target_policy": f"deterministic_scorer_top_{target_k}",
        "recommendations": len(grouped),
        "candidate_rows": len(rows),
        "target_items": target_items,
        "observed_target_items": observed_target_items,
        "unsupported_target_items": unsupported,
        "support_ok": support_ok,
        "observed_behavior_reward_mean": (
            sum(observed_rewards) / len(observed_rewards)
            if observed_rewards else None
        ),
        "ips_reward_per_item": ips if support_ok else None,
        "snips_reward_per_item": snips if support_ok else None,
        "effective_sample_size": ess,
        "warning": (
            None if support_ok
            else "Target policy has zero-propensity actions; enable exploration "
                 "before using OPE for this target."
        ),
    }


def diversity_rerank(
    candidates: Sequence[dict],
    target_k: int,
    diversity_weight: float = 0.15,
) -> list[dict]:
    """Greedily trade scorer probability for previously uncovered genres."""
    if target_k <= 0:
        raise ValueError("target_k must be positive")
    if diversity_weight < 0.0:
        raise ValueError("diversity_weight cannot be negative")
    remaining = [dict(row) for row in candidates]
    selected = []
    covered: set[str] = set()
    while remaining and len(selected) < target_k:
        def utility(row):
            genres = {
                genre for genre in str(row.get("genres") or "").split("|")
                if genre
            }
            novelty = len(genres - covered) / max(1, len(genres))
            return (
                float(row["score"]) + diversity_weight * novelty,
                -int(row["candidate_rank"]),
                -int(row["movie_id"]),
            )

        best = max(remaining, key=utility)
        remaining.remove(best)
        best["target_rank"] = len(selected) + 1
        selected.append(best)
        covered.update(
            genre for genre in str(best.get("genres") or "").split("|")
            if genre
        )
    return selected


def _evaluate_selected_policy(
    rows: Sequence[dict],
    target_k: int,
    policy_name: str,
    selector,
) -> tuple[dict, list[float]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["recommendation_id"])].append(row)
    if not grouped:
        raise ValueError("dataset is empty")

    contributions = []
    weighted_reward = 0.0
    weight_sum = 0.0
    weight_sq_sum = 0.0
    target_items = 0
    observed_target_items = 0
    unsupported = 0
    for candidates in grouped.values():
        candidates.sort(key=lambda row: int(row["candidate_rank"]))
        selected = selector(candidates, min(target_k, len(candidates)))
        contribution = 0.0
        for row in selected:
            target_items += 1
            propensity = float(row["behavior_propensity"])
            if propensity <= 0.0:
                unsupported += 1
                continue
            if row.get("impressed"):
                observed_target_items += 1
                weight = 1.0 / propensity
                reward = float(row.get("reward") or 0.0)
                contribution += weight * reward
                weighted_reward += weight * reward
                weight_sum += weight
                weight_sq_sum += weight * weight
        contributions.append(contribution / max(1, len(selected)))

    support_ok = unsupported == 0
    return {
        "estimator": "item_inclusion_ips_v2",
        "target_policy": policy_name,
        "recommendations": len(grouped),
        "target_items": target_items,
        "observed_target_items": observed_target_items,
        "unsupported_target_items": unsupported,
        "support_ok": support_ok,
        "ips_reward_per_item": (
            sum(contributions) / len(contributions) if support_ok else None
        ),
        "snips_reward_per_item": (
            weighted_reward / weight_sum
            if support_ok and weight_sum else None
        ),
        "effective_sample_size": (
            weight_sum * weight_sum / weight_sq_sum
            if weight_sq_sum else 0.0
        ),
    }, contributions


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> list[float] | None:
    """Return a deterministic percentile bootstrap interval for a mean."""
    if not values:
        return None
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lower = means[max(0, math.floor(alpha * (samples - 1)))]
    upper = means[min(samples - 1, math.ceil(
        (1.0 - alpha) * (samples - 1)))]
    return [lower, upper]


def bootstrap_cluster_mean_interval(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> list[float] | None:
    """Bootstrap actor clusters while retaining all recommendations per actor."""
    if len(values) != len(clusters):
        raise ValueError("values and clusters must have equal length")
    if not values:
        return None
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters):
        grouped[str(cluster)].append(float(value))
    cluster_ids = sorted(grouped)
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        sampled_values = []
        for _ in cluster_ids:
            sampled_values.extend(grouped[rng.choice(cluster_ids)])
        means.append(sum(sampled_values) / len(sampled_values))
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lower = means[max(0, math.floor(alpha * (samples - 1)))]
    upper = means[min(samples - 1, math.ceil(
        (1.0 - alpha) * (samples - 1)))]
    return [lower, upper]


def compare_logged_policies(
    rows: Sequence[dict],
    *,
    target_k: int,
    diversity_weight: float = 0.15,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
    min_recommendations: int = 200,
    min_actors: int = 5,
    min_effective_sample_size: float = 100.0,
    min_impression_coverage: float = 0.95,
    seed: int = 42,
) -> dict:
    """Formally gate baseline Top-k against a diversity-aware reranker."""
    baseline, baseline_values = _evaluate_selected_policy(
        rows,
        target_k,
        f"deepfm_top_{target_k}",
        lambda candidates, k: candidates[:k],
    )
    candidate, candidate_values = _evaluate_selected_policy(
        rows,
        target_k,
        f"deepfm_diversity_top_{target_k}_w{diversity_weight:g}",
        lambda candidates, k: diversity_rerank(
            candidates, k, diversity_weight),
    )
    paired_delta = [
        candidate_value - baseline_value
        for baseline_value, candidate_value
        in zip(baseline_values, candidate_values)
    ]
    recommendation_actors = []
    seen_recommendations = set()
    for row in rows:
        recommendation_id = str(row["recommendation_id"])
        if recommendation_id in seen_recommendations:
            continue
        seen_recommendations.add(recommendation_id)
        recommendation_actors.append(str(row.get("actor_id") or "__unknown__"))
    support_ok = baseline["support_ok"] and candidate["support_ok"]
    baseline["ips_confidence_interval"] = (
        bootstrap_cluster_mean_interval(
            baseline_values, recommendation_actors,
            samples=bootstrap_samples, confidence=confidence, seed=seed)
        if baseline["support_ok"] else None
    )
    candidate["ips_confidence_interval"] = (
        bootstrap_cluster_mean_interval(
            candidate_values, recommendation_actors,
            samples=bootstrap_samples, confidence=confidence, seed=seed + 1)
        if candidate["support_ok"] else None
    )

    served = sum(row.get("served_rank") is not None for row in rows)
    impressed = sum(bool(row.get("impressed")) for row in rows)
    recommendation_count = baseline["recommendations"]
    actor_count = len(set(recommendation_actors))
    impression_coverage = impressed / served if served else 0.0
    delta_interval = (
        bootstrap_cluster_mean_interval(
            paired_delta, recommendation_actors,
            samples=bootstrap_samples, confidence=confidence, seed=seed + 2)
        if support_ok else None
    )
    delta_mean = (
        sum(paired_delta) / len(paired_delta) if support_ok else None)

    blockers = []
    if recommendation_count < min_recommendations:
        blockers.append(
            f"recommendations {recommendation_count} < {min_recommendations}")
    if actor_count < min_actors:
        blockers.append(f"actors {actor_count} < {min_actors}")
    if impression_coverage < min_impression_coverage:
        blockers.append(
            f"impression_coverage {impression_coverage:.3f} "
            f"< {min_impression_coverage:.3f}")
    if not support_ok:
        blockers.append("target policy has zero-propensity items")
    minimum_ess = min(
        baseline["effective_sample_size"],
        candidate["effective_sample_size"],
    )
    if minimum_ess < min_effective_sample_size:
        blockers.append(
            f"effective_sample_size {minimum_ess:.1f} "
            f"< {min_effective_sample_size:.1f}")

    ready = not blockers
    if not ready:
        decision = "collect_more_data"
    elif delta_interval[0] > 0.0:
        decision = "promote_candidate"
    elif delta_interval[1] < 0.0:
        decision = "keep_baseline"
    else:
        decision = "continue_experiment"
    return {
        "protocol": {
            "comparison": "paired item-inclusion IPS by recommendation",
            "bootstrap_unit": "actor_id cluster",
            "confidence": confidence,
            "bootstrap_samples": bootstrap_samples,
            "diversity_weight": diversity_weight,
        },
        "data_quality": {
            "recommendations": recommendation_count,
            "actors": actor_count,
            "served_items": served,
            "impressed_items": impressed,
            "impression_coverage": impression_coverage,
            "minimum_effective_sample_size": minimum_ess,
        },
        "baseline": baseline,
        "candidate": candidate,
        "paired_delta": {
            "ips_reward_per_item": delta_mean,
            "confidence_interval": delta_interval,
        },
        "gate": {
            "formal_ready": ready,
            "min_recommendations": min_recommendations,
            "min_actors": min_actors,
            "min_effective_sample_size": min_effective_sample_size,
            "min_impression_coverage": min_impression_coverage,
            "blockers": blockers,
            "decision": decision,
        },
    }


def write_jsonl(rows: Sequence[dict], path: str | Path) -> Path:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    return output
