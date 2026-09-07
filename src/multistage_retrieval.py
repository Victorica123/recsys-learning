# -*- coding: utf-8 -*-
"""Multi-route retrieval fusion and a sequence-aware residual listwise ranker."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


FEATURE_NAMES = (
    "two_tower_reciprocal_rank",
    "item_knn_reciprocal_rank",
    "popular_reciprocal_rank",
    "recent_genre_affinity",
    "last_item_genre_overlap",
    "normalized_log_popularity",
    "is_long_tail",
)


def reciprocal_rank_fusion(
    routes: Mapping[str, Sequence[int]],
    *,
    route_weights: Mapping[str, float] | None = None,
    rrf_constant: float = 60.0,
    limit: int | None = None,
) -> tuple[list[int], np.ndarray]:
    """Fuse ranked routes with deterministic weighted reciprocal-rank fusion."""
    if rrf_constant < 0:
        raise ValueError("rrf_constant must be non-negative")
    weights = route_weights or {}
    scores: dict[int, float] = {}
    for name, ranking in routes.items():
        weight = float(weights.get(name, 1.0))
        if weight < 0:
            raise ValueError("route weights must be non-negative")
        for rank, item in enumerate(ranking, start=1):
            item = int(item)
            scores[item] = scores.get(item, 0.0) + weight / (rrf_constant + rank)
    ordered = sorted(scores, key=lambda item: (-scores[item], item))
    if limit is not None:
        ordered = ordered[:limit]
    return ordered, np.asarray([scores[item] for item in ordered], dtype=np.float32)


def _rank_map(items: Sequence[int]) -> dict[int, int]:
    return {int(item): rank for rank, item in enumerate(items, start=1)}


def sequence_candidate_features(
    candidates: Sequence[int],
    *,
    routes: Mapping[str, Sequence[int]],
    recent_items: Sequence[int],
    item_genres: Sequence[set[int]],
    item_counts: np.ndarray,
    long_tail_items: set[int],
    recency_decay: float = 0.85,
) -> np.ndarray:
    """Build route + recent-sequence features without using future labels."""
    if not 0.0 < recency_decay <= 1.0:
        raise ValueError("recency_decay must be in (0, 1]")
    route_maps = {name: _rank_map(items) for name, items in routes.items()}
    recent = [int(item) for item in recent_items]
    genre_pref: dict[int, float] = {}
    total_weight = 0.0
    for age, item in enumerate(reversed(recent)):
        weight = recency_decay ** age
        genres = item_genres[item]
        if not genres:
            continue
        per_genre = weight / len(genres)
        for genre in genres:
            genre_pref[genre] = genre_pref.get(genre, 0.0) + per_genre
        total_weight += weight
    if total_weight:
        genre_pref = {key: value / total_weight for key, value in genre_pref.items()}
    last_genres = item_genres[recent[-1]] if recent else set()
    max_log_pop = float(np.log1p(item_counts).max()) if len(item_counts) else 1.0
    max_log_pop = max(max_log_pop, 1.0)

    rows = []
    for item in candidates:
        item = int(item)
        genres = item_genres[item]
        affinity = (
            float(np.mean([genre_pref.get(genre, 0.0) for genre in genres]))
            if genres else 0.0)
        rows.append([
            1.0 / route_maps.get("two_tower", {}).get(item, 1_000_000),
            1.0 / route_maps.get("item_knn", {}).get(item, 1_000_000),
            1.0 / route_maps.get("popular", {}).get(item, 1_000_000),
            affinity,
            float(bool(genres & last_genres)),
            float(np.log1p(item_counts[item]) / max_log_pop),
            float(item in long_tail_items),
        ])
    return np.asarray(rows, dtype=np.float32)


@dataclass
class ListwiseGroup:
    candidate_ids: np.ndarray
    base_scores: np.ndarray
    features: np.ndarray
    target_index: int | None
    user_id: int
    event_timestamp: int
    window: int


@dataclass
class ResidualListwiseRanker:
    weights: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    residual_scale: float = 0.20

    @classmethod
    def zero(cls, n_features: int, residual_scale: float = 0.20):
        return cls(
            weights=np.zeros(n_features, dtype=np.float32),
            feature_mean=np.zeros(n_features, dtype=np.float32),
            feature_std=np.ones(n_features, dtype=np.float32),
            residual_scale=residual_scale,
        )

    def scores(self, group: ListwiseGroup) -> np.ndarray:
        features = (group.features - self.feature_mean) / self.feature_std
        residual = self.residual_scale * np.tanh(features @ self.weights)
        return group.base_scores + residual

    def rank(self, group: ListwiseGroup, k: int) -> list[int]:
        scores = self.scores(group)
        order = np.lexsort((group.candidate_ids, -scores))[:k]
        return [int(group.candidate_ids[index]) for index in order]


def validate_prior_window_groups(
    groups: Sequence[ListwiseGroup], evaluation_window: int
) -> None:
    """Reject same-window/future labels before ranker fitting."""
    invalid = sorted({group.window for group in groups
                      if group.window >= evaluation_window})
    if invalid:
        raise ValueError(
            f"ranker leakage: evaluation window {evaluation_window} cannot "
            f"train on windows {invalid}")


def fit_residual_listwise_ranker(
    groups: Sequence[ListwiseGroup],
    *,
    epochs: int = 100,
    lr: float = 0.03,
    l2: float = 0.01,
    residual_scale: float = 0.20,
    seed: int = 42,
) -> tuple[ResidualListwiseRanker, dict]:
    """Fit a bounded residual with listwise softmax on prior-window groups."""
    eligible = [group for group in groups if group.target_index is not None]
    if not eligible:
        raise ValueError("no ranker groups contain the positive target")
    feature_stack = np.concatenate([group.features for group in eligible], axis=0)
    mean = feature_stack.mean(axis=0).astype(np.float32)
    std = feature_stack.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0

    torch.manual_seed(seed)
    weights = torch.zeros(len(mean), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([weights], lr=lr)
    losses: list[float] = []
    for _epoch in range(epochs):
        optimizer.zero_grad()
        group_losses = []
        for group in eligible:
            base = torch.as_tensor(group.base_scores, dtype=torch.float32)
            features = torch.as_tensor(
                (group.features - mean) / std, dtype=torch.float32)
            scores = base + residual_scale * torch.tanh(features @ weights)
            target = torch.as_tensor([group.target_index], dtype=torch.long)
            group_losses.append(F.cross_entropy(scores.unsqueeze(0), target))
        loss = torch.stack(group_losses).mean() + l2 * weights.square().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    ranker = ResidualListwiseRanker(
        weights=weights.detach().numpy(), feature_mean=mean,
        feature_std=std, residual_scale=residual_scale)
    return ranker, {
        "eligible_groups": len(eligible),
        "epochs": epochs,
        "initial_loss": round(losses[0], 6),
        "final_loss": round(losses[-1], 6),
        "feature_weights": {
            name: round(float(value), 6)
            for name, value in zip(FEATURE_NAMES, ranker.weights)},
    }
