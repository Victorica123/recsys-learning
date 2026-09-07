# -*- coding: utf-8 -*-
"""TIGER-inspired semantic-ID generative retrieval primitives.

This is a controlled MovieLens baseline, not an exact TIGER reproduction.  It
keeps the defining retrieval mechanism -- residual semantic IDs followed by
autoregressive Transformer generation -- while replacing SentenceT5 + RQ-VAE
with repository-local metadata hashing + residual K-means because ML-1M has
only titles and genres.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


TITLE_TOKEN = re.compile(r"[a-z0-9]+")


def _stable_bucket(token: str, buckets: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


def build_item_content_vectors(
    movies_path: Path,
    item_map: dict[int, int],
    *,
    title_buckets: int = 192,
) -> np.ndarray:
    """Build deterministic content-only vectors aligned with internal item IDs.

    All catalog metadata may be used: no interaction label or timestamp enters
    this representation.  Hashed title TF-IDF, explicit genres and decade
    buckets make the transductive catalog assumption visible and reproducible.
    """
    if title_buckets < 8:
        raise ValueError("title_buckets must be at least 8")
    movies = pd.read_csv(
        movies_path, sep="::", engine="python",
        names=["movie_id", "title", "genres"], encoding="latin-1")
    genres = sorted({
        genre for raw in movies["genres"].astype(str)
        for genre in raw.split("|")})
    genre_index = {genre: idx for idx, genre in enumerate(genres)}
    decade_buckets = 12  # pre-1900, 1900s ... 2000s, unknown/overflow.
    width = title_buckets + len(genres) + decade_buckets
    matrix = np.zeros((len(item_map), width), dtype=np.float32)
    document_frequency = np.zeros(title_buckets, dtype=np.int64)

    token_rows: list[tuple[int, list[int]]] = []
    for row in movies.itertuples(index=False):
        if int(row.movie_id) not in item_map:
            continue
        item = int(item_map[int(row.movie_id)])
        title = str(row.title)
        tokens = TITLE_TOKEN.findall(re.sub(r"\(\d{4}\)\s*$", "", title).lower())
        token_ids = [_stable_bucket(token, title_buckets) for token in tokens]
        token_rows.append((item, token_ids))
        for bucket in set(token_ids):
            document_frequency[bucket] += 1
        for genre in str(row.genres).split("|"):
            matrix[item, title_buckets + genre_index[genre]] = 1.0
        year_match = re.search(r"\((\d{4})\)\s*$", title)
        if year_match:
            year = int(year_match.group(1))
            decade = min(10, max(0, (year - 1900) // 10 + 1))
        else:
            decade = 11
        matrix[item, title_buckets + len(genres) + decade] = 1.0

    idf = np.log((len(item_map) + 1.0) / (document_frequency + 1.0)) + 1.0
    for item, token_ids in token_rows:
        for bucket in token_ids:
            matrix[item, bucket] += float(idf[bucket])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= np.maximum(norms, 1e-8)
    return matrix


def _kmeans_plus_plus(
    values: np.ndarray,
    clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    centers = np.empty((clusters, values.shape[1]), dtype=np.float32)
    first = int(rng.integers(len(values)))
    centers[0] = values[first]
    closest = np.sum((values - centers[0]) ** 2, axis=1)
    for index in range(1, clusters):
        total = float(closest.sum())
        if total <= 1e-12:
            choice = int(rng.integers(len(values)))
        else:
            choice = int(rng.choice(len(values), p=closest / total))
        centers[index] = values[choice]
        distance = np.sum((values - centers[index]) ** 2, axis=1)
        closest = np.minimum(closest, distance)
    return centers


def residual_kmeans(
    vectors: np.ndarray,
    codebook_sizes: Sequence[int],
    *,
    seed: int = 2026,
    iterations: int = 25,
) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    """Quantize vectors level by level and subtract each selected centroid."""
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("vectors must be a non-empty 2D array")
    sizes = tuple(int(size) for size in codebook_sizes)
    if not sizes or any(size < 2 or size > len(values) for size in sizes):
        raise ValueError("every codebook size must be in [2, n_items]")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    rng = np.random.default_rng(seed)
    residual = values.copy()
    codes = np.zeros((len(values), len(sizes)), dtype=np.int64)
    codebooks: list[np.ndarray] = []
    reconstruction_mse: list[float] = []
    for level, clusters in enumerate(sizes):
        centers = _kmeans_plus_plus(residual, clusters, rng)
        assignment = np.zeros(len(values), dtype=np.int64)
        for _ in range(iterations):
            distances = (
                np.sum(residual ** 2, axis=1, keepdims=True)
                - 2.0 * residual @ centers.T
                + np.sum(centers ** 2, axis=1)[None, :])
            new_assignment = np.argmin(distances, axis=1)
            if np.array_equal(new_assignment, assignment):
                assignment = new_assignment
                break
            assignment = new_assignment
            min_distance = distances[np.arange(len(values)), assignment]
            for cluster in range(clusters):
                members = residual[assignment == cluster]
                if len(members):
                    centers[cluster] = members.mean(axis=0)
                else:
                    # Deterministically split the currently worst represented point.
                    centers[cluster] = residual[int(np.argmax(min_distance))]
        codes[:, level] = assignment
        residual -= centers[assignment]
        codebooks.append(centers.copy())
        reconstruction_mse.append(float(np.mean(residual ** 2)))
    return codes, codebooks, reconstruction_mse


def append_collision_tokens(codes: np.ndarray) -> tuple[np.ndarray, int]:
    """Append a deterministic within-tuple ordinal so every item ID is unique."""
    base = np.asarray(codes, dtype=np.int64)
    if base.ndim != 2 or len(base) == 0:
        raise ValueError("codes must be a non-empty 2D array")
    groups: dict[tuple[int, ...], list[int]] = {}
    for item, row in enumerate(base):
        groups.setdefault(tuple(int(value) for value in row), []).append(item)
    collision = np.zeros((len(base), 1), dtype=np.int64)
    for items in groups.values():
        for ordinal, item in enumerate(sorted(items)):
            collision[item, 0] = ordinal
    cardinality = max(int(collision.max()) + 1, 1)
    output = np.concatenate([base, collision], axis=1)
    if len({tuple(row) for row in output.tolist()}) != len(output):
        raise AssertionError("collision disambiguation failed to create unique IDs")
    return output, cardinality


@dataclass(frozen=True)
class SemanticIDIndex:
    codes: np.ndarray
    cardinalities: tuple[int, ...]
    reconstruction_mse: tuple[float, ...]

    @classmethod
    def fit(
        cls,
        vectors: np.ndarray,
        codebook_sizes: Sequence[int],
        *,
        seed: int = 2026,
        iterations: int = 25,
    ) -> "SemanticIDIndex":
        codes, _books, mse = residual_kmeans(
            vectors, codebook_sizes, seed=seed, iterations=iterations)
        unique_codes, collision_cardinality = append_collision_tokens(codes)
        return cls(
            codes=unique_codes,
            cardinalities=tuple(int(size) for size in codebook_sizes)
            + (collision_cardinality,),
            reconstruction_mse=tuple(mse),
        )

    def profile(self) -> dict:
        base = self.codes[:, :-1]
        tuple_counts: dict[tuple[int, ...], int] = {}
        for row in base:
            key = tuple(int(value) for value in row)
            tuple_counts[key] = tuple_counts.get(key, 0) + 1
        return {
            "items": int(len(self.codes)),
            "levels": int(self.codes.shape[1]),
            "cardinalities": list(self.cardinalities),
            "unique_final_ids": int(len({tuple(row) for row in self.codes.tolist()})),
            "unique_base_tuples": int(len(tuple_counts)),
            "colliding_base_tuples": int(sum(value > 1 for value in tuple_counts.values())),
            "max_collision_size": int(max(tuple_counts.values())),
            "level_utilization": [
                round(len(np.unique(self.codes[:, level])) / cardinality, 6)
                for level, cardinality in enumerate(self.cardinalities)
            ],
            "residual_reconstruction_mse": [
                round(float(value), 8) for value in self.reconstruction_mse],
        }


class SemanticSequenceDataset(Dataset):
    """Next-item examples built strictly from one temporal training window."""

    def __init__(
        self,
        train: pd.DataFrame,
        semantic_codes: np.ndarray,
        *,
        max_history_items: int,
        max_examples: int = 0,
        seed: int = 42,
        target_after_timestamp: float | None = None,
    ) -> None:
        if max_history_items < 1:
            raise ValueError("max_history_items must be positive")
        self.codes = np.asarray(semantic_codes, dtype=np.int64)
        self.max_history_items = int(max_history_items)
        groups = [
            group for _, group in train.sort_values("timestamp").groupby("u", sort=True)
            if len(group) >= 2
        ]
        self.sequences = [
            [int(item) for item in group["i"].tolist()] for group in groups]
        self.timestamps = [
            [float(value) for value in group["timestamp"].tolist()] for group in groups]
        references = [
            (sequence_index, position)
            for sequence_index, sequence in enumerate(self.sequences)
            for position in range(1, len(sequence))
            if (target_after_timestamp is None
                or self.timestamps[sequence_index][position] > target_after_timestamp)
        ]
        if max_examples and len(references) > max_examples:
            chosen = np.random.default_rng(seed).choice(
                len(references), max_examples, replace=False)
            references = [references[int(index)] for index in np.sort(chosen)]
        self.references = references

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_index, position = self.references[index]
        sequence = self.sequences[sequence_index]
        history = sequence[max(0, position - self.max_history_items):position]
        context = np.full(
            (self.max_history_items, self.codes.shape[1]), -1, dtype=np.int64)
        context[-len(history):] = self.codes[np.asarray(history, dtype=np.int64)]
        target = self.codes[sequence[position]]
        return torch.from_numpy(context), torch.from_numpy(target.copy())


class SemanticIDGenerator(nn.Module):
    """Transformer encoder-decoder that autoregressively emits one Semantic ID."""

    def __init__(
        self,
        cardinalities: Sequence[int],
        *,
        max_history_items: int = 20,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cardinalities = tuple(int(value) for value in cardinalities)
        if not self.cardinalities or any(value < 1 for value in self.cardinalities):
            raise ValueError("cardinalities must be positive")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.levels = len(self.cardinalities)
        self.max_history_items = int(max_history_items)
        offsets = np.cumsum((1,) + self.cardinalities[:-1]).astype(np.int64)
        self.register_buffer("offsets", torch.as_tensor(offsets), persistent=False)
        self.bos_id = 1 + sum(self.cardinalities)
        self.token_emb = nn.Embedding(self.bos_id + 1, d_model, padding_idx=0)
        self.encoder_pos = nn.Embedding(max_history_items * self.levels, d_model)
        self.decoder_pos = nn.Embedding(self.levels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, cardinality) for cardinality in self.cardinalities])
        self.final_norm = nn.LayerNorm(d_model)

    def _global_tokens(self, local_codes: torch.Tensor) -> torch.Tensor:
        offsets = self.offsets.to(local_codes.device)
        shape = (1,) * (local_codes.ndim - 1) + (self.levels,)
        global_codes = local_codes + offsets.view(shape)
        return torch.where(local_codes >= 0, global_codes, 0)

    def encode_context(
        self, context_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context_codes.ndim != 3 or context_codes.shape[-1] != self.levels:
            raise ValueError("context_codes must have shape [batch, history, levels]")
        tokens = self._global_tokens(context_codes).flatten(1)
        padding = tokens.eq(0)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        encoded = self.token_emb(tokens) + self.encoder_pos(positions)[None, :, :]
        return self.final_norm(
            self.encoder(encoded, src_key_padding_mask=padding)), padding

    def decode_prefix(
        self,
        memory: torch.Tensor,
        memory_padding: torch.Tensor,
        prefix: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        if not 0 <= level < self.levels:
            raise ValueError("invalid semantic level")
        if prefix.ndim != 2 or prefix.shape[1] != level:
            raise ValueError("prefix width must equal the requested level")
        batch = memory.shape[0]
        bos = torch.full((batch, 1), self.bos_id, dtype=torch.long,
                         device=memory.device)
        if level:
            offsets = self.offsets[:level].to(prefix.device)
            prefix_tokens = prefix + offsets[None, :]
            decoder_tokens = torch.cat([bos, prefix_tokens], dim=1)
        else:
            decoder_tokens = bos
        positions = torch.arange(decoder_tokens.shape[1], device=memory.device)
        target = self.token_emb(decoder_tokens) + self.decoder_pos(positions)[None, :, :]
        causal = torch.triu(torch.ones(
            len(positions), len(positions), dtype=torch.bool,
            device=memory.device), diagonal=1)
        decoded = self.decoder(
            target, memory, tgt_mask=causal,
            memory_key_padding_mask=memory_padding)
        return self.output_heads[level](decoded[:, -1])

    def forward(
        self, context_codes: torch.Tensor, target_codes: torch.Tensor,
    ) -> list[torch.Tensor]:
        memory, padding = self.encode_context(context_codes)
        return [
            self.decode_prefix(memory, padding, target_codes[:, :level], level)
            for level in range(self.levels)
        ]


def fit_semantic_generator(
    train: pd.DataFrame,
    semantic_index: SemanticIDIndex,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    max_history_items: int,
    max_examples: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    device: str,
    verbose: bool = True,
    validation_frame: pd.DataFrame | None = None,
    validation_after_timestamp: float | None = None,
    validation_max_examples: int = 20000,
) -> tuple[SemanticIDGenerator, list[dict], int]:
    dataset = SemanticSequenceDataset(
        train, semantic_index.codes, max_history_items=max_history_items,
        max_examples=max_examples, seed=seed)
    if not len(dataset):
        raise ValueError("training window has no next-item examples")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        generator=generator)
    validation_loader = None
    if validation_frame is not None:
        if validation_after_timestamp is None:
            raise ValueError(
                "validation_after_timestamp is required with validation_frame")
        validation_dataset = SemanticSequenceDataset(
            validation_frame, semantic_index.codes,
            max_history_items=max_history_items,
            max_examples=validation_max_examples, seed=seed,
            target_after_timestamp=validation_after_timestamp)
        if not len(validation_dataset):
            raise ValueError("temporal validation slice has no next-item examples")
        validation_loader = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0)
    model = SemanticIDGenerator(
        semantic_index.cardinalities,
        max_history_items=max_history_items, d_model=d_model,
        n_heads=n_heads, n_layers=n_layers, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history: list[dict] = []
    best_epoch = epochs
    best_validation_loss = math.inf
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for context, target in loader:
            context = context.to(device)
            target = target.to(device)
            logits = model(context, target)
            loss = torch.stack([
                F.cross_entropy(level_logits, target[:, level])
                for level, level_logits in enumerate(logits)
            ]).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(context)
            examples += len(context)
        row = {
            "epoch": epoch,
            "loss": round(total_loss / examples, 6),
            "examples": examples,
        }
        if validation_loader is not None:
            row["validation_loss"] = round(
                evaluate_generator_loss(model, validation_loader, device=device), 6)
            if row["validation_loss"] < best_validation_loss:
                best_validation_loss = row["validation_loss"]
                best_epoch = epoch
        history.append(row)
        if verbose:
            validation_text = (
                f" val={row['validation_loss']:.4f}"
                if "validation_loss" in row else "")
            print(
                f"    epoch={epoch} loss={row['loss']:.4f}"
                f"{validation_text} examples={examples}", flush=True)
    return model, history, best_epoch


@torch.no_grad()
def evaluate_generator_loss(
    model: SemanticIDGenerator,
    loader: DataLoader,
    *,
    device: str,
) -> float:
    model.eval()
    total_loss = 0.0
    examples = 0
    for context, target in loader:
        context = context.to(device)
        target = target.to(device)
        logits = model(context, target)
        loss = torch.stack([
            F.cross_entropy(level_logits, target[:, level])
            for level, level_logits in enumerate(logits)
        ]).mean()
        total_loss += float(loss.item()) * len(context)
        examples += len(context)
    return total_loss / examples


def user_context_codes(
    train: pd.DataFrame,
    users: Iterable[int],
    semantic_codes: np.ndarray,
    *,
    max_history_items: int,
) -> np.ndarray:
    histories = (
        train.sort_values("timestamp").groupby("u")["i"].apply(list).to_dict())
    users_list = [int(user) for user in users]
    output = np.full(
        (len(users_list), max_history_items, semantic_codes.shape[1]),
        -1, dtype=np.int64)
    for row, user in enumerate(users_list):
        history = histories.get(user, [])[-max_history_items:]
        if history:
            output[row, -len(history):] = semantic_codes[
                np.asarray(history, dtype=np.int64)]
    return output


def catalog_prefix_layouts(codes: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    output = []
    for level in range(codes.shape[1]):
        if level == 0:
            prefixes = np.empty((1, 0), dtype=np.int64)
            inverse = np.zeros(len(codes), dtype=np.int64)
        else:
            prefixes, inverse = np.unique(
                codes[:, :level], axis=0, return_inverse=True)
        output.append((prefixes.astype(np.int64), inverse.astype(np.int64)))
    return output


@torch.no_grad()
def score_catalog_exact(
    model: SemanticIDGenerator,
    context_codes: torch.Tensor,
    catalog_codes: np.ndarray,
    *,
    prefix_chunk: int = 128,
    prefix_layouts: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    """Exact log-probability for every catalog Semantic ID (no beam pruning)."""
    if prefix_chunk < 1:
        raise ValueError("prefix_chunk must be positive")
    model.eval()
    device = next(model.parameters()).device
    context_codes = context_codes.to(device)
    memory, memory_padding = model.encode_context(context_codes)
    batch = len(context_codes)
    item_codes = np.asarray(catalog_codes, dtype=np.int64)
    scores = torch.zeros((batch, len(item_codes)), device=device)
    layouts = prefix_layouts or catalog_prefix_layouts(item_codes)
    if len(layouts) != item_codes.shape[1]:
        raise ValueError("prefix layout count must equal semantic ID levels")

    for level, (prefixes, inverse) in enumerate(layouts):
        for start in range(0, len(prefixes), prefix_chunk):
            chunk = torch.as_tensor(
                prefixes[start:start + prefix_chunk], device=device)
            width = len(chunk)
            expanded_memory = memory[:, None].expand(
                batch, width, *memory.shape[1:]).reshape(
                    batch * width, *memory.shape[1:])
            expanded_padding = memory_padding[:, None].expand(
                batch, width, memory_padding.shape[1]).reshape(
                    batch * width, memory_padding.shape[1])
            expanded_prefix = chunk[None].expand(batch, width, level).reshape(
                batch * width, level)
            logits = model.decode_prefix(
                expanded_memory, expanded_padding, expanded_prefix, level)
            log_probs = F.log_softmax(logits, dim=-1).reshape(
                batch, width, model.cardinalities[level])
            items = np.flatnonzero(
                (inverse >= start) & (inverse < start + width))
            if not len(items):
                continue
            local_prefix = torch.as_tensor(inverse[items] - start, device=device)
            target = torch.as_tensor(item_codes[items, level], device=device)
            scores[:, torch.as_tensor(items, device=device)] += log_probs[
                :, local_prefix, target]
    return scores.cpu().numpy()


def semantic_recommendations(
    model: SemanticIDGenerator,
    train: pd.DataFrame,
    users: Iterable[int],
    semantic_codes: np.ndarray,
    *,
    k: int,
    max_history_items: int,
    user_batch: int = 16,
    prefix_chunk: int = 128,
) -> dict[int, list[int]]:
    """Rank all unique Semantic IDs exactly and exclude the complete history."""
    users_list = [int(user) for user in users]
    if k < 1 or k >= len(semantic_codes):
        raise ValueError("k must be positive and smaller than the catalog")
    contexts = user_context_codes(
        train, users_list, semantic_codes,
        max_history_items=max_history_items)
    layouts = catalog_prefix_layouts(semantic_codes)
    seen = train.groupby("u")["i"].apply(set).to_dict()
    item_ids = np.arange(len(semantic_codes))
    output: dict[int, list[int]] = {}
    for start in range(0, len(users_list), user_batch):
        batch_users = users_list[start:start + user_batch]
        batch_context = torch.as_tensor(contexts[start:start + user_batch])
        scores = score_catalog_exact(
            model, batch_context, semantic_codes, prefix_chunk=prefix_chunk,
            prefix_layouts=layouts)
        for row, user in enumerate(batch_users):
            user_seen = seen.get(user, set())
            if user_seen:
                scores[row, np.fromiter(user_seen, dtype=np.int64)] = -np.inf
            order = np.lexsort((item_ids, -scores[row]))
            recs = [int(item) for item in order if np.isfinite(scores[row, item])][:k]
            if len(recs) != k:
                raise ValueError(f"user {user} has fewer than k unseen catalog items")
            output[user] = recs
    return output
