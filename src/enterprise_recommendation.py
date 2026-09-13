"""Portable, standard-library knowledge ranking for an authorized Agent Service.

The caller supplies a principal resolved by its existing authentication layer.
This module neither authenticates JWTs nor exposes an additional HTTP service.
Synthetic oracle labels, topic IDs and logged scores are never ranking inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


POLICIES = ("baseline_v0", "semantic_v1", "hybrid_v1")
FEATURE_VERSION = "knowledge-text-history-v1"
MODEL_VERSION = "knowledge-bm25-v1"
ROLE_TYPES = {
    "product_manager": {"prd", "meeting_transcript"},
    "backend_engineer": {"knowledge", "incident"},
    "data_analyst": {"prd", "knowledge"},
    "security_reviewer": {"knowledge", "incident"},
    "operations": {"knowledge", "incident"},
}
WEIGHTS = {
    "baseline_v0": {"freshness": 0.55, "popularity": 0.45},
    "semantic_v1": {"content": 1.0},
    "hybrid_v1": {"content": 0.70, "freshness": 0.15, "popularity": 0.10, "role": 0.05},
}


class KnowledgeValidationError(ValueError):
    """An invalid or inconsistent recommendation record."""


class KnowledgeAuthorizationError(KnowledgeValidationError):
    """Principal and resource scopes do not agree."""


class ScorerUnavailable(RuntimeError):
    """An optional content scorer failed; the authorized baseline may be used."""


def timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError, TypeError) as exc:
        raise KnowledgeValidationError("expected an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise KnowledgeValidationError("timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def fingerprint(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def item_key(item: dict) -> tuple[str, str, int]:
    item_type, item_id = item.get("item_type"), item.get("item_id")
    version = item.get("item_version")
    if not isinstance(item_type, str) or not item_type.strip():
        raise KnowledgeValidationError("item_type is required")
    if not isinstance(item_id, str) or not item_id.strip():
        raise KnowledgeValidationError("item_id must be a nonempty string")
    if type(version) is not int or version < 1:
        raise KnowledgeValidationError("item_version must be a positive integer")
    return item_type, item_id, version


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    workspace_id: str
    actor_id: str
    role: str
    workspace_type: str = "team"

    def __post_init__(self):
        if any(not isinstance(value, str) or not value.strip() for value in (
                self.tenant_id, self.workspace_id, self.actor_id, self.role)):
            raise KnowledgeValidationError("principal scope and role are required")
        if self.workspace_type not in {"team", "personal"}:
            raise KnowledgeValidationError("invalid trusted workspace type")

    def check_request(self, request: dict) -> None:
        if any(request.get(key) != getattr(self, key)
               for key in ("tenant_id", "workspace_id", "workspace_type", "actor_id")):
            raise KnowledgeAuthorizationError("request does not match the trusted principal")


def active_at(item: dict, at: datetime) -> bool:
    """Evaluate a version's validity interval, including historical replay.

    A presently superseded version was usable before its recorded boundary.
    Unknown states and terminal states without a timestamp fail closed.
    """
    status = item.get("status")
    if status not in {"ACTIVE", "SUPERSEDED", "REVOKED"}:
        return False
    if status == "SUPERSEDED" and not item.get("superseded_at"):
        return False
    if status == "REVOKED" and not item.get("revoked_at"):
        return False
    if timestamp(item["created_at"]) > at:
        return False
    for field in ("superseded_at", "revoked_at"):
        if item.get(field) and at >= timestamp(item[field]):
            return False
    return True


def can_access(principal: Principal, item: dict, at: datetime) -> bool:
    if (item.get("tenant_id"), item.get("workspace_id")) != (
            principal.tenant_id, principal.workspace_id):
        return False
    if item.get("workspace_type") != principal.workspace_type:
        return False
    if item.get("visibility") not in {"workspace", "owner"}:
        return False
    if (item["visibility"] == "owner" or item["workspace_type"] == "personal"):
        if item.get("owner_id") != principal.actor_id:
            return False
    roles = item.get("visible_roles")
    if not isinstance(roles, list) or principal.role not in roles:
        return False
    return active_at(item, at)


def authorized_candidates(principal: Principal, catalog: list[dict], at: datetime) -> list[dict]:
    latest: dict[tuple[str, str], dict] = {}
    for item in catalog:
        # Authorization precedes text extraction, corpus statistics and scoring.
        if not can_access(principal, item, at):
            continue
        kind, identity, version = item_key(item)
        previous = latest.get((kind, identity))
        if previous and previous["item_version"] == version and previous != item:
            raise KnowledgeValidationError("conflicting content for the same item version")
        if previous is None or version > previous["item_version"]:
            latest[(kind, identity)] = item
    return sorted(latest.values(), key=item_key)


def tokenize(text: str) -> Counter:
    """Latin terms and Chinese bigrams; no model download or hidden topic map."""
    terms = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text.lower())
    tokens = []
    for term in terms:
        if re.fullmatch(r"[\u4e00-\u9fff]+", term):
            tokens.extend(term[i:i + 2] for i in range(max(1, len(term) - 1)))
        else:
            tokens.append(term)
    return Counter(tokens)


def document_tokens(item: dict) -> Counter:
    # Explicit allowlist prevents labels/logged features from entering scoring.
    title = item.get("title", "")
    summary = item.get("summary", "")
    content = item.get("content", "")
    tags = item.get("tags", [])
    if not all(isinstance(s, str) for s in (title, summary, content)):
        raise KnowledgeValidationError("document text fields must be strings")
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise KnowledgeValidationError("tags must be a list of strings")
    return tokenize(" ".join([title, title, summary, content, *tags]))


def bm25_scores(query: str, documents: list[Counter]) -> list[float]:
    if not documents:
        return []
    lengths = [sum(doc.values()) for doc in documents]
    average = sum(lengths) / len(lengths) or 1.0
    df = Counter(term for doc in documents for term in doc)
    query_terms = tokenize(query)
    scores = []
    for doc, length in zip(documents, lengths):
        score = 0.0
        for term in query_terms:
            frequency = doc[term]
            idf = math.log1p((len(documents) - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * frequency * 2.2 / (
                frequency + 1.2 * (0.25 + 0.75 * length / average))
        scores.append(score / (1.0 + score))
    return scores


def experiment_bucket(principal: Principal, experiment_id: str, buckets: int = 10000) -> int:
    if type(buckets) is not int or buckets <= 0 or not experiment_id:
        raise KnowledgeValidationError("experiment ID and positive bucket count are required")
    identity = [principal.tenant_id, principal.workspace_id, principal.actor_id, experiment_id]
    return int(fingerprint(identity)[:16], 16) % buckets


def policy_bundle() -> dict:
    return {
        "schema_version": 1, "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION, "content_scorer": "lexical_bm25_cjk_bigrams",
        "weights": json.loads(canonical_json(WEIGHTS)),
        "freshness_half_life_days": 75.0 * math.log(2),
        "hybrid_diversity_penalty": 0.06,
        "fallback_policy": "baseline_v0",
        "deployment_status": "synthetic_validation_only",
        "requires_trusted_principal": True,
    }


class KnowledgeRanker:
    def __init__(self, bundle: dict | None = None, content_scorer=None):
        self.bundle = json.loads(canonical_json(bundle if bundle is not None else policy_bundle()))
        expected = policy_bundle()
        for field in ("schema_version", "model_version", "feature_version",
                      "content_scorer", "fallback_policy", "requires_trusted_principal"):
            if self.bundle.get(field) != expected[field]:
                raise KnowledgeValidationError(f"unsupported policy artifact field: {field}")
        weights = self.bundle.get("weights", {})
        if set(weights) != set(POLICIES):
            raise KnowledgeValidationError("artifact must declare all three policies")
        for policy, values in weights.items():
            if set(values) != set(WEIGHTS[policy]):
                raise KnowledgeValidationError("unsupported ranking feature")
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                   for v in values.values()) or not math.isclose(sum(values.values()), 1.0):
                raise KnowledgeValidationError("weights must be finite, nonnegative and sum to one")
        for field in ("freshness_half_life_days", "hybrid_diversity_penalty"):
            value = self.bundle.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise KnowledgeValidationError(f"invalid {field}")
        if self.bundle["freshness_half_life_days"] == 0:
            raise KnowledgeValidationError("freshness half-life must be positive")
        self.content_scorer = content_scorer or bm25_scores

    @classmethod
    def from_artifact(cls, path: Path) -> KnowledgeRanker:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def recommend(
        self, principal: Principal, request: dict, catalog: list[dict], *,
        policy: str = "hybrid_v1", k: int = 5, popularity: dict | None = None,
        epsilon: float = 0.0, rng: random.Random | None = None,
    ) -> dict:
        principal.check_request(request)
        if policy not in POLICIES:
            raise KnowledgeValidationError("unknown ranking policy")
        if type(k) is not int or k < 1:
            raise KnowledgeValidationError("k must be positive")
        if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise KnowledgeValidationError("epsilon must be in [0, 1]")
        at = timestamp(request["requested_at"])
        items = authorized_candidates(principal, catalog, at)
        docs = [document_tokens(item) for item in items]
        if not isinstance(request.get("context"), dict):
            raise KnowledgeValidationError("context must be an object")
        query = request["context"].get("intent", "")
        if not isinstance(query, str):
            raise KnowledgeValidationError("context.intent must be a string")
        requested_policy, fallback = policy, None
        content = [0.0] * len(items)
        if policy != "baseline_v0":
            try:
                content = list(self.content_scorer(query, docs))
                if len(content) != len(items) or any(
                        type(s) not in (int, float) or not math.isfinite(s) or not 0 <= s <= 1
                        for s in content):
                    raise ScorerUnavailable("invalid scorer output")
            except ScorerUnavailable:
                policy, fallback = "baseline_v0", "content_scorer_unavailable"
                content = [0.0] * len(items)
        popularity = popularity or {}
        counts = [float(popularity.get(item_key(item), 0.0)) for item in items]
        if any(not math.isfinite(count) or count < 0 for count in counts):
            raise KnowledgeValidationError("popularity must be finite and nonnegative")
        scale = math.log1p(max(counts, default=0)) or 1.0
        rows = []
        for item, terms, text_score, count in zip(items, docs, content, counts):
            age = max(0.0, (at - timestamp(item["created_at"])).total_seconds() / 86400)
            features = {
                "content": text_score,
                "freshness": math.exp(-math.log(2) * age /
                                      self.bundle["freshness_half_life_days"]),
                "popularity": math.log1p(count) / scale,
                "role": float(item["item_type"] in ROLE_TYPES.get(principal.role, set())),
            }
            score = sum(features[name] * weight
                        for name, weight in self.bundle["weights"][policy].items())
            reason = ("与当前问题的文本相关" if text_score > 0 else "工作区内近期内容")
            if features["role"] and policy == "hybrid_v1":
                reason += "；内容类型与当前角色匹配"
            rows.append({
                "item_type": item["item_type"], "item_id": item["item_id"],
                "item_version": item["item_version"], "title": item.get("title", ""),
                "score": score, "features": features, "recommendation_reason": reason,
                "_terms": set(terms),
            })
        ordered = []
        penalty = self.bundle["hybrid_diversity_penalty"] if policy == "hybrid_v1" else 0.0
        while rows:
            for row in rows:
                overlap = max((len(row["_terms"] & old["_terms"]) /
                               max(1, len(row["_terms"] | old["_terms"]))
                               for old in ordered), default=0.0)
                row["selection_score"] = row["score"] - penalty * overlap
            chosen = min(rows, key=lambda row: (-row["selection_score"], item_key(row)))
            rows.remove(chosen)
            ordered.append(chosen)
        take = min(k, len(ordered))
        rng = rng or random.SystemRandom()
        explored = bool(take and epsilon > 0 and rng.random() < epsilon)
        selected = rng.sample(range(len(ordered)), take) if explored else list(range(take))
        positions = {index: rank for rank, index in enumerate(selected, 1)}
        for index, row in enumerate(ordered):
            row.pop("_terms")
            row["candidate_rank"] = index + 1
            row["served_rank"] = positions.get(index)
            row["behavior_propensity"] = (
                (1 - epsilon) * float(index < take) + epsilon * take / len(ordered))
        snapshot = fingerprint({
            "at": request["requested_at"], "principal": vars(principal), "query": query,
            "features": [{**{key: row[key] for key in ("item_type", "item_id", "item_version")},
                          "features": row["features"]} for row in ordered],
            "artifact": self.bundle,
        })
        return {
            "schema_version": 1, "recommendation_id": request["recommendation_id"],
            "tenant_id": principal.tenant_id, "workspace_id": principal.workspace_id,
            "workspace_type": request.get("workspace_type", "team"),
            "actor_id": principal.actor_id, "requested_at": request["requested_at"],
            "context_type": request.get("context_type", "analysis_result"),
            "context_id": request.get("context_id"), "context": {"intent": query},
            "requested_policy_version": requested_policy, "policy_version": policy,
            "model_version": MODEL_VERSION, "feature_snapshot_version": snapshot,
            "feature_version": FEATURE_VERSION, "fallback_reason": fallback,
            "requested_k": k, "epsilon": epsilon, "explored": explored,
            "candidate_count": len(ordered), "candidates": ordered,
            "synthetic": request.get("synthetic"),
        }
