"""Validate and adapt the original enterprise synthetic seed to a generic contract."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from enterprise_recommendation import (
    KnowledgeValidationError, Principal, can_access, fingerprint, item_key, timestamp,
)


DATA_FILES = (
    "tenants.jsonl", "users.jsonl", "items.jsonl", "relations.jsonl",
    "recommendations.jsonl", "events.jsonl", "ground_truth.jsonl",
)
EVENT_TYPES = frozenset({
    "impression", "open", "dwell", "cite", "save", "dismiss",
    "create_action", "complete_action",
})
SPLITS = ("train", "validation", "test")


def require(condition, message: str) -> None:
    if not condition:
        raise KnowledgeValidationError(message)


def nonempty(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def unique_index(rows: list[dict], fields: tuple[str, ...]) -> dict:
    result = {}
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        require(all(nonempty(value) for value in key), f"missing identity fields: {fields}")
        require(key not in result, f"duplicate identity: {key}")
        result[key] = row
    return result


def request_key(row: dict) -> tuple[str, str, str]:
    return row["tenant_id"], row["workspace_id"], row["recommendation_id"]


def validate_request(row: dict) -> None:
    require(row.get("schema_version") == 1, "unsupported knowledge request schema")
    for name in ("tenant_id", "workspace_id", "actor_id", "recommendation_id",
                 "context_type", "context_id", "policy_version", "model_version",
                 "feature_snapshot_version"):
        require(nonempty(row.get(name)), f"missing request field: {name}")
    require(row.get("workspace_type") in {"team", "personal"}, "invalid workspace type")
    require(type(row.get("synthetic")) is bool, "synthetic provenance must be explicit")
    at = timestamp(row.get("requested_at"))
    require(row.get("split") in SPLITS, "invalid request split")
    if row.get("split_start"):
        require(timestamp(row["split_start"]) <= at, "request precedes its split")
    if row.get("split_end"):
        require(at < timestamp(row["split_end"]), "request is outside its split")
    candidates = row.get("candidates")
    require(isinstance(candidates, list), "candidates must be a list")
    require(type(row.get("candidate_count")) is int and
            row["candidate_count"] == len(candidates), "candidate count mismatch")
    k = row.get("requested_k")
    require(type(k) is int and k > 0, "requested_k must be positive")
    epsilon = row.get("epsilon")
    require(type(epsilon) in (int, float) and math.isfinite(epsilon)
            and 0 <= epsilon <= 1, "invalid exploration rate")
    require(type(row.get("explored")) is bool, "explored must be a boolean")
    require(not row["explored"] or epsilon > 0, "exploration at zero epsilon")
    take = min(k, len(candidates))
    seen, ranks, served = set(), [], []
    for candidate in candidates:
        identity = item_key(candidate)
        require(identity[:2] not in seen, "duplicate candidate or logical item version")
        seen.add(identity[:2])
        rank = candidate.get("candidate_rank")
        require(type(rank) is int and rank > 0, "invalid candidate rank")
        ranks.append(rank)
        served_rank = candidate.get("served_rank")
        if served_rank is not None:
            require(type(served_rank) is int and served_rank > 0, "invalid served rank")
            served.append(served_rank)
        for field in ("score", "behavior_propensity"):
            value = candidate.get(field)
            require(type(value) in (int, float) and math.isfinite(value),
                    f"invalid candidate {field}")
        propensity = candidate["behavior_propensity"]
        require(0 <= propensity <= 1, "propensity outside [0, 1]")
        expected = (1 - epsilon) * float(rank <= take) + epsilon * take / len(candidates)
        require(math.isclose(propensity, expected, abs_tol=1e-8),
                "propensity disagrees with epsilon-slate marginal inclusion probability")
        require(served_rank is None or propensity > 0, "served candidate has zero support")
        if not row["explored"]:
            require(served_rank == (rank if rank <= take else None),
                    "non-exploratory slate must equal deterministic top-k")
    require(sorted(ranks) == list(range(1, len(candidates) + 1)), "candidate ranks are not contiguous")
    require(sorted(served) == list(range(1, take + 1)), "served slate size or ranks are invalid")


def validate_event(event: dict, request: dict) -> None:
    require(event.get("schema_version") == 1, "unsupported knowledge event schema")
    require(nonempty(event.get("event_id")), "event_id is required")
    for key in ("tenant_id", "workspace_id", "workspace_type", "actor_id",
                "recommendation_id", "policy_version", "model_version",
                "feature_snapshot_version", "context_type", "context_id", "split", "synthetic"):
        require(event.get(key) == request.get(key), f"event lineage mismatch: {key}")
    require(event.get("event_type") in EVENT_TYPES, "unsupported event type")
    identity = item_key(event)
    candidate = next((item for item in request["candidates"] if item_key(item) == identity), None)
    require(candidate is not None and candidate["served_rank"] is not None,
            "event must refer to a served item version")
    require(event.get("served_rank") == candidate["served_rank"], "event position mismatch")
    require(timestamp(event.get("occurred_at")) >= timestamp(request["requested_at"]),
            "event precedes recommendation")
    value = event.get("value")
    require(type(value) in (int, float) and math.isfinite(value), "event value must be finite")
    require(isinstance(event.get("metadata", {}), dict), "event metadata must be an object")
    if event["event_type"] == "dwell":
        require(value >= 0, "dwell must be nonnegative")
    if event["event_type"] in {"create_action", "complete_action"}:
        require(nonempty(event.get("metadata", {}).get("action_id")), "action_id is required")


def validate_event_lineage(requests: list[dict], events: list[dict]) -> None:
    by_request = {request_key(row): row for row in requests}
    ids, impressions, actions, completions = set(), {}, {}, set()
    for event in sorted(events, key=lambda e: (timestamp(e["occurred_at"]),
                                             e["event_type"] != "impression", e["event_id"])):
        identity = (*request_key(event)[:2], event["event_id"])
        require(identity not in ids, "duplicate event_id")
        ids.add(identity)
        request = by_request.get(request_key(event))
        require(request is not None, "event refers to an unknown recommendation")
        validate_event(event, request)
        key = (*request_key(event), *item_key(event))
        if event["event_type"] == "impression":
            require(key not in impressions, "duplicate impression")
            impressions[key] = timestamp(event["occurred_at"])
        else:
            require(key in impressions and impressions[key] <= timestamp(event["occurred_at"]),
                    "feedback precedes or lacks an impression")
        if event["event_type"] in {"create_action", "complete_action"}:
            action = (*request_key(event)[:2], event["metadata"]["action_id"])
            if event["event_type"] == "create_action":
                require(action not in actions, "duplicate action creation")
                actions[action] = key
            else:
                require(actions.get(action) == key, "action completion has no matching creation")
                require(action not in completions, "duplicate action completion")
                completions.add(action)
    expected = {(*request_key(row), *item_key(candidate))
                for row in requests for candidate in row["candidates"]
                if candidate["served_rank"] is not None}
    require(set(impressions) == expected, "snapshot must contain one impression per served item")


@dataclass
class EnterpriseDataset:
    manifest: dict
    tenants: dict
    users: dict
    items: list[dict]
    requests: list[dict]
    events: list[dict]
    labels: dict
    train_end: str
    test_start: str
    source_hashes: dict

    def principal(self, request: dict) -> Principal:
        user = self.users[(request["tenant_id"], request["actor_id"])]
        return Principal(request["tenant_id"], request["workspace_id"],
                         request["actor_id"], user["role"])

    @classmethod
    def load(cls, directory: Path) -> EnterpriseDataset:
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        require(manifest.get("synthetic") is True, "seed adapter accepts explicitly synthetic data only")
        require(set(manifest.get("files", {})) == set(DATA_FILES), "manifest file set is incomplete")
        raw, hashes = {}, {}
        for name in DATA_FILES:
            data = (directory / name).read_bytes()
            hashes[name] = hashlib.sha256(data).hexdigest()
            expected = manifest["files"][name]
            require(hashes[name] == expected["sha256"], f"hash mismatch: {name}")
            rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
            require(len(rows) == expected["rows"], f"row count mismatch: {name}")
            require(all(isinstance(row, dict) and row.get("synthetic") is True for row in rows),
                    f"missing synthetic provenance: {name}")
            raw[name] = rows
        return cls.from_rows(raw, manifest, hashes)

    @classmethod
    def from_rows(cls, raw: dict, manifest: dict, hashes: dict | None = None) -> EnterpriseDataset:
        require(set(raw) == set(DATA_FILES), "incomplete dataset")
        require(all(isinstance(row, dict) and row.get("synthetic") is True
                    for rows in raw.values() for row in rows), "missing synthetic row provenance")
        tenants = unique_index(raw["tenants.jsonl"], ("tenant_id",))
        users = unique_index(raw["users.jsonl"], ("tenant_id", "user_id"))
        raw_items = unique_index(raw["items.jsonl"], ("tenant_id", "item_id"))
        start = timestamp(manifest["time_range"]["start"])
        end = timestamp(manifest["time_range"]["end"])
        train_end, test_start = start + timedelta(days=60), start + timedelta(days=75)
        require(end == start + timedelta(days=90), "unsupported seed time protocol")
        require(manifest.get("synthetic") is True, "missing synthetic manifest provenance")
        require(manifest.get("dataset") == "enterprise_knowledge_recommendation_synthetic_pilot"
                and manifest.get("version") == "1.0.0", "unsupported upstream dataset schema")
        for tenant in tenants.values():
            require(nonempty(tenant.get("workspace_id")), "missing workspace_id")
        for (tenant, actor), user in users.items():
            require((tenant,) in tenants and nonempty(user.get("role")), "invalid user membership")
            require(user.get("workspace_id", tenants[(tenant,)]["workspace_id"]) ==
                    tenants[(tenant,)]["workspace_id"], "user workspace mismatch")
        items, item_lookup = [], {}
        for (tenant, identity), item in raw_items.items():
            require((tenant,) in tenants, "item has unknown tenant")
            require((tenant, item["owner_id"]) in users, "item has unknown owner")
            require(item.get("workspace_id", tenants[(tenant,)]["workspace_id"]) ==
                    tenants[(tenant,)]["workspace_id"], "item workspace mismatch")
            require(item.get("workspace_type", "team") == "team",
                    "upstream v1 adapter supports team workspaces only")
            normalized = {key: item[key] for key in (
                "tenant_id", "owner_id", "item_type", "item_id", "title", "summary",
                "content", "tags", "status", "created_at", "visible_roles",
                "superseded_at", "revoked_at", "synthetic")}
            normalized.update(
                item_version=item["version"], workspace_id=tenants[(tenant,)]["workspace_id"],
                workspace_type="team", visibility=item.get("visibility", "workspace"))
            item_key(normalized)
            require(item["status"] in {"ACTIVE", "DRAFT", "SUPERSEDED", "REVOKED"},
                    "unknown item state")
            created = timestamp(item["created_at"])
            for field in ("superseded_at", "revoked_at"):
                if item.get(field):
                    require(timestamp(item[field]) >= created, "invalid item validity interval")
            if item["status"] == "SUPERSEDED":
                require(bool(item["superseded_at"]), "superseded version has no effective time")
            if item["status"] == "REVOKED":
                require(bool(item["revoked_at"]), "revoked version has no effective time")
            require(isinstance(item["visible_roles"], list) and
                    all(nonempty(role) for role in item["visible_roles"]), "invalid item roles")
            items.append(normalized)
            item_lookup[(tenant, identity)] = normalized
        for relation in raw["relations.jsonl"]:
            require((relation["tenant_id"], relation["source_item_id"]) in item_lookup
                    and (relation["tenant_id"], relation["target_item_id"]) in item_lookup,
                    "dangling or cross-tenant relation")
        raw_requests = unique_index(raw["recommendations.jsonl"], ("tenant_id", "recommendation_id"))
        requests, lookup = [], {}
        for (tenant, identity), row in raw_requests.items():
            user = users.get((tenant, row["user_id"]))
            require(user is not None and user["role"] == row["user_role"], "request membership mismatch")
            at = timestamp(row["requested_at"])
            require(start <= at < end, "request outside dataset window")
            split = "train" if at < train_end else "validation" if at < test_start else "test"
            require(row["split"] == split, "request split disagrees with its actual timestamp")
            workspace = tenants[(tenant,)]["workspace_id"]
            require(row.get("workspace_id", workspace) == workspace and
                    row.get("workspace_type", "team") == "team", "request workspace mismatch")
            principal = Principal(tenant, workspace, row["user_id"], user["role"])
            candidates = []
            for candidate in row["candidates"]:
                item = item_lookup.get((tenant, candidate["item_id"]))
                require(item is not None and can_access(principal, item, at),
                        "unauthorized, missing or inactive candidate")
                require(candidate.get("item_version", item["item_version"]) == item["item_version"]
                        and candidate.get("item_type", item["item_type"]) == item["item_type"],
                        "candidate item type/version mismatch")
                candidates.append({
                    "item_type": item["item_type"], "item_id": item["item_id"],
                    "item_version": item["item_version"],
                    **{key: candidate[key] for key in (
                        "candidate_rank", "served_rank", "score", "behavior_propensity")},
                    "recommendation_reason": "upstream synthetic behavior policy",
                })
            boundaries = {"train": (start, train_end), "validation": (train_end, test_start),
                          "test": (test_start, end)}
            request = {
                "schema_version": 1, "recommendation_id": identity, "tenant_id": tenant,
                "workspace_id": workspace, "workspace_type": "team", "actor_id": row["user_id"],
                "requested_at": row["requested_at"], "context_type": row["context"]["surface"],
                # The source has no real PRD identifier; this is explicitly a synthetic context.
                "context_id": f"synthetic-context:{identity}",
                "context": {"intent": row["context"]["intent"]},
                "policy_version": row["policy_version"], "model_version": row["model_version"],
                "feature_snapshot_version": f"upstream:{fingerprint(row['candidates'])}",
                "requested_k": 5, "candidate_count": row["candidate_count"],
                "epsilon": row["epsilon"], "explored": row["explored"], "candidates": candidates,
                "split": split, "split_start": boundaries[split][0].isoformat(),
                "split_end": boundaries[split][1].isoformat(), "synthetic": True,
            }
            validate_request(request)
            requests.append(request)
            lookup[(tenant, identity)] = request
        require(manifest["request_count"] == len(requests), "manifest request_count mismatch")
        require(manifest["split_counts"] == dict(Counter(r["split"] for r in requests)),
                "manifest split counts mismatch")
        events = []
        for row in raw["events.jsonl"]:
            rec = lookup.get((row["tenant_id"], row["recommendation_id"]))
            require(rec is not None, "orphan feedback event")
            require(row["user_id"] == rec["actor_id"] and row["split"] == rec["split"],
                    "event actor/split mismatch")
            item = item_lookup.get((row["tenant_id"], row["item_id"]))
            require(item is not None, "unknown event item")
            event = {key: rec[key] for key in (
                "schema_version", "tenant_id", "workspace_id", "workspace_type",
                "actor_id", "recommendation_id", "context_type", "context_id", "policy_version",
                "model_version", "feature_snapshot_version", "split", "synthetic")}
            event.update(
                event_id=row["event_id"], item_type=item["item_type"], item_id=item["item_id"],
                item_version=item["item_version"], event_type=row["event_type"],
                occurred_at=row["occurred_at"], served_rank=row["position"], value=row["value"],
                metadata=row.get("metadata", {}))
            for field in ("workspace_id", "workspace_type", "actor_id", "policy_version",
                          "model_version", "item_type", "item_version"):
                require(field not in row or row[field] == event[field],
                        f"upstream event lineage mismatch: {field}")
            events.append(event)
        require(manifest["event_counts"] == dict(Counter(e["event_type"] for e in events)),
                "manifest event counts mismatch")
        validate_event_lineage(requests, events)
        truth = unique_index(raw["ground_truth.jsonl"], ("tenant_id", "recommendation_id"))
        require(set(truth) == set(raw_requests), "ground truth request coverage mismatch")
        labels = {}
        for key, row in truth.items():
            request = lookup[key]
            require(row["split"] == request["split"] and row["label_source"] == "synthetic_rule_v1",
                    "invalid ground truth lineage")
            grades = {}
            for label in row["labels"]:
                identity = label["item_id"]
                require(identity not in grades, "duplicate ground truth label")
                require(type(label["relevance_grade"]) is int and
                        0 <= label["relevance_grade"] <= 3, "invalid relevance grade")
                grades[identity] = label["relevance_grade"]
            require(set(grades) == {c["item_id"] for c in request["candidates"]},
                    "ground truth must cover exactly the logged pool")
            labels[request_key(request)] = {
                item_key(candidate): grades[candidate["item_id"]] for candidate in request["candidates"]}
        return cls(manifest, tenants, users, items, requests, events, labels,
                   train_end.isoformat(), test_start.isoformat(), hashes or {})
