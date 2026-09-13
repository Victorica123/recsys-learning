"""Local, versioned enterprise replay store; production writes belong in EIP/MySQL.

This is a separate schema from the established MovieLens feedback database.
Every lookup is scoped by tenant and workspace; event ownership is checked
against a trusted principal and an immutable recommendation snapshot.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from enterprise_dataset import (
    EnterpriseDataset, nonempty, require, request_key, validate_event, validate_request,
)
from enterprise_recommendation import (
    KnowledgeAuthorizationError, KnowledgeValidationError, Principal,
    canonical_json, fingerprint, item_key, timestamp,
)


class KnowledgeConflictError(KnowledgeValidationError):
    """An idempotency key was reused with different content."""


class KnowledgeFeedbackStore:
    def __init__(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS knowledge_requests (
                tenant_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                recommendation_id TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                requested_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                digest TEXT NOT NULL,
                PRIMARY KEY (tenant_id, workspace_id, recommendation_id)
            );
            CREATE TABLE IF NOT EXISTS knowledge_candidates (
                tenant_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                recommendation_id TEXT NOT NULL,
                item_type TEXT NOT NULL,
                item_id TEXT NOT NULL,
                item_version INTEGER NOT NULL,
                candidate_rank INTEGER NOT NULL,
                served_rank INTEGER,
                score REAL NOT NULL,
                behavior_propensity REAL NOT NULL,
                PRIMARY KEY (tenant_id, workspace_id, recommendation_id,
                             item_type, item_id, item_version),
                FOREIGN KEY (tenant_id, workspace_id, recommendation_id)
                  REFERENCES knowledge_requests ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS knowledge_events (
                tenant_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                recommendation_id TEXT NOT NULL,
                item_type TEXT NOT NULL,
                item_id TEXT NOT NULL,
                item_version INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                action_id TEXT,
                payload TEXT NOT NULL,
                digest TEXT NOT NULL,
                PRIMARY KEY (tenant_id, workspace_id, event_id),
                FOREIGN KEY (tenant_id, workspace_id, recommendation_id,
                             item_type, item_id, item_version)
                  REFERENCES knowledge_candidates ON DELETE CASCADE
            );
            CREATE UNIQUE INDEX IF NOT EXISTS knowledge_one_impression
              ON knowledge_events (tenant_id, workspace_id, recommendation_id,
                                   item_type, item_id, item_version)
              WHERE event_type='impression';
            CREATE UNIQUE INDEX IF NOT EXISTS knowledge_one_action_transition
              ON knowledge_events (tenant_id, workspace_id, action_id, event_type)
              WHERE event_type IN ('create_action', 'complete_action');
            CREATE INDEX IF NOT EXISTS knowledge_actor
              ON knowledge_requests (tenant_id, workspace_id, actor_id);
            CREATE INDEX IF NOT EXISTS knowledge_event_replay
              ON knowledge_events (tenant_id, workspace_id, recommendation_id, occurred_at);
        """)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.db.close()

    def _record_request(self, request: dict, principal: Principal) -> bool:
        principal.check_request(request)
        validate_request(request)
        key = request_key(request)
        payload, digest = canonical_json(request), fingerprint(request)
        previous = self.db.execute(
            "SELECT digest FROM knowledge_requests "
            "WHERE tenant_id=? AND workspace_id=? AND recommendation_id=?", key).fetchone()
        if previous:
            if previous["digest"] != digest:
                raise KnowledgeConflictError("recommendation ID already has different content")
            return False
        self.db.execute(
            "INSERT INTO knowledge_requests VALUES (?, ?, ?, ?, ?, ?, ?)",
            (*key, principal.actor_id, timestamp(request["requested_at"]).isoformat(), payload, digest))
        self.db.executemany(
            "INSERT INTO knowledge_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(*key, *item_key(candidate), candidate["candidate_rank"], candidate["served_rank"],
              candidate["score"], candidate["behavior_propensity"])
             for candidate in request["candidates"]])
        return True

    def record_request(self, request: dict, principal: Principal) -> bool:
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            return self._record_request(request, principal)

    def _record_event(self, event: dict, principal: Principal) -> bool:
        principal.check_request(event)
        scope = (principal.tenant_id, principal.workspace_id)
        request = self.db.execute(
            "SELECT payload FROM knowledge_requests WHERE tenant_id=? AND workspace_id=? "
            "AND recommendation_id=? AND actor_id=?",
            (*scope, event["recommendation_id"], principal.actor_id)).fetchone()
        if request is None:
            raise KnowledgeAuthorizationError("no recommendation belonging to this principal")
        request = json.loads(request["payload"])
        validate_event(event, request)
        digest = fingerprint(event)
        existing = self.db.execute(
            "SELECT digest FROM knowledge_events WHERE tenant_id=? AND workspace_id=? AND event_id=?",
            (*scope, event["event_id"])).fetchone()
        if existing:
            if existing["digest"] != digest:
                raise KnowledgeConflictError("event ID already has different content")
            return False
        identity = (*request_key(event), *item_key(event))
        if event["event_type"] != "impression":
            impression = self.db.execute(
                "SELECT occurred_at FROM knowledge_events WHERE tenant_id=? AND workspace_id=? "
                "AND recommendation_id=? AND item_type=? AND item_id=? AND item_version=? "
                "AND event_type='impression'", identity).fetchone()
            require(impression is not None and
                    timestamp(impression["occurred_at"]) <= timestamp(event["occurred_at"]),
                    "feedback precedes or lacks an impression")
        action = event.get("metadata", {}).get("action_id")
        if event["event_type"] == "complete_action":
            created = self.db.execute(
                "SELECT payload FROM knowledge_events WHERE tenant_id=? AND workspace_id=? "
                "AND action_id=? AND event_type='create_action'", (*scope, action)).fetchone()
            require(created is not None, "action completion has no creation")
            created = json.loads(created["payload"])
            require((*request_key(created), *item_key(created)) == identity and
                    timestamp(created["occurred_at"]) <= timestamp(event["occurred_at"]),
                    "action lineage or time mismatch")
        try:
            self.db.execute(
                "INSERT INTO knowledge_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*scope, event["event_id"], event["recommendation_id"], *item_key(event),
                 event["event_type"], timestamp(event["occurred_at"]).isoformat(),
                 action, canonical_json(event), digest))
        except sqlite3.IntegrityError as exc:
            raise KnowledgeConflictError("duplicate impression or action transition") from exc
        return True

    def record_event(self, event: dict, principal: Principal) -> bool:
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            return self._record_event(event, principal)

    def import_dataset(self, dataset: EnterpriseDataset) -> dict:
        """One atomic transaction; replaying the same validated file is a no-op."""
        added_requests = added_events = 0
        by_request = {request_key(request): request for request in dataset.requests}
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            for request in dataset.requests:
                added_requests += self._record_request(request, dataset.principal(request))
            for event in sorted(dataset.events, key=lambda row: (
                    timestamp(row["occurred_at"]), row["event_type"] != "impression", row["event_id"])):
                principal = dataset.principal(by_request[request_key(event)])
                added_events += self._record_event(event, principal)
        return {"requests_added": added_requests, "events_added": added_events}

    def replay(self, tenant_id: str, workspace_id: str):
        """Privileged offline export of one explicitly selected workspace."""
        require(nonempty(tenant_id) and nonempty(workspace_id), "export scope is required")
        requests = self.db.execute(
            "SELECT payload FROM knowledge_requests WHERE tenant_id=? AND workspace_id=? "
            "ORDER BY requested_at, recommendation_id", (tenant_id, workspace_id)).fetchall()
        for row in requests:
            request = json.loads(row["payload"])
            events = self.db.execute(
                "SELECT payload FROM knowledge_events WHERE tenant_id=? AND workspace_id=? "
                "AND recommendation_id=? ORDER BY occurred_at, event_id",
                request_key(request)).fetchall()
            yield {
                "schema_version": 1, "record_type": "knowledge_replay",
                "recommendation": request, "events": [json.loads(event["payload"]) for event in events],
            }

    def delete_actor(self, principal: Principal) -> int:
        """Local retention primitive; the integrating service authorizes deletion."""
        with self.db:
            result = self.db.execute(
                "DELETE FROM knowledge_requests WHERE tenant_id=? AND workspace_id=? AND actor_id=?",
                (principal.tenant_id, principal.workspace_id, principal.actor_id))
        return result.rowcount

    def purge_before(self, tenant_id: str, workspace_id: str, cutoff: str) -> int:
        require(nonempty(tenant_id) and nonempty(workspace_id), "retention scope is required")
        at = timestamp(cutoff).isoformat()
        with self.db:
            result = self.db.execute(
                "DELETE FROM knowledge_requests WHERE tenant_id=? AND workspace_id=? AND requested_at<?",
                (tenant_id, workspace_id, at))
        return result.rowcount

    def counts(self) -> dict:
        return {name: self.db.execute(f"SELECT COUNT(*) FROM knowledge_{name}").fetchone()[0]
                for name in ("requests", "candidates", "events")}
