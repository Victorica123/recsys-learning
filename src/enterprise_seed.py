#!/usr/bin/env python3
"""Reproduce the user-provided ChatGPT enterprise knowledge seed (2026-09-13).

The synthetic topics, preferences, oracle labels and logged scores deliberately
share rules. They are fixture generation code, NOT production ranking features.
Source hashes and public references: artifacts/enterprise_seed_provenance.json.
Only the entry point was adapted: explicit output and no recursive deletion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


START = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
ROLES = ["product_manager", "backend_engineer", "data_analyst", "security_reviewer", "operations"]

TENANTS = [
    ("tenant_aurora", "极光零售实验室", "retail"),
    ("tenant_lantern", "灯塔金融科技", "fintech"),
    ("tenant_pine", "青松健康服务", "healthtech"),
    ("tenant_orbit", "轨道协作软件", "saas"),
]

TOPICS = {
    "identity_oidc": {
        "label": "统一身份与 OIDC",
        "keywords": ["OIDC", "ID Token", "issuer", "audience", "单点登录"],
        "decision": "统一使用授权码流程，服务端校验签发者、受众、签名和有效期",
        "risk": "回调地址配置错误或令牌校验不完整会造成账号串联风险",
        "acceptance": "测试环境覆盖过期令牌、错误受众、签名轮换和重复回调",
    },
    "tenant_authorization": {
        "label": "多租户与对象授权",
        "keywords": ["tenant_id", "owner_id", "最小权限", "对象授权", "越权"],
        "decision": "每次资源访问同时校验租户、主体和对象归属，不信任客户端过滤",
        "risk": "仅校验登录态会导致跨租户读取或修改资源",
        "acceptance": "所有读写接口具备跨租户负向用例并记录拒绝审计",
    },
    "resumable_upload": {
        "label": "大文件分片上传",
        "keywords": ["分片", "校验和", "幂等", "断点续传", "对象存储"],
        "decision": "上传会话保存分片清单与校验和，完成阶段执行幂等合并",
        "risk": "重复分片、超时重试和并发完成可能产生脏对象",
        "acceptance": "模拟断网、重复请求、乱序分片和合并重试仍得到唯一对象",
    },
    "search_rag": {
        "label": "证据化 RAG",
        "keywords": ["向量检索", "关键词检索", "证据", "引用", "重排"],
        "decision": "采用关键词与向量混合召回，生成回答必须绑定可回溯证据片段",
        "risk": "过期知识或无权限片段进入上下文会产生错误且不可审计的回答",
        "acceptance": "答案逐条引用授权片段，撤销文档在下一次检索中不可见",
    },
    "observability": {
        "label": "可观测性与追踪",
        "keywords": ["trace", "metric", "log", "correlation_id", "SLO"],
        "decision": "请求、异步任务和模型调用共享关联标识，并分别记录追踪、指标与日志",
        "risk": "只看应用日志无法定位跨服务延迟和队列堆积",
        "acceptance": "一次分析任务可从入口追踪到模型调用、持久化和前端推送",
    },
    "audit_retention": {
        "label": "审计与数据留存",
        "keywords": ["审计", "留存", "删除", "导出", "不可抵赖"],
        "decision": "业务数据和审计数据采用独立留存策略，删除操作记录主体与原因",
        "risk": "留存边界不清会导致数据长期滞留或审计证据缺失",
        "acceptance": "到期任务可重试、可追踪，合法保留和用户删除互不冲突",
    },
    "notification": {
        "label": "通知与事件投递",
        "keywords": ["outbox", "幂等消费", "重试", "死信", "通知偏好"],
        "decision": "业务事务写入 outbox，消费者以事件 ID 幂等处理并执行有限重试",
        "risk": "直接同步通知会让外部故障扩大到核心事务",
        "acceptance": "重复投递不产生重复通知，失败事件可进入人工处理队列",
    },
    "billing": {
        "label": "用量计费与配额",
        "keywords": ["usage", "quota", "账单", "限额", "对账"],
        "decision": "原始用量事件不可变，聚合账单可重算，并对租户设置软硬配额",
        "risk": "事件重复或迟到会造成账单偏差和额度误判",
        "acceptance": "同一事件重复消费不重复计费，迟到事件进入下一次对账",
    },
}

RELATED = {
    "identity_oidc": {"tenant_authorization"},
    "tenant_authorization": {"identity_oidc", "audit_retention"},
    "resumable_upload": {"observability"},
    "search_rag": {"audit_retention", "observability"},
    "observability": {"resumable_upload", "search_rag", "notification"},
    "audit_retention": {"tenant_authorization", "search_rag"},
    "notification": {"observability"},
    "billing": {"audit_retention"},
}


def iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def split_for_day(day: int) -> str:
    if day < 60:
        return "train"
    if day < 75:
        return "validation"
    return "test"


def active_at(item: dict, at: datetime) -> bool:
    created = datetime.fromisoformat(item["created_at"].replace("Z", "+00:00"))
    if created > at or item["status"] == "DRAFT":
        return False
    for key in ("superseded_at", "revoked_at"):
        if item.get(key):
            boundary = datetime.fromisoformat(item[key].replace("Z", "+00:00"))
            if at >= boundary:
                return False
    return True


def make_tenants() -> list[dict]:
    return [
        {
            "tenant_id": tenant_id,
            "name": name,
            "industry": industry,
            "workspace_id": f"workspace_{tenant_id.removeprefix('tenant_')}",
            "synthetic": True,
        }
        for tenant_id, name, industry in TENANTS
    ]


def make_users(rng: random.Random) -> list[dict]:
    users = []
    topic_ids = list(TOPICS)
    for tenant_index, (tenant_id, _, _) in enumerate(TENANTS):
        for local_index in range(10):
            role = ROLES[local_index % len(ROLES)]
            primary = topic_ids[(local_index + tenant_index) % len(topic_ids)]
            alternatives = [topic for topic in topic_ids if topic != primary]
            secondary = rng.choice(alternatives)
            users.append(
                {
                    "user_id": f"user_{tenant_index + 1:02d}_{local_index + 1:02d}",
                    "tenant_id": tenant_id,
                    "role": role,
                    "team": {
                        "product_manager": "product",
                        "backend_engineer": "engineering",
                        "data_analyst": "data",
                        "security_reviewer": "security",
                        "operations": "operations",
                    }[role],
                    "preferred_topics": [primary, secondary],
                    "synthetic": True,
                }
            )
    return users


def content_for(item_type: str, topic_id: str, tenant_name: str, version: int) -> tuple[str, str, str]:
    topic = TOPICS[topic_id]
    label = topic["label"]
    suffix = f" v{version}" if version > 1 else ""
    if item_type == "meeting_transcript":
        title = f"{label}方案评审纪要{suffix}"
        summary = f"{tenant_name}围绕{label}确认边界、风险与验收方式。"
        body = (
            f"产品说明：本季度需要把{label}接入现有业务，不新增孤立演示服务。"
            f"技术决定：{topic['decision']}。"
            f"评审风险：{topic['risk']}。"
            f"验收共识：{topic['acceptance']}。"
            "行动项：后端补充失败路径测试，产品准备灰度名单，安全负责人复核审计字段。"
        )
    elif item_type == "prd":
        title = f"PRD：{label}企业版改造{suffix}"
        summary = f"定义{label}的目标用户、业务规则、异常路径与可观测指标。"
        body = (
            f"背景：{tenant_name}需要在既有流程中落地{label}，目标是减少人工查找与重复确认。"
            f"核心规则：{topic['decision']}。"
            f"非目标：本期不替换核心账号体系，也不把合成指标作为业务收益。"
            f"风险控制：{topic['risk']}。"
            f"验收标准：{topic['acceptance']}。"
        )
    elif item_type == "knowledge":
        title = f"知识库：{label}实施手册{suffix}"
        summary = f"面向研发和运维的{label}配置、验证与故障处置说明。"
        body = (
            f"适用范围：{tenant_name}的生产与预发布工作区。"
            f"标准做法：{topic['decision']}。"
            f"检查清单：{topic['acceptance']}。"
            f"常见故障：{topic['risk']}。"
            "变更需由模块负责人审核，并在发布后观察错误率、延迟和拒绝审计。"
        )
    else:
        title = f"事故复盘：{label}链路异常{suffix}"
        summary = f"一次与{label}相关的合成事故复盘，包含影响、根因和修复项。"
        body = (
            f"影响：{tenant_name}部分内部请求延迟升高，未发现真实数据泄露。"
            f"根因：测试环境遗漏了一个关键失败分支；已知风险为“{topic['risk']}”。"
            f"立即措施：回滚变更、限制入口、补充审计。"
            f"长期修复：{topic['decision']}；验收采用“{topic['acceptance']}”。"
        )
    return title, summary, body


def make_items_and_relations(rng: random.Random) -> tuple[list[dict], list[dict]]:
    items: list[dict] = []
    relations: list[dict] = []
    topic_ids = list(TOPICS)
    for tenant_index, (tenant_id, tenant_name, _) in enumerate(TENANTS):
        for topic_index, topic_id in enumerate(topic_ids):
            base_day = 2 + topic_index * 2
            ids: dict[str, str] = {}
            for offset, item_type in enumerate(("meeting_transcript", "prd", "knowledge", "incident")):
                item_id = f"item_{tenant_index + 1:02d}_{topic_index + 1:02d}_{offset + 1:02d}"
                ids[item_type] = item_id
                created_at = START + timedelta(days=base_day + offset, hours=offset)
                status = "ACTIVE"
                superseded_at = None
                revoked_at = None
                if item_type == "knowledge" and topic_index in (0, 3):
                    status = "SUPERSEDED"
                    superseded_at = START + timedelta(days=55 + tenant_index)
                if item_type == "incident" and topic_index == 6:
                    status = "REVOKED"
                    revoked_at = START + timedelta(days=70 + tenant_index)
                title, summary, body = content_for(item_type, topic_id, tenant_name, 1)
                visible_roles = ROLES.copy()
                if item_type == "incident":
                    visible_roles = ["backend_engineer", "security_reviewer", "operations", "product_manager"]
                items.append(
                    {
                        "item_id": item_id,
                        "tenant_id": tenant_id,
                        "owner_id": f"user_{tenant_index + 1:02d}_{(topic_index % 10) + 1:02d}",
                        "item_type": item_type,
                        "topic": topic_id,
                        "title": title,
                        "summary": summary,
                        "content": body,
                        "tags": TOPICS[topic_id]["keywords"],
                        "version": 1,
                        "status": status,
                        "created_at": iso(created_at),
                        "superseded_at": iso(superseded_at) if superseded_at else None,
                        "revoked_at": iso(revoked_at) if revoked_at else None,
                        "visible_roles": visible_roles,
                        "popularity_prior": round(rng.uniform(0.15, 0.85), 4),
                        "synthetic": True,
                    }
                )
            relations.extend(
                [
                    {"tenant_id": tenant_id, "source_item_id": ids["prd"], "target_item_id": ids["meeting_transcript"], "relation_type": "derived_from", "synthetic": True},
                    {"tenant_id": tenant_id, "source_item_id": ids["knowledge"], "target_item_id": ids["prd"], "relation_type": "implements", "synthetic": True},
                    {"tenant_id": tenant_id, "source_item_id": ids["incident"], "target_item_id": ids["knowledge"], "relation_type": "references", "synthetic": True},
                ]
            )
            if topic_index in (0, 3):
                old_id = ids["knowledge"]
                new_id = f"item_{tenant_index + 1:02d}_{topic_index + 1:02d}_05"
                created_at = START + timedelta(days=55 + tenant_index)
                title, summary, body = content_for("knowledge", topic_id, tenant_name, 2)
                items.append(
                    {
                        "item_id": new_id,
                        "tenant_id": tenant_id,
                        "owner_id": f"user_{tenant_index + 1:02d}_{(topic_index % 10) + 1:02d}",
                        "item_type": "knowledge",
                        "topic": topic_id,
                        "title": title,
                        "summary": summary + " 本版补充了生产故障后的强制校验项。",
                        "content": body + " 新版要求灰度阶段保留逐请求验证记录，并复核回滚路径。",
                        "tags": TOPICS[topic_id]["keywords"] + ["新版"],
                        "version": 2,
                        "status": "ACTIVE",
                        "created_at": iso(created_at),
                        "superseded_at": None,
                        "revoked_at": None,
                        "visible_roles": ROLES.copy(),
                        "popularity_prior": round(rng.uniform(0.25, 0.75), 4),
                        "synthetic": True,
                    }
                )
                relations.append(
                    {"tenant_id": tenant_id, "source_item_id": new_id, "target_item_id": old_id, "relation_type": "supersedes", "synthetic": True}
                )
    return items, relations


def role_match(role: str, item_type: str) -> float:
    favored = {
        "product_manager": {"prd", "meeting_transcript"},
        "backend_engineer": {"knowledge", "incident"},
        "data_analyst": {"prd", "knowledge"},
        "security_reviewer": {"incident", "knowledge"},
        "operations": {"incident", "knowledge"},
    }
    return 1.0 if item_type in favored[role] else 0.35


def relevance(topic: str, item: dict, role: str) -> int:
    if item["topic"] == topic:
        base = 3
    elif item["topic"] in RELATED.get(topic, set()):
        base = 2
    else:
        base = 0
    if base == 2 and role_match(role, item["item_type"]) < 0.5:
        return 1
    return base


def make_recommendations(
    rng: random.Random,
    users: list[dict],
    items: list[dict],
    relations: list[dict],
    request_count: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    recommendations: list[dict] = []
    ground_truth: list[dict] = []
    events: list[dict] = []
    item_by_id = {item["item_id"]: item for item in items}
    relation_pairs = {(row["source_item_id"], row["target_item_id"]) for row in relations}
    relation_pairs |= {(b, a) for a, b in relation_pairs}
    topic_ids = list(TOPICS)
    for request_index in range(request_count):
        user = users[request_index % len(users)]
        day = 30 + (request_index * 59 // max(1, request_count - 1))
        minute = (request_index * 37) % (10 * 60)
        requested_at = START + timedelta(days=day, minutes=minute)
        topic = user["preferred_topics"][request_index % 2] if request_index % 3 else rng.choice(topic_ids)
        request_id = f"rec_{request_index + 1:06d}"
        split = split_for_day(day)
        policy_name = "semantic_hybrid_v1" if request_index % 2 else "recency_popularity_v1"
        epsilon = 0.15 if request_index % 5 == 0 else 0.0
        candidates = [
            item
            for item in items
            if item["tenant_id"] == user["tenant_id"]
            and user["role"] in item["visible_roles"]
            and active_at(item, requested_at)
        ]
        scored: list[dict] = []
        for item in candidates:
            created = datetime.fromisoformat(item["created_at"].replace("Z", "+00:00"))
            age_days = max(0.0, (requested_at - created).total_seconds() / 86400)
            recency = math.exp(-age_days / 75.0)
            topic_match = 1.0 if item["topic"] == topic else (0.55 if item["topic"] in RELATED.get(topic, set()) else 0.05)
            rmatch = role_match(user["role"], item["item_type"])
            popularity = item["popularity_prior"]
            context_items = [candidate["item_id"] for candidate in candidates if candidate["topic"] == topic and candidate["item_type"] == "meeting_transcript"]
            relation_boost = 1.0 if any((context_id, item["item_id"]) in relation_pairs for context_id in context_items) else 0.0
            if policy_name == "recency_popularity_v1":
                score = 0.55 * recency + 0.45 * popularity
            else:
                score = 0.52 * topic_match + 0.16 * rmatch + 0.14 * recency + 0.10 * popularity + 0.08 * relation_boost
            score += rng.uniform(-0.01, 0.01)
            scored.append(
                {
                    "item_id": item["item_id"],
                    "features": {
                        "topic_match": round(topic_match, 4),
                        "role_match": round(rmatch, 4),
                        "recency_score": round(recency, 4),
                        "popularity_score": round(popularity, 4),
                        "relation_boost": round(relation_boost, 4),
                    },
                    "score": round(score, 6),
                }
            )
        scored.sort(key=lambda row: (-row["score"], row["item_id"]))
        pool = scored[: min(16, len(scored))]
        k = min(5, len(pool))
        deterministic_ids = [row["item_id"] for row in pool[:k]]
        explored = epsilon > 0 and rng.random() < epsilon
        if explored:
            served_ids = [row["item_id"] for row in rng.sample(pool, k)]
        else:
            served_ids = deterministic_ids
        served_rank = {item_id: rank for rank, item_id in enumerate(served_ids, start=1)}
        n = len(pool)
        for candidate_rank, row in enumerate(pool, start=1):
            item_id = row["item_id"]
            propensity = (1 - epsilon) * (1.0 if item_id in deterministic_ids else 0.0) + epsilon * (k / n)
            row["candidate_rank"] = candidate_rank
            row["served_rank"] = served_rank.get(item_id)
            row["behavior_propensity"] = round(propensity, 8)
        recommendations.append(
            {
                "recommendation_id": request_id,
                "tenant_id": user["tenant_id"],
                "user_id": user["user_id"],
                "user_role": user["role"],
                "requested_at": iso(requested_at),
                "context": {
                    "surface": "analysis_result_sidebar",
                    "intent": f"查找与{TOPICS[topic]['label']}相关的可信知识和历史方案",
                    "topic": topic,
                },
                "model_version": "synthetic_ranker_2026_01",
                "policy_version": policy_name,
                "epsilon": epsilon,
                "explored": explored,
                "candidate_count": len(pool),
                "candidates": pool,
                "split": split,
                "synthetic": True,
            }
        )
        labels = []
        for row in pool:
            grade = relevance(topic, item_by_id[row["item_id"]], user["role"])
            labels.append({"item_id": row["item_id"], "relevance_grade": grade})
        ground_truth.append(
            {
                "recommendation_id": request_id,
                "tenant_id": user["tenant_id"],
                "labels": labels,
                "label_source": "synthetic_rule_v1",
                "split": split,
                "synthetic": True,
            }
        )
        event_sequence = 0
        for item_id in served_ids:
            event_sequence += 1
            event_time = requested_at + timedelta(seconds=event_sequence * 2)
            item = item_by_id[item_id]
            grade = relevance(topic, item, user["role"])
            preferred = item["topic"] in user["preferred_topics"]
            impression_id = f"evt_{request_index + 1:06d}_{event_sequence:02d}_imp"
            events.append(
                {
                    "event_id": impression_id,
                    "recommendation_id": request_id,
                    "tenant_id": user["tenant_id"],
                    "user_id": user["user_id"],
                    "item_id": item_id,
                    "event_type": "impression",
                    "occurred_at": iso(event_time),
                    "position": served_rank[item_id],
                    "value": 1.0,
                    "split": split,
                    "synthetic": True,
                }
            )
            open_p = min(0.92, 0.08 + 0.19 * grade + (0.10 if preferred else 0) + 0.04 * role_match(user["role"], item["item_type"]))
            opened = rng.random() < open_p
            if opened:
                event_sequence += 1
                dwell = int(18 + grade * 24 + rng.uniform(0, 55))
                events.append(
                    {
                        "event_id": f"evt_{request_index + 1:06d}_{event_sequence:02d}_open",
                        "recommendation_id": request_id,
                        "tenant_id": user["tenant_id"],
                        "user_id": user["user_id"],
                        "item_id": item_id,
                        "event_type": "open",
                        "occurred_at": iso(event_time + timedelta(seconds=3)),
                        "position": served_rank[item_id],
                        "value": dwell,
                        "metadata": {"unit": "dwell_seconds"},
                        "split": split,
                        "synthetic": True,
                    }
                )
                if rng.random() < 0.04 + 0.13 * grade:
                    event_sequence += 1
                    events.append(
                        {
                            "event_id": f"evt_{request_index + 1:06d}_{event_sequence:02d}_cite",
                            "recommendation_id": request_id,
                            "tenant_id": user["tenant_id"],
                            "user_id": user["user_id"],
                            "item_id": item_id,
                            "event_type": "cite",
                            "occurred_at": iso(event_time + timedelta(seconds=8)),
                            "position": served_rank[item_id],
                            "value": 1.0,
                            "split": split,
                            "synthetic": True,
                        }
                    )
                if rng.random() < 0.03 + 0.08 * grade:
                    event_sequence += 1
                    events.append(
                        {
                            "event_id": f"evt_{request_index + 1:06d}_{event_sequence:02d}_save",
                            "recommendation_id": request_id,
                            "tenant_id": user["tenant_id"],
                            "user_id": user["user_id"],
                            "item_id": item_id,
                            "event_type": "save",
                            "occurred_at": iso(event_time + timedelta(seconds=12)),
                            "position": served_rank[item_id],
                            "value": 1.0,
                            "split": split,
                            "synthetic": True,
                        }
                    )
                action_p = 0.02 + (0.14 if grade == 3 and user["role"] in {"product_manager", "operations"} else 0)
                if rng.random() < action_p:
                    event_sequence += 1
                    action_id = f"action_{request_index + 1:06d}_{served_rank[item_id]:02d}"
                    action_time = event_time + timedelta(seconds=20)
                    events.append(
                        {
                            "event_id": f"evt_{request_index + 1:06d}_{event_sequence:02d}_action",
                            "recommendation_id": request_id,
                            "tenant_id": user["tenant_id"],
                            "user_id": user["user_id"],
                            "item_id": item_id,
                            "event_type": "create_action",
                            "occurred_at": iso(action_time),
                            "position": served_rank[item_id],
                            "value": 1.0,
                            "metadata": {"action_id": action_id},
                            "split": split,
                            "synthetic": True,
                        }
                    )
                    if rng.random() < 0.48 + 0.10 * grade:
                        event_sequence += 1
                        events.append(
                            {
                                "event_id": f"evt_{request_index + 1:06d}_{event_sequence:02d}_complete",
                                "recommendation_id": request_id,
                                "tenant_id": user["tenant_id"],
                                "user_id": user["user_id"],
                                "item_id": item_id,
                                "event_type": "complete_action",
                                "occurred_at": iso(action_time + timedelta(hours=12 + rng.randint(0, 72))),
                                "position": served_rank[item_id],
                                "value": 1.0,
                                "metadata": {"action_id": action_id},
                                "split": split,
                                "synthetic": True,
                            }
                        )
            elif grade <= 1 and rng.random() < 0.16:
                event_sequence += 1
                events.append(
                    {
                        "event_id": f"evt_{request_index + 1:06d}_{event_sequence:02d}_dismiss",
                        "recommendation_id": request_id,
                        "tenant_id": user["tenant_id"],
                        "user_id": user["user_id"],
                        "item_id": item_id,
                        "event_type": "dismiss",
                        "occurred_at": iso(event_time + timedelta(seconds=5)),
                        "position": served_rank[item_id],
                        "value": -1.0,
                        "split": split,
                        "synthetic": True,
                    }
                )
    return recommendations, ground_truth, events


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset(seed: int = 20260913, requests: int = 640) -> dict[str, list[dict]]:
    """Reproduce the upstream synthetic fixture; never use its oracle in a ranker."""
    if requests < 40:
        raise ValueError("requests must be at least 40")
    rng = random.Random(seed)
    tenants = make_tenants()
    users = make_users(rng)
    items, relations = make_items_and_relations(rng)
    recommendations, ground_truth, events = make_recommendations(
        rng, users, items, relations, requests)
    return {
        "tenants.jsonl": tenants, "users.jsonl": users, "items.jsonl": items,
        "relations.jsonl": relations, "recommendations.jsonl": recommendations,
        "events.jsonl": events, "ground_truth.jsonl": ground_truth,
    }


def generate_dataset(output: Path, seed: int = 20260913, requests: int = 640) -> dict:
    """Write a new dataset directory, refusing to overwrite any existing path."""
    outputs = build_dataset(seed, requests)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in outputs.items():
        write_jsonl(output / name, rows)
    manifest = {
        "dataset": "enterprise_knowledge_recommendation_synthetic_pilot",
        "version": "1.0.0", "seed": seed, "request_count": requests,
        "generated_at": "deterministic-build-no-wall-clock",
        "time_range": {"start": iso(START), "end": iso(START + timedelta(days=90))},
        "split_counts": dict(Counter(r["split"] for r in outputs["recommendations.jsonl"])),
        "event_counts": dict(Counter(r["event_type"] for r in outputs["events.jsonl"])),
        "files": {name: {"rows": len(rows), "sha256": file_hash(output / name)}
                  for name, rows in outputs.items()},
        "synthetic": True,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--requests", type=int, default=640)
    args = parser.parse_args()
    manifest = generate_dataset(args.output, args.seed, args.requests)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
