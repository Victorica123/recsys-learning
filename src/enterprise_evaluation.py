"""Paired, time-aware evaluation; synthetic outcomes can never promote a policy."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from enterprise_dataset import EnterpriseDataset, request_key, require
from enterprise_recommendation import (
    POLICIES, KnowledgeRanker, authorized_candidates, item_key, timestamp,
)


HISTORY_WEIGHTS = {"open": 1.0, "cite": 3.0, "save": 1.0,
                   "create_action": 4.0, "complete_action": 4.0}


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def ranking_metrics(ranked: list[tuple], labels: dict, k: int = 5) -> dict:
    require(k > 0, "metric k must be positive")
    require(len(set(ranked)) == len(ranked), "duplicate item in evaluated slate")
    require(all(key in labels for key in ranked), "unjudged item in evaluated slate")
    ranked = ranked[:k]
    ideal = sum((2 ** grade - 1) / math.log2(i + 2)
                for i, grade in enumerate(sorted(labels.values(), reverse=True)[:k]))
    actual = sum((2 ** labels[key] - 1) / math.log2(i + 2)
                 for i, key in enumerate(ranked))
    relevant = {key for key, grade in labels.items() if grade >= 2}
    first = next((i for i, key in enumerate(ranked, 1) if key in relevant), None)
    return {
        f"ndcg_at_{k}": actual / ideal if ideal else 0.0,
        f"mrr_at_{k}": 1.0 / first if first else 0.0,
        f"recall_at_{k}": len(set(ranked) & relevant) / len(relevant) if relevant else 0.0,
        "has_relevant_target": bool(relevant),
    }


def history_popularity(dataset: EnterpriseDataset, request: dict) -> dict:
    """Use only training-cohort outcomes that actually arrived before the cutoff.

    Train replay uses each request's past. Validation/test share a frozen
    training cutoff. A delayed completion labelled 'train' is not automatically
    a training feature.
    """
    cutoff = min(timestamp(request["requested_at"]), timestamp(dataset.train_end))
    counts = Counter()
    used = set()
    for event in dataset.events:
        if (event["tenant_id"], event["workspace_id"]) != (
                request["tenant_id"], request["workspace_id"]):
            continue
        weight = HISTORY_WEIGHTS.get(event["event_type"], 0.0)
        if not weight or event["split"] != "train" or timestamp(event["occurred_at"]) >= cutoff:
            continue
        identity = (*request_key(event), *item_key(event), event["event_type"])
        if identity not in used:
            used.add(identity)
            counts[item_key(event)] += weight
    return dict(counts)


def business_metrics(dataset: EnterpriseDataset) -> dict:
    """Descriptive logging-policy cohorts, not counterfactual policy outcomes."""
    cohorts = defaultdict(list)
    observation_end = timestamp(dataset.manifest["time_range"]["end"])
    censored = 0
    for event in dataset.events:
        if timestamp(event["occurred_at"]) >= observation_end:
            censored += 1
            continue
        cohorts[(event["split"], event["policy_version"], event["model_version"])].append(event)
    report = {}
    for cohort, events in sorted(cohorts.items()):
        by_type = defaultdict(set)
        actions = defaultdict(set)
        for event in events:
            by_type[event["event_type"]].add((*request_key(event), *item_key(event)))
            if event["event_type"] in {"create_action", "complete_action"}:
                actions[event["event_type"]].add((
                    event["tenant_id"], event["workspace_id"], event["metadata"]["action_id"]))
        impressions = len(by_type["impression"])
        report["/".join(cohort)] = {
            "unique_item_impressions": impressions,
            **{f"{name}_per_impression": len(by_type[name]) / impressions if impressions else None
               for name in ("open", "cite", "save", "dismiss", "create_action")},
            "complete_per_created_action": (
                len(actions["complete_action"]) / len(actions["create_action"])
                if actions["create_action"] else None),
        }
    return {
        "cohorts": report, "observation_end": observation_end.isoformat(),
        "events_after_observation_end": censored,
        "interpretation": "synthetic descriptive cohorts; repeated events count once per item/request",
        "completion_caveat": "late action outcomes are right-censored at observation_end",
    }


def summarize_metrics(rows: list[dict], k: int) -> dict:
    return {
        "requests": len(rows),
        **{name: mean(row[name] for row in rows)
           for name in (f"ndcg_at_{k}", f"mrr_at_{k}", f"recall_at_{k}")},
        "requests_without_relevant_target": sum(not row["has_relevant_target"] for row in rows),
    }


def evaluate_dataset(dataset: EnterpriseDataset, ranker: KnowledgeRanker | None = None,
                     k: int = 5) -> tuple[dict, list[dict]]:
    require(dataset.manifest.get("synthetic") is True, "this benchmark is explicitly synthetic")
    ranker = ranker or KnowledgeRanker()
    paired = defaultdict(lambda: defaultdict(list))
    upstream = defaultdict(list)
    served_catalog = defaultdict(set)
    candidate_catalog = defaultdict(set)
    logged_coverage, audits = [], []
    policy_actors = defaultdict(set)
    unsupported = Counter()
    candidate_positions = Counter()
    slate_records = []
    for request in sorted(dataset.requests, key=lambda row: (
            timestamp(row["requested_at"]), row["recommendation_id"])):
        key, split = request_key(request), request["split"]
        labels = dataset.labels[key]
        principal = dataset.principal(request)
        pool_keys = {item_key(candidate) for candidate in request["candidates"]}
        full_catalog = authorized_candidates(principal, dataset.items, timestamp(request["requested_at"]))
        # These are the only judged candidates in the upstream dataset.
        pool = [item for item in full_catalog if item_key(item) in pool_keys]
        require({item_key(item) for item in pool} == pool_keys, "evaluation pool lost authorized items")
        logged_coverage.append(len(pool) / len(full_catalog) if full_catalog else 1.0)
        scoped = lambda identity: (request["tenant_id"], request["workspace_id"], *identity)
        candidate_catalog[split].update(scoped(identity) for identity in pool_keys)
        policy_actors[request["policy_version"]].add((request["tenant_id"], request["actor_id"]))
        upstream_order = [item_key(candidate) for candidate in sorted(
            (c for c in request["candidates"] if c["served_rank"] is not None),
            key=lambda c: c["served_rank"])]
        upstream[(split, request["policy_version"])].append(
            ranking_metrics(upstream_order, labels, k))
        popularity = history_popularity(dataset, request)
        propensities = {item_key(candidate): candidate["behavior_propensity"]
                        for candidate in request["candidates"]}
        outcomes = {}
        for policy in POLICIES:
            result = ranker.recommend(principal, request, pool, policy=policy, k=k,
                                      popularity=popularity)
            require(result["candidate_count"] == len(labels), "policies must share a judged pool")
            top = [item_key(candidate) for candidate in result["candidates"][:k]]
            outcomes[policy] = ranking_metrics(top, labels, k)
            paired[split][policy].append(outcomes[policy])
            served_catalog[(split, policy)].update(scoped(identity) for identity in top)
            unsupported[policy] += sum(propensities[identity] <= 0 for identity in top)
            candidate_positions[policy] += len(top)
            slate_records.append({
                "recommendation_id": request["recommendation_id"],
                "tenant_id": request["tenant_id"], "workspace_id": request["workspace_id"],
                "split": split, "policy_version": policy, "synthetic": True,
                "feature_snapshot_version": result["feature_snapshot_version"],
                "ranked_items": [{"item_type": key[0], "item_id": key[1], "item_version": key[2]}
                                 for key in top],
                "metrics": outcomes[policy],
            })
        audits.append({
            "split": split,
            "semantic_minus_baseline": outcomes["semantic_v1"][f"ndcg_at_{k}"] -
                                       outcomes["baseline_v0"][f"ndcg_at_{k}"],
            "hybrid_minus_baseline": outcomes["hybrid_v1"][f"ndcg_at_{k}"] -
                                     outcomes["baseline_v0"][f"ndcg_at_{k}"],
        })
    offline = {}
    for split, policies in paired.items():
        offline[split] = {}
        for policy, values in policies.items():
            offline[split][policy] = {
                **summarize_metrics(values, k),
                "judged_catalog_coverage": (
                    len(served_catalog[(split, policy)]) / len(candidate_catalog[split])
                    if candidate_catalog[split] else 0.0),
            }
    actors = list(policy_actors.values())
    shared_actors = set.intersection(*actors) if actors else set()
    train_events = [event for event in dataset.events if event["split"] == "train"]
    usable_train_events = [event for event in train_events
                           if timestamp(event["occurred_at"]) < timestamp(dataset.train_end)]
    report = {
        "schema_version": 1, "synthetic": True,
        "dataset": {
            "name": dataset.manifest["dataset"], "seed": dataset.manifest["seed"],
            "tenants": len(dataset.tenants), "actors": len(dataset.users), "items": len(dataset.items),
            "requests": len(dataset.requests), "events": len(dataset.events),
            "split_counts": dataset.manifest["split_counts"], "source_sha256": dataset.source_hashes,
        },
        "protocol": {
            "k": k, "candidate_scope": "same upstream logged pool for every policy and request",
            "label_source": "synthetic_rule_v1; observed ground truth only",
            "content_scorer": "BM25 over text/tags; semantic_v1 is not a neural embedding model",
            "ranker_excludes": ["topic IDs", "preferred_topics", "popularity_prior",
                                "upstream scores/features/ranks", "ground truth"],
            "history_source": "training-cohort events, occurred_at < min(request time, train_end)",
            "train_end_exclusive": dataset.train_end, "test_start_inclusive": dataset.test_start,
            "history_weights": HISTORY_WEIGHTS,
            "hyperparameters": "fixed before evaluating; no test-set tuning",
            "inference": "descriptive synthetic validation only; no real-business significance claims",
        },
        "source_audit": {
            "upstream_actors_per_policy": {policy: len(group) for policy, group in policy_actors.items()},
            "upstream_shared_actors_between_policies": len(shared_actors),
            "mean_logged_pool_fraction_of_authorized_catalog": mean(logged_coverage),
            "deterministic_request_fraction": mean(r["epsilon"] == 0 for r in dataset.requests),
            "train_cohort_events_censored_from_features": len(train_events) - len(usable_train_events),
            "latest_allowed_training_event": max(
                (event["occurred_at"] for event in usable_train_events), key=timestamp, default=None),
            "findings": [
                "Upstream policies preselect different top-16 pools, not the complete authorized catalog.",
                "Upstream policy assignment is confounded with actor identity.",
                "Oracle labels and upstream topic-match scores share the same synthetic rules.",
                "Most logged requests have zero propensity outside their deterministic top-k.",
                "Membership and role grants are static in this fixture; dynamic grants need real snapshots.",
            ],
        },
        "offline_paired": offline,
        "paired_ndcg_deltas": {
            split: {name: mean(row[name] for row in audits if row["split"] == split)
                    for name in ("semantic_minus_baseline", "hybrid_minus_baseline")}
            for split in paired
        },
        "upstream_logged_slates_not_a_paired_experiment": {
            "/".join(key): summarize_metrics(rows, k) for key, rows in sorted(upstream.items())},
        "business_logging_cohorts": business_metrics(dataset),
        "off_policy_support": {
            "status": "do_not_estimate_business_lift",
            "unsupported_target_positions": dict(unsupported),
            "target_positions": dict(candidate_positions),
            "reason": "synthetic outcomes, actor confounding, zero propensities and preselected pools",
        },
        "decision": {
            "status": "not_promoted_synthetic_only", "train_two_tower": False,
            "next_step": "collect authorized real pilot feedback; keep content baselines as candidates",
            "real_business_lift": None,
        },
    }
    return report, slate_records


def render_report(report: dict) -> str:
    k = report["protocol"]["k"]
    lines = [
        "# 企业知识推荐：合成种子验证报告", "",
        "本报告验证授权候选、排序与反馈数据链路。所有企业、用户、标签和行为均为合成；"
        "结果不代表真实用户收益，也不构成策略上线证据。", "",
        "## 数据与协议", "",
        f"- 种子：{report['dataset']['seed']}；"
        f"{report['dataset']['tenants']} 个租户、{report['dataset']['actors']} 个用户、"
        f"{report['dataset']['items']} 份文档。",
        f"- {report['dataset']['requests']} 次请求、{report['dataset']['events']} 条原始事件。",
        "- 每个请求的三种策略共享同一原始候选池；只评估有标签的候选，不将未标注文档当负例。",
        "- semantic_v1 使用中文二元词组与英文词项的 BM25，是文本相关性基线，未使用 Embedding。",
        "- 热度只来自训练期且在当时已经发生的反馈；验证与测试冻结训练窗口。",
        "- 三种策略的固定参数在运行前确定，测试集不参与调参。", "",
        "## 同请求、同候选池的测试结果", "",
        f"| 策略 | 请求数 | NDCG@{k} | MRR@{k} | Recall@{k} |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy, metrics in report["offline_paired"].get("test", {}).items():
        lines.append(f"| {policy} | {metrics['requests']} | {metrics[f'ndcg_at_{k}']:.4f} | "
                     f"{metrics[f'mrr_at_{k}']:.4f} | {metrics[f'recall_at_{k}']:.4f} |")
    audit = report["source_audit"]
    lines += [
        "", "## 原始包审计", "",
        f"- 原始两种策略的共同用户数为 {audit['upstream_shared_actors_between_policies']}，"
        "不能把两个用户群的差值解释为策略增益。",
        f"- 原始 16 项候选池平均覆盖授权目录的 "
        f"{audit['mean_logged_pool_fraction_of_authorized_catalog']:.1%}，不是全目录召回评估。",
        "- 原包的 topic_match 与真值标签使用同一套主题规则；新排序器仅读取允许的文本与历史字段。",
        f"- {audit['deterministic_request_fraction']:.1%} 的请求没有探索；"
        "未展示候选常有零倾向概率，不能直接做跨策略 IPS。",
        f"- 已从训练特征排除 {audit['train_cohort_events_censored_from_features']} 条"
        "虽然归属于训练请求、但实际发生在训练截止之后的延迟事件。",
        "", "## 决策", "",
        "完成合成数据与工程验证，保持未晋升状态。暂不训练企业双塔，下一阶段在企业平台"
        "收集真实、授权且版本完整的曝光、打开、引用与行动项反馈。",
        "在线鉴权、MySQL 落库和 React 埋点属于企业平台；本仓库交付离线回放、纯函数排序器"
        "与版本化策略包。", "",
        "完整分组指标、源文件哈希、OPE 支持检查及策略参数见同目录 report.json 与 policy.json。",
        "",
    ]
    return "\n".join(lines)
