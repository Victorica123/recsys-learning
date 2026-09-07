# -*- coding: utf-8 -*-
"""Train and evaluate budgeted multi-tool recommendation trajectories.

Each recommendation decision may call fatigue, diversity and preference tools
before STOP_AND_SERVE.  Tools have separate costs, a shared session budget and
at most two calls per decision.  Learned masked-GRPO is compared on same-seed
sessions with zero-tool, single-tool, budgeted-rule, random-unmasked and an
oracle one-step ceiling.  At least ten independent training seeds are required
before the paired learned-vs-rule result receives a formal interval/verdict.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import set_seed  # noqa: E402
from feedback import bootstrap_mean_interval  # noqa: E402
from train_dqn_rec import DEVICE, RecSimEnv, load_rec_assets, load_user_model  # noqa: E402
from research_v3.agentic_rec.multitool import (  # noqa: E402
    FATIGUE_TOOL,
    PREFERENCE_TOOL,
    STOP_AND_SERVE,
    MultiToolRecEnv,
    budgeted_rule_action,
    evaluate_multitool_policy,
    learned_multitool_action,
    train_multitool_policy,
)


DEFAULT_SEEDS = tuple(range(42, 52))
MIN_SEEDS_FOR_INFERENCE = 10
LEARNED_SAMPLE = "learned_masked_grpo_sample"
LEARNED_ARGMAX = "learned_masked_grpo_argmax"
RULE_POLICY = "budgeted_rule"


def parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def reserve(tag: str, seeds: tuple[int, ...]) -> tuple[dict[str, Path], dict[int, Path]]:
    paths = {
        "json": ROOT / "experiments" / f"agentic_multitool_{tag}.json",
        "csv": ROOT / "experiments" / f"agentic_multitool_{tag}.csv",
    }
    checkpoints = {
        seed: ROOT / "checkpoints" / f"agentic_multitool_{tag}_seed{seed}.pt"
        for seed in seeds
    }
    existing = [
        str(path) for path in [*paths.values(), *checkpoints.values()]
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "refusing to overwrite multi-tool artifacts; choose a new --tag:\n"
            + "\n".join(existing))
    return paths, checkpoints


def aggregate_policies(
    rows: list[dict],
    *,
    bootstrap_samples: int,
    min_seeds_for_inference: int = MIN_SEEDS_FOR_INFERENCE,
) -> list[dict]:
    output = []
    policies = sorted({str(row["policy"]) for row in rows})
    for policy in policies:
        selected = [row for row in rows if row["policy"] == policy]
        net = [float(row["avg_net_reward"]) for row in selected]
        enough = len(net) >= min_seeds_for_inference
        interval = (
            bootstrap_mean_interval(net, samples=bootstrap_samples)
            if enough else None)
        output.append({
            "policy": policy,
            "seeds": len(net),
            "mean_user_reward": round(float(np.mean([
                row["avg_user_reward"] for row in selected])), 6),
            "mean_net_reward": round(float(np.mean(net)), 6),
            "std_net_reward": round(float(np.std(net)), 6),
            "mean_session_length": round(float(np.mean([
                row["avg_session_length"] for row in selected])), 6),
            "mean_genre_diversity": round(float(np.mean([
                row["avg_genre_diversity"] for row in selected])), 6),
            "mean_budget_spent": round(float(np.mean([
                row["avg_budget_spent"] for row in selected])), 6),
            "mean_tool_calls_per_decision": round(float(np.mean([
                row["tool_calls_per_decision"] for row in selected])), 6),
            "mean_multi_tool_decision_rate": round(float(np.mean([
                row["multi_tool_decision_rate"] for row in selected])), 6),
            "mean_invalid_action_rate": round(float(np.mean([
                row["invalid_action_rate"] for row in selected])), 6),
            "net_reward_confidence_interval": (
                None if interval is None
                else [round(float(value), 6) for value in interval]),
            "inference_status": (
                "formal" if enough else "exploratory_insufficient_seeds"),
            "min_seeds_for_inference": min_seeds_for_inference,
        })
    return output


def paired_learned_vs_rule(
    rows: list[dict],
    *,
    bootstrap_samples: int,
    min_seeds_for_inference: int = MIN_SEEDS_FOR_INFERENCE,
) -> dict:
    by_seed: dict[int, dict[str, float]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), {})[str(row["policy"])] = float(
            row["avg_net_reward"])
    deltas = [
        policies[LEARNED_SAMPLE] - policies[RULE_POLICY]
        for _, policies in sorted(by_seed.items())
        if LEARNED_SAMPLE in policies and RULE_POLICY in policies
    ]
    enough = len(deltas) >= min_seeds_for_inference
    interval = (
        bootstrap_mean_interval(deltas, samples=bootstrap_samples)
        if enough else None)
    if not enough:
        verdict = "insufficient_seed_replication"
    elif interval[0] > 0.0:
        verdict = "learned_policy_wins"
    elif interval[1] < 0.0:
        verdict = "budgeted_rule_wins"
    else:
        verdict = "indistinguishable"
    return {
        "comparison": f"{LEARNED_SAMPLE} - {RULE_POLICY}",
        "paired_unit": "training seed; policies share evaluation session seeds",
        "seeds": len(deltas),
        "deltas": [round(float(value), 6) for value in deltas],
        "mean_delta_net_reward": (
            round(float(np.mean(deltas)), 6) if deltas else None),
        "confidence_interval": (
            None if interval is None
            else [round(float(value), 6) for value in interval]),
        "verdict": verdict,
        "inference_status": (
            "formal" if enough else "exploratory_insufficient_seeds"),
        "min_seeds_for_inference": min_seeds_for_inference,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--ckpt", default="checkpoints/sasrec_main.pt")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument("--groups-per-update", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--eval-sessions", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.04)
    parser.add_argument("--session-budget", type=float, default=0.60)
    parser.add_argument("--fatigue-cost", type=float, default=0.05)
    parser.add_argument("--diversity-cost", type=float, default=0.03)
    parser.add_argument("--preference-cost", type=float, default=0.12)
    parser.add_argument("--max-tools-per-decision", type=int, default=2)
    parser.add_argument("--invalid-action-penalty", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--min-seeds-for-inference", type=int,
                        default=MIN_SEEDS_FOR_INFERENCE)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    seeds = parse_ints(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("seeds must be a non-empty unique list")
    if args.min_seeds_for_inference < 2:
        parser.error("min-seeds-for-inference must be at least 2")
    if args.smoke:
        seeds = (seeds[0],)
        args.updates = 2
        args.groups_per_update = 2
        args.group_size = 3
        args.epochs = 2
        args.eval_sessions = 20
        args.bootstrap_samples = 100
    paths, checkpoints = reserve(args.tag, seeds)
    checkpoint = ROOT / args.ckpt
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing SASRec checkpoint: {checkpoint}")

    print(f"Loading frozen simulator once: {checkpoint}", flush=True)
    sasrec, maxlen, _ = load_user_model(checkpoint)
    assets = load_rec_assets()

    def make_env(seed: int) -> MultiToolRecEnv:
        return MultiToolRecEnv(
            RecSimEnv(sasrec, maxlen, assets, seed=seed),
            session_budget=args.session_budget,
            fatigue_cost=args.fatigue_cost,
            diversity_cost=args.diversity_cost,
            preference_cost=args.preference_cost,
            max_tools_per_decision=args.max_tools_per_decision,
            invalid_action_penalty=args.invalid_action_penalty,
        )

    rows: list[dict] = []
    training: list[dict] = []
    started = time.perf_counter()
    for index, seed in enumerate(seeds, start=1):
        print(f"[seed {index}/{len(seeds)}] training seed={seed}", flush=True)
        set_seed(seed)
        policy, history = train_multitool_policy(
            make_env=make_env, seed=seed, updates=args.updates,
            groups_per_update=args.groups_per_update,
            group_size=args.group_size, lr=args.lr,
            entropy_coef=args.entropy, clip_eps=args.clip,
            kl_coef=args.kl_coef, epochs=args.epochs,
            device=DEVICE, verbose=True)
        torch.save({
            "model": policy.state_dict(), "seed": seed,
            "config": vars(args), "training_final": history[-1],
        }, checkpoints[seed])

        random_rng = np.random.default_rng(seed + 8_000_000)
        policies = {
            "stop_and_serve": (
                lambda state, mask, env: STOP_AND_SERVE),
            "fatigue_then_serve": (
                lambda state, mask, env: FATIGUE_TOOL
                if mask[FATIGUE_TOOL] and FATIGUE_TOOL not in env.called_tools
                else STOP_AND_SERVE),
            "preference_then_serve": (
                lambda state, mask, env: PREFERENCE_TOOL
                if mask[PREFERENCE_TOOL] and PREFERENCE_TOOL not in env.called_tools
                else STOP_AND_SERVE),
            RULE_POLICY: budgeted_rule_action,
            "oracle_one_step": (
                lambda state, mask, env: env.oracle_action()),
            "random_unmasked": (
                lambda state, mask, env: int(random_rng.integers(4))),
            LEARNED_ARGMAX: learned_multitool_action(
                policy, DEVICE, sample=False, seed=seed + 7_000_000),
            LEARNED_SAMPLE: learned_multitool_action(
                policy, DEVICE, sample=True, seed=seed + 7_000_000),
        }
        eval_seed = seed + 9_000_000
        for name, action_fn in policies.items():
            result = evaluate_multitool_policy(
                name, action_fn, make_env=make_env,
                seed=eval_seed, sessions=args.eval_sessions)
            rows.append({"seed": seed, **result})
        training.append({
            "seed": seed,
            "checkpoint": str(checkpoints[seed].relative_to(ROOT)),
            "history": history,
        })
        learned_row = next(
            row for row in rows
            if row["seed"] == seed and row["policy"] == LEARNED_SAMPLE)
        rule_row = next(
            row for row in rows
            if row["seed"] == seed and row["policy"] == RULE_POLICY)
        print(
            f"  eval learned={learned_row['avg_net_reward']:.3f} "
            f"rule={rule_row['avg_net_reward']:.3f} "
            f"delta={learned_row['avg_net_reward'] - rule_row['avg_net_reward']:+.3f}",
            flush=True)

    summary = aggregate_policies(
        rows, bootstrap_samples=args.bootstrap_samples,
        min_seeds_for_inference=args.min_seeds_for_inference)
    paired = paired_learned_vs_rule(
        rows, bootstrap_samples=args.bootstrap_samples,
        min_seeds_for_inference=args.min_seeds_for_inference)
    deployable = [
        row for row in summary
        if row["policy"] not in ("oracle_one_step", "random_unmasked")]
    best_deployable = max(deployable, key=lambda row: row["mean_net_reward"])
    learned_promoted = (
        paired["verdict"] == "learned_policy_wins"
        and best_deployable["policy"] in (LEARNED_SAMPLE, LEARNED_ARGMAX))
    report = {
        "experiment": "budgeted multi-tool Agentic recommendation trajectories",
        "scope_warning": (
            "frozen SASRec + hand-designed hidden preference/fatigue simulator; "
            "not an LLM-agent reproduction or online uplift claim"),
        "protocol": {
            "trajectory": (
                "zero/one/two tool calls followed by STOP_AND_SERVE for each "
                "recommendation; user session ends at 20 recommendations or 3 dislikes"),
            "tools": {
                "fatigue_scan": args.fatigue_cost,
                "diversity_search": args.diversity_cost,
                "preference_probe": args.preference_cost,
            },
            "session_budget": args.session_budget,
            "invalid_action": (
                "penalized, recorded, and next action forced to STOP_AND_SERVE"),
            "training": "masked trajectory-level GRPO with same-context groups",
            "evaluation": "all policies use identical session seeds within each training seed",
        },
        "config": vars(args) | {"resolved_seeds": list(seeds), "device": DEVICE},
        "rows": rows,
        "summary": summary,
        "paired_learned_vs_rule": paired,
        "training": training,
        "decision": {
            "best_deployable_mean_net_reward": best_deployable["policy"],
            "learned_policy_promoted": learned_promoted,
            "criterion": (
                "learned sample must beat budgeted rule with paired formal interval "
                "above zero and be the best deployable mean-net policy"),
            "reason": (
                "promotion criterion satisfied" if learned_promoted
                else "promotion criterion not satisfied; keep the best interpretable policy"),
        },
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    with paths["json"].open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    csv_rows = []
    for row in rows:
        flat = dict(row)
        flat["tool_mix"] = json.dumps(flat["tool_mix"], sort_keys=True)
        flat["selected_source_share"] = json.dumps(
            flat["selected_source_share"], sort_keys=True)
        csv_rows.append(flat)
    with paths["csv"].open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n=== mean net reward across training seeds ===")
    for row in sorted(summary, key=lambda item: -item["mean_net_reward"]):
        print(
            f"  {row['policy']:<28} net={row['mean_net_reward']:.3f} "
            f"user={row['mean_user_reward']:.3f} "
            f"tools/decision={row['mean_tool_calls_per_decision']:.2f}")
    print(json.dumps(paired, ensure_ascii=False, indent=2))
    print(f"Report: {paths['json'].relative_to(ROOT)}")


if __name__ == "__main__":
    main()
