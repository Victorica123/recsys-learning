# -*- coding: utf-8 -*-
"""工具成本相变扫描：找出「学出来的门控」开始打败「写死的规则」的临界成本。

【为什么要做这个】
`research_v3/agentic_rec` 的既有结论是「规则门 17.42 > GRPO 16.67 > always_tool
17.35」，即学出来的策略打不过一条 if 语句。但那只是**单个成本点**（tool_cost=0.03）
的结论。成本这么低时，工具几乎总是划算的，最优解退化成「无脑一直调」——
于是「学会何时调工具」这个问题本身就不成立。

本脚本沿 tool_cost 轴扫一遍，把单点结论升级成一条相变曲线，回答三个问题：

  1. 成本多高时，`always_tool` 不再是最优？（理论交叉点：user 奖励差 / 会话步数）
  2. 成本多高时，学出来的门控开始超过 `heuristic_gain_gate`？
  3. 每个成本点上，RL 相对规则的增量方向是什么；达到足够 seed 后才做正式推断。

【与既有实验的关系】
复用 `research_v3.agentic_rec.train` 的训练器与评估器，不复制逻辑。SASRec 只加载
一次，在所有 (cost, algo, seed) 组合间共享，因此 36 组只需一次 8 秒冷启动。

【诚实边界】
仿真器证据，不是线上保证。用户模型是冻结 SASRec + 手写疲劳/流失规则，成本轴本身
也是人为设定的——本脚本量化的是「在这个仿真器里，RL 相对规则的增量如何随成本变化」，
不是「推荐系统里工具调用应该收多少钱」。默认 3 seed 只用于探索相变方向；少于
`MIN_SEEDS_FOR_INFERENCE` 时不输出置信区间，也不允许使用“显著胜出”等正式表述。

【运行】
    .venv/Scripts/python.exe scripts/agentic_cost_sweep.py --tag cost-sweep-v1
    .venv/Scripts/python.exe scripts/agentic_cost_sweep.py --tag smoke --smoke
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
from train_dqn_rec import (  # noqa: E402
    DEVICE,
    RecSimEnv,
    load_rec_assets,
    load_user_model,
)
from research_v3.agentic_rec.env import (  # noqa: E402
    BASELINE_ACTION,
    TOOL_ACTION,
    AgenticRecEnv,
)
from research_v3.agentic_rec.train import (  # noqa: E402
    evaluate_policy,
    learned_action,
    learned_sample_action,
    train_group_relative_policy,
)

# 扫描区间的选取依据（会话最长 20 步，like=+0.9 / dislike=-0.1）：
# always_tool 净奖励 ≈ 18.0 - 20·cost，always_baseline ≈ 2.81（不付成本）。
# 两者理论交叉点 cost ≈ (18.0 - 2.81) / 20 ≈ 0.76，因此扫到 0.8 足以覆盖相变。
DEFAULT_COSTS = (0.0, 0.03, 0.15, 0.30, 0.50, 0.80)
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_ALGOS = ("reinforce", "grpo")
MIN_SEEDS_FOR_INFERENCE = 10

# 参与对比的固定策略；学出来的两种部署方式在运行时追加。
RULE_POLICIES = ("always_baseline", "always_tool", "heuristic_gain_gate")
LEARNED_SAMPLE = "learned_sample"
LEARNED_ARGMAX = "learned_argmax"


def build_env_factory(sasrec, maxlen, assets, *, tool_cost, tool_bonus, min_gain):
    """返回一个 seed -> AgenticRecEnv 的工厂，成本参数已绑定。"""

    def make_env(seed: int) -> AgenticRecEnv:
        base = RecSimEnv(sasrec, maxlen, assets, seed=seed)
        return AgenticRecEnv(
            base, tool_cost=tool_cost, tool_bonus=tool_bonus, min_gain=min_gain)

    return make_env


def run_one_cell(
    *,
    make_env,
    seed: int,
    algo: str,
    args,
) -> tuple[dict[str, dict], dict]:
    """训练一个门控并在同一批 seed 会话上评估全部策略。"""
    set_seed(seed)
    policy, history = train_group_relative_policy(
        make_env=make_env,
        seed=seed,
        updates=args.updates,
        groups_per_update=args.groups_per_update,
        group_size=args.group_size,
        lr=args.lr,
        entropy_coef=args.entropy,
        algo=algo,
        clip_eps=args.clip,
        kl_coef=args.kl_coef,
        # K=1 时 ratio 恒等于 1、裁剪永不触发，GRPO 的梯度与 REINFORCE 逐位
        # 相同（见 notes/14）。所以只有 grpo 分支才用 --epochs。
        epochs=args.epochs if algo == "grpo" else 1,
        device=DEVICE,
        log_path=None,
        verbose=False,
    )

    action_fns = {
        "always_baseline": lambda state, env: BASELINE_ACTION,
        "always_tool": lambda state, env: TOOL_ACTION,
        "heuristic_gain_gate": (
            lambda state, env: TOOL_ACTION
            if env.current_plan is not None
            and env.current_plan.expected_gain >= args.heuristic_gain
            else BASELINE_ACTION
        ),
        LEARNED_ARGMAX: learned_action(policy, DEVICE),
        LEARNED_SAMPLE: learned_sample_action(
            policy, DEVICE, seed + 7_000_000),
    }
    eval_seed = seed + 9_000_000
    return {
        name: evaluate_policy(
            name, fn, make_env=make_env, seed=eval_seed,
            sessions=args.eval_sessions)
        for name, fn in action_fns.items()
    }, history[-1]


def aggregate(
    rows: list[dict],
    bootstrap_samples: int,
    min_seeds_for_inference: int = MIN_SEEDS_FOR_INFERENCE,
) -> dict:
    """聚合多 seed 结果；重复数达门槛后才给 bootstrap 区间。"""
    grouped: dict[tuple[float, str, str], list[float]] = {}
    for row in rows:
        key = (row["tool_cost"], row["algo"], row["policy"])
        grouped.setdefault(key, []).append(float(row["avg_net_reward"]))

    summary = {}
    for (cost, algo, policy), values in sorted(grouped.items()):
        enough_seeds = len(values) >= min_seeds_for_inference
        interval = (
            bootstrap_mean_interval(values, samples=bootstrap_samples)
            if enough_seeds else None)
        summary[f"{cost:g}|{algo}|{policy}"] = {
            "tool_cost": cost,
            "algo": algo,
            "policy": policy,
            "seeds": len(values),
            "mean_net_reward": round(float(np.mean(values)), 6),
            "std_net_reward": round(float(np.std(values)), 6),
            "confidence_interval": (
                None if interval is None
                else [round(bound, 6) for bound in interval]),
            "inference_status": (
                "formal" if enough_seeds
                else "exploratory_insufficient_seeds"),
            "min_seeds_for_inference": min_seeds_for_inference,
        }
    return summary


def paired_delta_vs_rule(
    rows: list[dict],
    bootstrap_samples: int,
    min_seeds_for_inference: int = MIN_SEEDS_FOR_INFERENCE,
) -> list[dict]:
    """每个 (cost, algo) 上，学出来的采样策略相对规则门的配对增量。

    配对单位是 seed：同一个 seed 下两种策略跑的是同一批会话，直接相减能降低
    仿真器噪声。seed 数少于推断门槛时只报告均值方向，不计算/解读置信区间。
    """
    by_key: dict[tuple[float, str, int], dict[str, float]] = {}
    for row in rows:
        key = (row["tool_cost"], row["algo"], row["seed"])
        by_key.setdefault(key, {})[row["policy"]] = float(row["avg_net_reward"])

    pairs: dict[tuple[float, str], list[float]] = {}
    for (cost, algo, _seed), policies in by_key.items():
        if LEARNED_SAMPLE in policies and "heuristic_gain_gate" in policies:
            pairs.setdefault((cost, algo), []).append(
                policies[LEARNED_SAMPLE] - policies["heuristic_gain_gate"])

    output = []
    for (cost, algo), deltas in sorted(pairs.items()):
        mean = float(np.mean(deltas))
        direction = (
            "learned_higher" if mean > 0.0
            else "rule_higher" if mean < 0.0
            else "tie")
        enough_seeds = len(deltas) >= min_seeds_for_inference
        interval = (
            bootstrap_mean_interval(deltas, samples=bootstrap_samples)
            if enough_seeds else None)
        if not enough_seeds:
            verdict = "insufficient_seed_replication"
        elif interval is None:
            verdict = "insufficient_data"
        elif interval[0] > 0.0:
            verdict = "learned_gate_wins"
        elif interval[1] < 0.0:
            verdict = "rule_gate_wins"
        else:
            verdict = "indistinguishable"
        output.append({
            "tool_cost": cost,
            "algo": algo,
            "seeds": len(deltas),
            "mean_delta_net_reward": round(mean, 6),
            "confidence_interval": (
                None if interval is None
                else [round(bound, 6) for bound in interval]),
            "verdict": verdict,
            "direction": direction,
            "inference_status": (
                "formal" if enough_seeds
                else "exploratory_insufficient_seeds"),
            "min_seeds_for_inference": min_seeds_for_inference,
        })
    return output


def find_phase_transitions(summary: dict) -> dict:
    """找出每个 algo 下「最优策略换人」的成本点。"""
    by_algo: dict[str, dict[float, tuple[str, float]]] = {}
    for entry in summary.values():
        algo, cost = entry["algo"], entry["tool_cost"]
        best = by_algo.setdefault(algo, {}).get(cost)
        if best is None or entry["mean_net_reward"] > best[1]:
            by_algo[algo][cost] = (entry["policy"], entry["mean_net_reward"])

    transitions = {}
    for algo, per_cost in by_algo.items():
        ordered = sorted(per_cost.items())
        winners = [
            {"tool_cost": cost, "winner": winner,
             "mean_net_reward": round(reward, 6)}
            for cost, (winner, reward) in ordered
        ]
        switches = [
            {"from_cost": ordered[i - 1][0], "to_cost": ordered[i][0],
             "from_winner": ordered[i - 1][1][0], "to_winner": ordered[i][1][0]}
            for i in range(1, len(ordered))
            if ordered[i][1][0] != ordered[i - 1][1][0]
        ]
        transitions[algo] = {"winner_by_cost": winners, "switches": switches}
    return transitions


def reserve(tag: str) -> dict[str, Path]:
    paths = {
        "csv": ROOT / "experiments" / f"agentic_cost_sweep_{tag}.csv",
        "json": ROOT / "experiments" / f"agentic_cost_sweep_{tag}.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite sweep artifacts; choose a new --tag:\n"
            + "\n".join(existing))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True,
                        help="产物标签；已存在则拒绝覆盖")
    parser.add_argument("--ckpt", default="checkpoints/sasrec_main.pt")
    parser.add_argument("--costs", default=None,
                        help="逗号分隔的 tool_cost 列表")
    parser.add_argument("--seeds", default=None, help="逗号分隔的随机种子")
    parser.add_argument("--algos", default=",".join(DEFAULT_ALGOS))
    parser.add_argument("--updates", type=int, default=15)
    parser.add_argument("--groups-per-update", type=int, default=5)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--eval-sessions", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.04)
    parser.add_argument("--epochs", type=int, default=4,
                        help="GRPO 的 rollout 复用轮数 K；K=1 时退化为 REINFORCE")
    parser.add_argument("--tool-bonus", type=float, default=0.05)
    parser.add_argument("--min-gain", type=float, default=0.01)
    parser.add_argument("--heuristic-gain", type=float, default=0.02)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--min-seeds-for-inference", type=int,
        default=MIN_SEEDS_FOR_INFERENCE,
        help="输出置信区间/胜负推断所需的最少独立 seed（默认 10）")
    parser.add_argument("--smoke", action="store_true",
                        help="最小网格，只验证脚本能跑通")
    args = parser.parse_args()
    if args.min_seeds_for_inference < 2:
        parser.error("--min-seeds-for-inference must be at least 2")

    costs = (
        tuple(float(value) for value in args.costs.split(","))
        if args.costs else DEFAULT_COSTS)
    seeds = (
        tuple(int(value) for value in args.seeds.split(","))
        if args.seeds else DEFAULT_SEEDS)
    algos = tuple(value.strip() for value in args.algos.split(",") if value.strip())
    for algo in algos:
        if algo not in ("reinforce", "grpo"):
            parser.error(f"unknown algo {algo!r}")
    if args.smoke:
        costs, seeds = (0.03, 0.80), (42,)
        algos = ("reinforce",)
        args.updates, args.groups_per_update = 2, 2
        args.group_size, args.eval_sessions = 3, 15

    paths = reserve(args.tag)
    checkpoint = ROOT / args.ckpt
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing SASRec checkpoint: {checkpoint}")

    print(f"Loading frozen SASRec simulator once: {checkpoint}", flush=True)
    sasrec, maxlen, _ = load_user_model(checkpoint)
    assets = load_rec_assets()

    total = len(costs) * len(algos) * len(seeds)
    print(f"Sweeping {total} cells "
          f"= {len(costs)} costs x {len(algos)} algos x {len(seeds)} seeds "
          f"(device={DEVICE})", flush=True)

    rows: list[dict] = []
    started = time.perf_counter()
    cell = 0
    for cost in costs:
        make_env = build_env_factory(
            sasrec, maxlen, assets, tool_cost=cost,
            tool_bonus=args.tool_bonus, min_gain=args.min_gain)
        for algo in algos:
            for seed in seeds:
                cell += 1
                evaluations, final_train = run_one_cell(
                    make_env=make_env, seed=seed, algo=algo, args=args)
                for name, result in evaluations.items():
                    rows.append({
                        "tool_cost": cost,
                        "algo": algo,
                        "seed": seed,
                        "policy": name,
                        "avg_user_reward": result["avg_user_reward"],
                        "avg_net_reward": result["avg_net_reward"],
                        "avg_session_length": result["avg_session_length"],
                        "avg_genre_diversity": result["avg_genre_diversity"],
                        "tool_rate": result["tool_rate"],
                    })
                best = max(evaluations.values(),
                           key=lambda item: item["avg_net_reward"])
                print(
                    f"  [{cell:>2}/{total}] cost={cost:<5g} {algo:<9} "
                    f"seed={seed} | best={best['policy']:<20} "
                    f"net={best['avg_net_reward']:>7.3f} | "
                    f"learned_sample_tool_rate="
                    f"{evaluations[LEARNED_SAMPLE]['tool_rate']:.1%} "
                    f"| {time.perf_counter() - started:.0f}s",
                    flush=True)

    with paths["csv"].open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = aggregate(
        rows, args.bootstrap_samples, args.min_seeds_for_inference)
    deltas = paired_delta_vs_rule(
        rows, args.bootstrap_samples, args.min_seeds_for_inference)
    transitions = find_phase_transitions(summary)
    report = {
        "experiment": "agentic tool-gate cost phase transition sweep",
        "question": (
            "at which tool cost does a learned gate start beating the "
            "hand-written gain-threshold rule?"),
        "scope_warning": (
            "simulator evidence only; frozen SASRec user model with "
            "hand-designed fatigue/churn and a hand-chosen cost axis; "
            "confidence intervals and win/loss inference are withheld below "
            f"{args.min_seeds_for_inference} independent seeds"),
        "config": vars(args) | {
            "costs": list(costs), "seeds": list(seeds), "algos": list(algos)},
        "grid": {"cells": total, "rows": len(rows)},
        "summary_by_cost_algo_policy": summary,
        "paired_delta_learned_sample_vs_rule": deltas,
        "phase_transitions": transitions,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    with paths["json"].open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("\n=== 每个成本点的最优策略 ===")
    for algo, data in transitions.items():
        print(f"[{algo}]")
        for entry in data["winner_by_cost"]:
            print(f"  cost={entry['tool_cost']:<5g} -> {entry['winner']:<22}"
                  f" net={entry['mean_net_reward']:>7.3f}")
        for switch in data["switches"]:
            print(f"  ** 相变: cost {switch['from_cost']:g} -> "
                  f"{switch['to_cost']:g} 时最优策略由 "
                  f"{switch['from_winner']} 变为 {switch['to_winner']}")

    print("\n=== 学出来的门控 vs 规则门（按 seed 配对）===")
    print(f"推断门槛: >= {args.min_seeds_for_inference} seeds；低于门槛只报均值方向")
    for entry in deltas:
        interval = entry["confidence_interval"]
        text = "n/a" if interval is None else f"[{interval[0]:+.3f}, {interval[1]:+.3f}]"
        print(f"  cost={entry['tool_cost']:<5g} {entry['algo']:<9} "
              f"Δnet={entry['mean_delta_net_reward']:>+7.3f} {text:<20} "
              f"{entry['verdict']}")
    print(f"\nReport: {paths['json'].relative_to(ROOT)}")


if __name__ == "__main__":
    main()
