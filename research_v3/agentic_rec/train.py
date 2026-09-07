"""Train and evaluate the minimal active-tool recommendation loop.

The policy is optimized with *group-relative REINFORCE*: several rollouts start
from the same seeded user context, their final shaped returns are normalized
inside that group, and the better trajectories receive positive policy-gradient
weight.  This deliberately does not claim to be full GRPO because it has no
old-policy importance ratio or clipped surrogate objective.

Examples
--------
Smoke the complete loop with real SASRec checkpoint::

    python research_v3/agentic_rec/train.py --smoke --tag local-smoke

Run the interview experiment (new tag required; artifacts are immutable)::

    python research_v3/agentic_rec/train.py --tag deepeyes-minloop-v1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import set_seed  # noqa: E402
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
from research_v3.agentic_rec.policy import (  # noqa: E402
    ToolPolicy,
    clipped_surrogate_loss,
    group_relative_advantages,
    kl_from_logits,
)


@dataclass
class Trajectory:
    states: list[np.ndarray]
    actions: list[int]
    log_probs: list[float]
    objective_return: float
    user_return: float
    net_return: float
    tool_calls: int
    steps: int


def rollout(
    policy: ToolPolicy,
    env: AgenticRecEnv,
    *,
    device: str,
    sample: bool,
) -> Trajectory:
    state = env.reset()
    states: list[np.ndarray] = []
    actions: list[int] = []
    log_probs: list[float] = []
    objective_return = 0.0
    user_return = 0.0
    net_return = 0.0
    tool_calls = 0

    while True:
        tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = policy(tensor)
        if sample:
            distribution = torch.distributions.Categorical(logits=logits)
            action = int(distribution.sample())
            log_prob = float(
                distribution.log_prob(torch.tensor(action, device=device))
            )
        else:
            distribution = torch.distributions.Categorical(logits=logits)
            action = int(logits.argmax(dim=1).item())
            log_prob = float(distribution.log_prob(torch.tensor(action, device=device)))
        next_state, reward, done, info = env.step(action)
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        objective_return += float(reward)
        user_return += float(info["user_reward"])
        net_return += float(info["net_reward"])
        tool_calls += int(info["tool_used"])
        state = next_state
        if done:
            break

    return Trajectory(
        states=states,
        actions=actions,
        log_probs=log_probs,
        objective_return=objective_return,
        user_return=user_return,
        net_return=net_return,
        tool_calls=tool_calls,
        steps=len(actions),
    )


def train_group_relative_policy(
    *,
    make_env: Callable[[int], AgenticRecEnv],
    seed: int,
    updates: int,
    groups_per_update: int,
    group_size: int,
    lr: float,
    entropy_coef: float,
    algo: str,
    clip_eps: float,
    kl_coef: float,
    epochs: int,
    device: str,
    log_path: Path | None = None,
    verbose: bool = True,
) -> tuple[ToolPolicy, list[dict]]:
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    set_seed(seed)
    policy = ToolPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    reference = None
    if algo == "grpo":
        # GRPO 的参考策略 = 训练起始时的冻结快照（论文里通常是 SFT 模型；
        # 这里没有 SFT 门控，退而用初始策略，KL 梯度因此非零且随训练增长）
        reference = ToolPolicy().to(device)
        reference.load_state_dict(policy.state_dict())
        reference.eval()
    history: list[dict] = []
    started = time.perf_counter()

    for update in range(1, updates + 1):
        weighted_trajectories: list[tuple[Trajectory, float]] = []
        update_trajectories: list[Trajectory] = []

        for group in range(groups_per_update):
            # Every rollout in a group starts from the same user context and
            # random stream.  Only the sampled policy actions differ initially.
            context_seed = seed + update * 100_000 + group
            trajectories = [
                rollout(policy, make_env(context_seed), device=device, sample=True)
                for _ in range(group_size)
            ]
            advantages = group_relative_advantages(
                [trajectory.objective_return for trajectory in trajectories]
            )
            weighted_trajectories.extend(zip(trajectories, advantages.tolist()))
            update_trajectories.extend(trajectories)

        reuse_epochs = epochs if algo == "grpo" else 1
        for _ in range(reuse_epochs):
            losses = []
            entropies = []
            kls = []
            for trajectory, advantage in weighted_trajectories:
                states = torch.from_numpy(np.stack(trajectory.states)).float().to(device)
                actions = torch.tensor(
                    trajectory.actions, dtype=torch.long, device=device
                )
                distribution = torch.distributions.Categorical(
                    logits=policy(states)
                )
                mean_log_probability = distribution.log_prob(actions).mean()
                mean_entropy = distribution.entropy().mean()
                if algo == "grpo":
                    old_log_probs = torch.tensor(
                        trajectory.log_probs, dtype=torch.float32, device=device
                    )
                    advantage_tensor = torch.tensor(
                        float(advantage), dtype=torch.float32, device=device
                    ).expand(actions.numel())
                    with torch.no_grad():
                        reference_logits = reference(states)
                    grpo_loss = clipped_surrogate_loss(
                        distribution.log_prob(actions),
                        old_log_probs,
                        advantage_tensor,
                        clip_eps,
                    )
                    kls.append(
                        kl_from_logits(distribution.logits, reference_logits)
                    )
                    losses.append(grpo_loss)
                else:
                    losses.append(-float(advantage) * mean_log_probability)
                entropies.append(mean_entropy)

            policy_loss = torch.stack(losses).mean()
            entropy = torch.stack(entropies).mean()
            kl = (
                torch.stack(kls).mean()
                if kls
                else torch.tensor(0.0, device=device)
            )
            loss = policy_loss + kl_coef * kl - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optimizer.step()

        total_steps = sum(item.steps for item in update_trajectories)
        total_tools = sum(item.tool_calls for item in update_trajectories)
        row = {
            "update": update,
            "objective_return": round(
                float(np.mean([item.objective_return for item in update_trajectories])), 6
            ),
            "user_return": round(
                float(np.mean([item.user_return for item in update_trajectories])), 6
            ),
            "net_return": round(
                float(np.mean([item.net_return for item in update_trajectories])), 6
            ),
            "tool_rate": round(total_tools / max(1, total_steps), 6),
            "entropy": round(float(entropy.item()), 6),
            "kl": round(float(kl.item()), 6),
            "loss": round(float(loss.item()), 6),
            "elapsed_s": round(time.perf_counter() - started, 3),
            "algo": algo,
        }
        history.append(row)
        if verbose and (
            update == 1 or update == updates
            or update % max(1, updates // 5) == 0
        ):
            print(
                f"  update {update:>3}/{updates} | user={row['user_return']:.3f} "
                f"net={row['net_return']:.3f} tool={row['tool_rate']:.1%} "
                f"entropy={row['entropy']:.3f}",
                flush=True,
            )

    # Sweeps reuse this trainer dozens of times and only want the returned
    # history in memory; only the tagged single runs persist a CSV.
    if log_path is not None:
        with log_path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    return policy, history


def learned_action(policy: ToolPolicy, device: str):
    def act(state: np.ndarray, env: AgenticRecEnv) -> int:
        del env
        with torch.no_grad():
            logits = policy(torch.from_numpy(state).float().unsqueeze(0).to(device))
        return int(logits.argmax(dim=1).item())

    return act


def learned_sample_action(policy: ToolPolicy, device: str, seed: int):
    """Reproducible stochastic deployment for the learned two-action gate."""
    rng = np.random.default_rng(seed)

    def act(state: np.ndarray, env: AgenticRecEnv) -> int:
        del env
        with torch.no_grad():
            logits = policy(torch.from_numpy(state).float().unsqueeze(0).to(device))
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return int(rng.choice(len(probabilities), p=probabilities))

    return act


def evaluate_policy(
    name: str,
    action_fn,
    *,
    make_env: Callable[[int], AgenticRecEnv],
    seed: int,
    sessions: int,
) -> dict:
    session_user_rewards = []
    session_net_rewards = []
    session_objectives = []
    session_lengths = []
    session_diversity = []
    tool_calls = 0
    tool_changed = 0
    beneficial_calls = 0
    successful_calls = 0
    total_steps = 0
    gains = []

    for session in range(sessions):
        env = make_env(seed + session)
        state = env.reset()
        user_reward = 0.0
        net_reward = 0.0
        objective = 0.0
        genres = []
        steps = 0
        while True:
            action = int(action_fn(state, env))
            state, reward, done, info = env.step(action)
            user_reward += float(info["user_reward"])
            net_reward += float(info["net_reward"])
            objective += float(reward)
            genres.append(int(info["genre"]))
            steps += 1
            if info["tool_used"]:
                tool_calls += 1
                gains.append(float(info["expected_gain"]))
                tool_changed += int(info["tool_changed_item"])
                beneficial_calls += int(info["beneficial_tool"])
                successful_calls += int(info["bonus_earned"])
            if done:
                break
        total_steps += steps
        session_user_rewards.append(user_reward)
        session_net_rewards.append(net_reward)
        session_objectives.append(objective)
        session_lengths.append(steps)
        session_diversity.append(len(set(genres)))

    return {
        "policy": name,
        "sessions": sessions,
        "avg_user_reward": round(float(np.mean(session_user_rewards)), 6),
        "avg_net_reward": round(float(np.mean(session_net_rewards)), 6),
        "avg_training_objective": round(float(np.mean(session_objectives)), 6),
        "avg_session_length": round(float(np.mean(session_lengths)), 6),
        "avg_genre_diversity": round(float(np.mean(session_diversity)), 6),
        "tool_rate": round(tool_calls / max(1, total_steps), 6),
        "tool_changed_item_rate": round(tool_changed / max(1, tool_calls), 6),
        "beneficial_tool_rate": round(beneficial_calls / max(1, tool_calls), 6),
        "successful_tool_rate": round(successful_calls / max(1, tool_calls), 6),
        "avg_expected_gain_when_called": round(
            float(np.mean(gains)) if gains else 0.0, 6
        ),
    }


def reserve_artifacts(tag: str, include_ablation: bool) -> dict[str, Path]:
    prefix = f"agentic_rec_{tag}"
    artifacts = {
        "train_log": ROOT / "experiments" / f"{prefix}_train.csv",
        "eval_csv": ROOT / "experiments" / f"{prefix}_eval.csv",
        "report": ROOT / "experiments" / f"{prefix}_report.json",
        "checkpoint": ROOT / "checkpoints" / f"{prefix}.pt",
    }
    if include_ablation:
        artifacts.update(
            {
                "ablation_log": ROOT / "experiments" / f"{prefix}_no_bonus_train.csv",
                "ablation_checkpoint": ROOT
                / "checkpoints"
                / f"{prefix}_no_bonus.pt",
            }
        )
    existing = [str(path) for path in artifacts.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite experiment artifacts; choose a new --tag:\n"
            + "\n".join(existing)
        )
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", default="checkpoints/sasrec_main.pt")
    parser.add_argument("--tag", default="agentic-minloop")
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--groups-per-update", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--eval-sessions", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument(
        "--algo",
        choices=["reinforce", "grpo"],
        default="reinforce",
        help="reinforce = 分组相对 REINFORCE（原版）；grpo = 完整 GRPO "
        "（旧策略比裁剪 + KL 正则，DeepSeekMath/R1 同款）",
    )
    parser.add_argument("--clip", type=float, default=0.2,
                        help="GRPO 重要性采样比裁剪 ε")
    parser.add_argument("--kl-coef", type=float, default=0.04,
                        help="GRPO 对参考策略的 KL 正则系数 β")
    parser.add_argument("--epochs", type=int, default=1,
                        help="GRPO 每份 rollout 复用的梯度轮数 K（>1 时裁剪才真正生效）")
    parser.add_argument("--tool-cost", type=float, default=0.03)
    parser.add_argument("--tool-bonus", type=float, default=0.05)
    parser.add_argument("--min-gain", type=float, default=0.01)
    parser.add_argument("--heuristic-gain", type=float, default=0.02)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.updates = 2
        args.groups_per_update = 2
        args.group_size = 3
        args.eval_sessions = 20

    include_ablation = not args.skip_ablation
    artifacts = reserve_artifacts(args.tag, include_ablation)
    set_seed(args.seed)
    checkpoint_path = ROOT / args.ckpt
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing SASRec checkpoint: {checkpoint_path}")

    print(f"Loading frozen SASRec simulator: {checkpoint_path}", flush=True)
    sasrec, maxlen, _ = load_user_model(checkpoint_path)
    assets = load_rec_assets()

    def make_env(seed: int, bonus: float = args.tool_bonus) -> AgenticRecEnv:
        base = RecSimEnv(sasrec, maxlen, assets, seed=seed)
        return AgenticRecEnv(
            base,
            tool_cost=args.tool_cost,
            tool_bonus=bonus,
            min_gain=args.min_gain,
        )

    print(
        "Training group-relative policy "
        f"(updates={args.updates}, groups={args.groups_per_update}, "
        f"group_size={args.group_size}, device={DEVICE})",
        flush=True,
    )
    policy, train_history = train_group_relative_policy(
        make_env=lambda seed: make_env(seed, args.tool_bonus),
        seed=args.seed,
        updates=args.updates,
        groups_per_update=args.groups_per_update,
        group_size=args.group_size,
        lr=args.lr,
        entropy_coef=args.entropy,
        algo=args.algo,
        clip_eps=args.clip,
        kl_coef=args.kl_coef,
        epochs=args.epochs,
        device=DEVICE,
        log_path=artifacts["train_log"],
    )
    torch.save(
        {"model": policy.state_dict(), "args": vars(args)},
        artifacts["checkpoint"],
    )

    no_bonus_policy = None
    no_bonus_history = None
    if include_ablation:
        print("Training reward ablation (conditional tool bonus = 0)", flush=True)
        no_bonus_policy, no_bonus_history = train_group_relative_policy(
            make_env=lambda seed: make_env(seed, 0.0),
            seed=args.seed,
            updates=args.updates,
            groups_per_update=args.groups_per_update,
            group_size=args.group_size,
            lr=args.lr,
            entropy_coef=args.entropy,
            algo=args.algo,
            clip_eps=args.clip,
            kl_coef=args.kl_coef,
            epochs=args.epochs,
            device=DEVICE,
            log_path=artifacts["ablation_log"],
        )
        torch.save(
            {"model": no_bonus_policy.state_dict(), "args": vars(args)},
            artifacts["ablation_checkpoint"],
        )

    policies = {
        "always_baseline": lambda state, env: BASELINE_ACTION,
        "always_tool": lambda state, env: TOOL_ACTION,
        "heuristic_gain_gate": (
            lambda state, env: TOOL_ACTION
            if env.current_plan is not None
            and env.current_plan.expected_gain >= args.heuristic_gain
            else BASELINE_ACTION
        ),
        "learned_group_relative_argmax": learned_action(policy, DEVICE),
        "learned_group_relative_sample": learned_sample_action(
            policy, DEVICE, args.seed + 7_000_000
        ),
    }
    if no_bonus_policy is not None:
        policies["learned_no_bonus_argmax"] = learned_action(
            no_bonus_policy, DEVICE
        )
        policies["learned_no_bonus_sample"] = learned_sample_action(
            no_bonus_policy, DEVICE, args.seed + 7_000_000
        )

    eval_seed = args.seed + 9_000_000
    print(
        f"Evaluating {len(policies)} policies on {args.eval_sessions} "
        "same-seed simulator sessions",
        flush=True,
    )
    rows = [
        evaluate_policy(
            name,
            action_fn,
            make_env=lambda seed: make_env(seed, args.tool_bonus),
            seed=eval_seed,
            sessions=args.eval_sessions,
        )
        for name, action_fn in policies.items()
    ]
    with artifacts["eval_csv"].open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "experiment": "DeepEyes-inspired active recommendation tool gate",
        "algorithm": (
            f"{'GRPO (group advantage + clipped surrogate + KL, K=' + str(args.epochs) + ')' if args.algo == 'grpo' else 'group-relative REINFORCE'}"
        ),
        "simulator": (
            "MovieLens-1M frozen SASRec RecSimEnv with hand-designed "
            "genre fatigue and churn"
        ),
        "scope_warning": "simulator evidence only; not an online uplift claim",
        "config": vars(args),
        "artifacts": {key: str(path.relative_to(ROOT)) for key, path in artifacts.items()},
        "training_final": train_history[-1],
        "ablation_training_final": (
            no_bonus_history[-1] if no_bonus_history is not None else None
        ),
        "evaluation": rows,
        "decision": {
            "criterion": "highest average net reward while preserving user reward",
            "selected_policy": max(
                rows, key=lambda item: item["avg_net_reward"]
            )["policy"],
            "learned_gate_promoted": False,
            "reason": (
                "the learned argmax gate collapsed to always-tool in the verified "
                "run; use the interpretable gain gate unless new evidence changes it"
            ),
        },
    }
    with artifacts["report"].open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("\nEvaluation (raw user reward is the business-facing simulator metric):")
    for row in sorted(rows, key=lambda item: -item["avg_user_reward"]):
        print(
            f"  {row['policy']:<28} user={row['avg_user_reward']:>7.3f} "
            f"net={row['avg_net_reward']:>7.3f} "
            f"tool={row['tool_rate']:>6.1%} "
            f"len={row['avg_session_length']:>5.2f}",
            flush=True,
        )
    print(f"Report: {artifacts['report'].relative_to(ROOT)}")


if __name__ == "__main__":
    main()
