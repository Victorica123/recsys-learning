"""Step 4: Training loop -- put the three pieces together on CartPole-v1.

The loop is the neural-network analogue of ``train_q_learning`` from Phase 0:

    for each environment step:
        a = epsilon-greedy(online net)
        observe (r, s', done); store the transition
        once the buffer is warm, sample a mini-batch and take one gradient step
        periodically hard-sync the target network
    periodically evaluate the greedy policy

CartPole-v1 gives +1 reward per step and terminates at 500 steps, so the best
possible return is 500. Anything reliably above ~475 is "solved".

Run::

    .venv/Scripts/python.exe -m research_v2.phase1_dqn.train --steps 20000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch

from .dqn_agent import DQNAgent
from .replay_buffer import ReplayBuffer


@dataclass
class DQNConfig:
    env_id: str = "CartPole-v1"
    total_steps: int = 20_000
    buffer_capacity: int = 50_000
    batch_size: int = 64
    warmup_steps: int = 1_000
    gamma: float = 0.99
    lr: float = 1e-3
    target_sync: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    eval_every: int = 2_000
    eval_episodes: int = 20
    seed: int = 42
    device: str = "cpu"


@dataclass
class TrainResult:
    history: list[dict] = field(default_factory=list)
    final_eval: float = 0.0


def linear_epsilon(step: int, cfg: DQNConfig) -> float:
    """Linearly anneal epsilon from start to end over ``epsilon_decay_steps``."""
    fraction = min(step / max(1, cfg.epsilon_decay_steps), 1.0)
    return cfg.epsilon_start + fraction * (cfg.epsilon_end - cfg.epsilon_start)


def evaluate(agent: DQNAgent, cfg: DQNConfig, seed: int) -> float:
    """Average return of the greedy (epsilon=0) policy over several episodes."""
    env = gym.make(cfg.env_id)
    rng = np.random.default_rng(seed)
    returns = []
    for episode in range(cfg.eval_episodes):
        state, _ = env.reset(seed=seed + episode)
        total, done = 0.0, False
        while not done:
            action = agent.select_action(state, epsilon=0.0, rng=rng)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
    env.close()
    return float(np.mean(returns))


def train(cfg: DQNConfig) -> TrainResult:
    """Train a DQN agent and return the evaluation history."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    env = gym.make(cfg.env_id)
    state_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)

    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        gamma=cfg.gamma,
        lr=cfg.lr,
        target_sync=cfg.target_sync,
        device=cfg.device,
    )
    buffer = ReplayBuffer(cfg.buffer_capacity)
    result = TrainResult()

    state, _ = env.reset(seed=cfg.seed)
    episode_return = 0.0
    for step in range(1, cfg.total_steps + 1):
        epsilon = linear_epsilon(step, cfg)
        action = agent.select_action(state, epsilon, rng)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # Only "terminated" (pole fell) is a true terminal for bootstrapping;
        # "truncated" (time limit) should still bootstrap from next_state.
        buffer.push(state, action, reward, next_state, terminated)
        episode_return += reward

        if done:
            state, _ = env.reset()
            episode_return = 0.0
        else:
            state = next_state

        if len(buffer) >= max(cfg.warmup_steps, cfg.batch_size):
            agent.learn(buffer.sample(cfg.batch_size, rng))

        if cfg.eval_every and step % cfg.eval_every == 0:
            avg_return = evaluate(agent, cfg, seed=cfg.seed + step)
            result.history.append(
                {"step": step, "epsilon": round(epsilon, 3), "avg_return": avg_return}
            )
            print(
                f"step={step:6d}  epsilon={epsilon:.3f}  eval_return={avg_return:6.1f}"
            )

    env.close()
    result.final_eval = result.history[-1]["avg_return"] if result.history else 0.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=2_000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = DQNConfig(
        total_steps=args.steps,
        seed=args.seed,
        lr=args.lr,
        eval_every=args.eval_every,
        device=args.device,
    )
    print("Phase 1 - DQN on CartPole-v1")
    result = train(cfg)
    print(f"\nfinal greedy eval return: {result.final_eval:.1f} (max possible 500)")


if __name__ == "__main__":
    main()
