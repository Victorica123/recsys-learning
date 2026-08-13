"""Tabular Q-learning with explicit, testable update functions."""

from __future__ import annotations

from typing import Any

import numpy as np

from .gridworld import GridWorld


def q_learning_update(
    q_table: np.ndarray,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    done: bool,
    alpha: float,
    gamma: float,
) -> float:
    """Apply one Bellman update and return the temporal-difference error."""
    next_value = 0.0 if done else float(q_table[next_state].max())
    target = reward + gamma * next_value
    td_error = target - q_table[state, action]
    q_table[state, action] += alpha * td_error
    return float(td_error)


def epsilon_greedy(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    best = np.flatnonzero(np.isclose(q_values, q_values.max()))
    return int(rng.choice(best))


def evaluate_policy(
    q_table: np.ndarray | None,
    episodes: int = 300,
    seed: int = 123,
    max_steps: int = 50,
    random_policy: bool = False,
) -> dict[str, float]:
    env = GridWorld()
    rng = np.random.default_rng(seed)
    successes, returns, lengths = 0, [], []
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        for step in range(1, max_steps + 1):
            if random_policy:
                action = int(rng.integers(env.n_actions))
            else:
                if q_table is None:
                    raise ValueError("q_table is required for a greedy policy")
                action = epsilon_greedy(q_table[state], 0.0, rng)
            state, reward, done = env.step(action)
            total_reward += reward
            if done:
                successes += int(state == env.goal_state)
                break
        returns.append(total_reward)
        lengths.append(step)
    return {
        "success_rate": successes / episodes,
        "avg_return": float(np.mean(returns)),
        "avg_steps": float(np.mean(lengths)),
    }


def train_q_learning(
    episodes: int = 2_000,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    seed: int = 42,
    max_steps: int = 50,
    eval_every: int = 200,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    env = GridWorld()
    rng = np.random.default_rng(seed)
    q_table = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    history: list[dict[str, Any]] = []
    decay_episodes = max(1, int(episodes * 0.8))

    for episode in range(1, episodes + 1):
        state = env.reset()
        fraction = min(episode / decay_episodes, 1.0)
        epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)
        for _ in range(max_steps):
            action = epsilon_greedy(q_table[state], epsilon, rng)
            next_state, reward, done = env.step(action)
            q_learning_update(
                q_table, state, action, reward, next_state, done, alpha, gamma
            )
            state = next_state
            if done:
                break

        if eval_every and (episode % eval_every == 0 or episode == episodes):
            metrics = evaluate_policy(q_table, seed=seed + episode)
            history.append({"episode": episode, "epsilon": epsilon, **metrics})
    return q_table, history


def render_policy(q_table: np.ndarray) -> str:
    env = GridWorld()
    arrows = ("^", ">", "v", "<")
    cells = []
    for state in range(env.n_states):
        if state == env.goal_state:
            cells.append("G")
        elif state in env.trap_states:
            cells.append("X")
        else:
            cells.append(arrows[int(np.argmax(q_table[state]))])
    size = env.config.size
    return "\n".join(" ".join(cells[row * size : (row + 1) * size]) for row in range(size))


def format_metrics(name: str, metrics: dict[str, float]) -> str:
    return (
        f"{name}: success={metrics['success_rate']:.1%}, "
        f"avg_return={metrics['avg_return']:.3f}, "
        f"avg_steps={metrics['avg_steps']:.1f}"
    )
