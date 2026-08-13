"""Command-line entry point for the Phase 0 Q-learning experiment."""

from __future__ import annotations

import argparse

from .q_learning import evaluate_policy, format_metrics, render_policy, train_q_learning


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2_000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random_metrics = evaluate_policy(None, seed=args.seed, random_policy=True)
    q_table, history = train_q_learning(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_end=args.epsilon_end,
        seed=args.seed,
    )
    learned_metrics = evaluate_policy(q_table, seed=args.seed)

    print("Phase 0 - GridWorld Q-learning")
    print(format_metrics("random ", random_metrics))
    for row in history:
        print(
            f"episode={row['episode']:4d}, epsilon={row['epsilon']:.3f}, "
            f"success={row['success_rate']:.1%}, avg_return={row['avg_return']:.3f}"
        )
    print(format_metrics("learned", learned_metrics))
    print("\nLearned policy (^ > v <, X=trap, G=goal):")
    print(render_policy(q_table))


if __name__ == "__main__":
    main()
