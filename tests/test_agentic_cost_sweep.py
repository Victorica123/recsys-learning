# -*- coding: utf-8 -*-
"""Tests for the agentic tool-cost sweep aggregation.

The sweep's exploratory phase direction and its inference guard are entirely
produced by these pure functions, so they are pinned independently of the
(slow, GPU-bound) training loop.
"""
import unittest

from scripts.agentic_cost_sweep import (
    aggregate,
    find_phase_transitions,
    paired_delta_vs_rule,
)


def make_rows(spec):
    """spec: {(cost, algo, seed): {policy: net_reward}} -> flat sweep rows."""
    return [
        {"tool_cost": cost, "algo": algo, "seed": seed,
         "policy": policy, "avg_net_reward": reward,
         "avg_user_reward": reward, "avg_session_length": 20.0,
         "avg_genre_diversity": 3.0, "tool_rate": 0.5}
        for (cost, algo, seed), policies in spec.items()
        for policy, reward in policies.items()
    ]


class AggregateTests(unittest.TestCase):
    def test_averages_across_seeds_and_reports_interval(self):
        rows = make_rows({
            (0.5, "grpo", 42): {"learned_sample": 10.0},
            (0.5, "grpo", 43): {"learned_sample": 12.0},
            (0.5, "grpo", 44): {"learned_sample": 14.0},
        })
        entry = aggregate(
            rows, bootstrap_samples=200,
            min_seeds_for_inference=3)["0.5|grpo|learned_sample"]
        self.assertEqual(entry["seeds"], 3)
        self.assertAlmostEqual(entry["mean_net_reward"], 12.0)
        lower, upper = entry["confidence_interval"]
        self.assertLessEqual(lower, 12.0)
        self.assertGreaterEqual(upper, 12.0)

    def test_identical_seeds_give_a_degenerate_interval(self):
        # 所有 seed 结果相同 -> 区间退化成一个点，而不是假装有不确定性。
        rows = make_rows({
            (0.0, "grpo", seed): {"always_tool": 17.0} for seed in (1, 2, 3)})
        entry = aggregate(
            rows, bootstrap_samples=200,
            min_seeds_for_inference=3)["0|grpo|always_tool"]
        self.assertEqual(entry["std_net_reward"], 0.0)
        self.assertEqual(entry["confidence_interval"], [17.0, 17.0])

    def test_default_three_seed_result_is_exploratory_without_interval(self):
        rows = make_rows({
            (0.8, "grpo", seed): {"learned_sample": 5.0 + seed / 100.0}
            for seed in (42, 43, 44)})
        entry = aggregate(rows, bootstrap_samples=200)["0.8|grpo|learned_sample"]
        self.assertIsNone(entry["confidence_interval"])
        self.assertEqual(
            entry["inference_status"], "exploratory_insufficient_seeds")
        self.assertEqual(entry["min_seeds_for_inference"], 10)

    def test_costs_and_algos_are_kept_separate(self):
        rows = make_rows({
            (0.1, "grpo", 1): {"learned_sample": 5.0},
            (0.1, "reinforce", 1): {"learned_sample": 9.0},
            (0.2, "grpo", 1): {"learned_sample": 1.0},
        })
        summary = aggregate(rows, bootstrap_samples=50)
        self.assertEqual(len(summary), 3)
        self.assertAlmostEqual(
            summary["0.1|reinforce|learned_sample"]["mean_net_reward"], 9.0)
        self.assertAlmostEqual(
            summary["0.2|grpo|learned_sample"]["mean_net_reward"], 1.0)


class PairedDeltaTests(unittest.TestCase):
    def test_rule_wins_when_every_seed_favours_the_rule(self):
        rows = make_rows({
            (0.03, "grpo", seed): {
                "learned_sample": 14.0, "heuristic_gain_gate": 17.0}
            for seed in (42, 43, 44)})
        entry = paired_delta_vs_rule(
            rows, bootstrap_samples=200, min_seeds_for_inference=3)[0]
        self.assertAlmostEqual(entry["mean_delta_net_reward"], -3.0)
        self.assertEqual(entry["verdict"], "rule_gate_wins")

    def test_learned_wins_when_every_seed_favours_the_policy(self):
        rows = make_rows({
            (0.8, "grpo", seed): {
                "learned_sample": 5.9, "heuristic_gain_gate": 4.0}
            for seed in (42, 43, 44)})
        entry = paired_delta_vs_rule(
            rows, bootstrap_samples=200, min_seeds_for_inference=3)[0]
        self.assertGreater(entry["mean_delta_net_reward"], 0)
        self.assertEqual(entry["verdict"], "learned_gate_wins")

    def test_straddling_zero_is_reported_as_indistinguishable(self):
        """跨 0 的区间必须说"分不出"，不能挑均值的符号当结论。"""
        rows = make_rows({
            (0.5, "grpo", 42): {
                "learned_sample": 9.0, "heuristic_gain_gate": 9.3},
            (0.5, "grpo", 43): {
                "learned_sample": 9.6, "heuristic_gain_gate": 9.1},
            (0.5, "grpo", 44): {
                "learned_sample": 8.8, "heuristic_gain_gate": 9.2},
        })
        entry = paired_delta_vs_rule(
            rows, bootstrap_samples=500, min_seeds_for_inference=3)[0]
        self.assertEqual(entry["verdict"], "indistinguishable")

    def test_three_seed_direction_cannot_be_called_a_win_by_default(self):
        rows = make_rows({
            (0.8, "grpo", seed): {
                "learned_sample": 5.9, "heuristic_gain_gate": 4.0}
            for seed in (42, 43, 44)})
        entry = paired_delta_vs_rule(rows, bootstrap_samples=200)[0]
        self.assertEqual(entry["direction"], "learned_higher")
        self.assertIsNone(entry["confidence_interval"])
        self.assertEqual(entry["verdict"], "insufficient_seed_replication")
        self.assertEqual(
            entry["inference_status"], "exploratory_insufficient_seeds")

    def test_delta_is_paired_within_seed(self):
        # 若按各自均值相减而非配对，两组均值相同会得到 0；配对后应为 -1。
        rows = make_rows({
            (0.1, "grpo", 1): {
                "learned_sample": 1.0, "heuristic_gain_gate": 2.0},
            (0.1, "grpo", 2): {
                "learned_sample": 9.0, "heuristic_gain_gate": 10.0},
        })
        entry = paired_delta_vs_rule(rows, bootstrap_samples=100)[0]
        self.assertAlmostEqual(entry["mean_delta_net_reward"], -1.0)


class PhaseTransitionTests(unittest.TestCase):
    def test_detects_the_cost_where_the_winner_changes(self):
        rows = make_rows({
            (0.03, "grpo", 1): {
                "heuristic_gain_gate": 17.4, "learned_sample": 14.7},
            (0.80, "grpo", 1): {
                "heuristic_gain_gate": 4.0, "learned_sample": 5.9},
        })
        transitions = find_phase_transitions(aggregate(rows, 100))["grpo"]
        self.assertEqual(
            [entry["winner"] for entry in transitions["winner_by_cost"]],
            ["heuristic_gain_gate", "learned_sample"])
        self.assertEqual(len(transitions["switches"]), 1)
        switch = transitions["switches"][0]
        self.assertEqual(switch["from_cost"], 0.03)
        self.assertEqual(switch["to_cost"], 0.80)
        self.assertEqual(switch["to_winner"], "learned_sample")

    def test_no_switch_reported_when_one_policy_dominates_everywhere(self):
        rows = make_rows({
            (cost, "grpo", 1): {
                "heuristic_gain_gate": 10.0, "learned_sample": 1.0}
            for cost in (0.0, 0.1, 0.2)})
        transitions = find_phase_transitions(aggregate(rows, 100))["grpo"]
        self.assertEqual(transitions["switches"], [])
        self.assertEqual(len(transitions["winner_by_cost"]), 3)

    def test_algos_are_analysed_independently(self):
        rows = make_rows({
            (0.5, "grpo", 1): {"a": 2.0, "b": 1.0},
            (0.5, "reinforce", 1): {"a": 1.0, "b": 2.0},
        })
        transitions = find_phase_transitions(aggregate(rows, 50))
        self.assertEqual(transitions["grpo"]["winner_by_cost"][0]["winner"], "a")
        self.assertEqual(
            transitions["reinforce"]["winner_by_cost"][0]["winner"], "b")


if __name__ == "__main__":
    unittest.main()
