import unittest

from scripts.agentic_multitool_experiment import (
    LEARNED_SAMPLE,
    RULE_POLICY,
    aggregate_policies,
    paired_learned_vs_rule,
)


def row(seed, policy, net):
    return {
        "seed": seed,
        "policy": policy,
        "avg_user_reward": net,
        "avg_net_reward": net,
        "avg_session_length": 10.0,
        "avg_genre_diversity": 4.0,
        "avg_budget_spent": 0.2,
        "tool_calls_per_decision": 0.5,
        "multi_tool_decision_rate": 0.1,
        "invalid_action_rate": 0.0,
    }


class InferenceGuardrailTests(unittest.TestCase):
    def test_three_seeds_are_exploratory(self):
        rows = [row(seed, RULE_POLICY, 1.0) for seed in range(3)]
        summary = aggregate_policies(rows, bootstrap_samples=50)
        self.assertEqual(summary[0]["inference_status"],
                         "exploratory_insufficient_seeds")
        self.assertIsNone(summary[0]["net_reward_confidence_interval"])

    def test_ten_paired_seeds_can_formally_select_learned_policy(self):
        rows = []
        for seed in range(10):
            rows.extend([
                row(seed, RULE_POLICY, 1.0),
                row(seed, LEARNED_SAMPLE, 1.5),
            ])
        result = paired_learned_vs_rule(rows, bootstrap_samples=100)
        self.assertEqual(result["inference_status"], "formal")
        self.assertEqual(result["verdict"], "learned_policy_wins")
        self.assertEqual(result["confidence_interval"], [0.5, 0.5])

    def test_paired_delta_uses_matching_seed(self):
        rows = [
            row(1, RULE_POLICY, 10.0),
            row(1, LEARNED_SAMPLE, 11.0),
            row(2, RULE_POLICY, -3.0),
            row(2, LEARNED_SAMPLE, -1.0),
        ]
        result = paired_learned_vs_rule(
            rows, bootstrap_samples=20, min_seeds_for_inference=2)
        self.assertEqual(result["deltas"], [1.0, 2.0])
        self.assertAlmostEqual(result["mean_delta_net_reward"], 1.5)


if __name__ == "__main__":
    unittest.main()
