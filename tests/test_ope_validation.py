from __future__ import annotations

import unittest

from feedback import validate_replay_rows
from scripts.validate_ope_estimators import (
    generate_synthetic_exposures,
    oracle_policy_values,
)


class SyntheticExposureTests(unittest.TestCase):
    def test_logged_marginal_propensities_match_epsilon_slate_formula(self):
        rows = generate_synthetic_exposures(
            recommendations=1, pool_size=10, slate_k=2,
            exploration_rate=0.3, actors=1, seed=1)
        by_rank = {row["candidate_rank"]: row for row in rows}
        self.assertAlmostEqual(by_rank[1]["behavior_propensity"], 0.76)
        self.assertAlmostEqual(by_rank[2]["behavior_propensity"], 0.76)
        self.assertAlmostEqual(by_rank[3]["behavior_propensity"], 0.06)
        self.assertEqual(sum(row["impressed"] for row in rows), 2)
        self.assertEqual(validate_replay_rows(rows)["schema_versions"], [2])

    def test_oracle_value_is_independent_of_realized_clicks(self):
        first = generate_synthetic_exposures(
            recommendations=20, pool_size=5, slate_k=2,
            exploration_rate=0.3, actors=2, seed=1)
        second = generate_synthetic_exposures(
            recommendations=20, pool_size=5, slate_k=2,
            exploration_rate=0.3, actors=2, seed=2)
        first_value = oracle_policy_values(
            first, target_k=2, diversity_weight=0.15)
        second_value = oracle_policy_values(
            second, target_k=2, diversity_weight=0.15)
        # Scores vary by recommendation index, not RNG seed; oracle is identical.
        self.assertEqual(first_value, second_value)


if __name__ == "__main__":
    unittest.main()
