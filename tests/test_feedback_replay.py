from __future__ import annotations

import unittest

from scripts.feedback_replay import select_replay_cohort


class ReplayCohortTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"policy_name": "p1", "model_version": "m1", "x": 1},
            {"policy_name": "p2", "model_version": "m2", "x": 2},
        ]

    def test_unfiltered_mixed_versions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "would mix"):
            select_replay_cohort(
                self.rows, policy_name=None, model_version=None)

    def test_explicit_policy_and_model_select_one_cohort(self):
        rows, cohort = select_replay_cohort(
            self.rows, policy_name="p2", model_version="m2")
        self.assertEqual([row["x"] for row in rows], [2])
        self.assertEqual(
            cohort, {"policy_name": "p2", "model_version": "m2"})

    def test_empty_filter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no replay rows"):
            select_replay_cohort(
                self.rows, policy_name="missing", model_version=None)


if __name__ == "__main__":
    unittest.main()
