import unittest

from scripts.validate_external_survey import (
    bootstrap_difference,
    holm_adjust,
    permutation_p_value,
)


class ExternalSurveyStatisticsTests(unittest.TestCase):
    def test_bootstrap_constant_difference_is_exact(self):
        interval = bootstrap_difference(
            [3.0, 3.0], [1.0, 1.0], samples=100, seed=7)
        self.assertEqual(interval, [2.0, 2.0])

    def test_permutation_detects_identical_samples(self):
        p_value = permutation_p_value(
            [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], samples=100, seed=7)
        self.assertEqual(p_value, 1.0)

    def test_holm_adjustment_is_monotone_in_sorted_order(self):
        adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.5})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertAlmostEqual(adjusted["c"], 0.5)


if __name__ == "__main__":
    unittest.main()
