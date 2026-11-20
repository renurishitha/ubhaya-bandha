"""Unit tests for BSPAI mathematical formulas and RAG model validation."""

import math
import unittest

from bspai.data import compute_candidate_pair_count
from bspai.features.activity import compute_double_positive_ratio, compute_single_positive_ratio
from bspai.features.corr import compute_pearson_correlation
from bspai.features.embedding import compute_euclidean_distance
from bspai.features.safety import compute_safety_score
from bspai.rag import resolve_and_validate_gemini_model


class TestBspaiFormulas(unittest.TestCase):
    def test_safety_score(self) -> None:
        self.assertAlmostEqual(compute_safety_score(2.0, 6.0), 1.5, places=6)
        self.assertEqual(compute_safety_score(0.0, 0.0), 0.0)
        self.assertLess(compute_safety_score(-1.0, 2.0), 0.0)

    def test_activity_ratios(self) -> None:
        e1 = [0.2, 0.8, 1.5, 0.5, 1.1]
        e2 = [0.9, 0.7, 0.1, 1.2, 0.6]
        self.assertAlmostEqual(compute_single_positive_ratio(e1, 0.5), 0.6, places=6)
        self.assertAlmostEqual(compute_double_positive_ratio(e1, e2, 0.5), 0.4, places=6)

    def test_euclidean_distance(self) -> None:
        self.assertAlmostEqual(compute_euclidean_distance([1.0, 2.0, 3.0], [4.0, 6.0, 3.0]), 5.0, places=6)

    def test_pearson_correlation(self) -> None:
        self.assertAlmostEqual(compute_pearson_correlation([1, 2, 3, 4], [2, 4, 6, 8]), 1.0, places=6)
        self.assertAlmostEqual(compute_pearson_correlation([1, 2, 3, 4], [8, 6, 4, 2]), -1.0, places=6)

    def test_candidate_pair_count(self) -> None:
        self.assertEqual(compute_candidate_pair_count(223), 24753)
        self.assertEqual(compute_candidate_pair_count(327), 53301)

    def test_gemini_model_validation(self) -> None:
        self.assertEqual(resolve_and_validate_gemini_model("gemini-4-hydrogen"), "gemini-4-hydrogen")
        with self.assertRaises(ValueError):
            resolve_and_validate_gemini_model("gemini-1.5-pro")
        with self.assertRaises(ValueError):
            resolve_and_validate_gemini_model("invalid-model")


if __name__ == "__main__":
    unittest.main()
