import unittest
import numpy as np
from copycat.evaluation.metrics import (
    compute_spearman,
    compute_novelty_gap,
    compute_memorization_diagnostics,
)
from copycat.evaluation.identifiability import HungarianFactorMatcher

class TestEvaluation(unittest.TestCase):
    def test_metrics(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([12.0, 19.0, 35.0, 38.0, 55.0])
        rho = compute_spearman(y_true, y_pred)
        self.assertAlmostEqual(rho, 1.0, places=5)
        
        gap = compute_novelty_gap(0.85, 0.40)
        self.assertAlmostEqual(gap, 0.45, places=5)
        
        diag = compute_memorization_diagnostics(
            predictions=y_pred,
            surface_similarity_to_hits=np.array([0.9, 0.8, 0.7, 0.6, 0.5]),
            true_taste_match=np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
        )
        self.assertIn("memorization_bias", diag)

    def test_hungarian_matching(self):
        np.random.seed(42)
        n_samples = 200
        true_f = np.random.randn(n_samples, 5)
        perm = [3, 0, 4, 1, 2]
        learned_f = true_f[:, perm] + np.random.normal(0, 0.05, (n_samples, 5))
        
        matcher = HungarianFactorMatcher()
        res = matcher.match(true_f, learned_f)
        self.assertGreater(res["mean_recovery"], 0.90)
        self.assertEqual(len(res["matched_pairs"]), 5)

if __name__ == "__main__":
    unittest.main()
