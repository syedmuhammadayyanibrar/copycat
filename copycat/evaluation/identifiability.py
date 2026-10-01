from typing import List, Tuple, Dict, Any
import numpy as np
from scipy.optimize import linear_sum_assignment
from copycat.data.schema import LATENT_FACTOR_NAMES

class HungarianFactorMatcher:
    def __init__(self, factor_names: List[str] = LATENT_FACTOR_NAMES):
        self.factor_names = factor_names

    def match(
        self,
        true_factors: np.ndarray,
        learned_factors: np.ndarray,
    ) -> Dict[str, Any]:
        k_true = true_factors.shape[1]
        k_learned = learned_factors.shape[1]
        
        corr_matrix = np.zeros((k_true, k_learned))
        for i in range(k_true):
            t_col = true_factors[:, i]
            t_centered = t_col - np.mean(t_col)
            t_norm = np.linalg.norm(t_centered) + 1e-9
            
            for j in range(k_learned):
                l_col = learned_factors[:, j]
                l_centered = l_col - np.mean(l_col)
                l_norm = np.linalg.norm(l_centered) + 1e-9
                
                r = np.dot(t_centered, l_centered) / (t_norm * l_norm)
                corr_matrix[i, j] = abs(float(r))
                
        row_ind, col_ind = linear_sum_assignment(-corr_matrix)
        
        matched_pairs = []
        matched_correlations = []
        for r, c in zip(row_ind, col_ind):
            name = self.factor_names[r] if r < len(self.factor_names) else f"factor_{r}"
            corr_val = float(corr_matrix[r, c])
            matched_pairs.append({
                "factor_name": name,
                "true_index": int(r),
                "learned_index": int(c),
                "absolute_correlation": corr_val,
            })
            matched_correlations.append(corr_val)
            
        mean_recovery = float(np.mean(matched_correlations))
        
        return {
            "mean_recovery": mean_recovery,
            "matched_pairs": matched_pairs,
            "correlation_matrix": corr_matrix.tolist(),
        }
