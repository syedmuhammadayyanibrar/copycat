from typing import Dict, Any, Tuple
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, ConstantInputWarning

def compute_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConstantInputWarning)
        res = spearmanr(y_true, y_pred)
    val = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
    return 0.0 if np.isnan(val) else val

def compute_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConstantInputWarning)
        res = pearsonr(x, y)
    val = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
    return 0.0 if np.isnan(val) else val

def compute_novelty_gap(spearman_similar: float, spearman_novel: float) -> float:
    return float(spearman_similar - spearman_novel)

def compute_memorization_diagnostics(
    predictions: np.ndarray,
    surface_similarity_to_hits: np.ndarray,
    true_taste_match: np.ndarray,
) -> Dict[str, float]:
    corr_surface = compute_spearman(predictions, surface_similarity_to_hits)
    corr_taste = compute_spearman(predictions, true_taste_match)
    bias = float(corr_surface - corr_taste)
    return {
        "corr_surface_similarity": corr_surface,
        "corr_true_taste": corr_taste,
        "memorization_bias": bias,
    }
