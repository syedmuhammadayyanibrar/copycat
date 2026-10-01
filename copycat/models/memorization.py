from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from copycat.models.base import BaseCreatorModel
from copycat.data.schema import OBSERVABLE_CATEGORICAL_COLS, OBSERVABLE_NUMERIC_COLS

class MemorizationBaseline(BaseCreatorModel):
    def __init__(self, top_k: int = 5, hit_quantile: float = 0.65, gamma: float = 0.15):
        self.top_k = top_k
        self.hit_quantile = hit_quantile
        self.gamma = gamma
        self.encoder: Optional[OneHotEncoder] = None
        self.scaler: Optional[StandardScaler] = None
        self.creator_hits: Dict[str, np.ndarray] = {}
        self.creator_hit_perfs: Dict[str, np.ndarray] = {}
        self.creator_baselines: Dict[str, float] = {}
        self.global_hits: Optional[np.ndarray] = None
        self.global_hit_perfs: Optional[np.ndarray] = None
        self.global_baseline: float = 10.0

    def _transform_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        cat_data = df[OBSERVABLE_CATEGORICAL_COLS].values
        num_data = df[OBSERVABLE_NUMERIC_COLS].values
        
        if fit:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            self.scaler = StandardScaler()
            cat_enc = self.encoder.fit_transform(cat_data)
            num_enc = self.scaler.fit_transform(num_data)
        else:
            cat_enc = self.encoder.transform(cat_data)
            num_enc = self.scaler.transform(num_data)
            
        scaled_cat = cat_enc * 2.0
        return np.hstack([scaled_cat, num_enc])

    def fit(self, train_df: pd.DataFrame) -> "MemorizationBaseline":
        X = self._transform_features(train_df, fit=True)
        y = train_df["performance"].values
        log_y = np.log(np.maximum(y, 1.0))
        creators = train_df["creator_id"].values
        
        self.global_baseline = float(np.mean(log_y))
        glob_thresh = np.quantile(y, self.hit_quantile)
        glob_mask = y >= glob_thresh
        self.global_hits = X[glob_mask]
        self.global_hit_perfs = log_y[glob_mask]
        
        for cid in np.unique(creators):
            c_mask = creators == cid
            X_c = X[c_mask]
            y_c = y[c_mask]
            log_y_c = log_y[c_mask]
            
            c_thresh = np.quantile(y_c, self.hit_quantile)
            hit_mask = y_c >= c_thresh
            if np.sum(hit_mask) == 0:
                hit_mask = y_c >= np.median(y_c)
                
            self.creator_hits[cid] = X_c[hit_mask]
            self.creator_hit_perfs[cid] = log_y_c[hit_mask]
            self.creator_baselines[cid] = float(np.mean(log_y_c))
            
        return self

    def compute_similarity_to_hits(self, df: pd.DataFrame) -> np.ndarray:
        X = self._transform_features(df, fit=False)
        creators = df["creator_id"].values
        sims = np.zeros(len(df))
        
        for idx in range(len(df)):
            cid = creators[idx]
            x_vec = X[idx : idx + 1]
            if cid in self.creator_hits and len(self.creator_hits[cid]) > 0:
                hits = self.creator_hits[cid]
            else:
                hits = self.global_hits
                
            dists_sq = np.sum((hits - x_vec) ** 2, axis=1)
            rbf_sims = np.exp(-self.gamma * dists_sq)
            k = min(self.top_k, len(rbf_sims))
            top_k_sims = np.partition(rbf_sims, -k)[-k:]
            sims[idx] = float(np.mean(top_k_sims))
            
        return sims

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        X = self._transform_features(test_df, fit=False)
        creators = test_df["creator_id"].values
        preds = np.zeros(len(test_df))
        
        for idx in range(len(test_df)):
            cid = creators[idx]
            x_vec = X[idx : idx + 1]
            if cid in self.creator_hits and len(self.creator_hits[cid]) > 0:
                hits = self.creator_hits[cid]
                hit_perfs = self.creator_hit_perfs[cid]
                base = self.creator_baselines[cid]
            else:
                hits = self.global_hits
                hit_perfs = self.global_hit_perfs
                base = self.global_baseline
                
            dists_sq = np.sum((hits - x_vec) ** 2, axis=1)
            rbf_sims = np.exp(-self.gamma * dists_sq)
            k = min(self.top_k, len(rbf_sims))
            top_idx = np.argpartition(rbf_sims, -k)[-k:]
            
            top_sims = rbf_sims[top_idx]
            top_perfs = hit_perfs[top_idx]
            
            mean_sim = float(np.mean(top_sims))
            weighted_perf = float(np.average(top_perfs, weights=top_sims + 1e-6))
            
            predicted_log = base + 1.2 * (weighted_perf - base) * mean_sim + 0.5 * (mean_sim - 0.3)
            preds[idx] = float(np.exp(predicted_log))
            
        return preds
