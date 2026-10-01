from typing import Dict, Optional
import numpy as np
import pandas as pd
from copycat.models.base import BaseCreatorModel
from copycat.data.schema import CreatorProfile, LATENT_FACTOR_NAMES

class OracleModel(BaseCreatorModel):
    def __init__(
        self,
        creators: Optional[Dict[str, CreatorProfile]] = None,
        beta_taste: float = 1.8,
        beta_platform: float = 0.8,
    ):
        self.creators = creators or {}
        self.beta_taste = beta_taste
        self.beta_platform = beta_platform

    def set_creators(self, creators: Dict[str, CreatorProfile]) -> None:
        self.creators = creators

    def fit(self, train_df: pd.DataFrame) -> "OracleModel":
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(test_df))
        creators = test_df["creator_id"].values
        
        latent_cols = [f"latent_{name}" for name in LATENT_FACTOR_NAMES]
        has_latents = all(col in test_df.columns for col in latent_cols)
        
        for idx in range(len(test_df)):
            cid = creators[idx]
            creator = self.creators.get(cid)
            
            if creator is not None and has_latents:
                v_vec = test_df.iloc[idx][latent_cols].values.astype(float)
                v_norm = v_vec / (np.linalg.norm(v_vec) + 1e-9)
                taste_match = float(np.dot(creator.taste_vector, v_norm))
                baseline = creator.baseline_scale
                
                if "platform_trend" in test_df.columns:
                    trend = float(test_df.iloc[idx]["platform_trend"])
                else:
                    trend = 0.0
                    
                log_pred = baseline + self.beta_taste * taste_match + self.beta_platform * trend
                preds[idx] = float(np.exp(log_pred))
            elif "performance" in test_df.columns:
                preds[idx] = float(test_df.iloc[idx]["performance"])
            else:
                preds[idx] = 10000.0
                
        return preds
