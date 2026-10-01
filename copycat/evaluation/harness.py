from typing import Dict, Any, List
import numpy as np
import pandas as pd
from copycat.models.memorization import MemorizationBaseline
from copycat.models.taste import LatentTasteModel
from copycat.models.oracle import OracleModel
from copycat.evaluation.metrics import (
    compute_spearman,
    compute_novelty_gap,
    compute_memorization_diagnostics,
)
from copycat.evaluation.identifiability import HungarianFactorMatcher
from copycat.data.schema import CreatorProfile, LATENT_FACTOR_NAMES

class EvaluationHarness:
    def __init__(self, creators: Dict[str, CreatorProfile]):
        self.creators = creators
        self.matcher = HungarianFactorMatcher(LATENT_FACTOR_NAMES)

    def run_evaluation(
        self,
        train_df: pd.DataFrame,
        similar_df: pd.DataFrame,
        novel_df: pd.DataFrame,
        is_held_out: bool = False,
        pretrained_taste_model: LatentTasteModel = None,
    ) -> Dict[str, Any]:
        memo_model = MemorizationBaseline().fit(train_df)
        
        if is_held_out and pretrained_taste_model is not None:
            taste_model = pretrained_taste_model
            for cid in np.unique(train_df["creator_id"].values):
                c_train = train_df[train_df["creator_id"] == cid]
                taste_model.fit_creator_adaptation(cid, c_train)
        else:
            taste_model = LatentTasteModel().fit(train_df)
            
        oracle_model = OracleModel(creators=self.creators).fit(train_df)
        
        models = {
            "Memorization": memo_model,
            "TasteModel": taste_model,
            "Oracle": oracle_model,
        }
        
        sim_surface_sim = memo_model.compute_similarity_to_hits(similar_df)
        nov_surface_sim = memo_model.compute_similarity_to_hits(novel_df)
        
        results: Dict[str, Any] = {"models": {}, "creator_breakdown": {}}
        unique_creators = sorted(list(np.unique(train_df["creator_id"].values)))
        
        combined_eval_df = pd.concat([similar_df, novel_df], ignore_index=True)
        combined_surface_sim = np.concatenate([sim_surface_sim, nov_surface_sim])
        
        for m_name, model in models.items():
            pred_sim = model.predict(similar_df)
            pred_nov = model.predict(novel_df)
            combined_preds = np.concatenate([pred_sim, pred_nov])
            
            sim_rhos = []
            nov_rhos = []
            gaps = []
            
            for cid in unique_creators:
                c_mask_sim = similar_df["creator_id"] == cid
                c_mask_nov = novel_df["creator_id"] == cid
                
                y_sim = similar_df.loc[c_mask_sim, "performance"].values
                p_sim = pred_sim[c_mask_sim.values]
                rho_sim = compute_spearman(y_sim, p_sim)
                
                y_nov = novel_df.loc[c_mask_nov, "performance"].values
                p_nov = pred_nov[c_mask_nov.values]
                rho_nov = compute_spearman(y_nov, p_nov)
                
                gap = compute_novelty_gap(rho_sim, rho_nov)
                
                sim_rhos.append(rho_sim)
                nov_rhos.append(rho_nov)
                gaps.append(gap)
                
                if cid not in results["creator_breakdown"]:
                    results["creator_breakdown"][cid] = {}
                results["creator_breakdown"][cid][m_name] = {
                    "spearman_similar": rho_sim,
                    "spearman_novel": rho_nov,
                    "novelty_gap": gap,
                }
                
            mem_diag = compute_memorization_diagnostics(
                predictions=combined_preds,
                surface_similarity_to_hits=combined_surface_sim,
                true_taste_match=combined_eval_df["taste_match"].values,
            )
            
            results["models"][m_name] = {
                "mean_spearman_similar": float(np.mean(sim_rhos)),
                "std_spearman_similar": float(np.std(sim_rhos)),
                "mean_spearman_novel": float(np.mean(nov_rhos)),
                "std_spearman_novel": float(np.std(nov_rhos)),
                "mean_novelty_gap": float(np.mean(gaps)),
                "std_novelty_gap": float(np.std(gaps)),
                "memorization_diagnostics": mem_diag,
            }
            
        latent_cols = [f"latent_{name}" for name in LATENT_FACTOR_NAMES]
        if all(col in combined_eval_df.columns for col in latent_cols):
            true_factors = combined_eval_df[latent_cols].values
            learned_factors = taste_model.get_latent_factors(combined_eval_df)
            matching_results = self.matcher.match(true_factors, learned_factors)
            results["factor_recovery"] = matching_results
            
        results["fitted_taste_model"] = taste_model
        return results
