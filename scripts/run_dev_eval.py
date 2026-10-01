import json
import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from copycat.data.schema import CreatorProfile
from copycat.evaluation.harness import EvaluationHarness

def load_dev_data():
    dev_dir = os.path.join("data", "dev")
    train_df = pd.read_csv(os.path.join(dev_dir, "train.csv"))
    similar_df = pd.read_csv(os.path.join(dev_dir, "held_out_similar.csv"))
    novel_df = pd.read_csv(os.path.join(dev_dir, "held_out_novel.csv"))
    
    with open(os.path.join(dev_dir, "creators.json"), "r") as f:
        raw_creators = json.load(f)
        
    creators = {}
    for cid, data in raw_creators.items():
        creators[cid] = CreatorProfile(
            creator_id=cid,
            taste_vector=np.array(data["taste_vector"]),
            baseline_scale=data["baseline_scale"],
            familiar_formats=data["familiar_formats"],
            familiar_thumbnails=data["familiar_thumbnails"],
        )
    return creators, train_df, similar_df, novel_df

def main():
    creators, train_df, similar_df, novel_df = load_dev_data()
    harness = EvaluationHarness(creators=creators)
    
    print("=" * 60)
    print("RUNNING DEVELOPMENT SET EVALUATION (20 CREATORS)")
    print("=" * 60)
    
    results = harness.run_evaluation(
        train_df=train_df,
        similar_df=similar_df,
        novel_df=novel_df,
        is_held_out=False,
    )
    
    print("\nMODEL PERFORMANCE SUMMARY:")
    print("-" * 85)
    print(f"{'Model':<16} | {'Sim Spearman':<14} | {'Nov Spearman':<14} | {'Novelty Gap':<12} | {'Memo Bias':<10}")
    print("-" * 85)
    
    for m_name, metrics in results["models"].items():
        sim_s = f"{metrics['mean_spearman_similar']:.3f} +/- {metrics['std_spearman_similar']:.3f}"
        nov_s = f"{metrics['mean_spearman_novel']:.3f} +/- {metrics['std_spearman_novel']:.3f}"
        gap_s = f"{metrics['mean_novelty_gap']:.3f}"
        bias_s = f"{metrics['memorization_diagnostics']['memorization_bias']:.3f}"
        print(f"{m_name:<16} | {sim_s:<14} | {nov_s:<14} | {gap_s:<12} | {bias_s:<10}")
        
    print("-" * 85)
    
    if "factor_recovery" in results:
        rec = results["factor_recovery"]
        print(f"\nHUNGARIAN FACTOR RECOVERY (Mean Correlation: {rec['mean_recovery']:.3f}):")
        print("-" * 65)
        print(f"{'True Factor':<25} | {'Learned Factor':<16} | {'Absolute Corr':<14}")
        print("-" * 65)
        for pair in rec["matched_pairs"]:
            print(f"{pair['factor_name']:<25} | factor_{pair['learned_index']:<9} | {pair['absolute_correlation']:.3f}")
        print("-" * 65)
        
    os.makedirs("checkpoints", exist_ok=True)
    with open(os.path.join("checkpoints", "dev_taste_model.pkl"), "wb") as f:
        pickle.dump(results["fitted_taste_model"], f)
        
    with open(os.path.join("checkpoints", "dev_eval_results.json"), "w") as f:
        clean_res = {
            "models": results["models"],
            "factor_recovery": results.get("factor_recovery"),
        }
        json.dump(clean_res, f, indent=2)

if __name__ == "__main__":
    main()
