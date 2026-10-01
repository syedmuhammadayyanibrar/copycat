import json
import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from copycat.data.schema import CreatorProfile
from copycat.evaluation.harness import EvaluationHarness

def load_held_out_data():
    held_dir = os.path.join("data", "held_out")
    train_df = pd.read_csv(os.path.join(held_dir, "train.csv"))
    similar_df = pd.read_csv(os.path.join(held_dir, "held_out_similar.csv"))
    novel_df = pd.read_csv(os.path.join(held_dir, "held_out_novel.csv"))
    
    with open(os.path.join(held_dir, "creators.json"), "r") as f:
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

def generate_report(results: dict, output_path: str = "EVAL_REPORT.md"):
    with open(output_path, "w") as f:
        f.write("# Final Evaluation Report: Copycat Benchmark\n\n")
        f.write("## 1. Framing & Scope\n\n")
        f.write(
            "This document presents the final, un-tuned evaluation of the copycat benchmark "
            "on the 5 quarantined held-out creators (`creator_20` through `creator_24`).\n\n"
            "**Crucial Framing Notice**: This evaluation benchmarks a **measurement methodology** "
            "using synthetic ground truth. It establishes that this measurement protocol can "
            "successfully detect a novelty gap when present. It does NOT evaluate or prove the presence "
            "or absence of a novelty gap in Grocalo's commercial product or any proprietary tool.\n\n"
        )
        
        f.write("## 2. Held-Out Evaluation Summary\n\n")
        f.write("| Model | Similar Spearman | Novel Spearman | Novelty Gap | Memorization Bias |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for m_name, metrics in results["models"].items():
            sim_s = f"{metrics['mean_spearman_similar']:.3f} +/- {metrics['std_spearman_similar']:.3f}"
            nov_s = f"{metrics['mean_spearman_novel']:.3f} +/- {metrics['std_spearman_novel']:.3f}"
            gap_s = f"{metrics['mean_novelty_gap']:.3f}"
            bias_s = f"{metrics['memorization_diagnostics']['memorization_bias']:.3f}"
            f.write(f"| **{m_name}** | {sim_s} | {nov_s} | {gap_s} | {bias_s} |\n")
            
        f.write("\n\n## 3. Hungarian Factor Recovery (Latent Identifiability)\n\n")
        rec = results.get("factor_recovery", {})
        mean_rec = rec.get("mean_recovery", 0.0)
        f.write(f"**Mean Factor Alignment Correlation**: `{mean_rec:.3f}`\n\n")
        f.write("| True Named Factor | Matched Latent Dimension | Absolute Correlation |\n")
        f.write("| :--- | :--- | :--- |\n")
        for pair in rec.get("matched_pairs", []):
            f.write(f"| `{pair['factor_name']}` | `factor_{pair['learned_index']}` | {pair['absolute_correlation']:.3f} |\n")
            
        f.write("\n\n## 4. Per-Creator Granular Breakdown\n\n")
        f.write("| Creator ID | Model | Similar Spearman | Novel Spearman | Novelty Gap |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for cid, m_dict in results["creator_breakdown"].items():
            for m_name, met in m_dict.items():
                f.write(f"| `{cid}` | {m_name} | {met['spearman_similar']:.3f} | {met['spearman_novel']:.3f} | {met['novelty_gap']:.3f} |\n")
                
        f.write("\n\n## 5. Narrative Determination Against Pre-Registered Criteria\n\n")
        memo_gap = results["models"]["Memorization"]["mean_novelty_gap"]
        taste_gap = results["models"]["TasteModel"]["mean_novelty_gap"]
        taste_nov = results["models"]["TasteModel"]["mean_spearman_novel"]
        memo_nov = results["models"]["Memorization"]["mean_spearman_novel"]
        
        is_success = (taste_nov > memo_nov + 0.05) and (mean_rec >= 0.30)
        if is_success:
            f.write("### Outcome: Narrative A (Success Case Validated)\n\n")
            f.write(
                "The held-out results confirm Narrative A. The Latent Taste Model outperforms the "
                "Memorization Baseline on novel content, achieving higher novel rank correlation "
                f"({taste_nov:.3f} vs {memo_nov:.3f}) and maintaining healthy factor recovery "
                f"({mean_rec:.3f}). This confirms that the evaluation protocol reliably distinguishes "
                "taste-based latent inference from surface template memorization."
            )
        else:
            f.write("### Outcome: Narrative B (Null Case Encountered)\n\n")
            f.write(
                "The held-out results reflect Narrative B. The Taste Model did not demonstrate a decisive "
                "advantage over memorization on novel content under this parameter regime. "
                "As pre-registered, this outcome reflects limitations in the synthetic generator or model "
                "capacity, NOT empirical evidence about Grocalo's proprietary system."
            )

def main():
    creators, train_df, similar_df, novel_df = load_held_out_data()
    harness = EvaluationHarness(creators=creators)
    
    with open(os.path.join("checkpoints", "dev_taste_model.pkl"), "rb") as f:
        dev_taste_model = pickle.load(f)
        
    print("=" * 65)
    print("RUNNING FINAL HELD-OUT EVALUATION (5 QUARANTINED CREATORS)")
    print("=" * 65)
    
    results = harness.run_evaluation(
        train_df=train_df,
        similar_df=similar_df,
        novel_df=novel_df,
        is_held_out=True,
        pretrained_taste_model=dev_taste_model,
    )
    
    print("\nFINAL HELD-OUT PERFORMANCE SUMMARY:")
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
        
    generate_report(results, "EVAL_REPORT.md")
    print("\nEvaluation report written to EVAL_REPORT.md")

if __name__ == "__main__":
    main()
