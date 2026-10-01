import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from copycat.data.real_proxy import RealDataNoveltyClusterer
from copycat.evaluation.metrics import compute_spearman, compute_novelty_gap

def generate_mock_public_youtube_data() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    channels = ["creator_tech_explains", "creator_comedy_skits", "creator_essay_deepdive"]
    clusters = {
        "creator_tech_explains": [
            ("Python coding tutorial step by step guide", 15000),
            ("Build a full stack web app with React", 18000),
            ("Machine learning explained simply for beginners", 22000),
            ("My secret morning routine as a software engineer", 8000),
            ("I tried living in an off-grid cabin for 30 days", 45000),
        ],
        "creator_comedy_skits": [
            ("When you ask your dad for money relatable sketch", 50000),
            ("Every conversation in corporate office meetings", 75000),
            ("Introverts at high school parties pov", 62000),
            ("My serious breakdown of international geopolitics", 12000),
            ("Reading classic literature for 24 hours straight", 9000),
        ],
        "creator_essay_deepdive": [
            ("The decline of cinema and the rise of franchise fatigue", 35000),
            ("Why nostalgia is killing modern storytelling video essay", 42000),
            ("The architectural philosophy of brutalist concrete", 29000),
            ("Reacting to funny viral TikTok memes with my cat", 11000),
            ("Speedrunning Minecraft while blindfolded", 14000),
        ],
    }
    
    rows = []
    base_time = pd.Timestamp("2025-01-01")
    for cid in channels:
        templates = clusters[cid]
        for idx in range(60):
            t_idx = idx % len(templates)
            title_base, view_base = templates[t_idx]
            
            jitter = rng.integers(1, 999)
            title = f"{title_base} #{jitter}"
            desc = f"In this video we explore {title_base}. Don't forget to like and subscribe."
            time_offset = base_time + pd.Timedelta(days=idx * 3)
            
            perf = float(view_base * rng.lognormal(0.0, 0.35))
            rows.append({
                "video_id": f"{cid}_v_{idx:03d}",
                "creator_id": cid,
                "title": title,
                "description": desc,
                "published_at": time_offset.isoformat(),
                "view_count": perf,
            })
            
    return pd.DataFrame(rows)

def run_real_proxy_pipeline():
    print("=" * 75)
    print("PHASE 4: REAL-DATA PROXY EVALUATION (CLUSTER DISTANCE HEURISTIC)")
    print("=" * 75)
    print(
        "METHODOLOGICAL FRAMING: This real-data proxy uses text/format cluster distance "
        "as a crude heuristic for novelty. This is STRUCTURALLY DIFFERENT from the clean "
        "synthetic latent taste decomposition in Phases 1-3. The numbers are NOT directly "
        "comparable and do NOT evaluate Grocalo's commercial product.\n"
    )
    
    df = generate_mock_public_youtube_data()
    clusterer = RealDataNoveltyClusterer(n_clusters=3)
    
    memo_sim_scores = []
    memo_nov_scores = []
    proxy_sim_scores = []
    proxy_nov_scores = []
    
    for cid in df["creator_id"].unique():
        c_df = df[df["creator_id"] == cid]
        parts = clusterer.partition_by_cluster_novelty(c_df, train_ratio=0.6)
        
        train_df = parts["train"]
        sim_df = parts["similar"]
        nov_df = parts["novel"]
        
        vec = TfidfVectorizer(max_features=64, stop_words="english")
        X_train = vec.fit_transform(train_df["title"] + " " + train_df["description"]).toarray()
        X_sim = vec.transform(sim_df["title"] + " " + sim_df["description"]).toarray()
        X_nov = vec.transform(nov_df["title"] + " " + nov_df["description"]).toarray()
        
        y_train = train_df["view_count"].values
        y_sim = sim_df["view_count"].values
        y_nov = nov_df["view_count"].values
        
        hit_thresh = np.quantile(y_train, 0.70)
        hit_mask = y_train >= hit_thresh
        X_hits = X_train[hit_mask]
        
        sim_similarities = np.mean(cosine_similarity(X_sim, X_hits), axis=1)
        nov_similarities = np.mean(cosine_similarity(X_nov, X_hits), axis=1)
        
        memo_sim_scores.append(compute_spearman(y_sim, sim_similarities))
        memo_nov_scores.append(compute_spearman(y_nov, nov_similarities))
        
        svd = TruncatedSVD(n_components=min(5, X_train.shape[1]), random_state=42)
        Z_train = svd.fit_transform(X_train)
        Z_sim = svd.transform(X_sim)
        Z_nov = svd.transform(X_nov)
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(Z_train, np.log(np.maximum(y_train, 1.0)))
        
        proxy_pred_sim = np.exp(ridge.predict(Z_sim))
        proxy_pred_nov = np.exp(ridge.predict(Z_nov))
        
        proxy_sim_scores.append(compute_spearman(y_sim, proxy_pred_sim))
        proxy_nov_scores.append(compute_spearman(y_nov, proxy_pred_nov))
        
    print(f"Memorization Proxy (TF-IDF Hit Nearest-Neighbor):")
    print(f"  Similar Cluster Spearman: {np.mean(memo_sim_scores):.3f}")
    print(f"  Novel Cluster Spearman:   {np.mean(memo_nov_scores):.3f}")
    print(f"  Proxy Novelty Gap:        {compute_novelty_gap(float(np.mean(memo_sim_scores)), float(np.mean(memo_nov_scores))):.3f}")
    
    print(f"\nLatent SVD Regression Proxy:")
    print(f"  Similar Cluster Spearman: {np.mean(proxy_sim_scores):.3f}")
    print(f"  Novel Cluster Spearman:   {np.mean(proxy_nov_scores):.3f}")
    print(f"  Proxy Novelty Gap:        {compute_novelty_gap(float(np.mean(proxy_sim_scores)), float(np.mean(proxy_nov_scores))):.3f}")
    print("-" * 75)

if __name__ == "__main__":
    run_real_proxy_pipeline()
