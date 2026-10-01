import os
import json
import urllib.request
import urllib.parse
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

class YouTubePublicDataFetcher:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("YOUTUBE_API_KEY", "")

    def fetch_channel_videos(self, channel_id: str, max_results: int = 50) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
            
        search_url = "https://www.googleapis.com/youtube/v3/search?" + urllib.parse.urlencode({
            "key": self.api_key,
            "channelId": channel_id,
            "part": "snippet",
            "order": "date",
            "maxResults": max_results,
            "type": "video",
        })
        
        req = urllib.request.Request(search_url)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            
        items = data.get("items", [])
        video_ids = [it["id"]["videoId"] for it in items]
        if not video_ids:
            return []
            
        stats_url = "https://www.googleapis.com/youtube/v3/videos?" + urllib.parse.urlencode({
            "key": self.api_key,
            "part": "snippet,statistics,contentDetails",
            "id": ",".join(video_ids),
        })
        
        with urllib.request.urlopen(urllib.request.Request(stats_url)) as resp:
            stats_data = json.loads(resp.read().decode("utf-8"))
            
        records = []
        for v in stats_data.get("items", []):
            snip = v["snippet"]
            stats = v.get("statistics", {})
            records.append({
                "video_id": v["id"],
                "creator_id": channel_id,
                "title": snip.get("title", ""),
                "description": snip.get("description", ""),
                "published_at": snip.get("publishedAt", ""),
                "view_count": float(stats.get("viewCount", 0)),
                "like_count": float(stats.get("likeCount", 0)),
                "comment_count": float(stats.get("commentCount", 0)),
            })
        return records

class RealDataNoveltyClusterer:
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(max_features=64, stop_words="english")
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    def partition_by_cluster_novelty(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.6,
        novel_threshold_percentile: float = 75.0,
    ) -> Dict[str, pd.DataFrame]:
        df_sorted = df.sort_values("published_at").reset_index(drop=True)
        n_train = int(len(df_sorted) * train_ratio)
        train_df = df_sorted.iloc[:n_train].copy()
        eval_df = df_sorted.iloc[n_train:].copy()
        
        train_text = train_df["title"] + " " + train_df["description"]
        X_train = self.vectorizer.fit_transform(train_text).toarray()
        self.kmeans.fit(X_train)
        
        eval_text = eval_df["title"] + " " + eval_df["description"]
        X_eval = self.vectorizer.transform(eval_text).toarray()
        
        eval_dists = np.min(self.kmeans.transform(X_eval), axis=1)
        thresh = np.percentile(eval_dists, novel_threshold_percentile)
        
        eval_df["cluster_distance"] = eval_dists
        sim_mask = eval_dists <= thresh
        novel_mask = eval_dists > thresh
        
        similar_df = eval_df[sim_mask].copy()
        novel_df = eval_df[novel_mask].copy()
        
        train_df["split"] = "train"
        similar_df["split"] = "real_proxy_similar"
        novel_df["split"] = "real_proxy_novel"
        
        return {
            "train": train_df,
            "similar": similar_df,
            "novel": novel_df,
        }
