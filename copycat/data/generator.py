import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from copycat.data.schema import (
    LATENT_FACTOR_NAMES,
    FORMATS,
    TOPICS,
    THUMBNAIL_STYLES,
    CreatorProfile,
    VideoRecord,
    DatasetBundle,
)

class SyntheticDatasetGenerator:
    def __init__(
        self,
        seed: int = 42,
        k_factors: int = 5,
        beta_taste: float = 2.0,
        beta_platform: float = 0.6,
        beta_novelty: float = 0.6,
        noise_std: float = 0.30,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.k_factors = k_factors
        self.beta_taste = beta_taste
        self.beta_platform = beta_platform
        self.beta_novelty = beta_novelty
        self.noise_std = noise_std
        self.topic_biases = {
            "tech": 0.20,
            "productivity": 0.15,
            "comedy": 0.10,
            "science": 0.05,
            "philosophy": -0.05,
            "culture": -0.05,
        }

    def generate_creator_profile(self, creator_id: str) -> CreatorProfile:
        raw_taste = self.rng.normal(0.0, 1.0, size=self.k_factors)
        taste_vector = raw_taste / np.linalg.norm(raw_taste)
        baseline_scale = float(self.rng.normal(10.0, 0.4))
        
        familiar_formats = self.rng.choice(FORMATS, size=2, replace=False).tolist()
        familiar_thumbnails = self.rng.choice(THUMBNAIL_STYLES, size=1, replace=False).tolist()
        
        return CreatorProfile(
            creator_id=creator_id,
            taste_vector=taste_vector,
            baseline_scale=baseline_scale,
            familiar_formats=familiar_formats,
            familiar_thumbnails=familiar_thumbnails,
        )

    def generate_video(
        self,
        creator: CreatorProfile,
        video_id: str,
        split: str,
    ) -> VideoRecord:
        novel_formats = [f for f in FORMATS if f not in creator.familiar_formats]
        novel_thumbnails = [t for t in THUMBNAIL_STYLES if t not in creator.familiar_thumbnails]
        
        if split in ["train", "held_out_similar"]:
            fmt = str(self.rng.choice(creator.familiar_formats))
            if self.rng.random() < 0.85:
                thumb = str(creator.familiar_thumbnails[0])
            else:
                thumb = str(self.rng.choice(THUMBNAIL_STYLES))
            novelty_dist = 0.0
        else:
            fmt = str(self.rng.choice(novel_formats))
            thumb = str(self.rng.choice(novel_thumbnails))
            novelty_dist = 1.0

        topic = str(self.rng.choice(TOPICS))
        
        raw_latent = self.rng.normal(0.0, 1.0, size=self.k_factors)
        latent_vector = raw_latent / np.linalg.norm(raw_latent)
        
        deadpan = latent_vector[0]
        observational = latent_vector[1]
        rigor = latent_vector[2]
        absurd = latent_vector[3]
        craft = latent_vector[4]
        
        duration_base = 320.0 + 100.0 * rigor + 140.0 * observational - 80.0 * absurd
        duration_sec = float(np.clip(self.rng.lognormal(np.log(max(duration_base, 60.0)), 0.20), 45.0, 1800.0))
        
        cuts_mean = 14.0 - 5.0 * deadpan - 4.0 * observational + 6.0 * absurd
        cuts_per_min = float(np.clip(self.rng.gamma(shape=max(cuts_mean, 2.0), scale=1.0), 1.0, 45.0))
        
        energy_alpha = max(2.0 - 1.2 * deadpan + 1.2 * absurd - 0.8 * observational, 0.4)
        energy_beta = max(2.0 + 1.2 * deadpan - 1.2 * absurd + 0.8 * observational, 0.4)
        audio_energy = float(self.rng.beta(energy_alpha, energy_beta))
        
        contrast_alpha = max(2.0 + 1.5 * craft + 0.5 * observational, 0.4)
        contrast_beta = max(2.0 - 1.5 * craft - 0.5 * observational, 0.4)
        visual_contrast = float(self.rng.beta(contrast_alpha, contrast_beta))
        
        taste_match = float(np.dot(creator.taste_vector, latent_vector))
        
        topic_bias = self.topic_biases.get(topic, 0.0)
        trend_shock = float(self.rng.normal(0.0, 0.15))
        platform_trend = topic_bias + trend_shock
        
        epsilon = float(self.rng.normal(0.0, self.noise_std))
        
        log_perf = (
            creator.baseline_scale
            + self.beta_taste * taste_match
            + self.beta_platform * platform_trend
            + self.beta_novelty * (novelty_dist * taste_match)
            + epsilon
        )
        performance = float(np.exp(log_perf))
        
        return VideoRecord(
            video_id=video_id,
            creator_id=creator.creator_id,
            split=split,
            format=fmt,
            topic=topic,
            thumbnail_style=thumb,
            duration_sec=duration_sec,
            cuts_per_min=cuts_per_min,
            audio_energy=audio_energy,
            visual_contrast=visual_contrast,
            performance=performance,
            taste_match=taste_match,
            novelty_dist=novelty_dist,
            platform_trend=platform_trend,
            latent_vector=latent_vector,
        )

    def generate_creator_dataset(
        self,
        creator: CreatorProfile,
        n_train: int = 60,
        n_similar: int = 25,
        n_novel: int = 25,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_records = []
        similar_records = []
        novel_records = []
        
        for idx in range(n_train):
            vid = f"{creator.creator_id}_tr_{idx:03d}"
            rec = self.generate_video(creator, vid, "train")
            train_records.append(rec.to_full_dict())
            
        for idx in range(n_similar):
            vid = f"{creator.creator_id}_sim_{idx:03d}"
            rec = self.generate_video(creator, vid, "held_out_similar")
            similar_records.append(rec.to_full_dict())
            
        for idx in range(n_novel):
            vid = f"{creator.creator_id}_nov_{idx:03d}"
            rec = self.generate_video(creator, vid, "held_out_novel")
            novel_records.append(rec.to_full_dict())
            
        return (
            pd.DataFrame(train_records),
            pd.DataFrame(similar_records),
            pd.DataFrame(novel_records),
        )

    def generate_bundle(
        self,
        n_creators: int = 25,
        n_train: int = 60,
        n_similar: int = 25,
        n_novel: int = 25,
    ) -> DatasetBundle:
        creators = {}
        train_dfs = []
        similar_dfs = []
        novel_dfs = []
        
        for cid in range(n_creators):
            creator_id = f"creator_{cid:02d}"
            profile = self.generate_creator_profile(creator_id)
            creators[creator_id] = profile
            tr_df, sim_df, nov_df = self.generate_creator_dataset(
                profile, n_train, n_similar, n_novel
            )
            train_dfs.append(tr_df)
            similar_dfs.append(sim_df)
            novel_dfs.append(nov_df)
            
        return DatasetBundle(
            creators=creators,
            train_df=pd.concat(train_dfs, ignore_index=True),
            similar_df=pd.concat(similar_dfs, ignore_index=True),
            novel_df=pd.concat(novel_dfs, ignore_index=True),
        )
