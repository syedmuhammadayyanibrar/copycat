from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np
import pandas as pd

LATENT_FACTOR_NAMES = [
    "deadpan_delivery",
    "observational_depth",
    "technical_rigor",
    "absurdist_humor",
    "visual_craft",
]

FORMATS = [
    "talking_head",
    "tutorial",
    "screen_breakdown",
    "sketch",
    "field_vlog",
    "mini_doc",
    "silent_essay",
]

TOPICS = [
    "tech",
    "philosophy",
    "productivity",
    "comedy",
    "science",
    "culture",
]

THUMBNAIL_STYLES = [
    "face_closeup",
    "minimalist_text",
    "high_contrast_action",
    "diagram_schematic",
    "meme_edit",
]

OBSERVABLE_CATEGORICAL_COLS = [
    "format",
    "topic",
    "thumbnail_style",
]

OBSERVABLE_NUMERIC_COLS = [
    "duration_sec",
    "cuts_per_min",
    "audio_energy",
    "visual_contrast",
]

@dataclass
class CreatorProfile:
    creator_id: str
    taste_vector: np.ndarray
    baseline_scale: float
    familiar_formats: List[str]
    familiar_thumbnails: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "creator_id": self.creator_id,
            "taste_vector": self.taste_vector.tolist(),
            "baseline_scale": float(self.baseline_scale),
            "familiar_formats": self.familiar_formats,
            "familiar_thumbnails": self.familiar_thumbnails,
        }

@dataclass
class VideoRecord:
    video_id: str
    creator_id: str
    split: str
    format: str
    topic: str
    thumbnail_style: str
    duration_sec: float
    cuts_per_min: float
    audio_energy: float
    visual_contrast: float
    performance: float
    taste_match: float
    novelty_dist: float
    platform_trend: float
    latent_vector: np.ndarray

    def to_observable_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "creator_id": self.creator_id,
            "split": self.split,
            "format": self.format,
            "topic": self.topic,
            "thumbnail_style": self.thumbnail_style,
            "duration_sec": self.duration_sec,
            "cuts_per_min": self.cuts_per_min,
            "audio_energy": self.audio_energy,
            "visual_contrast": self.visual_contrast,
            "performance": self.performance,
        }

    def to_full_dict(self) -> Dict[str, Any]:
        base = self.to_observable_dict()
        base.update({
            "taste_match": self.taste_match,
            "novelty_dist": self.novelty_dist,
            "platform_trend": self.platform_trend,
        })
        for i, name in enumerate(LATENT_FACTOR_NAMES):
            base[f"latent_{name}"] = float(self.latent_vector[i])
        return base

@dataclass
class DatasetBundle:
    creators: Dict[str, CreatorProfile]
    train_df: pd.DataFrame
    similar_df: pd.DataFrame
    novel_df: pd.DataFrame
