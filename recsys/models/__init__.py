"""Model components for the extended fusion study."""

from .attention_variants import (  # noqa: F401
    AttentionFusion,
    FixedFusion,
    ConcatMLPFusion,
    CrossAttentionFusion,  # New for RecSys 2026
    compute_item_popularity,
    compute_user_activity,
    compute_item_cf_confidence,
    compute_item_text_quality,
    compute_item_recency,
)
from .gbaf import GBAFFusion, GBAFAdaptiveFusion, GBAFv2Fusion

__all__ = [
    "AttentionFusion",
    "FixedFusion",
    "ConcatMLPFusion",
    "CrossAttentionFusion",
    "GBAFFusion",
    "GBAFAdaptiveFusion",
    "GBAFv2Fusion",
    "compute_item_popularity",
    "compute_user_activity",
    "compute_item_cf_confidence",
    "compute_item_text_quality",
    "compute_item_recency",
]
