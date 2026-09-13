"""Fusion models for the extended LLM feature study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import numpy as np


def _build_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation '{name}'")


def _build_attention_mlp(
    input_dim: int,
    hidden_sizes: Iterable[int],
    activation: str,
    dropout: float,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = input_dim
    act_layer = _build_activation(activation)
    hidden_sizes = list(hidden_sizes)
    for hidden in hidden_sizes:
        layers.append(nn.Linear(prev, hidden))
        layers.append(act_layer)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = hidden
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


@dataclass
class AttentionConfig:
    input_mode: str = "popularity"  # or "concat"
    features: Iterable[str] = ()
    hidden_sizes: Iterable[int] = (32, 16)
    activation: str = "relu"
    dropout: float = 0.1
    context_features: Iterable[str] = ()
    temperature: float = 1.0
    lambda_min: Optional[float] = None
    lambda_max: Optional[float] = None

    @classmethod
    def from_dict(cls, cfg: Optional[Dict]) -> "AttentionConfig":
        if cfg is None:
            return cls()
        return cls(
            input_mode=cfg.get("input_mode", cls.input_mode),
            features=cfg.get("features", cls.features),
            hidden_sizes=cfg.get("hidden_sizes", cls.hidden_sizes),
            activation=cfg.get("activation", cls.activation),
            dropout=float(cfg.get("dropout", cls.dropout)),
            context_features=cfg.get("context_features", ()),
            temperature=float(cfg.get("temperature", cls.temperature)),
            lambda_min=cfg.get("lambda_min", cls.lambda_min),
            lambda_max=cfg.get("lambda_max", cls.lambda_max),
        )


class AttentionFusion(nn.Module):
    """Adaptive fusion with configurable gating inputs."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        text_dim: int = 384,
        attention: Optional[Dict] = None,
        text_adapter: Optional[Dict] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.text_dim = text_dim
        self.device = torch.device(device)

        self.attn_cfg = AttentionConfig.from_dict(attention)
        self.input_mode = self.attn_cfg.input_mode.lower()
        if self.input_mode not in {"popularity", "concat"}:
            raise ValueError("attention.input_mode must be 'popularity' or 'concat'")

        self.features = tuple(str(f).lower() for f in (self.attn_cfg.features or ()))
        if not self.features:
            if self.input_mode == "popularity":
                self.features = ("popularity",)
            else:
                self.features = ("concat_embeddings",)

        supported_item_features = {
            "popularity",
            "cf_confidence",
            "text_quality",
            "recency",
            "concat_embeddings",
        }
        for feat in self.features:
            if feat not in supported_item_features:
                raise ValueError(
                    f"Unsupported attention feature '{feat}'. Supported: {sorted(supported_item_features)}"
                )

        self.context_features = tuple(str(f).lower() for f in self.attn_cfg.context_features)
        supported_context = {"user_activity"}
        for feature in self.context_features:
            if feature not in supported_context:
                raise ValueError(f"Unsupported attention context feature '{feature}'")
        self.has_user_activity = "user_activity" in self.context_features

        self.temperature = max(float(self.attn_cfg.temperature), 1e-3)
        self.lambda_min = None if self.attn_cfg.lambda_min is None else float(self.attn_cfg.lambda_min)
        self.lambda_max = None if self.attn_cfg.lambda_max is None else float(self.attn_cfg.lambda_max)
        if self.lambda_min is not None and self.lambda_max is not None:
            if self.lambda_min > self.lambda_max:
                raise ValueError("lambda_min must be <= lambda_max")

        # Embeddings for collaborative filtering branch
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Projection for text branch
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # Optional text adapter (bottleneck / LoRA style)
        self.text_adapter_type: Optional[str] = None
        self.text_adapter_layers: Optional[nn.Module] = None
        self.text_adapter_residual: bool = False
        self.text_adapter_down: Optional[nn.Linear] = None
        self.text_adapter_up: Optional[nn.Linear] = None
        self.text_adapter_scale: float = 1.0
        self.text_adapter_dropout: Optional[nn.Dropout] = None

        adapter_cfg = text_adapter or {}
        if adapter_cfg.get("enable", False):
            adapter_type = str(adapter_cfg.get("type", "bottleneck")).lower()
            if adapter_type == "bottleneck":
                bottleneck_dim = int(adapter_cfg.get("bottleneck_dim", 128))
                adapter_activation = _build_activation(adapter_cfg.get("activation", "relu"))
                adapter_dropout = float(adapter_cfg.get("dropout", 0.1))
                layers: List[nn.Module] = [nn.Linear(text_dim, bottleneck_dim), adapter_activation]
                if adapter_dropout > 0:
                    layers.append(nn.Dropout(adapter_dropout))
                layers.append(nn.Linear(bottleneck_dim, text_dim))
                self.text_adapter_layers = nn.Sequential(*layers)
                self.text_adapter_residual = bool(adapter_cfg.get("residual", True))
                self.text_adapter_type = "bottleneck"
            elif adapter_type == "lora":
                rank = int(adapter_cfg.get("rank", 16))
                alpha = float(adapter_cfg.get("alpha", 1.0))
                dropout = float(adapter_cfg.get("dropout", 0.0))
                self.text_adapter_down = nn.Linear(text_dim, rank, bias=False)
                self.text_adapter_up = nn.Linear(rank, text_dim, bias=False)
                nn.init.kaiming_uniform_(self.text_adapter_down.weight, a=np.sqrt(5))
                nn.init.zeros_(self.text_adapter_up.weight)
                self.text_adapter_scale = alpha / max(rank, 1)
                self.text_adapter_dropout = nn.Dropout(dropout) if dropout > 0 else None
                self.text_adapter_residual = bool(adapter_cfg.get("residual", True))
                self.text_adapter_type = "lora"
            else:
                raise ValueError(f"Unsupported text adapter type '{adapter_type}'")

        def _feature_dim(feature: str) -> int:
            if feature in {"popularity", "cf_confidence", "text_quality", "recency"}:
                return 1
            if feature == "concat_embeddings":
                return embedding_dim * 2
            raise ValueError(f"Unhandled feature '{feature}'")

        base_attn_dim = sum(_feature_dim(f) for f in self.features)
        context_dim = 1 if self.has_user_activity else 0
        attn_input_dim = base_attn_dim + context_dim

        self.attention_net = _build_attention_mlp(
            input_dim=attn_input_dim,
            hidden_sizes=self.attn_cfg.hidden_sizes,
            activation=self.attn_cfg.activation,
            dropout=self.attn_cfg.dropout,
        )
        self.sigmoid = nn.Sigmoid()

        # Feature buffers
        self.register_buffer("item_popularity", torch.zeros(n_items))
        self.register_buffer("popularity_normalized", torch.zeros(n_items))
        self.register_buffer("item_cf_confidence", torch.zeros(n_items))
        self.register_buffer("item_cf_confidence_normalized", torch.zeros(n_items))
        self.register_buffer("item_text_quality", torch.zeros(n_items))
        self.register_buffer("item_text_quality_normalized", torch.zeros(n_items))
        self.register_buffer("item_recency", torch.zeros(n_items))
        self.register_buffer("item_recency_normalized", torch.zeros(n_items))

        # User-context buffers
        self.register_buffer("user_activity_counts", torch.zeros(n_users))
        self.register_buffer("user_activity_normalized", torch.zeros(n_users))

        self.to(self.device)

    def _apply_text_adapter(self, text_tensor: torch.Tensor) -> torch.Tensor:
        if self.text_adapter_type is None:
            return text_tensor

        if self.text_adapter_type == "bottleneck" and self.text_adapter_layers is not None:
            adapted = self.text_adapter_layers(text_tensor)
            if self.text_adapter_residual:
                return text_tensor + adapted
            return adapted

        if (
            self.text_adapter_type == "lora"
            and self.text_adapter_down is not None
            and self.text_adapter_up is not None
        ):
            input_tensor = text_tensor
            if self.text_adapter_dropout is not None:
                input_tensor = self.text_adapter_dropout(input_tensor)
            delta = self.text_adapter_up(self.text_adapter_down(input_tensor)) * self.text_adapter_scale
            if self.text_adapter_residual:
                return text_tensor + delta
            return delta

        return text_tensor

    def set_item_popularity(self, popularity_counts: torch.Tensor) -> None:
        """Store normalized popularity (required for popularity mode)."""

        pop_tensor = popularity_counts.float().to(self.device)
        self.item_popularity = pop_tensor
        max_pop = pop_tensor.max().clamp(min=1.0)
        self.popularity_normalized = pop_tensor / max_pop

    def set_item_cf_confidence(self, confidence: torch.Tensor) -> None:
        tensor = confidence.float().to(self.device)
        if tensor.numel() != self.n_items:
            raise ValueError("item cf confidence tensor must have length equal to n_items")
        max_val = tensor.max().clamp(min=1e-6)
        self.item_cf_confidence = tensor
        self.item_cf_confidence_normalized = tensor / max_val

    def set_item_text_quality(self, stats: torch.Tensor) -> None:
        tensor = stats.float().to(self.device)
        if tensor.numel() != self.n_items:
            raise ValueError("item text quality tensor must have length equal to n_items")
        max_val = tensor.max().clamp(min=1e-6)
        self.item_text_quality = tensor
        self.item_text_quality_normalized = tensor / max_val

    def set_item_recency(self, recency: torch.Tensor) -> None:
        tensor = recency.float().to(self.device)
        if tensor.numel() != self.n_items:
            raise ValueError("item recency tensor must have length equal to n_items")
        max_val = tensor.max().clamp(min=1e-6)
        self.item_recency = tensor
        self.item_recency_normalized = tensor / max_val

    def set_user_activity(self, activity_counts: torch.Tensor) -> None:
        """Store normalized user activity statistics for context-aware gating."""

        activity_tensor = activity_counts.float().to(self.device)
        if activity_tensor.numel() != self.n_users:
            raise ValueError("user activity tensor must have length equal to n_users")

        max_val = activity_tensor.max().clamp(min=1.0)
        self.user_activity_counts = activity_tensor
        self.user_activity_normalized = activity_tensor / max_val

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> torch.Tensor:
        batch_size = user_ids.size(0)

        user_emb = self.user_embedding(user_ids)
        item_cf_emb = self.item_embedding(item_ids)

        if text_emb is None:
            text_emb = torch.zeros(batch_size, self.text_dim, device=self.device)
        else:
            text_emb = text_emb.to(self.device)

        text_emb = self._apply_text_adapter(text_emb)
        item_text_emb = self.text_projection(text_emb)

        feature_inputs: List[torch.Tensor] = []
        for feat in self.features:
            if feat == "popularity":
                feature_inputs.append(self.popularity_normalized[item_ids].unsqueeze(-1))
            elif feat == "cf_confidence":
                feature_inputs.append(self.item_cf_confidence_normalized[item_ids].unsqueeze(-1))
            elif feat == "text_quality":
                feature_inputs.append(self.item_text_quality_normalized[item_ids].unsqueeze(-1))
            elif feat == "recency":
                feature_inputs.append(self.item_recency_normalized[item_ids].unsqueeze(-1))
            elif feat == "concat_embeddings":
                feature_inputs.append(torch.cat([item_cf_emb, item_text_emb], dim=-1))
            else:
                raise RuntimeError(f"Unsupported feature '{feat}'")

        if not feature_inputs:
            raise RuntimeError("No attention features configured")
        if len(feature_inputs) == 1:
            attn_input = feature_inputs[0]
        else:
            attn_input = torch.cat(feature_inputs, dim=-1)

        context_inputs: List[torch.Tensor] = []
        if self.has_user_activity:
            context_inputs.append(self.user_activity_normalized[user_ids].unsqueeze(-1))
        if context_inputs:
            context_tensor = torch.cat(context_inputs, dim=-1)
            attn_input = torch.cat([attn_input, context_tensor], dim=-1)

        lambda_pre = self.attention_net(attn_input)
        if self.temperature != 1.0:
            lambda_pre = lambda_pre / self.temperature
        lambda_raw = self.sigmoid(lambda_pre)
        lambda_values = lambda_raw
        if self.lambda_min is not None or self.lambda_max is not None:
            lambda_values = torch.clamp(
                lambda_values,
                min=self.lambda_min if self.lambda_min is not None else 0.0,
                max=self.lambda_max if self.lambda_max is not None else 1.0,
            )

        item_fused = item_cf_emb + lambda_values * item_text_emb
        scores = (user_emb * item_fused).sum(dim=-1)

        if return_details:
            details = {
                "lambda_values": lambda_values.squeeze(-1),
                "lambda_values_raw": lambda_raw.squeeze(-1),
                "lambda_pre": lambda_pre.squeeze(-1),
                "attn_input": attn_input,
                "user_emb": user_emb,
                "item_cf_emb": item_cf_emb,
                "item_text_emb": item_text_emb,
                "item_fused": item_fused,
            }
            return scores, lambda_values.squeeze(-1), details

        return scores

    @torch.no_grad()
    def predict(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        return self.forward(user_ids, item_ids, text_emb)

    @torch.no_grad()
    def get_learned_lambdas(
        self,
        item_ids: Optional[torch.Tensor] = None,
        user_ids: Optional[torch.Tensor] = None,
        text_embeddings: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Inspect learned λ for diagnostic plots."""

        self.eval()
        if item_ids is None:
            item_ids = torch.arange(self.n_items, device=self.device)
        else:
            item_ids = item_ids.to(self.device)

        if self.has_user_activity:
            if user_ids is None:
                raise ValueError("user_ids must be provided when using user activity context")
            user_ids = user_ids.to(self.device)
            if user_ids.shape != item_ids.shape:
                raise ValueError("user_ids must have the same shape as item_ids")
        else:
            if user_ids is None:
                user_ids = torch.zeros_like(item_ids)
            else:
                user_ids = user_ids.to(self.device)

        cf_emb = self.item_embedding(item_ids)
        proj_text = None
        feature_inputs: List[torch.Tensor] = []
        for feat in self.features:
            if feat == "popularity":
                feature_inputs.append(self.popularity_normalized[item_ids].unsqueeze(-1))
            elif feat == "cf_confidence":
                feature_inputs.append(self.item_cf_confidence_normalized[item_ids].unsqueeze(-1))
            elif feat == "text_quality":
                feature_inputs.append(self.item_text_quality_normalized[item_ids].unsqueeze(-1))
            elif feat == "recency":
                feature_inputs.append(self.item_recency_normalized[item_ids].unsqueeze(-1))
            elif feat == "concat_embeddings":
                if text_embeddings is None:
                    raise ValueError("text_embeddings must be provided for concat_embeddings feature")
                if proj_text is None:
                    text_emb = torch.stack(
                        [text_embeddings[int(idx.item())].to(self.device) for idx in item_ids]
                    )
                    text_emb = self._apply_text_adapter(text_emb)
                    proj_text = self.text_projection(text_emb)
                feature_inputs.append(torch.cat([cf_emb, proj_text], dim=-1))
            else:
                raise RuntimeError(f"Unsupported feature '{feat}'")

        if len(feature_inputs) == 1:
            attn_input = feature_inputs[0]
        else:
            attn_input = torch.cat(feature_inputs, dim=-1)

        context_inputs: List[torch.Tensor] = []
        if self.has_user_activity:
            context_inputs.append(self.user_activity_normalized[user_ids].unsqueeze(-1))
        if context_inputs:
            context_tensor = torch.cat(context_inputs, dim=-1)
            attn_input = torch.cat([attn_input, context_tensor], dim=-1)

        lambda_pre = self.attention_net(attn_input)
        if self.temperature != 1.0:
            lambda_pre = lambda_pre / self.temperature
        lambda_values = self.sigmoid(lambda_pre)
        if self.lambda_min is not None or self.lambda_max is not None:
            lambda_values = torch.clamp(
                lambda_values,
                min=self.lambda_min if self.lambda_min is not None else 0.0,
                max=self.lambda_max if self.lambda_max is not None else 1.0,
            )
        return lambda_values.squeeze(-1)


class FixedFusion(nn.Module):
    """Fixed λ fusion baseline."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        text_dim: int = 384,
        lambda_fixed: float = 0.5,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.text_dim = text_dim
        self.lambda_fixed = lambda_fixed
        self.device = torch.device(device)

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.to(self.device)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> torch.Tensor:
        batch_size = user_ids.size(0)

        user_emb = self.user_embedding(user_ids)
        item_cf_emb = self.item_embedding(item_ids)

        if text_emb is None:
            text_emb = torch.zeros(batch_size, self.text_dim, device=self.device)
        item_text_emb = self.text_projection(text_emb)

        item_fused = item_cf_emb + self.lambda_fixed * item_text_emb
        scores = (user_emb * item_fused).sum(dim=-1)

        if return_details:
            lambda_values = torch.full((batch_size,), self.lambda_fixed, device=self.device)
            details = {
                "lambda_values": lambda_values,
                "user_emb": user_emb,
                "item_cf_emb": item_cf_emb,
                "item_text_emb": item_text_emb,
                "item_fused": item_fused,
            }
            return scores, lambda_values, details

        return scores

    @torch.no_grad()
    def predict(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        return self.forward(user_ids, item_ids, text_emb)


class ConcatMLPFusion(nn.Module):
    """Non-gated fusion via MLP over [e_cf || e_text_proj], with residual to e_cf.

    Design goals:
    - Minimal extra params; stable training (LayerNorm + residual)
    - Stronger representational capacity than scalar gate
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        text_dim: int = 384,
        mlp_hidden: Iterable[int] = (32, 16),
        activation: str = "relu",
        dropout: float = 0.1,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.text_dim = text_dim
        self.device = torch.device(device)

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # Build small MLP mapping [e_cf || e_text_proj] -> delta in R^d
        act = _build_activation(activation)
        layers: List[nn.Module] = []
        prev = embedding_dim * 2
        for h in mlp_hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, embedding_dim))
        self.mlp = nn.Sequential(*layers)

        # Optional LayerNorm after fusion for stability
        self.post_norm = nn.LayerNorm(embedding_dim)

        self.to(self.device)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> torch.Tensor:
        batch_size = user_ids.size(0)

        user_emb = self.user_embedding(user_ids)
        item_cf_emb = self.item_embedding(item_ids)

        if text_emb is None:
            text_emb = torch.zeros(batch_size, self.text_dim, device=self.device)
        item_text_emb = self.text_projection(text_emb)

        mlp_in = torch.cat([item_cf_emb, item_text_emb], dim=-1)
        delta = self.mlp(mlp_in)
        item_fused = self.post_norm(item_cf_emb + delta)

        scores = (user_emb * item_fused).sum(dim=-1)

        if return_details:
            details = {
                "user_emb": user_emb,
                "item_cf_emb": item_cf_emb,
                "item_text_emb": item_text_emb,
                "delta": delta,
                "item_fused": item_fused,
            }
            return scores, None, details

        return scores

    @torch.no_grad()
    def predict(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        return self.forward(user_ids, item_ids, text_emb)


def compute_item_popularity(train_data, n_items: int) -> torch.Tensor:
    """Count item interactions for popularity-based gating."""

    popularity = torch.zeros(n_items)

    if hasattr(train_data, "values"):
        item_ids = train_data["item_idx"].values
        for item_id in item_ids:
            popularity[int(item_id)] += 1
    elif isinstance(train_data, list):
        for sample in train_data:
            item_id = sample[1] if isinstance(sample, (tuple, list)) else sample.item_id
            popularity[int(item_id)] += 1
    else:
        for sample in train_data:
            popularity[int(sample[1])] += 1

    return popularity


def compute_user_activity(train_data, n_users: int) -> torch.Tensor:
    """Count user interaction frequencies for context-aware gating."""

    activity = torch.zeros(n_users)

    if hasattr(train_data, "values"):
        user_ids = train_data["user_idx"].values
        for user_id in user_ids:
            activity[int(user_id)] += 1
    elif isinstance(train_data, list):
        for sample in train_data:
            user_id = sample[0] if isinstance(sample, (tuple, list)) else sample.user_id
            activity[int(user_id)] += 1
    else:
        for sample in train_data:
            activity[int(sample[0])] += 1

    return activity


def compute_item_cf_confidence(popularity: torch.Tensor, method: str = "log") -> torch.Tensor:
    """Derive item CF confidence proxy from popularity counts."""

    if method == "log":
        return torch.log1p(popularity.float())
    if method == "sqrt":
        return torch.sqrt(popularity.float().clamp(min=0.0) + 1.0)
    raise ValueError(f"Unsupported confidence method '{method}'")


def compute_item_text_quality(item_embeddings, n_items: int) -> torch.Tensor:
    """Compute L2 norms of precomputed text embeddings per item."""

    quality = torch.zeros(n_items, dtype=torch.float32)

    if hasattr(item_embeddings, "items"):
        iterator = item_embeddings.items()
    elif isinstance(item_embeddings, dict):
        iterator = item_embeddings.items()
    else:
        iterator = enumerate(item_embeddings)

    for item_id, emb in iterator:
        if emb is None:
            continue
        vec = torch.as_tensor(emb, dtype=torch.float32)
        if vec.ndim == 0:
            continue
        quality[int(item_id)] = vec.norm(p=2)

    return quality


def compute_item_recency(train_data, n_items: int) -> torch.Tensor:
    """Compute per-item recency score based on max timestamp/order."""

    recency = np.zeros(n_items, dtype=np.float32)

    item_ids: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None

    if hasattr(train_data, "columns"):
        df = train_data
        if "item_idx" in df.columns:
            item_ids = df["item_idx"].to_numpy(dtype=np.int64)
        ts_col = next((c for c in ["timestamp", "time", "ts", "review_time"] if c in df.columns), None)
        if ts_col is not None:
            timestamps = df[ts_col].to_numpy(dtype=np.float64)
    else:
        records = list(train_data)
        if records and isinstance(records[0], (tuple, list)) and len(records[0]) >= 2:
            item_ids = np.array([int(r[1]) for r in records], dtype=np.int64)
            timestamps = np.arange(len(records), dtype=np.float64)
        elif records and isinstance(records[0], dict):
            item_ids = np.array([int(r.get("item_idx", 0)) for r in records], dtype=np.int64)
            timestamps = np.array([float(r.get("timestamp", idx)) for idx, r in enumerate(records)], dtype=np.float64)

    if item_ids is None:
        return torch.from_numpy(recency)

    if timestamps is None:
        timestamps = np.linspace(0.0, 1.0, num=item_ids.shape[0], dtype=np.float64)

    np.maximum.at(recency, item_ids, timestamps)

    max_val = np.max(recency)
    min_val = np.min(recency)
    if max_val > min_val:
        recency = (recency - min_val) / (max_val - min_val + 1e-8)
    else:
        recency.fill(0.0)

    return torch.from_numpy(recency)


class CrossAttentionFusion(nn.Module):
    """
    Semantic-Adaptive Cross-Attention Fusion (SACAF)
    
    New method for RecSys 2026 submission - addresses reviewer feedback:
    - Uses cross-attention between CF and Text embeddings (not just metadata)
    - Learns semantic relevance rather than relying on simple popularity signal
    - More principled than popularity-gated fusion
    
    Architecture:
        - CF embedding as Query (user behavior context)
        - Text embedding as Key/Value (semantic information)
        - Attention score measures semantic relevance
        - Adaptive gate based on attention + learnable MLP
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        text_dim: int = 4096,  # For E5-Mistral-7B (4096), or 384 for MiniLM
        hidden_dim: int = 128,
        gate_hidden: Iterable[int] = (64, 32),
        activation: str = "relu",
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_item_popularity: bool = False,  # Optional: include popularity as auxiliary feature
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.temperature = max(temperature, 1e-3)
        self.use_item_popularity = use_item_popularity
        self.device = torch.device(device)

        # Collaborative filtering embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Text projection to embedding space (with LayerNorm for stability)
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # Cross-attention components
        # Query: project CF embedding to hidden space
        self.W_q = nn.Linear(embedding_dim, hidden_dim, bias=False)
        
        # Key/Value: project text embedding to hidden space
        self.W_k = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(embedding_dim, hidden_dim, bias=False)
        
        # Scaling factor for attention (similar to Transformer)
        self.scale = hidden_dim ** -0.5
        
        # Adaptive gate network
        # Input: [attention_score, Q, V, optional_popularity]
        gate_input_dim = hidden_dim * 2 + 1  # Q, V, attention_score
        if use_item_popularity:
            gate_input_dim += 1  # Add popularity feature
            self.register_buffer("item_popularity", torch.zeros(n_items))
            self.register_buffer("popularity_normalized", torch.zeros(n_items))
        
        # Build gate MLP
        act = _build_activation(activation)
        gate_layers: List[nn.Module] = []
        prev_dim = gate_input_dim
        for h in gate_hidden:
            gate_layers.append(nn.Linear(prev_dim, h))
            gate_layers.append(act)
            if dropout > 0:
                gate_layers.append(nn.Dropout(dropout))
            prev_dim = h
        gate_layers.append(nn.Linear(prev_dim, 1))
        gate_layers.append(nn.Sigmoid())
        
        self.gate_net = nn.Sequential(*gate_layers)
        
        # Value projection back to embedding space
        self.W_out = nn.Linear(hidden_dim, embedding_dim)
        
        # Post-fusion normalization for stability
        self.post_norm = nn.LayerNorm(embedding_dim)
        
        self.to(self.device)

    def set_item_popularity(self, popularity_counts: torch.Tensor) -> None:
        """Store normalized popularity (optional auxiliary feature)."""
        if not self.use_item_popularity:
            return
        
        pop_tensor = popularity_counts.float().to(self.device)
        self.item_popularity = pop_tensor
        max_pop = pop_tensor.max().clamp(min=1.0)
        self.popularity_normalized = pop_tensor / max_pop

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with cross-attention fusion.
        
        Args:
            user_ids: [batch_size]
            item_ids: [batch_size]
            text_emb: [batch_size, text_dim] - raw text embeddings
            return_details: if True, return detailed diagnostics
        
        Returns:
            scores: [batch_size] - predicted scores
            (optional) lambda_values: [batch_size] - fusion weights
            (optional) details: dict with intermediate values
        """
        batch_size = user_ids.size(0)

        # Get embeddings
        user_emb = self.user_embedding(user_ids)  # [B, D]
        item_cf_emb = self.item_embedding(item_ids)  # [B, D]

        # Handle missing text embeddings
        if text_emb is None:
            text_emb = torch.zeros(batch_size, self.text_dim, device=self.device)
        else:
            text_emb = text_emb.to(self.device)

        # Project text to embedding space
        item_text_emb = self.text_projection(text_emb)  # [B, D]

        # Cross-attention mechanism
        # Query from CF (user behavior context)
        Q = self.W_q(item_cf_emb)  # [B, H]
        
        # Key/Value from Text (semantic information)
        K = self.W_k(item_text_emb)  # [B, H]
        V = self.W_v(item_text_emb)  # [B, H]
        
        # Compute attention scores (semantic relevance)
        # Higher score = text embedding is more relevant to CF context
        attention_score = torch.sum(Q * K, dim=1, keepdim=True) * self.scale  # [B, 1]
        
        # Apply temperature scaling (optional, for controlling sharpness)
        if self.temperature != 1.0:
            attention_score = attention_score / self.temperature
        
        # Softmax not needed here since we're doing element-wise fusion
        # But we normalize for interpretability
        attention_weight = torch.sigmoid(attention_score)  # [B, 1]
        
        # Attended text representation
        attended_text = attention_weight * V  # [B, H]
        
        # Adaptive gate: decide how much to use text vs CF
        # Gate input: [attention_score, Q, V, optional_popularity]
        gate_inputs = [attention_score, Q, V]
        
        if self.use_item_popularity:
            pop_feature = self.popularity_normalized[item_ids].unsqueeze(-1)  # [B, 1]
            gate_inputs.append(pop_feature)
        
        gate_input = torch.cat(gate_inputs, dim=-1)  # [B, gate_input_dim]
        lambda_values = self.gate_net(gate_input)  # [B, 1]
        
        # Project attended text back to embedding space
        text_contribution = self.W_out(attended_text)  # [B, D]
        
        # Fusion: weighted combination
        # item_fused = CF_emb + λ * attended_text_emb
        item_fused = item_cf_emb + lambda_values * text_contribution
        
        # Post-normalization for stability
        item_fused = self.post_norm(item_fused)
        
        # Final score: dot product with user embedding
        scores = (user_emb * item_fused).sum(dim=-1)  # [B]

        if return_details:
            details = {
                "lambda_values": lambda_values.squeeze(-1),
                "attention_score": attention_score.squeeze(-1),
                "attention_weight": attention_weight.squeeze(-1),
                "user_emb": user_emb,
                "item_cf_emb": item_cf_emb,
                "item_text_emb": item_text_emb,
                "Q": Q,
                "K": K,
                "V": V,
                "attended_text": attended_text,
                "text_contribution": text_contribution,
                "item_fused": item_fused,
            }
            return scores, lambda_values.squeeze(-1), details

        return scores

    @torch.no_grad()
    def predict(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inference mode prediction."""
        self.eval()
        return self.forward(user_ids, item_ids, text_emb)

    @torch.no_grad()
    def get_attention_weights(
        self,
        item_ids: torch.Tensor,
        text_embeddings: Dict[int, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Diagnostic method: extract attention weights and lambda values for analysis.
        
        Args:
            item_ids: [N] item indices
            text_embeddings: dict mapping item_id -> text embedding
        
        Returns:
            attention_weights: [N] - semantic relevance scores
            lambda_values: [N] - fusion weights
        """
        self.eval()
        
        item_ids = item_ids.to(self.device)
        batch_size = item_ids.size(0)
        
        # Get CF embeddings
        item_cf_emb = self.item_embedding(item_ids)  # [N, D]
        
        # Get text embeddings
        text_emb = torch.stack([
            text_embeddings[int(idx.item())].to(self.device) 
            for idx in item_ids
        ])  # [N, text_dim]
        
        item_text_emb = self.text_projection(text_emb)  # [N, D]
        
        # Compute attention
        Q = self.W_q(item_cf_emb)
        K = self.W_k(item_text_emb)
        V = self.W_v(item_text_emb)
        
        attention_score = torch.sum(Q * K, dim=1, keepdim=True) * self.scale
        if self.temperature != 1.0:
            attention_score = attention_score / self.temperature
        attention_weight = torch.sigmoid(attention_score)
        
        # Compute lambda
        gate_inputs = [attention_score, Q, V]
        if self.use_item_popularity:
            pop_feature = self.popularity_normalized[item_ids].unsqueeze(-1)
            gate_inputs.append(pop_feature)
        
        gate_input = torch.cat(gate_inputs, dim=-1)
        lambda_values = self.gate_net(gate_input)
        
        return attention_weight.squeeze(-1), lambda_values.squeeze(-1)
