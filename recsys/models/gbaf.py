"""Gradient-Balanced Adaptive Fusion (GBAF) models."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from .attention_variants import AttentionFusion

class GBAFFusion(AttentionFusion):
    """
    Gradient-Balanced Adaptive Fusion (GBAF).
    
    Components:
    1. Confidence-Aware Conditioning: MLP([p_i, c_i, p_i * c_i]) -> lambda_i
    2. Gradient Normalization: (Handled in training loop)
    3. Regularized Gating: (Handled in training loop via L_balance)
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        text_dim: int = 384,
        hidden_sizes: Tuple[int, ...] = (32, 16),
        device: str = "cpu",
    ) -> None:
        # Parent provides embeddings/projection and feature buffers.
        super().__init__(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            text_dim=text_dim,
            attention={
                "input_mode": "popularity",
                "features": ["popularity", "cf_confidence"],
                "hidden_sizes": hidden_sizes
            },
            device=device
        )
        
        # Strict GBAF gate: MLP([p_i, c_i, p_i*c_i]) -> lambda_i
        layers = []
        input_dim = 3
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        
        self.attention_net = nn.Sequential(*layers).to(self.device)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> torch.Tensor:
        batch_size = user_ids.size(0)

        # 1) Embeddings
        user_emb = self.user_embedding(user_ids)
        item_cf_emb = self.item_embedding(item_ids)

        if text_emb is None:
            text_emb = torch.zeros(batch_size, self.text_dim, device=self.device)
        else:
            text_emb = text_emb.to(self.device)

        text_emb = self._apply_text_adapter(text_emb)
        item_text_emb = self.text_projection(text_emb)

        # 2) GBAF gate inputs
        p_i = self.popularity_normalized[item_ids].unsqueeze(-1)  # [B, 1]
        c_i = self.item_cf_confidence_normalized[item_ids].unsqueeze(-1)  # [B, 1]
        interaction = p_i * c_i  # [B, 1]
        gbaf_input = torch.cat([p_i, c_i, interaction], dim=-1)

        lambda_pre = self.attention_net(gbaf_input)
        lambda_values = self.sigmoid(lambda_pre)

        # Apply range constraint if configured (from parent config)
        if self.lambda_min is not None or self.lambda_max is not None:
            lambda_values = torch.clamp(
                lambda_values,
                min=self.lambda_min if self.lambda_min is not None else 0.0,
                max=self.lambda_max if self.lambda_max is not None else 1.0,
            )

        # 3) Fusion
        item_fused = item_cf_emb + lambda_values * item_text_emb
        scores = (user_emb * item_fused).sum(dim=-1)

        if return_details:
            details = {
                "lambda_values": lambda_values.squeeze(-1),
                # Keep unclamped pre-sigmoid logits for diagnostics when needed.
                "lambda_values_raw": lambda_pre.squeeze(-1),
                "user_emb": user_emb,
                "item_cf_emb": item_cf_emb,
                "item_text_emb": item_text_emb,
                "item_fused": item_fused,
                "p_i": p_i,
                "c_i": c_i
            }
            return scores, lambda_values.squeeze(-1), details

        return scores

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

        # 1) GBAF conditioning
        p_i = self.popularity_normalized[item_ids].unsqueeze(-1)
        c_i = self.item_cf_confidence_normalized[item_ids].unsqueeze(-1)
        interaction = p_i * c_i

        gbaf_input = torch.cat([p_i, c_i, interaction], dim=-1)

        lambda_pre = self.attention_net(gbaf_input)
        lambda_values = self.sigmoid(lambda_pre)

        if self.lambda_min is not None or self.lambda_max is not None:
            lambda_values = torch.clamp(
                lambda_values,
                min=self.lambda_min if self.lambda_min is not None else 0.0,
                max=self.lambda_max if self.lambda_max is not None else 1.0,
            )
        return lambda_values.squeeze(-1)


class GBAFv2Fusion(GBAFFusion):
    """
    GBAF-v2:
    - Temperature-annealed gating
    - User-aware confidence conditioning
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        text_dim: int = 384,
        hidden_sizes: Tuple[int, ...] = (32, 16),
        tau_start: float = 5.0,
        tau_end: float = 1.0,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            text_dim=text_dim,
            hidden_sizes=hidden_sizes,
            device=device,
        )

        # Rebuild gate with user-aware features:
        # [p_i, c_i, p_i*c_i, a_u, a_u*p_i, a_u*c_i]
        layers = []
        input_dim = 6
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.attention_net = nn.Sequential(*layers).to(self.device)

        self.tau_start = max(float(tau_start), 1e-3)
        self.tau_end = max(float(tau_end), 1e-3)
        self.current_tau = self.tau_start

        # Expose user context for statistics setup in trainer.
        self.context_features = ("user_activity",)
        self.has_user_activity = True

    def set_training_progress(self, epoch: int, total_epochs: int) -> None:
        if total_epochs <= 1:
            self.current_tau = self.tau_end
            return
        progress = min(max(float(epoch) / float(total_epochs - 1), 0.0), 1.0)
        self.current_tau = self.tau_start + (self.tau_end - self.tau_start) * progress
        self.current_tau = max(self.current_tau, 1e-3)

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

        p_i = self.popularity_normalized[item_ids].unsqueeze(-1)
        c_i = self.item_cf_confidence_normalized[item_ids].unsqueeze(-1)
        a_u = self.user_activity_normalized[user_ids].unsqueeze(-1)
        gbaf_input = torch.cat([p_i, c_i, p_i * c_i, a_u, a_u * p_i, a_u * c_i], dim=-1)

        lambda_pre = self.attention_net(gbaf_input)
        lambda_values = torch.sigmoid(lambda_pre / self.current_tau)

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
                "lambda_values_raw": lambda_pre.squeeze(-1),
                "user_emb": user_emb,
                "item_cf_emb": item_cf_emb,
                "item_text_emb": item_text_emb,
                "item_fused": item_fused,
                "p_i": p_i,
                "c_i": c_i,
                "a_u": a_u,
                "tau": torch.full_like(lambda_values.squeeze(-1), self.current_tau),
            }
            return scores, lambda_values.squeeze(-1), details
        return scores


class GBAFAdaptiveFusion(GBAFFusion):
    """
    Route-C incremental variant:
    - Keep original 3-dim gate input
    - Use adaptive regularization in training loop (no user-aware gate / no temperature annealing)
    """

    pass
