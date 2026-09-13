"""
训练脚本 - Attention vs Fixed Fusion
PAKDD 2026 补充实验
"""

import os
import sys
import yaml
import pickle
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from models import (
    AttentionFusion,
    FixedFusion,
    ConcatMLPFusion,
    CrossAttentionFusion,
    GBAFFusion,
    GBAFAdaptiveFusion,
    GBAFv2Fusion,
    compute_item_popularity,
    compute_user_activity,
    compute_item_cf_confidence,
    compute_item_text_quality,
    compute_item_recency,
)


def _branch_parameters(model):
    branches = {
        'cf': [model.user_embedding.weight, model.item_embedding.weight],
        'text': list(model.text_projection.parameters()),
    }
    if isinstance(model, AttentionFusion):
        branches['attention'] = list(model.attention_net.parameters())
    return branches


def _compute_grad_norms(branch_params):
    norms = {}
    for name, params in branch_params.items():
        total = 0.0
        for param in params:
            if param.grad is None:
                continue
            total += param.grad.detach().pow(2).sum().item()
        norms[name] = total ** 0.5 if total > 0 else 0.0
    return norms


def _pcgrad_between_branches(branch_params, target_ratio=1.0, eps=1e-8, max_scale=5.0):
    """
    Branch-level PCGrad proxy for disjoint parameter groups.
    Since CF/text branches do not share tensors, we apply conflict mitigation by
    scaling only the dominant branch when ratio deviates from target_ratio.
    """
    grad_norms = _compute_grad_norms(branch_params)
    cf_norm = grad_norms.get('cf', 0.0)
    text_norm = grad_norms.get('text', 0.0)
    if cf_norm <= 0 or text_norm <= 0:
        return None

    ratio = text_norm / (cf_norm + eps)
    desired = max(float(target_ratio), eps)
    if ratio <= desired:
        return {
            'pcgrad_ratio': float(ratio),
            'pcgrad_scale': 1.0,
            'pcgrad_adjusted': 0.0,
        }

    scale = desired / (ratio + eps)
    if max_scale > 0:
        scale = min(scale, max_scale)
    scale = max(scale, 0.0)
    for param in branch_params.get('text', []):
        if param.grad is not None:
            param.grad.mul_(scale)
    return {
        'pcgrad_ratio': float(ratio),
        'pcgrad_scale': float(scale),
        'pcgrad_adjusted': 1.0,
    }


class RecommendationDataset(Dataset):
    """推荐数据集"""
    
    def __init__(self, interactions, text_embeddings, n_items, negative_sampling=4, text_dim=384):
        # 如果interactions是DataFrame，提取(user_idx, item_idx)
        if hasattr(interactions, 'values'):  # pandas DataFrame
            self.interactions = interactions[['user_idx', 'item_idx']].values  # shape: (N, 2)
        else:
            self.interactions = interactions
        
        self.text_embeddings = text_embeddings  # {item_id: embedding}
        self.n_items = n_items
        self.negative_sampling = negative_sampling
        self.text_dim = text_dim
        
        # 构建user的positive items集合
        self.user_pos_items = {}
        for user_id, item_id in self.interactions:
            if user_id not in self.user_pos_items:
                self.user_pos_items[user_id] = set()
            self.user_pos_items[user_id].add(item_id)
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        user_id, pos_item_id = self.interactions[idx]
        
        # 负采样
        neg_items = []
        user_pos = self.user_pos_items[user_id]
        
        while len(neg_items) < self.negative_sampling:
            neg_item = np.random.randint(0, self.n_items)
            if neg_item not in user_pos:
                neg_items.append(neg_item)
        
        # 获取text embeddings
        pos_text = self.text_embeddings.get(pos_item_id, torch.zeros(self.text_dim))
        neg_texts = [self.text_embeddings.get(neg_id, torch.zeros(self.text_dim)) for neg_id in neg_items]
        
        return {
            'user_id': int(user_id),
            'pos_item': int(pos_item_id),
            'neg_items': torch.tensor(neg_items, dtype=torch.long),  # 转换为tensor
            'pos_text': pos_text,
            'neg_texts': torch.stack(neg_texts)
        }


def load_data(data_dir, config, config_dir: Path):
    """加载MovieLens-1M数据"""
    print(f"📂 Loading data from {data_dir}...")
    
    data_dir_path = Path(data_dir)
    if not data_dir_path.is_absolute():
        data_dir_path = (config_dir / data_dir_path).resolve()

    with open(data_dir_path / "train.pkl", 'rb') as f:
        train_data = pickle.load(f)
    
    with open(data_dir_path / "val.pkl", 'rb') as f:
        val_data = pickle.load(f)
    
    with open(data_dir_path / "test.pkl", 'rb') as f:
        test_data = pickle.load(f)
    
    # 从配置文件加载embeddings文件
    embedding_file = config['text_encoder']['embedding_file']
    embedding_path = Path(embedding_file)
    if not embedding_path.is_absolute():
        embedding_path = (config_dir / embedding_path).resolve()
    print(f"   Loading embeddings from: {embedding_path}")
    with open(embedding_path, 'rb') as f:
        item_embeddings = pickle.load(f)
    for k, v in item_embeddings.items():
        if not isinstance(v, torch.Tensor):
            item_embeddings[k] = torch.tensor(v, dtype=torch.float32)
    
    with open(data_dir_path / "stats.json", 'r') as f:
        stats = json.load(f)
    
    print(f"✅ Data loaded:")
    print(f"   Train: {len(train_data)} interactions")
    print(f"   Val: {len(val_data)} interactions")
    print(f"   Test: {len(test_data)} interactions")
    print(f"   Users: {stats['n_users']}, Items: {stats['n_items']}")
    print(f"   Embeddings: {len(item_embeddings)} items")
    
    return train_data, val_data, test_data, item_embeddings, stats


class RecommendationDataset(Dataset):
    """推荐数据集"""
    
    def __init__(self, interactions, text_embeddings, n_items, negative_sampling=4, text_dim=384):
        # 如果interactions是DataFrame，提取(user_idx, item_idx)
        if hasattr(interactions, 'values'):  # pandas DataFrame
            self.interactions = interactions[['user_idx', 'item_idx']].values  # shape: (N, 2)
        else:
            self.interactions = interactions
        
        self.text_embeddings = text_embeddings  # {item_id: embedding}
        self.n_items = n_items
        self.negative_sampling = negative_sampling
        self.text_dim = text_dim
        
        # 构建user的positive items集合
        self.user_pos_items = {}
        for user_id, item_id in self.interactions:
            if user_id not in self.user_pos_items:
                self.user_pos_items[user_id] = set()
            self.user_pos_items[user_id].add(item_id)
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        user_id, pos_item_id = self.interactions[idx]
        
        # 负采样
        neg_items = []
        user_pos = self.user_pos_items[user_id]
        
        while len(neg_items) < self.negative_sampling:
            neg_item = np.random.randint(0, self.n_items)
            if neg_item not in user_pos:
                neg_items.append(neg_item)
        
        # 获取text embeddings
        pos_text = self.text_embeddings.get(pos_item_id, torch.zeros(self.text_dim))
        neg_texts = [self.text_embeddings.get(neg_id, torch.zeros(self.text_dim)) for neg_id in neg_items]
        
        return {
            'user_id': int(user_id),
            'pos_item': int(pos_item_id),
            'neg_items': torch.tensor(neg_items, dtype=torch.long),  # 转换为tensor
            'pos_text': pos_text,
            'neg_texts': torch.stack(neg_texts)
        }


def load_data(data_dir, config, config_dir: Path):
    """加载MovieLens-1M数据"""
    print(f"📂 Loading data from {data_dir}...")
    
    data_dir_path = Path(data_dir)
    if not data_dir_path.is_absolute():
        data_dir_path = (config_dir / data_dir_path).resolve()

    with open(data_dir_path / "train.pkl", 'rb') as f:
        train_data = pickle.load(f)
    
    with open(data_dir_path / "val.pkl", 'rb') as f:
        val_data = pickle.load(f)
    
    with open(data_dir_path / "test.pkl", 'rb') as f:
        test_data = pickle.load(f)
    
    # 从配置文件加载embeddings文件
    embedding_file = config['text_encoder']['embedding_file']
    embedding_path = Path(embedding_file)
    if not embedding_path.is_absolute():
        embedding_path = (config_dir / embedding_path).resolve()
    print(f"   Loading embeddings from: {embedding_path}")
    with open(embedding_path, 'rb') as f:
        item_embeddings = pickle.load(f)
    for k, v in item_embeddings.items():
        if not isinstance(v, torch.Tensor):
            item_embeddings[k] = torch.tensor(v, dtype=torch.float32)
    
    with open(data_dir_path / "stats.json", 'r') as f:
        stats = json.load(f)
    
    print(f"✅ Data loaded:")
    print(f"   Train: {len(train_data)} interactions")
    print(f"   Val: {len(val_data)} interactions")
    print(f"   Test: {len(test_data)} interactions")
    print(f"   Users: {stats['n_users']}, Items: {stats['n_items']}")
    print(f"   Embeddings: {len(item_embeddings)} items")
    
    return train_data, val_data, test_data, item_embeddings, stats


def build_model(config, device):
    """构建模型"""
    model_type = config['model']['type']
    n_users = config['model']['n_users']
    n_items = config['model']['n_items']
    embedding_dim = config['model']['embedding_dim']
    text_dim = config['model']['text_dim']
    
    print(f"\n🔨 Building {model_type} model...")
    
    if model_type == 'attention_fusion':
        attention_cfg = config['model'].get('attention', {})
        legacy_dropout = config['model'].get('dropout')
        if legacy_dropout is not None and 'dropout' not in attention_cfg:
            attention_cfg = dict(attention_cfg)
            attention_cfg['dropout'] = legacy_dropout

        model = AttentionFusion(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            text_dim=text_dim,
            attention=attention_cfg,
            text_adapter=config['model'].get('text_adapter'),
            device=device
        )
        print(f"   Attention input mode: {model.input_mode}")
    elif model_type == 'fixed_fusion':
        model = FixedFusion(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            text_dim=text_dim,
            lambda_fixed=config['model']['lambda_fixed'],
            device=device
        )
    elif model_type == 'gbaf_fusion':
        hidden_sizes = config['model'].get('hidden_sizes', [32, 16])
        model = GBAFFusion(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            text_dim=text_dim,
            hidden_sizes=hidden_sizes,
            device=device
        )
    elif model_type in {'gbaf_adaptive_fusion', 'gbaf_adaptive'}:
        hidden_sizes = config['model'].get('hidden_sizes', [32, 16])
        model = GBAFAdaptiveFusion(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            text_dim=text_dim,
            hidden_sizes=hidden_sizes,
            device=device
        )
    elif model_type in {'gbaf_v2_fusion', 'gbafv2_fusion'}:
        hidden_sizes = config['model'].get('hidden_sizes', [32, 16])
        tau_cfg = config['model'].get('temperature', {}) or {}
        model = GBAFv2Fusion(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            text_dim=text_dim,
            hidden_sizes=hidden_sizes,
            tau_start=float(tau_cfg.get('tau_start', 5.0)),
            tau_end=float(tau_cfg.get('tau_end', 1.0)),
            device=device
        )
    elif model_type == 'concat_mlp_fusion':
        mlp_cfg = config['model'].get('mlp', {}) or {}
        model = ConcatMLPFusion(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            text_dim=text_dim,
            mlp_hidden=mlp_cfg.get('hidden_sizes', [32, 16]),
            activation=mlp_cfg.get('activation', 'relu'),
            dropout=float(mlp_cfg.get('dropout', 0.1)),
            device=device,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def build_model(config, device):
    """构建模型"""
    model_type = config['model']['type']
    n_users = config['model']['n_users']
    n_items = config['model']['n_items']
    embedding_dim = config['model']['embedding_dim']
    text_dim = config['model']['text_dim']
    
    print(f"\n🔨 Building {model_type} model...")
    
    if model_type == 'attention_fusion':
        attention_cfg = config['model'].get('attention', {})
        legacy_dropout = config['model'].get('dropout')
        if legacy_dropout is not None and 'dropout' not in attention_cfg:
            attention_cfg = dict(attention_cfg)
            attention_cfg['dropout'] = legacy_dropout

        model = AttentionFusion(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            text_dim=text_dim,
            attention=attention_cfg,
            text_adapter=config['model'].get('text_adapter'),
            device=device
        )
        print(f"   Attention input mode: {model.input_mode}")
    elif model_type == 'fixed_fusion':
        model = FixedFusion(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            text_dim=text_dim,
            lambda_fixed=config['model']['lambda_fixed'],
            device=device
        )
    elif model_type == 'gbaf_fusion':
        hidden_sizes = config['model'].get('hidden_sizes', [32, 16])
        model = GBAFFusion(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            text_dim=text_dim,
            hidden_sizes=hidden_sizes,
            device=device
        )
    elif model_type in {'gbaf_adaptive_fusion', 'gbaf_adaptive'}:
        hidden_sizes = config['model'].get('hidden_sizes', [32, 16])
        model = GBAFAdaptiveFusion(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            text_dim=text_dim,
            hidden_sizes=hidden_sizes,
            device=device,
        )
    elif model_type in {'gbaf_v2_fusion', 'gbafv2_fusion'}:
        hidden_sizes = config['model'].get('hidden_sizes', [32, 16])
        tau_cfg = config['model'].get('temperature', {}) or {}
        model = GBAFv2Fusion(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            text_dim=text_dim,
            hidden_sizes=hidden_sizes,
            tau_start=float(tau_cfg.get('tau_start', 5.0)),
            tau_end=float(tau_cfg.get('tau_end', 1.0)),
            device=device,
        )
    elif model_type == 'concat_mlp_fusion':
        mlp_cfg = config['model'].get('mlp', {}) or {}
        model = ConcatMLPFusion(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            text_dim=text_dim,
            mlp_hidden=mlp_cfg.get('hidden_sizes', [32, 16]),
            activation=mlp_cfg.get('activation', 'relu'),
            dropout=float(mlp_cfg.get('dropout', 0.1)),
            device=device,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def train_epoch(
    model,
    train_loader,
    optimizer,
    device,
    epoch,
    lambda_reg_cfg=None,
    grad_balance_cfg=None,
    total_epochs=None,
):
    """训练一个epoch（含梯度平衡、动态正则和温度调度）"""
    model.train()
    total_loss = 0.0
    lambda_reg_total = 0.0

    lambda_alpha = float((lambda_reg_cfg or {}).get('lambda_alpha', 0.0))
    lambda_beta = float((lambda_reg_cfg or {}).get('lambda_beta', 0.0))
    lambda_penalty_weight = float((lambda_reg_cfg or {}).get('boundary_penalty', 0.0))
    lambda_margin = float((lambda_reg_cfg or {}).get('boundary_margin', 0.0))
    adaptive_alpha = bool((lambda_reg_cfg or {}).get('adaptive_alpha', False))
    alpha_base = float((lambda_reg_cfg or {}).get('alpha_base', lambda_alpha))
    alpha_max_scale = float((lambda_reg_cfg or {}).get('alpha_max_scale', 3.0))

    apply_lambda_reg = isinstance(model, AttentionFusion) and (lambda_alpha > 0 or lambda_beta > 0)
    reg_enabled = isinstance(model, AttentionFusion) and (
        lambda_alpha > 0 or lambda_beta > 0 or lambda_penalty_weight > 0
    )

    grad_cfg = grad_balance_cfg or {}
    grad_method = str(grad_cfg.get('method', 'none')).lower()
    log_gradients = bool(grad_cfg.get('log_grads', False))
    target_ratio = float(grad_cfg.get('target_ratio', 1.0))
    max_scale = float(grad_cfg.get('max_scale', 5.0))
    min_scale = float(grad_cfg.get('min_scale', 0.0))
    scale_attention = bool(grad_cfg.get('scale_attention', False))
    log_lambdas = bool(grad_cfg.get('log_lambdas', False))
    grad_eps = float(grad_cfg.get('eps', 1e-8))
    running_ratio = max(target_ratio, grad_eps)

    if hasattr(model, "set_training_progress") and total_epochs is not None:
        model.set_training_progress(epoch=epoch, total_epochs=total_epochs)

    branch_params = _branch_parameters(model) if (log_gradients or grad_method != 'none') else None
    grad_metrics = {} if (log_gradients or grad_method != 'none' or log_lambdas) else None
    need_lambda_details = isinstance(model, AttentionFusion) and (reg_enabled or log_lambdas or lambda_penalty_weight > 0)

    for batch_idx, batch in enumerate(train_loader):
        user_ids = batch['user_id'].to(device)
        pos_items = batch['pos_item'].to(device)
        neg_items = batch['neg_items'].to(device)
        pos_texts = batch['pos_text'].to(device)
        neg_texts = batch['neg_texts'].to(device)
        batch_size, n_neg = neg_items.shape

        if need_lambda_details:
            pos_scores, pos_lambda, pos_details = model(user_ids, pos_items, pos_texts, return_details=True)
        else:
            pos_scores = model(user_ids, pos_items, pos_texts)
            pos_lambda, pos_details = None, None

        user_ids_expanded = user_ids.unsqueeze(1).expand(-1, n_neg).reshape(-1)
        neg_items_flat = neg_items.reshape(-1)
        neg_texts_flat = neg_texts.reshape(-1, neg_texts.size(-1))
        if need_lambda_details:
            neg_scores_flat, neg_lambda_flat, neg_details = model(
                user_ids_expanded, neg_items_flat, neg_texts_flat, return_details=True
            )
            neg_scores = neg_scores_flat.reshape(batch_size, n_neg)
            neg_lambda = neg_lambda_flat.reshape(batch_size, n_neg)
        else:
            neg_scores = model(user_ids_expanded, neg_items_flat, neg_texts_flat).reshape(batch_size, n_neg)
            neg_lambda, neg_details = None, None

        bpr_loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-10).mean()

        lambda_values = None
        lambda_raw_values = None
        if need_lambda_details:
            lambda_chunks = []
            raw_chunks = []
            if pos_lambda is not None:
                lambda_chunks.append(pos_lambda.view(-1))
            if neg_lambda is not None:
                lambda_chunks.append(neg_lambda.reshape(-1))
            if pos_details and "lambda_values_raw" in pos_details:
                raw_chunks.append(pos_details["lambda_values_raw"].view(-1))
            if neg_details and "lambda_values_raw" in neg_details:
                raw_chunks.append(neg_details["lambda_values_raw"].view(-1))
            if lambda_chunks:
                lambda_values = torch.cat(lambda_chunks)
            if raw_chunks:
                lambda_raw_values = torch.cat(raw_chunks)

        reg_loss = torch.tensor(0.0, device=device)
        lambda_alpha_t = lambda_alpha
        if apply_lambda_reg and adaptive_alpha and alpha_base > 0:
            ratio = max(float(running_ratio), grad_eps)
            target = max(float(target_ratio), grad_eps)
            deviation = max(ratio / target, target / ratio)
            ratio_scale = min(max(deviation, 1.0), alpha_max_scale)
            lambda_alpha_t = alpha_base * ratio_scale

        if apply_lambda_reg and lambda_values is not None:
            if lambda_alpha_t > 0:
                reg_loss = reg_loss + lambda_alpha_t * ((lambda_values - 0.5) ** 2).mean()
            if lambda_beta > 0:
                entropy = -(
                    lambda_values * torch.log(lambda_values + 1e-8)
                    + (1 - lambda_values) * torch.log(1 - lambda_values + 1e-8)
                )
                reg_loss = reg_loss + lambda_beta * entropy.mean()

        if lambda_penalty_weight > 0 and lambda_values is not None:
            boundary_loss = torch.tensor(0.0, device=device)
            lambda_min = getattr(model, "lambda_min", None)
            lambda_max = getattr(model, "lambda_max", None)
            if lambda_min is not None:
                lower = lambda_min + lambda_margin
                boundary_loss = boundary_loss + ((lower - lambda_values).clamp(min=0) ** 2).mean()
            if lambda_max is not None:
                upper = lambda_max - lambda_margin
                boundary_loss = boundary_loss + ((lambda_values - upper).clamp(min=0) ** 2).mean()
            reg_loss = reg_loss + lambda_penalty_weight * boundary_loss

        if grad_metrics is not None and log_lambdas and lambda_values is not None and lambda_values.numel() > 0:
            grad_metrics.setdefault('lambda_mean', []).append(float(lambda_values.mean().detach().cpu()))
            grad_metrics.setdefault('lambda_std', []).append(float(lambda_values.std(unbiased=False).detach().cpu()))
            if lambda_raw_values is not None and lambda_raw_values.numel() > 0:
                grad_metrics.setdefault('lambda_logit_mean', []).append(float(lambda_raw_values.mean().detach().cpu()))

        total_batch_loss = bpr_loss + reg_loss
        optimizer.zero_grad()
        total_batch_loss.backward()

        if branch_params is not None:
            grad_norms = _compute_grad_norms(branch_params)
            if grad_metrics is not None and log_gradients:
                for name, value in grad_norms.items():
                    grad_metrics.setdefault(name, []).append(value)

            if grad_method in {'gradnorm', 'ratio', 'independent', 'pcgrad'}:
                cf_norm = grad_norms.get('cf', 0.0)
                text_norm = grad_norms.get('text', 0.0)
                running_ratio = text_norm / (cf_norm + grad_eps) if text_norm > 0 else 0.0
                if grad_metrics is not None:
                    grad_metrics.setdefault('ratio_text_cf', []).append(float(running_ratio))
                scale = None

                if grad_method == 'independent':
                    if cf_norm > 0:
                        cf_scale = 1.0 / (cf_norm + grad_eps)
                        for param in branch_params.get('cf', []):
                            if param.grad is not None:
                                param.grad.mul_(cf_scale)
                    if text_norm > 0:
                        text_scale = 1.0 / (text_norm + grad_eps)
                        for param in branch_params.get('text', []):
                            if param.grad is not None:
                                param.grad.mul_(text_scale)
                elif grad_method == 'gradnorm' and text_norm > 0:
                    desired = target_ratio * cf_norm
                    scale = (desired + grad_eps) / (text_norm + grad_eps)
                elif grad_method == 'ratio' and text_norm > 0 and cf_norm > 0:
                    current_ratio = text_norm / (cf_norm + grad_eps)
                    scale = target_ratio / (current_ratio + grad_eps)
                elif grad_method == 'pcgrad':
                    pcgrad_info = _pcgrad_between_branches(
                        branch_params,
                        target_ratio=target_ratio,
                        eps=grad_eps,
                        max_scale=max_scale,
                    )
                    if grad_metrics is not None and pcgrad_info is not None:
                        for key, value in pcgrad_info.items():
                            grad_metrics.setdefault(key, []).append(value)

                if scale is not None:
                    if max_scale > 0:
                        scale = min(scale, max_scale)
                    if min_scale > 0:
                        scale = max(scale, min_scale)
                    for param in branch_params.get('text', []):
                        if param.grad is not None:
                            param.grad.mul_(scale)
                    if scale_attention and 'attention' in branch_params:
                        for param in branch_params['attention']:
                            if param.grad is not None:
                                param.grad.mul_(scale)
                    if grad_metrics is not None:
                        grad_metrics.setdefault('scale', []).append(float(scale))

        if grad_metrics is not None and adaptive_alpha:
            grad_metrics.setdefault('lambda_alpha_t', []).append(float(lambda_alpha_t))
        optimizer.step()

        total_loss += total_batch_loss.item()
        if reg_enabled:
            lambda_reg_total += reg_loss.item()
        if (batch_idx + 1) % 100 == 0:
            if reg_enabled:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}], Loss: {total_batch_loss.item():.4f}, Reg: {reg_loss.item():.4f}")
            else:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}], Loss: {bpr_loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    avg_reg = (lambda_reg_total / len(train_loader)) if reg_enabled else None
    grad_summary = {name: float(np.mean(values)) for name, values in grad_metrics.items() if values} if grad_metrics else None
    return avg_loss, avg_reg, grad_summary


def _to_pairs(data):
    """Convert DataFrame/array-like interactions to (user, item) pairs."""
    if hasattr(data, "values"):
        return data[["user_idx", "item_idx"]].values
    return data


def _build_user_item_sets(data):
    """Build user -> interacted item set mapping."""
    pairs = _to_pairs(data)
    user_items = {}
    for user_id, item_id in pairs:
        user_id = int(user_id)
        item_id = int(item_id)
        if user_id not in user_items:
            user_items[user_id] = set()
        user_items[user_id].add(item_id)
    return user_items


@torch.no_grad()
def evaluate(
    model,
    eval_data,
    item_texts,
    n_items,
    k=10,
    device='cpu',
    n_neg_samples=99,  # kept for backward compatibility, unused in full-ranking
    text_dim=384,
    train_data=None,
    max_users=None,
    item_chunk_size=4096,
):
    """
    Full-ranking evaluation (MMRec-compatible):
    - Per-user ranking over all items
    - Mask train interactions
    - Compute user-level Recall@K / NDCG@K
    """
    del n_neg_samples
    model.eval()

    eval_user_items = _build_user_item_sets(eval_data)
    if not eval_user_items:
        return 0.0, 0.0

    train_user_items = _build_user_item_sets(train_data) if train_data is not None else {}

    users = sorted(eval_user_items.keys())
    if max_users is not None and max_users > 0:
        users = users[: int(max_users)]
    if not users:
        return 0.0, 0.0

    item_ids_all = torch.arange(n_items, dtype=torch.long, device=device)
    text_fallback = torch.zeros(text_dim)
    item_text_matrix = torch.stack(
        [item_texts.get(item_id, text_fallback) for item_id in range(n_items)]
    ).to(device)

    recalls = []
    ndcgs = []
    for user_id in users:
        score_chunks = []
        for start in range(0, n_items, item_chunk_size):
            end = min(start + item_chunk_size, n_items)
            item_chunk = item_ids_all[start:end]
            text_chunk = item_text_matrix[start:end]
            user_chunk = torch.full((end - start,), int(user_id), dtype=torch.long, device=device)
            score_chunks.append(model(user_chunk, item_chunk, text_chunk))
        scores = torch.cat(score_chunks, dim=0)

        seen_items = train_user_items.get(int(user_id), set())
        if seen_items:
            seen_idx = torch.tensor(sorted(seen_items), dtype=torch.long, device=device)
            scores[seen_idx] = -1e9

        topk = min(int(k), n_items)
        topk_idx = torch.topk(scores, topk).indices.cpu().tolist()
        gt_items = eval_user_items[int(user_id)]

        hit_count = sum(1 for item_id in topk_idx if item_id in gt_items)
        recalls.append(hit_count / max(len(gt_items), 1))

        dcg = 0.0
        for rank, item_id in enumerate(topk_idx):
            if item_id in gt_items:
                dcg += 1.0 / np.log2(rank + 2)
        ideal_hits = min(len(gt_items), topk)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(ideal_hits))
        ndcgs.append((dcg / idcg) if idcg > 0 else 0.0)

    return float(np.mean(recalls)), float(np.mean(ndcgs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 加载配置
    config_path = Path(args.config).resolve()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config_dir = config_path.parent
    
    print("="*60)
    print(f"🚀 Starting Training - Seed {args.seed}")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Model: {config['model']['type']}")
    
    # 设置设备
    device_cfg = config.get('device', 'auto')
    if str(device_cfg).lower() == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_cfg)
    print(f"Device: {device}")
    
    # 加载数据
    train_data, val_data, test_data, item_embeddings, stats = load_data(
        config['data']['data_dir'], config, config_dir
    )
    
    # 构建模型
    model = build_model(config, device)

    lambda_reg_cfg = config.get('regularization', {})
    if isinstance(model, AttentionFusion):
        lambda_alpha = float(lambda_reg_cfg.get('lambda_alpha', 0.0))
        lambda_beta = float(lambda_reg_cfg.get('lambda_beta', 0.0))
        if lambda_alpha > 0 or lambda_beta > 0:
            print(f"   λ-regularization enabled (alpha={lambda_alpha}, beta={lambda_beta})")

    grad_balance_cfg = config.get('grad_balance', {})
    grad_method = str(grad_balance_cfg.get('method', 'none')).lower()
    if grad_method != 'none':
        target_ratio = grad_balance_cfg.get('target_ratio', 1.0)
        print(f"   Gradient balancing: {grad_method} (target_ratio={target_ratio})")
    elif grad_balance_cfg.get('log_grads', False):
        print("   Gradient norms will be logged (log_grads=True)")
    
    # 如果是attention模型，设置必要的统计特征
    if isinstance(model, AttentionFusion):
        n_items = config['model']['n_items']
        n_users = config['model']['n_users']
        att_features = set(getattr(model, 'features', ()) or [])
        feature_params = config['model'].get('feature_params', {})

        popularity = None
        needs_popularity = (
            model.input_mode == 'popularity'
            or 'popularity' in att_features
            or 'cf_confidence' in att_features
        )
        if needs_popularity:
            print("\n📊 Computing item popularity...")
            popularity = compute_item_popularity(train_data, n_items)
            model.set_item_popularity(popularity)
        else:
            popularity = None

        if 'cf_confidence' in att_features:
            if popularity is None:
                popularity = compute_item_popularity(train_data, n_items)
            method = str(feature_params.get('cf_confidence_method', 'log')).lower()
            cf_confidence = compute_item_cf_confidence(popularity, method=method)
            model.set_item_cf_confidence(cf_confidence)

        if 'text_quality' in att_features:
            print("\n📊 Computing text quality stats...")
            text_quality = compute_item_text_quality(item_embeddings, n_items)
            model.set_item_text_quality(text_quality)

        if 'recency' in att_features:
            print("\n📊 Computing item recency...")
            item_recency = compute_item_recency(train_data, n_items)
            model.set_item_recency(item_recency)

        if getattr(model, 'context_features', None) and 'user_activity' in model.context_features:
            print("\n📊 Computing user activity...")
            user_activity = compute_user_activity(train_data, n_users)
            model.set_user_activity(user_activity)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), 
                          lr=float(config['training']['learning_rate']),
                          weight_decay=float(config['training']['weight_decay']))
    
    # 创建数据加载器
    train_dataset = RecommendationDataset(
        train_data, item_embeddings, config['model']['n_items'],
        negative_sampling=config['training']['negative_sampling'],
        text_dim=config['model']['text_dim']
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0  # CPU模式下使用0
    )
    
    # 训练
    best_val_recall = -1.0
    patience_counter = 0
    save_path = f"{args.output_dir}/best_model_seed{args.seed}.pt"
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("📈 Starting Training Loop")
    print("="*60)
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch [{epoch+1}/{config['training']['num_epochs']}]")
        
        # 训练
        train_loss, train_reg, grad_summary = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            lambda_reg_cfg=lambda_reg_cfg,
            grad_balance_cfg=grad_balance_cfg,
            total_epochs=int(config['training']['num_epochs']),
        )
        if train_reg is not None:
            print(f"  Train Loss: {train_loss:.4f} (Reg: {train_reg:.4f})")
        else:
            print(f"  Train Loss: {train_loss:.4f}")
        if grad_summary:
            summary_str = ", ".join(f"{k}={v:.4f}" for k, v in grad_summary.items())
            print(f"  Grad Stats: {summary_str}")
        
        # 验证（full-ranking）
        if (epoch + 1) % 5 == 0:
            print("  Evaluating on validation set...")
            val_max_users = int(config.get("evaluation", {}).get("val_max_users", 1000))
            val_recall, val_ndcg = evaluate(
                model,
                val_data,
                item_embeddings,
                config['model']['n_items'], k=10, device=device,
                text_dim=config['model']['text_dim'],
                train_data=train_data,
                max_users=val_max_users,
                item_chunk_size=int(config.get("evaluation", {}).get("item_chunk_size", 4096)),
            )
            print(f"  Val Recall@10: {val_recall:.4f}, NDCG@10: {val_ndcg:.4f}")
            
            # Early stopping
            if val_recall > best_val_recall:
                best_val_recall = val_recall
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), save_path)
                print(f"  ✅ Best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= config['training']['early_stopping_patience']:
                    print(f"  ⚠️ Early stopping triggered!")
                    break
    
    # 最终测试
    print("\n" + "="*60)
    print("🎯 Final Evaluation on Test Set")
    print("="*60)
    
    # 加载最佳模型；若未触发验证保存，则保存并使用最后一个epoch模型
    if not os.path.exists(save_path):
        torch.save(model.state_dict(), save_path)
        print("  ℹ️ No validation checkpoint found, saved last-epoch model.")
    model.load_state_dict(torch.load(save_path, map_location=device))
    
    test_recall, test_ndcg = evaluate(
        model,
        test_data,
        item_embeddings,
        config['model']['n_items'], k=10, device=device,
        text_dim=config['model']['text_dim'],
        train_data=train_data,
        max_users=config.get("evaluation", {}).get("test_max_users"),
        item_chunk_size=int(config.get("evaluation", {}).get("item_chunk_size", 4096)),
    )
    
    print(f"\n📊 Final Results (Seed {args.seed}):")
    print(f"   Recall@10: {test_recall:.4f}")
    print(f"   NDCG@10: {test_ndcg:.4f}")
    
    # 保存结果
    results = {
        'seed': args.seed,
        'model_type': config['model']['type'],
        'eval_protocol': 'full_ranking',
        'recall@10': float(test_recall),
        'ndcg@10': float(test_ndcg),
        'test_recall@10': float(test_recall),
        'test_ndcg@10': float(test_ndcg),
        'best_val_recall': float(best_val_recall)
    }
    
    result_file = f"{args.output_dir}/results_seed{args.seed}.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {result_file}")
    print("="*60)


if __name__ == "__main__":
    main()
