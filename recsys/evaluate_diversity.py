"""
Batch evaluation script: load saved checkpoints and compute accuracy + diversity metrics.
Usage:
    python evaluate_diversity.py --config configs/extended/yelp_gbaf.yaml \
        --checkpoint results/yelp/gbaf/best_model_seed42.pt \
        --seed 42 --output_dir diversity_results/yelp/gbaf
"""

import os
import sys
import yaml
import pickle
import argparse
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from models import (
    AttentionFusion, FixedFusion, ConcatMLPFusion, CrossAttentionFusion,
    GBAFFusion, GBAFAdaptiveFusion, GBAFv2Fusion,
    compute_item_popularity, compute_user_activity,
)
from diversity_metrics import (
    compute_all_diversity_metrics,
    compute_item_popularity_array,
)


MODEL_REGISTRY = {
    "attention_fusion": AttentionFusion,
    "fixed_fusion": FixedFusion,
    "concat_mlp_fusion": ConcatMLPFusion,
    "cross_attention_fusion": CrossAttentionFusion,
    "gbaf_fusion": GBAFFusion,
    "gbaf_adaptive_fusion": GBAFAdaptiveFusion,
    "gbafv2_fusion": GBAFv2Fusion,
}


def load_data(data_dir):
    """Load train/val/test splits."""
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)
    return train_data, val_data, test_data


def load_embeddings(embedding_file):
    """Load pre-computed text embeddings."""
    with open(embedding_file, "rb") as f:
        embeddings = pickle.load(f)
    if isinstance(embeddings, dict):
        return embeddings
    elif isinstance(embeddings, np.ndarray):
        return {i: torch.tensor(embeddings[i], dtype=torch.float32) for i in range(len(embeddings))}
    else:
        raise ValueError(f"Unsupported embedding format: {type(embeddings)}")


def build_model(config, device):
    """Instantiate model from config."""
    model_type = config["model"]["type"]
    model_cls = MODEL_REGISTRY.get(model_type)
    if model_cls is None:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")

    model_cfg = config["model"]
    kwargs = {
        "n_users": model_cfg["n_users"],
        "n_items": model_cfg["n_items"],
        "embedding_dim": model_cfg["embedding_dim"],
        "text_dim": model_cfg["text_dim"],
    }
    if model_type == "fixed_fusion":
        kwargs["lambda_fixed"] = model_cfg.get("lambda_fixed", 0.5)
    if model_type == "concat_mlp_fusion":
        kwargs["hidden_dims"] = model_cfg.get("mlp_hidden_dims", [32, 16])
        kwargs["dropout"] = model_cfg.get("mlp_dropout", 0.1)

    if model_type in ("gbaf_fusion", "gbaf_adaptive_fusion", "gbafv2_fusion"):
        gate_cfg = model_cfg.get("gate", {})
        if "hidden_dim" in gate_cfg:
            kwargs["gate_hidden_dim"] = gate_cfg["hidden_dim"]
        ctx = model_cfg.get("context_features", [])
        if ctx:
            kwargs["context_features"] = ctx

    model = model_cls(**kwargs).to(device)
    return model


def _build_user_item_sets(data):
    if hasattr(data, "values"):
        pairs = data[["user_idx", "item_idx"]].values
    else:
        pairs = data
    user_items = {}
    for user_id, item_id in pairs:
        user_items.setdefault(int(user_id), set()).add(int(item_id))
    return user_items


@torch.no_grad()
def generate_topk_and_evaluate(
    model, test_data, train_data, item_embeddings,
    n_items, n_users, k=10, device="cpu", text_dim=384,
    item_chunk_size=4096, max_users=None,
):
    """Generate per-user top-K lists and compute accuracy metrics."""
    model.eval()
    eval_user_items = _build_user_item_sets(test_data)
    train_user_items = _build_user_item_sets(train_data)

    users = sorted(eval_user_items.keys())
    if max_users and max_users > 0:
        users = users[:int(max_users)]

    item_ids_all = torch.arange(n_items, dtype=torch.long, device=device)
    text_fallback = torch.zeros(text_dim)
    item_text_matrix = torch.stack(
        [item_embeddings.get(i, text_fallback) for i in range(n_items)]
    ).to(device)

    topk_lists = {}
    recalls, ndcgs = [], []

    for user_id in users:
        score_chunks = []
        for start in range(0, n_items, item_chunk_size):
            end_idx = min(start + item_chunk_size, n_items)
            item_chunk = item_ids_all[start:end_idx]
            text_chunk = item_text_matrix[start:end_idx]
            user_chunk = torch.full((end_idx - start,), int(user_id), dtype=torch.long, device=device)
            score_chunks.append(model(user_chunk, item_chunk, text_chunk))
        scores = torch.cat(score_chunks, dim=0)

        seen_items = train_user_items.get(int(user_id), set())
        if seen_items:
            seen_idx = torch.tensor(sorted(seen_items), dtype=torch.long, device=device)
            scores[seen_idx] = -1e9

        topk = min(k, n_items)
        topk_idx = torch.topk(scores, topk).indices.cpu().tolist()
        topk_lists[int(user_id)] = topk_idx

        gt_items = eval_user_items[int(user_id)]
        hit_count = sum(1 for i in topk_idx if i in gt_items)
        recalls.append(hit_count / max(len(gt_items), 1))
        dcg = sum(1.0 / np.log2(r + 2) for r, i in enumerate(topk_idx) if i in gt_items)
        ideal_hits = min(len(gt_items), topk)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(ideal_hits))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    accuracy = {
        "recall@10": float(np.mean(recalls)),
        "ndcg@10": float(np.mean(ndcgs)),
        "n_eval_users": len(users),
    }
    return topk_lists, accuracy


def main():
    parser = argparse.ArgumentParser(description="Diversity evaluation for text-CF fusion models")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint .pt path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./diversity_results")
    parser.add_argument("--k", type=int, default=10, help="Top-K for evaluation")
    parser.add_argument("--max_users", type=int, default=None, help="Max users to evaluate (None=all)")
    parser.add_argument("--save_lists", action="store_true", help="Save per-user top-K lists")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Diversity Eval] Device: {device}")
    print(f"[Diversity Eval] Config: {args.config}")
    print(f"[Diversity Eval] Checkpoint: {args.checkpoint}")

    # Load data
    data_dir = config["data"]["data_dir"]
    train_data, val_data, test_data = load_data(data_dir)
    n_items = config["model"]["n_items"]
    n_users = config["model"]["n_users"]
    text_dim = config["model"]["text_dim"]
    print(f"[Diversity Eval] Dataset: {config['data']['dataset']}, Users: {n_users}, Items: {n_items}")

    # Load embeddings
    emb_file = config["text_encoder"]["embedding_file"]
    item_embeddings = load_embeddings(emb_file)
    print(f"[Diversity Eval] Loaded {len(item_embeddings)} item embeddings")

    # Build and load model
    model = build_model(config, device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print(f"[Diversity Eval] Model loaded: {config['model']['type']}")

    # Set auxiliary features if needed
    if hasattr(model, "item_popularity") and model.item_popularity is None:
        pop = compute_item_popularity(train_data, n_items)
        model.set_item_popularity(pop)
    if hasattr(model, "user_activity") and getattr(model, "user_activity", None) is None:
        if getattr(model, "context_features", None) and "user_activity" in model.context_features:
            act = compute_user_activity(train_data, n_users)
            model.set_user_activity(act)

    # Generate top-K and accuracy
    chunk_size = int(config.get("evaluation", {}).get("item_chunk_size", 4096))
    topk_lists, accuracy = generate_topk_and_evaluate(
        model, test_data, train_data, item_embeddings,
        n_items, n_users, k=args.k, device=device, text_dim=text_dim,
        item_chunk_size=chunk_size, max_users=args.max_users,
    )
    print(f"[Accuracy] Recall@{args.k}: {accuracy['recall@10']:.4f}, NDCG@{args.k}: {accuracy['ndcg@10']:.4f}")

    # Compute diversity metrics
    item_pop = compute_item_popularity_array(train_data, n_items)

    text_emb_matrix = np.stack([
        item_embeddings.get(i, torch.zeros(text_dim)).numpy()
        if isinstance(item_embeddings.get(i, torch.zeros(text_dim)), torch.Tensor)
        else item_embeddings.get(i, np.zeros(text_dim))
        for i in range(n_items)
    ])

    cf_emb_matrix = None
    if hasattr(model, "item_embedding"):
        cf_emb_matrix = model.item_embedding.weight.detach().cpu().numpy()

    diversity = compute_all_diversity_metrics(
        topk_lists, n_items, n_users, item_pop,
        text_embeddings=text_emb_matrix,
        cf_embeddings=cf_emb_matrix,
        k=args.k,
    )

    for metric, val in diversity.items():
        if isinstance(val, float):
            print(f"[Diversity] {metric}: {val:.4f}")
        else:
            print(f"[Diversity] {metric}: {val}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "seed": args.seed,
        "dataset": config["data"]["dataset"],
        "model_type": config["model"]["type"],
        "config_file": str(args.config),
        "checkpoint_file": str(args.checkpoint),
        **accuracy,
        **diversity,
    }
    result_file = os.path.join(args.output_dir, f"diversity_seed{args.seed}.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] {result_file}")

    if args.save_lists:
        lists_file = os.path.join(args.output_dir, f"topk_lists_seed{args.seed}.json")
        serializable = {str(k): v for k, v in topk_lists.items()}
        with open(lists_file, "w") as f:
            json.dump(serializable, f)
        print(f"[Saved] {lists_file}")


if __name__ == "__main__":
    main()
