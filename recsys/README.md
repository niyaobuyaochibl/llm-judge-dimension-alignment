# Recommender models

The training and evaluation code for the recommenders whose top-10 lists the LLM judges
compared, together with the seed-42 checkpoints that produced every list in
[`../data/topk_results/`](../data/topk_results/). This is the code the paper refers to as
"recommendation model code and pre-trained checkpoints."

The recommenders are not the object of study; the paper needs a set of models spanning a wide
accuracy–diversity range (Coverage 0.006–0.939, NDCG@10 0.006–0.031) so that the judge's
preference can be compared against metric winners. Any model set with a comparable spread
would serve.

## Models

| Name in paper | Checkpoint folder | Config file | `model.type` | Notes |
|---|---|---|---|---|
| GBAF | `gbaf` | `<dataset>_gbaf.yaml` | `gbaf_fusion` | gradient-balanced text–CF fusion; highest NDCG |
| Fixed | `fixed` | `<dataset>_fixed_baseline.yaml` | `fixed_fusion` | static fusion weight; high Coverage |
| Concat-MLP | `concat_mlp` | `<dataset>_concat_mlp.yaml` | `concat_mlp_fusion` | MLP over concatenated CF and text |
| GradNorm | `gradnorm_only` | `<dataset>_gradnorm_only.yaml` | `attention_fusion` | multi-objective gradient balancing |
| PCGrad | `pcgrad` | `<dataset>_pcgrad.yaml` | `attention_fusion` + `pcgrad` | MIND and Yelp only |
| LightGCN | `lightgcn_only` | `<dataset>_lightgcn_only.yaml` | `fixed_fusion`, text branch off | pure CF; very low Coverage |

`<dataset>` is one of `mind`, `yelp`, `amazon_books`, `amazon_cds`. Model classes live in
`models/gbaf.py` (`GBAFFusion`) and `models/attention_variants.py` (`FixedFusion`,
`ConcatMLPFusion`, `AttentionFusion`). GBAF, Fixed, and Concat-MLP are our own fusion
variants; LightGCN, GradNorm, and PCGrad follow their original formulations.

## Checkpoints

The 22 seed-42 checkpoints (418 MB) are attached to the GitHub release rather than committed:

| Archive | Models | Size |
|---|---|---|
| `checkpoints_mind_seed42.zip` | 6 | 91 MB |
| `checkpoints_yelp_seed42.zip` | 6 | 248 MB |
| `checkpoints_amazon_books_seed42.zip` | 5 | 36 MB |
| `checkpoints_amazon_cds_seed42.zip` | 5 | 43 MB |

Each unpacks to `checkpoints/<dataset>/<model>/best_model_seed42.pt`. Verify with
`sha256sum -c CHECKPOINT_SHA256SUMS.txt`, which lists both the archives and every individual
`.pt` file. Each checkpoint is a `torch.save` of the model `state_dict`; the matching config
gives the architecture and the `n_users` / `n_items` it was trained with.

## Regenerate the top-10 lists from a checkpoint

```bash
pip install torch numpy pandas pyyaml
python evaluate_diversity.py \
    --config configs/extended/yelp_gbaf.yaml \
    --checkpoint checkpoints/yelp/gbaf/best_model_seed42.pt \
    --seed 42 --k 10 --save_lists \
    --output_dir out/yelp/gbaf
```

This writes `topk_lists_seed42.json` in the same format as `../data/topk_results/yelp/gbaf/`,
plus the accuracy and diversity metrics (NDCG@10, Recall@10, Coverage, Entropy). Repeating it
for all 22 config/checkpoint pairs reproduces `../data/topk_results/` in full.

## Retrain from scratch

```bash
python train_fusion.py --config configs/extended/yelp_gbaf.yaml --seed 42 --output_dir results
```

Training needs a GPU; the Yelp models (99,165 users, 56,696 items) take the longest.
`run_diversity_eval.sh` is the original batch driver and is kept for reference; it also lists
jobs for two datasets from a companion paper that are not used here.

## Data and paths the configs expect

The configs carry absolute paths from the machine the models were trained on, in three keys:

```yaml
data:
  data_dir: /root/autodl-tmp/extended_fusion/datasets/yelp
text_encoder:
  embedding_file: /root/autodl-tmp/extended_fusion/embeddings/yelp/all-MiniLM-L6-v2.pkl
logging:
  checkpoint_dir: /root/autodl-tmp/extended_fusion/checkpoints
```

Either edit them or note that the loader resolves relative paths against the config file's own
directory. `data_dir` must contain `train.pkl`, `val.pkl`, `test.pkl`, and `stats.json`; see
[`../data/README.md`](../data/README.md) for how to obtain the sources and what the files hold.

The text branch reads one pre-computed 384-d embedding per item, produced with
`sentence-transformers/all-MiniLM-L6-v2` over each item's display text and stored as a pickle
`{item_id: vector}`. The embeddings are derived from the source corpora and are therefore not
redistributed; regenerate them with `sentence-transformers` once you have `item_texts.pkl`.
