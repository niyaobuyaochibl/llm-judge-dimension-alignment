# Data

## What is here

`topk_results/` holds the top-10 recommendation lists that every recommender produced for
every user, as **item IDs only** (for example `[15851, 7146, ...]`). These are our own model
outputs and contain no third-party item text, so they are included in full: 4 datasets x 5–6
recommenders, seed 42.

## What is not here, and why

The processed datasets (`train.pkl`, `test.pkl`, `item_texts.pkl`) are **not redistributed**.
`item_texts.pkl` embeds item titles and descriptions taken from the source corpora, and the
Yelp Open Dataset licence in particular forbids redistribution. Obtain the sources yourself:

| Dataset | Domain | Items | Source |
|---|---|---|---|
| MIND | News | 15,697 | https://msnews.github.io/ |
| Yelp | Business reviews | 17,219 | https://www.yelp.com/dataset |
| Amazon Books | Books | 12,410 | https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/ |
| Amazon CDs & Vinyl | Music | 14,390 | https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/ |

## Expected layout

Once you have the sources, build this tree and point `DIVERSITY_ROOT` in
`code/experiment_config.py` at it:

```
<DIVERSITY_ROOT>/datasets/
  mind/          train.pkl  test.pkl  item_texts.pkl  stats.json
  yelp/          train.pkl  test.pkl  item_texts.pkl  stats.json
  amazon_books/  train.pkl  test.pkl  item_texts.pkl  stats.json
  amazon_cds/    train.pkl  test.pkl  item_texts.pkl  stats.json
```

`train.pkl` and `test.pkl` are user-item interaction splits keyed by contiguous integer user
and item IDs. `item_texts.pkl` maps each item ID to its display text; that text is what the
judge sees in the prompt, so the numbering must match the IDs in `topk_results/`.
`code/data_loader.py` is the reference reader for all four files.

## Recommenders

The five to six recommenders per dataset (GBAF, Fixed, Concat-MLP, GradNorm, PCGrad,
LightGCN) come from a separate multi-task recommendation codebase. Their trained outputs are
what `topk_results/` contains, so the judge experiments can be reproduced from the lists here
without retraining any recommender.
