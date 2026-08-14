"""
Mixed-effects logistic regression for LLM judge alignment analysis.

Replaces the simple N=15 binomial test with a user-level GLMM that properly
accounts for the nested structure: users within model-pairs.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_config import RESULTS_DIR, DATASETS, EXPERIMENT_PAIRS


def build_user_level_dataframe():
    """
    Build a long-format DataFrame with one row per user-pair-prompt.
    Columns:
        user_id, dataset, model_a, model_b, pair_id, prompt,
        fwd_verdict, rev_verdict, consistent, effective_pref,
        prefers_ndcg_winner (binary outcome for GLMM),
        delta_ndcg, delta_cov, delta_nov
    """
    rows = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name == "all_results_7b.json":
            continue
        if f.parent.name != "results":
            continue

        with open(f) as fh:
            data = json.load(fh)

        ds = data["dataset"]
        ma, mb = data["model_a"], data["model_b"]
        prompt = data["dimension"]
        ndcg_winner = data["ndcg_winner"]
        cov_winner = data["cov_winner"]

        delta_ndcg = data["gt_ndcg_a"] - data["gt_ndcg_b"]
        delta_cov = data["gt_cov_a"] - data["gt_cov_b"]
        pair_id = f"{ds}_{ma}_vs_{mb}"

        is_conflict = (ndcg_winner != cov_winner)

        for trial in data["per_trial"]:
            ep = trial["effective_preference"]
            if ep == "INCONSISTENT":
                continue

            if ep == "TIE":
                prefers_ndcg = np.nan
            elif ep == ndcg_winner:
                prefers_ndcg = 1.0
            elif ep == cov_winner:
                prefers_ndcg = 0.0
            elif ep in (ma, mb):
                prefers_ndcg = 1.0 if ep == ndcg_winner else 0.0
            else:
                prefers_ndcg = np.nan

            rows.append({
                "user_id": trial["user_id"],
                "dataset": ds,
                "model_a": ma,
                "model_b": mb,
                "pair_id": pair_id,
                "prompt": prompt,
                "is_conflict": is_conflict,
                "fwd_verdict": trial["fwd_verdict"],
                "rev_verdict": trial["rev_verdict"],
                "consistent": trial["consistent"],
                "effective_pref": ep,
                "prefers_ndcg": prefers_ndcg,
                "abs_delta_ndcg": abs(delta_ndcg),
                "abs_delta_cov": abs(delta_cov),
            })

    df = pd.DataFrame(rows)
    return df


def run_mixed_effects_analysis(df):
    """
    Run mixed-effects logistic regression (GLMM) on conflict-pair data.

    Model: Pr(prefers_ndcg=1) ~ C(prompt) + abs_delta_ndcg + abs_delta_cov
                                 + (1 | pair_id)

    Uses GEE (Generalized Estimating Equations) as a robust alternative
    since statsmodels' BinomialBayesMixedGLM can be unstable.
    """
    conflict_df = df[(df["is_conflict"]) & (df["prefers_ndcg"].notna())].copy()
    conflict_df["prefers_ndcg"] = conflict_df["prefers_ndcg"].astype(int)

    print(f"\n{'='*70}")
    print("MIXED-EFFECTS ANALYSIS: User-Level GLMM")
    print(f"{'='*70}")
    print(f"Total consistent user-level observations (conflict pairs): {len(conflict_df)}")
    print(f"Unique pairs: {conflict_df['pair_id'].nunique()}")
    print(f"Unique users: {conflict_df['user_id'].nunique()}")

    print("\n--- Per-prompt descriptive statistics ---")
    for prompt in ["balanced", "relevance", "diversity", "novelty"]:
        sub = conflict_df[conflict_df["prompt"] == prompt]
        n = len(sub)
        n_ndcg = sub["prefers_ndcg"].sum()
        pct = n_ndcg / n * 100 if n > 0 else 0
        print(f"  {prompt:12s}: N={n:4d}, prefers_NDCG={int(n_ndcg):4d} ({pct:.1f}%)")

    # --- GEE with exchangeable correlation within pair clusters ---
    print("\n--- GEE Logistic Regression (cluster = pair_id) ---")
    conflict_df["prompt"] = pd.Categorical(
        conflict_df["prompt"],
        categories=["balanced", "relevance", "diversity", "novelty"],
    )
    conflict_df = conflict_df.sort_values("pair_id").reset_index(drop=True)

    fam = sm_families()
    gee_model = smf.gee(
        "prefers_ndcg ~ C(prompt, Treatment(reference='balanced'))"
        " + abs_delta_ndcg + abs_delta_cov",
        groups="pair_id",
        data=conflict_df,
        family=fam,
        cov_struct=sm_exchangeable(),
    )
    gee_result = gee_model.fit()
    print(gee_result.summary())

    print("\n--- Marginal effects (prompt-specific predicted probabilities) ---")
    for prompt in ["balanced", "relevance", "diversity", "novelty"]:
        sub = conflict_df[conflict_df["prompt"] == prompt]
        pred = gee_result.predict(sub)
        print(f"  {prompt:12s}: predicted Pr(NDCG) = {pred.mean():.3f} "
              f"[observed = {sub['prefers_ndcg'].mean():.3f}]")

    # --- Simple per-prompt user-level binomial test ---
    print(f"\n{'='*70}")
    print("USER-LEVEL BINOMIAL TESTS (one-sided, H0: p=0.5)")
    print(f"{'='*70}")
    for prompt in ["balanced", "relevance", "diversity", "novelty"]:
        sub = conflict_df[conflict_df["prompt"] == prompt]
        n = len(sub)
        k = int(sub["prefers_ndcg"].sum())
        pval = stats.binomtest(k, n, 0.5, alternative="greater").pvalue
        ci_lo, ci_hi = proportion_ci(k, n)
        print(f"  {prompt:12s}: {k}/{n} = {k/n:.3f}  p={pval:.4f}  "
              f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")

    # --- Cochran-Mantel-Haenszel test for prompt effect ---
    print(f"\n{'='*70}")
    print("COCHRAN-MANTEL-HAENSZEL: Prompt effect controlling for pair")
    print(f"{'='*70}")
    run_cmh_test(conflict_df)

    return gee_result, conflict_df


def sm_families():
    import statsmodels.genmod.families as fam
    return fam.Binomial()


def sm_exchangeable():
    from statsmodels.genmod.cov_struct import Exchangeable
    return Exchangeable()


def proportion_ci(k, n, alpha=0.05):
    """Wilson score interval for a binomial proportion."""
    from statsmodels.stats.proportion import proportion_confint
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return lo, hi


def run_cmh_test(df):
    """
    Cochran-Mantel-Haenszel test: is prompt type associated with
    NDCG-preference, controlling for pair_id?

    Stratified 2x4 test (NDCG-pref × prompt, stratified by pair).
    """
    prompts = ["balanced", "relevance", "diversity", "novelty"]
    pairs = sorted(df["pair_id"].unique())

    tables = []
    for pair in pairs:
        pair_df = df[df["pair_id"] == pair]
        table = np.zeros((2, len(prompts)), dtype=int)
        for j, prompt in enumerate(prompts):
            sub = pair_df[pair_df["prompt"] == prompt]
            table[1, j] = int(sub["prefers_ndcg"].sum())
            table[0, j] = len(sub) - table[1, j]
        tables.append(table)

    from scipy.stats import chi2
    numerator = np.zeros(len(prompts) - 1)
    denominator = np.zeros((len(prompts) - 1, len(prompts) - 1))

    for table in tables:
        n_k = table.sum()
        if n_k <= 1:
            continue
        r1 = table[1]  # NDCG-preferred counts per prompt
        c = table.sum(axis=0)  # total per prompt
        n1 = table[1].sum()  # total NDCG-preferred

        for j in range(len(prompts) - 1):
            numerator[j] += r1[j] - n1 * c[j] / n_k
            for l in range(len(prompts) - 1):
                if j == l:
                    denominator[j, l] += (n1 * (n_k - n1) * c[j] * (n_k - c[j])) / (n_k**2 * (n_k - 1))
                else:
                    denominator[j, l] += -(n1 * (n_k - n1) * c[j] * c[l]) / (n_k**2 * (n_k - 1))

    try:
        inv_denom = np.linalg.inv(denominator)
        stat = numerator @ inv_denom @ numerator
        df_chi = len(prompts) - 1
        pval = 1 - chi2.cdf(stat, df_chi)
        print(f"  CMH statistic = {stat:.3f}, df = {df_chi}, p = {pval:.4f}")
    except np.linalg.LinAlgError:
        print("  CMH test: singular matrix, falling back to pairwise tests")

    # Pairwise: balanced vs diversity
    print("\n  Pairwise CMH (balanced vs diversity, stratified by pair):")
    _pairwise_cmh(df, "balanced", "diversity", pairs)
    print("  Pairwise CMH (balanced vs novelty, stratified by pair):")
    _pairwise_cmh(df, "balanced", "novelty", pairs)


def _pairwise_cmh(df, prompt1, prompt2, pairs):
    """2x2 CMH test for two prompts stratified by pair."""
    tables_2x2 = []
    for pair in pairs:
        sub1 = df[(df["pair_id"] == pair) & (df["prompt"] == prompt1)]
        sub2 = df[(df["pair_id"] == pair) & (df["prompt"] == prompt2)]
        if len(sub1) == 0 or len(sub2) == 0:
            continue
        a = int(sub1["prefers_ndcg"].sum())
        b = len(sub1) - a
        c = int(sub2["prefers_ndcg"].sum())
        d = len(sub2) - c
        tables_2x2.append(np.array([[a, b], [c, d]]))

    from statsmodels.stats.contingency_tables import StratifiedTable
    st = StratifiedTable(tables_2x2)
    result = st.test_null_odds()
    print(f"    stat = {result.statistic:.3f}, p = {result.pvalue:.4f}")
    or_pooled = st.oddsratio_pooled
    logodds_se = st.logodds_pooled_se
    print(f"    Common OR = {or_pooled:.3f}, "
          f"95% CI [{or_pooled * np.exp(-1.96*logodds_se):.3f}, "
          f"{or_pooled * np.exp(1.96*logodds_se):.3f}]")


if __name__ == "__main__":
    df = build_user_level_dataframe()
    print(f"Built DataFrame: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"\nPrompt distribution:\n{df['prompt'].value_counts()}")
    print(f"\nConflict pairs: {df['is_conflict'].sum()} / {len(df)}")

    gee_result, conflict_df = run_mixed_effects_analysis(df)
