"""
Benjamini-Hochberg FDR correction for all reported p-values.
Addresses reviewer concern about multiple hypothesis testing.
"""

import numpy as np
from scipy import stats


def benjamini_hochberg(p_values, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted[i] = sorted_p[i]
        else:
            adjusted[i] = min(adjusted[i + 1], sorted_p[i] * n / (i + 1))
    adjusted = np.minimum(adjusted, 1.0)

    result = np.zeros(n)
    result[sorted_indices] = adjusted
    return result


# All user-level binomial test p-values from the paper
# Qwen-7B primary analysis (Table 3 / text)
tests = {
    # Qwen-7B user-level binomial tests
    "Q7B_balanced":   0.000530,
    "Q7B_diversity":  0.000052,
    "Q7B_novelty":    0.084953,
    "Q7B_relevance":  0.449000,

    # GEE coefficients
    "GEE_intercept":      0.005,
    "GEE_div_vs_bal":     0.858,
    "GEE_nov_vs_bal":     0.075,
    "GEE_delta_cov":      0.035,

    # Qwen-14B user-level
    "Q14B_balanced":  0.000417,
    "Q14B_diversity": 0.001759,
    "Q14B_novelty":   0.768725,

    # Mistral-7B user-level
    "Mis_balanced":   0.019090,
    "Mis_diversity":  0.040713,
    "Mis_novelty":    0.344440,

    # DeepSeek-V3 user-level
    "DS_balanced":    0.000018,
    "DS_diversity":   0.000002,
    "DS_novelty":     0.974164,
}

labels = list(tests.keys())
p_vals = list(tests.values())
adjusted = benjamini_hochberg(p_vals)

print("Benjamini-Hochberg FDR Correction")
print("=" * 70)
print(f"{'Test':<22s} {'Raw p':>12s} {'BH-adjusted':>12s} {'Sig (0.05)':>10s}")
print("-" * 70)

for label, raw, adj in sorted(zip(labels, p_vals, adjusted), key=lambda x: x[1]):
    sig = "***" if adj < 0.001 else "**" if adj < 0.01 else "*" if adj < 0.05 else "n.s."
    print(f"{label:<22s} {raw:12.6f} {adj:12.6f} {sig:>10s}")

print("-" * 70)
print(f"Total tests: {len(tests)}")

# Check: do any originally significant results lose significance?
print("\n\nCRITICAL CHECK: Do any key results change after FDR correction?")
key_results = [
    ("Q7B_balanced",  "Relevance Dominance (Qwen-7B)"),
    ("Q7B_diversity", "Diversity Blindness (Qwen-7B)"),
    ("DS_balanced",   "Relevance Dominance (DeepSeek-V3)"),
    ("DS_diversity",  "Diversity Blindness (DeepSeek-V3)"),
    ("GEE_div_vs_bal","GEE: diversity = balanced"),
    ("GEE_delta_cov", "GEE: Coverage threshold effect"),
]

for key, desc in key_results:
    idx = labels.index(key)
    raw = p_vals[idx]
    adj = adjusted[idx]
    status = "STILL SIGNIFICANT" if adj < 0.05 else "LOST SIGNIFICANCE"
    print(f"  {desc:<40s}: raw={raw:.6f} -> adj={adj:.6f}  [{status}]")
