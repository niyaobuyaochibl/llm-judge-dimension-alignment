"""
Prompt protocols added for the TKDE major revision.

These protocols are intentionally separated from the original four prompts so
that revision experiments can test stronger local-diversity prompting and
system-level prompts with explicit global statistics.
"""

ANTI_POSITION_NOTE = (
    " The labels A and B are arbitrary and the order is randomized. Do not "
    "prefer either label because of its position. Base the verdict only on "
    "the stated criterion. If the same two lists were shown in the opposite "
    "order, a consistent evaluator would choose the opposite label for the "
    "same underlying list."
)


def _stats_block(label, metrics):
    return (
        f"{label}:\n"
        f"- Coverage@10: {metrics.get('coverage@k', 0):.4f}\n"
        f"- Entropy@10: {metrics.get('entropy@k', 0):.4f}\n"
        f"- Novelty@10: {metrics.get('novelty@k', 0):.4f}\n"
        f"- TailCoverage@10: {metrics.get('tail_coverage@k', 0):.4f}\n"
        f"- Gini@10: {metrics.get('gini@k', 0):.4f}\n"
    )


SYSTEM_PROMPTS = {
    "diversity_rubric": (
        "You are an expert recommender-system evaluator. Your task is to judge "
        "LOCAL LIST-LEVEL DIVERSITY only. Diversity means the displayed list "
        "covers a broad range of categories, genres, topics, item types, or "
        "styles, and avoids repeatedly recommending very similar items. Ignore "
        "relevance, popularity, description length, and overall quality. Do not "
        "try to infer catalog coverage across all users."
    ),
    "diversity_fewshot": (
        "You are an expert recommender-system evaluator. Judge LOCAL LIST-LEVEL "
        "DIVERSITY only. Example: if List A contains ten action movies and List B "
        "contains action, comedy, drama, documentary, and animation, choose B. "
        "Example: if List A contains restaurants from five cuisines and List B "
        "contains mostly pizza restaurants, choose A. Ignore relevance and judge "
        "only visible variety within the displayed list."
    ),
    "diversity_structured": (
        "You are an expert recommender-system evaluator. Judge LOCAL LIST-LEVEL "
        "DIVERSITY only. Silently follow this procedure: (1) identify the main "
        "category/topic/style of each item in List A and List B; (2) count how "
        "many distinct groups each list covers; (3) penalize repeated or highly "
        "similar items; (4) choose the list with broader visible variety. Ignore "
        "relevance and do not infer system-level catalog coverage."
    ),
    "global_diversity_stats": (
        "You are an expert recommender-system evaluator. Your task is to judge "
        "SYSTEM-LEVEL DISTRIBUTIONAL DIVERSITY. You are given aggregate metrics "
        "computed over all users for the two systems. Coverage@10 is the primary "
        "criterion; Entropy@10 and TailCoverage@10 are secondary criteria. The "
        "single displayed user list is illustrative and must not override the "
        "aggregate system statistics. Ignore relevance unless diversity metrics "
        "are exactly tied."
    ),
    "global_balanced_stats": (
        "You are an expert recommender-system evaluator. Judge overall system "
        "quality using both the displayed recommendation lists and aggregate "
        "system statistics. For diversity, rely on the supplied system-level "
        "Coverage@10, Entropy@10, Novelty@10, and TailCoverage@10 statistics. "
        "For relevance, use the user's history and displayed items. Make the "
        "trade-off explicit in your internal reasoning, then output only a final "
        "verdict."
    ),
}


INSTRUCTIONS = {
    "diversity_rubric": (
        "Using only visible list-level diversity, choose the list with broader "
        "category/topic/style coverage. Ignore relevance and system-level "
        "coverage. Respond with exactly: VERDICT: A, VERDICT: B, or VERDICT: TIE."
    ),
    "diversity_fewshot": (
        "Apply the examples from the system message. Choose the list with more "
        "visible variety among its 10 items. Respond with exactly: VERDICT: A, "
        "VERDICT: B, or VERDICT: TIE."
    ),
    "diversity_structured": (
        "Silently extract item categories/topics and compare breadth/repetition. "
        "Do not print the extraction. Respond with exactly: VERDICT: A, "
        "VERDICT: B, or VERDICT: TIE."
    ),
    "global_diversity_stats": (
        "Use the aggregate system-level diversity statistics as the authority. "
        "Prefer the system with higher Coverage@10; use Entropy@10 and "
        "TailCoverage@10 as tie-breakers. Respond with exactly: VERDICT: A, "
        "VERDICT: B, or VERDICT: TIE."
    ),
    "global_balanced_stats": (
        "Judge overall quality while using the supplied global statistics for "
        "diversity. Respond with exactly: VERDICT: A, VERDICT: B, or VERDICT: TIE."
    ),
}


USER_TEMPLATE = """Given a user's interaction history and two recommendation lists from different systems, determine which list is better under the specified protocol.

## User Profile
{user_profile}

{global_stats}
## List A
{list_a}

## List B
{list_b}

## Instructions
{instruction}

Remember: A/B order is randomized; do not choose A by default.

Your verdict:"""


def get_revision_prompt(protocol, user_profile, list_a, list_b,
                        metrics_a=None, metrics_b=None):
    if protocol not in SYSTEM_PROMPTS:
        raise KeyError(f"Unknown revision prompt protocol: {protocol}")

    global_stats = ""
    if protocol in {"global_diversity_stats", "global_balanced_stats"}:
        if metrics_a is None or metrics_b is None:
            raise ValueError(f"{protocol} requires metrics_a and metrics_b")
        global_stats = (
            "## Aggregate System Statistics\n"
            "These statistics are computed over all users, not just this user.\n"
            + _stats_block("System that produced List A", metrics_a)
            + _stats_block("System that produced List B", metrics_b)
            + "\n"
        )

    user = USER_TEMPLATE.format(
        user_profile=user_profile,
        global_stats=global_stats,
        list_a=list_a,
        list_b=list_b,
        instruction=INSTRUCTIONS[protocol],
    )
    return SYSTEM_PROMPTS[protocol] + ANTI_POSITION_NOTE, user


ALL_REVISION_PROTOCOLS = [
    "diversity_rubric",
    "diversity_fewshot",
    "diversity_structured",
    "global_diversity_stats",
    "global_balanced_stats",
]
