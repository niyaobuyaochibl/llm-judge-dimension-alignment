"""
Dimension-specific prompt templates for LLM-as-a-Judge evaluation.
Four evaluation dimensions: relevance, diversity, novelty, balanced (standard).
"""

SYSTEM_PROMPTS = {
    "balanced": (
        "You are an expert recommendation system evaluator. "
        "Your task is to judge which recommendation list better serves the user "
        "based on relevance, diversity, and overall quality. "
        "Be fair, objective, and base your judgment solely on content quality."
    ),
    "relevance": (
        "You are an expert recommendation system evaluator. "
        "Your SOLE task is to judge which recommendation list contains more RELEVANT items "
        "for this specific user based on their interaction history. "
        "Focus ONLY on how well items match the user's demonstrated interests and preferences. "
        "Ignore diversity, novelty, or any other factor. Only relevance matters."
    ),
    "diversity": (
        "You are an expert recommendation system evaluator. "
        "Your SOLE task is to judge which recommendation list provides more DIVERSE recommendations. "
        "Focus ONLY on whether the items cover a wider range of categories, genres, topics, or styles. "
        "A list with items from many different categories is better than one where all items are similar. "
        "Ignore relevance or accuracy. Only diversity matters."
    ),
    "novelty": (
        "You are an expert recommendation system evaluator. "
        "Your SOLE task is to judge which recommendation list introduces more NOVEL and SURPRISING items "
        "that the user has NOT seen before and might not discover on their own. "
        "Focus ONLY on how unexpected, fresh, and discovery-oriented the recommendations are. "
        "Ignore relevance and diversity. Only novelty and serendipity matter."
    ),
}

USER_TEMPLATE = (
    "Given a user's interaction history and two recommendation lists from different systems, "
    "determine which list is better.\n\n"
    "## User Profile\n{user_profile}\n\n"
    "## List A\n{list_a}\n\n"
    "## List B\n{list_b}\n\n"
    "## Instructions\n{instruction}\n\n"
    "Respond with ONLY \"A\", \"B\", or \"TIE\".\n\n"
    "Your verdict:"
)

INSTRUCTIONS = {
    "balanced": (
        "Compare the two lists based on: (1) Relevance to user interests, "
        "(2) Diversity of recommendations, (3) Overall quality. "
        "Choose the list that provides the best overall recommendations."
    ),
    "relevance": (
        "Compare the two lists based SOLELY on RELEVANCE. "
        "Which list contains items that better match this user's demonstrated preferences? "
        "Count how many items in each list align with the user's interests."
    ),
    "diversity": (
        "Compare the two lists based SOLELY on DIVERSITY. "
        "Which list covers a broader range of categories, genres, or topics? "
        "A diverse list avoids recommending too many similar items."
    ),
    "novelty": (
        "Compare the two lists based SOLELY on NOVELTY. "
        "Which list contains more surprising, unexpected, or discovery-oriented items? "
        "Items that are less mainstream or less obvious choices count as more novel."
    ),
}


def get_dimension_prompt(dimension: str, user_profile: str, list_a: str, list_b: str):
    """Build system + user prompts for a specific evaluation dimension."""
    system = SYSTEM_PROMPTS[dimension]
    user = USER_TEMPLATE.format(
        user_profile=user_profile,
        list_a=list_a,
        list_b=list_b,
        instruction=INSTRUCTIONS[dimension],
    )
    return system, user
