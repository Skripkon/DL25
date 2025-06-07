PRODUCT_CATEGORIES = {
    "Household": "Home, tools, outdoor, art works",
    "Books": "Publications, literature",
    "Clothing & Accessories": "Apparel, fashion",
    "Electronics": "Devices, gadgets"
}

SYSTEM_PROMPT = """
You are a product categorization expert. Your task is to classify product descriptions into predefined categories.

Available categories:
{}

Guidelines:
- Be precise and consistent in your categorization
- Consider the primary purpose and nature of the product
- If a product could fit multiple categories, choose the most dominant one
- For soft classification, provide probability scores that sum to 1
- For hard classification, select the single most appropriate category

""".strip().format("\n".join(f"- {cat}: {desc}" for cat, desc in PRODUCT_CATEGORIES.items()))

CLASSIFY_SOFT_PROMPT_TEMPLATE = """
Assign a probability score (0 < score < 1) to each category so they sum to 1.
Wrap your response in <answer></answer> tags.

# Expected format:
<answer>
{{
    "Household": <probability>,
    "Books": <probability>,
    "Clothing & Accessories": <probability>,
    "Electronics": <probability>
}}
</answer>

Product Description:
{description}

Provide only the JSON response without any additional text or explanations.
""".strip()

CLASSIFY_HARD_PROMPT_TEMPLATE = """
Select the most fitting category (among provided) for the given product description.
Wrap your response in <answer></answer> tags.

# Expected format:
<answer>
{{
    "Category": "<selected category>"
}}
</answer>

Product Description:
{description}

Provide only the JSON response without any additional text or explanations.
""".strip()
