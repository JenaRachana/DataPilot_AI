from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate

from utils.json_utils import extract_json
from utils.llm import get_llm


def define_problem(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses an LLM to infer:
    - possible target columns
    - problem type (classification/regression)

    Returns structured output:
    {
        "candidates": [...],
        "recommended": {...}
    }
    """
    
    # Extract inputs
    business_problem = inputs["business_problem"]
    columns = inputs["columns"]
    sample_data = inputs["dataset_sample"]
    feature_metadata = inputs["feature_metadata"]
    feature_stats = inputs["feature_stats"]

    # Define prompt

    prompt = ChatPromptTemplate.from_template(
        """
You are a data science assistant.

Your task:
Given a business problem and dataset information, identify:
1. Possible target columns
2. The problem type (classification or regression)

---

Business Problem:
{business_problem}

Columns:
{columns}

Sample Data:
{sample_data}

Feature Types:
{feature_metadata}

Feature Statistics:
{feature_stats}

---

Instructions:
- Suggest 2 to 3 possible target columns.
- For each, specify:
    - target_column
    - problem_type (classification or regression)
    - reason
- Also provide ONE recommended option

---

Return ONLY valid JSON in this exact format:

{{
    "candidates": [
        {{
            "target_column": "column_name",
            "problem_type": "classification/regression",
            "reason": "why this is suitable"
        }},
    ],
    "recommended": {{
        "target_column": "column_name",
        "problem_type": "classification",
        "reason": "why this is the best recommendation"
    }}
}}

DO NOT include any explanation outside JSON.
"""
    )

    # Initialize model (from the central LLM config)
    llm = get_llm(temperature=0)

    chain = prompt | llm

    # Call LLM
    response = chain.invoke({
        "business_problem": business_problem,
        "columns": columns,
        "sample_data": sample_data,
        "feature_metadata": feature_metadata,
        "feature_stats": feature_stats
    })

    parsed = extract_json(response.content)

    # Validation
    if "candidates" not in parsed or "recommended" not in parsed:
        raise ValueError(f"Invalid response format:\n{parsed}")

    return parsed