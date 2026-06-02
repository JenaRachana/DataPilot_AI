"""
Shared code-generating agent.

Every code phase (preprocessing, EDA, feature engineering, modeling, evaluation)
asks the LLM for a `{code, explanation}` payload that operates on a pandas
DataFrame named `df` and places results in an `outputs` dict. This helper builds
the common prompt scaffold (dataset context + output contract + optional
prior-error feedback) so each phase only supplies its task-specific instructions.
"""

import re
from typing import Any, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate

from utils.json_utils import extract_json
from utils.llm import get_llm

# A closed ```python (or bare ```) fenced block; captures its contents.
_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)
# An *unterminated* fence (truncated output): capture everything after the opener.
_OPEN_FENCE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*)$", re.DOTALL | re.IGNORECASE)
# Reasoning traces some models emit; stripped before parsing.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)


def _strip_reasoning(text: str) -> str:
    """Remove <think>...</think> reasoning traces (closed or truncated)."""
    text = _THINK_BLOCK.sub("", text)
    text = _OPEN_THINK.sub("", text)
    return text.strip()


def parse_code_response(raw: str) -> Dict[str, str]:
    """
    Extract ``{code, explanation}`` from a code-agent reply.

    Primary format: a plain-text explanation followed by a ```python code block.
    Tolerates reasoning traces and a truncated/unterminated code fence, and falls
    back to a legacy ``{"code", "explanation"}`` JSON object (including JSON wrongly
    nested in a fence). Raises ValueError only if no code can be recovered.
    """
    if not raw or not raw.strip():
        raise ValueError("Empty response from code agent.")

    text = _strip_reasoning(raw.strip())

    # 1. A properly closed fenced block.
    match = _CODE_BLOCK.search(text)
    if match:
        code = match.group(1).strip()
        explanation = text[: match.start()].strip()
        # Guard: a model may have placed a {code, explanation} JSON inside the fence.
        if code.lstrip().startswith("{") and '"code"' in code:
            try:
                obj = extract_json(code)
                if isinstance(obj, dict) and obj.get("code"):
                    return {
                        "code": obj["code"],
                        "explanation": obj.get("explanation", explanation),
                    }
            except ValueError:
                pass
        if code:
            return {"code": code, "explanation": explanation}

    # 2. An unterminated fence (output was truncated): take everything after it.
    open_match = _OPEN_FENCE.search(text)
    if open_match:
        code = open_match.group(1).strip().rstrip("`").strip()
        explanation = text[: open_match.start()].strip()
        if code:
            return {"code": code, "explanation": explanation}

    # 3. Fallback: a bare JSON object {code, explanation}.
    try:
        obj = extract_json(text)
        if isinstance(obj, dict) and obj.get("code"):
            return {"code": obj["code"], "explanation": obj.get("explanation", "")}
    except ValueError:
        pass

    raise ValueError(f"Could not extract code from agent output:\n{raw}")

def _installed_versions() -> str:
    """Report installed versions of the libraries generated code may use, so the
    LLM targets the actual environment instead of an outdated remembered API."""
    import importlib.metadata as md

    parts = []
    for name in ("scikit-learn", "pandas", "numpy", "scipy", "matplotlib", "seaborn"):
        try:
            parts.append(f"{name}=={md.version(name)}")
        except Exception:
            pass
    return ", ".join(parts)


_LIBRARY_VERSIONS = _installed_versions()


_PROMPT = ChatPromptTemplate.from_template(
    """
You are a senior data scientist writing Python for one stage of an ML pipeline.

A pandas DataFrame named `df` already exists in scope. The libraries `pd`
(pandas), `np` (numpy), `plt` (matplotlib.pyplot) and `sns` (seaborn) are
already imported. A dict named `outputs` already exists for you to fill in.

Dataset context
- Business problem: {business_problem}
- Target column: {target_column}
- Problem type: {problem_type}
- Columns: {columns}
- Feature types: {feature_metadata}
- Feature statistics: {feature_stats}
- Sample rows: {dataset_sample}
- Best model so far (if any): {best_model}
- Prior model results (if any): {model_results}

Your task for THIS stage:
{task_instructions}

{error_feedback}

Installed library versions (write code for THESE exact versions, not older APIs):
{library_versions}

Rules:
- Operate on the existing `df` (read and reassign it as needed). Do NOT read from
  or write to disk, and do NOT reload the data.
- Put any results the pipeline needs into the `outputs` dict exactly as the task
  describes.
- Keep the code self-contained and runnable top-to-bottom. Import only from
  pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, or joblib if needed.
- Use APIs compatible with the installed versions above. For example, in
  scikit-learn >= 1.2 `OneHotEncoder` uses `sparse_output=False` (the old
  `sparse=` argument was removed); prefer `pd.get_dummies` for simple encoding.
- When you build a DataFrame from a transformer's output, ensure the column count
  matches (use the transformer's output feature names), or keep it as an array.
- Do not call input(), exit(), or any networking/OS/subprocess APIs.

Respond in exactly this format and nothing else:
1. A clear, SPECIFIC explanation of what the code does — walk through it step by
   step and name the concrete columns/values it creates or changes and WHY,
   grounded in THIS dataset and problem. Do not be generic ("engineers some
   features"); say exactly which features/transformations and their purpose, and
   note any assumptions or caveats. A short paragraph or a bulleted list is ideal.
2. Then the complete Python in a SINGLE fenced code block:

```python
# your code here
```

Do NOT wrap your answer in JSON. Output only the explanation followed by one
```python code block.
"""
)


def generate_phase_code(
    task_instructions: str,
    context: Dict[str, Any],
    last_error: Optional[str] = None,
    api_hint: Optional[str] = None,
    user_comments: Optional[str] = None,
    attempt: int = 0,
) -> Dict[str, str]:
    """
    Ask the LLM for `{code, explanation}` for one pipeline stage.

    ``context`` provides the dataset framing (business_problem, target_column,
    problem_type, columns, feature_metadata, feature_stats, dataset_sample).
    When ``last_error`` is given, the model is asked to fix the prior failure;
    ``api_hint`` (the real signature of the offending symbol, from the installed
    package) is included so the fix is grounded in the actual environment.
    ``user_comments`` are reviewer instructions to follow in the generated code.
    """
    if user_comments and user_comments.strip():
        task_instructions = (
            f"{task_instructions}\n\nReviewer instructions — follow these exactly:\n"
            f"{user_comments.strip()}"
        )

    error_feedback = ""
    if last_error:
        error_feedback = (
            "IMPORTANT — your previous code FAILED with the error below. Write a "
            "DIFFERENT solution that specifically fixes this error; do not repeat "
            f"the same approach or the same mistake:\n{last_error}"
        )
        if api_hint:
            error_feedback += (
                "\n\nVerified API facts from the installed environment (use these "
                f"exactly):\n{api_hint}"
            )
        # An attempt counter keeps each retry's prompt distinct, so the response
        # cache can't return the same failing code, and a materially different
        # fix is requested each time.
        if attempt:
            error_feedback += (
                f"\n\n(Retry attempt #{attempt}. Previous fixes did not work — "
                "take a materially different approach.)"
            )

    # Deterministic on the first try; warmer on retries so a repeated failure
    # explores a genuinely different solution instead of regenerating the same code.
    temperature = 0.4 if last_error else 0.0

    llm = get_llm(temperature=temperature)
    chain = _PROMPT | llm

    response = chain.invoke(
        {
            "task_instructions": task_instructions,
            "error_feedback": error_feedback,
            "library_versions": _LIBRARY_VERSIONS,
            "business_problem": context.get("business_problem"),
            "target_column": context.get("target_column"),
            "problem_type": context.get("problem_type"),
            "columns": context.get("columns"),
            "feature_metadata": context.get("feature_metadata"),
            "feature_stats": context.get("feature_stats"),
            "dataset_sample": context.get("dataset_sample"),
            "best_model": context.get("best_model"),
            "model_results": context.get("model_results"),
        }
    )

    return parse_code_response(response.content)
