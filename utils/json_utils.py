"""
Robust JSON extraction from LLM responses.

LLMs frequently wrap JSON in ```json fences or add prose before/after the
object. This utility recovers the JSON payload from such responses so every
agent can share one resilient parser instead of fragile string stripping.
"""

import json
import re
from typing import Any


def extract_json(raw: str) -> Any:
    """
    Parse the first JSON object/array found in an LLM response.

    Handles:
      - Plain JSON
      - ```json ... ``` and ``` ... ``` fenced blocks
      - Leading/trailing prose around the JSON

    Raises ValueError if no valid JSON can be recovered.
    """
    if raw is None:
        raise ValueError("LLM returned no content to parse as JSON.")

    text = raw.strip()

    # 1. Prefer the contents of a fenced code block if present.
    fence_match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Fall through to brace scanning on the fenced content.
            text = candidate

    # 2. Try parsing the whole thing directly.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3. Fall back to the first balanced {...} or [...] span.
    span = _find_json_span(text)
    if span is not None:
        try:
            return json.loads(span)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from LLM output:\n{raw}")


def _find_json_span(text: str) -> str | None:
    """Return the first balanced JSON object/array substring, if any."""
    starts = {"{": "}", "[": "]"}
    for i, ch in enumerate(text):
        if ch in starts:
            closer = starts[ch]
            depth = 0
            in_string = False
            escape = False
            for j in range(i, len(text)):
                c = text[j]
                if escape:
                    escape = False
                    continue
                if c == "\\":
                    escape = True
                    continue
                if c == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == ch:
                    depth += 1
                elif c == closer:
                    depth -= 1
                    if depth == 0:
                        return text[i : j + 1]
            # Unbalanced from this start; try the next opener.
    return None
