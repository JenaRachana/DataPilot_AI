"""Unit tests for the code-agent response parser (no LLM required)."""

import pytest

from agents.code_agent import parse_code_response


def test_parses_explanation_plus_python_block():
    raw = (
        "This code adds a column.\n\n"
        "```python\n"
        "df['b'] = df['a'] * 2\n"
        "outputs['n'] = len(df)\n"
        "```"
    )
    result = parse_code_response(raw)
    assert "df['b'] = df['a'] * 2" in result["code"]
    assert result["explanation"].startswith("This code adds a column")


def test_parses_bare_fence_without_language():
    raw = "```\ndf = df.dropna()\n```"
    result = parse_code_response(raw)
    assert result["code"] == "df = df.dropna()"


def test_multiline_code_is_preserved():
    raw = "```python\nimport pandas as pd\nx = 1\ny = 2\n```"
    result = parse_code_response(raw)
    assert result["code"].count("\n") == 2  # three lines, two newlines


def test_legacy_json_object_still_works():
    raw = '{"code": "df = df.dropna()", "explanation": "drop NaNs"}'
    result = parse_code_response(raw)
    assert result["code"] == "df = df.dropna()"
    assert result["explanation"] == "drop NaNs"


def test_json_nested_inside_python_fence():
    # A model wrongly nests JSON in a code fence (with escaped newlines so it is
    # valid JSON) — we should still recover the code.
    raw = '```python\n{"code": "df = df.head()\\nx = 1", "explanation": "peek"}\n```'
    result = parse_code_response(raw)
    assert result["code"] == "df = df.head()\nx = 1"
    assert result["explanation"] == "peek"


def test_unterminated_fence_from_truncated_output():
    # Model ran out of tokens: opening fence, no closing ```.
    raw = (
        "This builds some plots.\n\n"
        "```python\n"
        "import pandas as pd\n"
        "fig, ax = plt.subplots()\n"
        "# truncated here"
    )
    result = parse_code_response(raw)
    assert "import pandas as pd" in result["code"]
    assert result["code"].rstrip().endswith("# truncated here")


def test_strips_reasoning_think_block():
    raw = (
        "<think>I should one-hot encode then scale.</think>\n"
        "Encodes and scales.\n"
        "```python\n"
        "df = df.dropna()\n"
        "```"
    )
    result = parse_code_response(raw)
    assert result["code"] == "df = df.dropna()"
    assert "I should one-hot" not in result["explanation"]


def test_empty_raises():
    with pytest.raises(ValueError):
        parse_code_response("")
