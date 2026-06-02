"""Unit tests for the code executor (no LLM required)."""

import pandas as pd

from utils.code_exec import run_code, scan_code, build_api_hint


def test_run_code_transforms_df_and_outputs():
    df = pd.DataFrame({"a": [1, 2, 3]})
    res = run_code(
        "df['b'] = df['a'] * 2\noutputs['total'] = int(df['b'].sum())",
        {"df": df},
    )
    assert res["ok"] is True
    assert "b" in res["df"].columns
    assert res["outputs"]["total"] == 12
    assert res["error"] is None


def test_run_code_captures_errors_without_raising():
    df = pd.DataFrame({"a": [1]})
    res = run_code("raise ValueError('boom')", {"df": df})
    assert res["ok"] is False
    assert "boom" in res["error"]
    # On error the original frame is preserved.
    assert res["df"] is df


def test_scan_code_flags_risky_references():
    warnings = scan_code("import os\nos.system('echo hi')")
    assert warnings  # at least one warning surfaced


def test_scan_code_clean_code_has_no_warnings():
    assert scan_code("df['b'] = df['a'] + 1") == []


def test_api_hint_reports_real_signature_for_bad_kwarg():
    # The exact failure seen in the app: OneHotEncoder no longer takes `sparse`.
    code = "from sklearn.preprocessing import OneHotEncoder\nenc = OneHotEncoder(sparse=False)"
    error = (
        "TypeError: OneHotEncoder.__init__() got an unexpected keyword argument 'sparse'"
    )
    hint = build_api_hint(error, code)
    assert "OneHotEncoder" in hint
    # The real signature in modern sklearn exposes sparse_output, not sparse.
    assert "sparse_output" in hint
    assert "sparse=" not in hint.replace("sparse_output", "")


def test_api_hint_empty_when_no_error():
    assert build_api_hint("", "x = 1") == ""
