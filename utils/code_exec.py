"""
Core code executor for DataPilot AI.

Each phase's agent generates Python code which the human reviews/edits and
approves; the execution nodes then run that approved code here.

SECURITY NOTE (deliberate, user-accepted risk): this runs LLM-generated code
in-process. The agreed mitigations are non-blocking and live around this module:
  * the exact code is shown and editable before it runs (UI),
  * code runs only after explicit human approval (UI),
  * `scan_code` flags risky references so the reviewer's attention is drawn to
    them (surfaced as a warning banner, not a block),
  * file writes are expected to stay under artifacts/<thread_id>/.
Do not run this against untrusted CSVs without reviewing the generated code.
"""

import ast
import io
import contextlib
import importlib
import inspect
import re
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Plotting is configured for headless/Streamlit use.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import seaborn as sns  # noqa: F401
except Exception:  # pragma: no cover - seaborn is a declared dependency
    sns = None


# Patterns that warrant a reviewer's attention. These do NOT block execution;
# they populate the warning banner in the approval UI.
_RISKY_PATTERNS = {
    "filesystem/system access": r"\b(?:import\s+os|import\s+sys|os\.|sys\.)",
    "subprocess execution": r"\b(?:subprocess|os\.system|os\.popen|pty)\b",
    "network access": r"\b(?:socket|requests|urllib|httpx|http\.client|ftplib|smtplib)\b",
    "dynamic code execution": r"\b(?:eval|exec|compile|__import__)\s*\(",
    "raw file I/O": r"(?<!pd\.)(?<!pandas\.)\bopen\s*\(",
    "shell/pip escapes": r"(?:^|\n)\s*[!%]",
}


def scan_code(code: Optional[str]) -> List[str]:
    """Return human-readable warnings for risky references found in ``code``."""
    if not code:
        return []
    warnings: List[str] = []
    for label, pattern in _RISKY_PATTERNS.items():
        if re.search(pattern, code):
            warnings.append(f"Code references {label} — review carefully before approving.")
    return warnings


def _base_namespace() -> Dict[str, Any]:
    """The libraries every generated snippet is allowed to assume are present."""
    ns: Dict[str, Any] = {
        "pd": pd,
        "np": np,
        "plt": plt,
    }
    if sns is not None:
        ns["sns"] = sns
    return ns


def run_code(code: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute approved code in a scoped namespace.

    The namespace is seeded with pandas/numpy/matplotlib (+seaborn), an empty
    ``outputs`` dict, and anything passed in ``context`` (typically ``df`` and
    ``target_column``). Generated code is expected to mutate/replace ``df`` and
    to place any results in ``outputs``.

    Returns a result dict (never raises for code errors):
        {
            "ok": bool,
            "df": pd.DataFrame | None,   # the working frame after execution
            "outputs": dict,             # whatever the code placed in `outputs`
            "stdout": str,
            "error": str | None,         # formatted traceback when ok is False
        }
    """
    namespace = _base_namespace()
    namespace["outputs"] = {}
    if context:
        namespace.update(context)

    stdout_buffer = io.StringIO()
    error: Optional[str] = None

    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code, namespace)  # noqa: S102 - intentional, see module docstring
        ok = True
    except Exception:
        ok = False
        error = traceback.format_exc()

    result_df = namespace.get("df")
    if not isinstance(result_df, pd.DataFrame):
        result_df = context.get("df") if context else None

    return {
        "ok": ok,
        "df": result_df,
        "outputs": namespace.get("outputs", {}) if ok else {},
        "stdout": stdout_buffer.getvalue(),
        "error": error,
    }


# --------------------------------------------------------------------------- #
# API introspection for the self-healing retry loop.
#
# When generated code fails with a version-mismatch error (e.g. a renamed/removed
# keyword argument), we resolve the offending symbol against the *installed*
# package — using the failing code's own imports — and report its real signature.
# This hands the model the exact corrective fact instead of hoping it guesses.
# Everything here is best-effort and must never raise.
# --------------------------------------------------------------------------- #

_KWARG_ERROR = re.compile(
    r"(\w[\w.]*?)(?:\.__init__)?\(\) got an unexpected keyword argument '(\w+)'"
)
_MODULE_ATTR_ERROR = re.compile(r"module '([\w.]+)' has no attribute '(\w+)'")


def _import_map(code: str) -> Dict[str, tuple]:
    """Map names used in ``code`` to (module_path, attr_or_None) via its imports."""
    mapping: Dict[str, tuple] = {}
    try:
        tree = ast.parse(code or "")
    except Exception:
        return mapping
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mapping[alias.asname or alias.name.split(".")[0]] = (alias.name, None)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                mapping[alias.asname or alias.name] = (node.module, alias.name)
    return mapping


def _resolve(name: str, import_map: Dict[str, tuple]) -> Optional[Any]:
    """Resolve a (possibly dotted) name to an object using the code's imports."""
    parts = name.split(".")
    head = parts[0]
    if head not in import_map:
        return None
    module_path, attr = import_map[head]
    try:
        obj: Any = importlib.import_module(module_path)
    except Exception:
        return None
    if attr:
        obj = getattr(obj, attr, None)
    for part in parts[1:]:
        if obj is None:
            return None
        obj = getattr(obj, part, None)
    return obj


def _safe_signature(obj: Any) -> Optional[str]:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def build_api_hint(error_text: Optional[str], code: Optional[str]) -> str:
    """
    Produce a corrective API hint for a failed execution, or "" if none applies.

    Resolves symbols named in the error against the installed packages (via the
    failing code's imports) and reports their real signatures / attributes.
    """
    if not error_text:
        return ""

    import_map = _import_map(code or "")
    hints: List[str] = []

    for symbol, bad_kwarg in _KWARG_ERROR.findall(error_text):
        obj = _resolve(symbol, import_map)
        sig = _safe_signature(obj)
        if sig:
            hints.append(
                f"`{symbol}` does not accept `{bad_kwarg}` in the installed version. "
                f"Its real signature here is: {symbol}{sig}"
            )

    for module_path, bad_attr in _MODULE_ATTR_ERROR.findall(error_text):
        try:
            mod = importlib.import_module(module_path)
        except Exception:
            continue
        available = [n for n in dir(mod) if not n.startswith("_")]
        close = [n for n in available if bad_attr.lower() in n.lower()]
        if close:
            hints.append(
                f"`{module_path}` has no attribute `{bad_attr}`. "
                f"Similar available names: {', '.join(close[:8])}"
            )

    # Preserve order, drop duplicates.
    return "\n".join(dict.fromkeys(hints))
