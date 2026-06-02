"""
Human-in-the-loop (HITL) helper.

Every phase that needs human approval pauses through this single function so
that the interrupt payload and the resume contract stay identical across the
whole pipeline. The Streamlit UI relies on this consistent shape to render one
generic review screen for all phases.

Interrupt payload shape:
    {
        "type": f"{phase}_review",
        "phase": phase,
        "title": str,
        "message": str,
        "code": str | None,         # python the human reviews/edits (None for plan reviews)
        "explanation": str | None,  # plain-language description of what the code does
        "warnings": list[str],      # risky-reference flags surfaced in the UI
        "review_logs": list[dict],  # log entries to show in the review panel
        **extra,                    # phase-specific extras (e.g. candidates for problem_definition)
    }

Resume contract (what the UI sends back via Command(resume=...)):
    {"action": "approve" | "modify", "code": "<approved/edited python>"}

For plan-style phases (e.g. problem definition) the resume value carries the
selection instead of code; see that node for its specific shape.
"""

from typing import Any, Dict, List, Optional

from langgraph.types import interrupt

from utils.code_exec import scan_code


def request_review(
    phase: str,
    title: str,
    message: str,
    code: Optional[str] = None,
    explanation: Optional[str] = None,
    logs: Optional[List[Dict]] = None,
    **extra: Any,
) -> Any:
    """
    Pause the graph for human review and return the resume value.

    If ``code`` is provided, it is scanned for risky references and the
    resulting warnings are attached to the payload so the UI can surface them.
    """
    warnings = scan_code(code) if code else []

    payload: Dict[str, Any] = {
        "type": f"{phase}_review",
        "phase": phase,
        "title": title,
        "message": message,
        "code": code,
        "explanation": explanation,
        "warnings": warnings,
        "review_logs": logs or [],
    }
    payload.update(extra)

    return interrupt(payload)
