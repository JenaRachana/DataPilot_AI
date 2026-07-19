"""
Factories for the standard phase pattern.

Every code phase is built from two nodes:
  * a **planner** node  : generate code (LLM) -> request_review() -> store approved code
  * an **execution** node: run the approved code -> map outputs into state / capture error

These factories remove the boilerplate so each phase file only declares its
task instructions and how to fold its `outputs` back into GraphState.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd

import numpy as np

from workflows.state import GraphState
from agents.code_agent import generate_phase_code
from utils.hitl import request_review
from utils.code_exec import run_code, build_api_hint
from utils.logging_utils import create_log


def _to_native(obj):
    """Recursively convert numpy scalars/arrays to native Python types.

    LLM-generated code often stores sklearn metrics as numpy.float64, which the
    LangGraph checkpointer (msgpack) cannot serialize. Applying this to whatever
    an executor writes into state keeps checkpoints serializable.
    """
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    # Anything else (a matplotlib Figure, a DataFrame, ...) isn't checkpoint-safe;
    # fall back to a display-safe string rather than letting the checkpointer crash.
    return str(obj)


def build_context(state: GraphState) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Assemble the dataset framing for an agent and the working DataFrame."""
    records = state.get("processed_dataset") or state.get("dataset") or []
    df = pd.DataFrame(records)
    context = {
        "business_problem": state.get("business_problem"),
        "target_column": state.get("target_column"),
        "problem_type": state.get("problem_type"),
        "columns": list(df.columns),
        "feature_metadata": state.get("feature_metadata"),
        "feature_stats": state.get("feature_stats"),
        "dataset_sample": df.head(6).to_dict(orient="records"),
        # Surfaced for the evaluation phase (harmless/empty for earlier phases).
        "best_model": state.get("best_model"),
        "model_results": state.get("model_results"),
    }
    return context, df


def _prepend_comments(code: str, comments: str) -> str:
    """Record reviewer comments as Python comments atop the approved code."""
    body = "\n".join(f"# {line}" for line in comments.strip().splitlines())
    return f"# --- Reviewer notes ---\n{body}\n# ----------------------\n{code}"


def make_planner_node(
    phase: str,
    title: str,
    message: str,
    task_instructions: str,
    code_field: str,
) -> Callable[[GraphState], Dict[str, Any]]:
    """Build a planner node that generates code and pauses for human review."""

    def node(state: GraphState) -> Dict[str, Any]:
        step = f"{phase}_node"
        logs = [create_log(step=step, message=f"Generating {phase} code")]

        context, _ = build_context(state)

        # Error context comes from a failed execution that routed back here.
        last_error = state.get("last_error")
        # Number of executions so far — grows on each retry (stable across an
        # interrupt replay), so it keeps retry prompts distinct from the cache.
        attempt = len(state.get("console_output") or [])
        api_hint = None
        if last_error:
            # Resolve the offending symbol's real signature from the installed
            # package so the retry is grounded in the actual environment.
            api_hint = build_api_hint(last_error, state.get("last_executed_code"))
            logs.append(
                create_log(
                    step=step,
                    message="Regenerating code after a previous execution error",
                    level="WARNING",
                )
            )

        # Review loop: the human can approve/edit the code, or send comments back
        # to regenerate. Regeneration stays inside this node (re-interrupting),
        # so it does not need a separate graph edge.
        user_comments = None
        while True:
            result = generate_phase_code(
                task_instructions, context, last_error, api_hint, user_comments, attempt
            )

            review_event = create_log(step=step, message="Awaiting code approval")
            decision = request_review(
                phase=phase,
                title=title,
                message=message,
                code=result["code"],
                explanation=result.get("explanation", ""),
                logs=[review_event],
                previous_error=last_error,
            )

            if isinstance(decision, dict) and decision.get("action") == "regenerate":
                user_comments = decision.get("comments") or ""
                # The user is steering now; drop the stale execution-error context.
                last_error = None
                api_hint = None
                logs.append(
                    create_log(step=step, message="Regenerating with reviewer comments")
                )
                continue

            # Approve / modify.
            if isinstance(decision, dict) and decision.get("code"):
                approved_code = decision["code"]
            else:
                approved_code = result["code"]

            comments = decision.get("comments") if isinstance(decision, dict) else None
            if comments and comments.strip():
                approved_code = _prepend_comments(approved_code, comments)
            break

        logs.append(create_log(step=step, message="Code approved by user"))

        return {
            code_field: approved_code,
            "event_logs": [
                create_log(step=step, message=f"Approved {phase} code")
            ],
            "execution_logs": logs,
        }

    node.__name__ = f"{phase}_node"
    return node


def make_execution_node(
    phase: str,
    code_field: str,
    apply_outputs: Callable[[Dict[str, Any], GraphState, Optional[Dict]], Dict[str, Any]],
) -> Callable[..., Dict[str, Any]]:
    """
    Build an execution node that runs the approved code for ``phase``.

    ``apply_outputs(result, state, config)`` maps the executor result into state
    updates on success. Errors are captured (never raised) and surfaced via
    ``last_error`` so the graph can route back to the planner for a retry.
    """

    def node(state: GraphState, config=None) -> Dict[str, Any]:
        step = f"{phase}_execution_node"
        code = state.get(code_field) or ""

        _, df = build_context(state)
        result = run_code(
            code,
            {"df": df, "target_column": state.get("target_column")},
        )

        if not result["ok"]:
            short = (result["error"] or "").strip().splitlines()
            short_msg = short[-1] if short else "unknown error"
            return {
                "last_error": result["error"],
                "last_executed_code": code,
                "console_output": [
                    {
                        "phase": phase,
                        "ok": False,
                        "stdout": result.get("stdout", ""),
                        "error": result.get("error", ""),
                    }
                ],
                "event_logs": [
                    create_log(
                        step=step,
                        message=f"{phase} execution failed: {short_msg}",
                        level="ERROR",
                    )
                ],
                "execution_logs": [
                    create_log(
                        step=step,
                        message=f"Execution error, will regenerate: {short_msg}",
                        level="ERROR",
                    )
                ],
            }

        updates: Dict[str, Any] = {
            "last_error": None,
            "last_executed_code": code,
            "transformation_history": [code],
            "console_output": [
                {
                    "phase": phase,
                    "ok": True,
                    "stdout": result.get("stdout", ""),
                    "error": "",
                    "outputs": _to_native(result.get("outputs") or {}),
                }
            ],
            "event_logs": [
                create_log(step=step, message=f"Completed {phase} execution")
            ],
            "execution_logs": [
                create_log(step=step, message=f"{phase} code executed successfully")
            ],
        }
        updates.update(_to_native(apply_outputs(result, state, config) or {}))
        return updates

    node.__name__ = f"{phase}_execution_node"
    return node


def make_error_router(
    planner_node_name: str,
    next_node_name: str,
) -> Callable[[GraphState], str]:
    """Route back to the planner if the last execution errored, else continue."""

    def route(state: GraphState) -> str:
        return planner_node_name if state.get("last_error") else next_node_name

    return route
