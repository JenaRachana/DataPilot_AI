import pandas as pd
from workflows.state import GraphState
from agents.problem_definition_agent import define_problem
from utils.logging_utils import create_log
from utils.hitl import request_review


def problem_definition_node(state: GraphState):
    """
    LLM recommends a target column + problem type, then pauses for human review.

    This is the one plan-style review in the pipeline (no code to run), so its
    resume value carries the selection rather than code:
        {"target_column": ..., "problem_type": ..., "reason": ...}
    """
    logs = [
        create_log(
            step="problem_definition_node",
            message="Started problem definition",
        )
    ]

    df = pd.DataFrame(state["dataset"])

    inputs = {
        "business_problem": state["business_problem"],
        "columns": list(df.columns),
        # Send only a few rows to the LLM to keep the prompt (and token use) small.
        "dataset_sample": (state.get("dataset_sample") or [])[:6],
        "feature_metadata": state["feature_metadata"],
        "feature_stats": state["feature_stats"],
    }

    logs.append(
        create_log(
            step="problem_definition_node",
            message="Calling problem definition LLM",
        )
    )

    result = define_problem(inputs)

    logs.append(
        create_log(
            step="problem_definition_node",
            message="Generated target column recommendations",
        )
    )

    workflow_event = create_log(
        step="problem_definition_node",
        message="Awaiting user approval",
    )
    logs.append(workflow_event)

    # Plan-style review: no code, but carries recommendation + candidates so the
    # UI can render a selector. Resume value is the chosen selection dict.
    approved_selection = request_review(
        phase="problem_definition",
        title="Review the recommended target column and problem type",
        message="Approve the recommendation or pick a different candidate.",
        code=None,
        explanation=result["recommended"].get("reason"),
        logs=[workflow_event],
        recommended=result["recommended"],
        candidates=result["candidates"],
    )

    logs.append(
        create_log(
            step="problem_definition_node",
            message=(
                f"Approved target column: {approved_selection['target_column']}"
            ),
        )
    )

    return {
        "problem_definition_candidates": result["candidates"],
        "target_column": approved_selection["target_column"],
        "problem_type": approved_selection["problem_type"],
        "problem_definition_reason": approved_selection.get("reason", ""),
        "event_logs": [
            create_log(
                step="problem_definition_node",
                message="Completed problem definition",
            )
        ],
        "execution_logs": logs,
    }
