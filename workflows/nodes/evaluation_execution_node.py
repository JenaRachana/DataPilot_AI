"""Evaluation execution node: run approved code, store report and figures."""

from typing import Any, Dict, Optional

from workflows.state import GraphState
from workflows.nodes.phase_base import make_execution_node
from utils.artifacts import get_artifacts_dir, save_figures


def _apply(result: Dict[str, Any], state: GraphState, config: Optional[Dict]) -> Dict[str, Any]:
    outputs = result.get("outputs", {})

    run_dir = get_artifacts_dir(config)
    artifacts = save_figures(outputs.get("figures", []), run_dir, phase="evaluation")

    report = outputs.get("evaluation_report", {}) or {}

    return {
        "evaluation_report": report,
        "eda_artifacts": artifacts,  # appended; tagged phase="evaluation"
    }


evaluation_execution_node = make_execution_node(
    phase="evaluation",
    code_field="evaluation_code",
    apply_outputs=_apply,
)
