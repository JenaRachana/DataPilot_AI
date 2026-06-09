"""EDA execution node: run approved code, persist figures and insights."""

from typing import Any, Dict, Optional

from workflows.state import GraphState
from workflows.nodes.phase_base import make_execution_node
from utils.artifacts import get_artifacts_dir, save_figures


def _apply(result: Dict[str, Any], state: GraphState, config: Optional[Dict]) -> Dict[str, Any]:
    outputs = result.get("outputs", {})

    run_dir = get_artifacts_dir(config)
    artifacts = save_figures(outputs.get("figures", []), run_dir, phase="eda")

    insights = outputs.get("insights", [])
    if isinstance(insights, str):
        insights = [insights]

    return {
        "eda_artifacts": artifacts,
        "eda_insights": list(insights),
    }


eda_execution_node = make_execution_node(
    phase="eda",
    code_field="eda_code",
    apply_outputs=_apply,
)
