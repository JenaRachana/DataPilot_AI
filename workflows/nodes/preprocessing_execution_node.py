"""Preprocessing execution node: run approved code, update the working frame."""

from typing import Any, Dict, Optional

from workflows.state import GraphState
from workflows.nodes.phase_base import make_execution_node


def _apply(result: Dict[str, Any], state: GraphState, config: Optional[Dict]) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    df = result.get("df")
    if df is not None:
        updates["processed_dataset"] = df.to_dict(orient="records")
    return updates


preprocessing_execution_node = make_execution_node(
    phase="preprocessing",
    code_field="preprocessing_code",
    apply_outputs=_apply,
)
