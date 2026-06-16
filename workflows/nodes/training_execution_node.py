"""Training execution node: run approved modeling code, persist the best model."""

from typing import Any, Dict, Optional

import joblib

from workflows.state import GraphState
from workflows.nodes.phase_base import make_execution_node
from utils.artifacts import get_artifacts_dir


def _apply(result: Dict[str, Any], state: GraphState, config: Optional[Dict]) -> Dict[str, Any]:
    outputs = result.get("outputs", {})

    model_results = outputs.get("model_results", []) or []
    best_name = outputs.get("best_model_name")

    best_metrics: Dict[str, Any] = {}
    for entry in model_results:
        if isinstance(entry, dict) and entry.get("name") == best_name:
            best_metrics = entry.get("metrics", {})
            break

    model = outputs.get("model")
    model_path = None
    if model is not None:
        run_dir = get_artifacts_dir(config)
        model_path = run_dir / "model.pkl"
        joblib.dump(model, model_path)

    best_model = {
        "name": best_name,
        "metrics": best_metrics,
        "path": str(model_path) if model_path else None,
        "feature_columns": outputs.get("feature_columns", []),
    }

    return {
        "model_results": model_results,
        "best_model": best_model,
    }


training_execution_node = make_execution_node(
    phase="modeling",
    code_field="modeling_code",
    apply_outputs=_apply,
)
