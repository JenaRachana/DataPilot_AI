
from typing import TypedDict, List, Dict, Optional, Any
from typing_extensions import Annotated

def append_list(old, new):
    return old + new

class GraphState(TypedDict):
    # ----- Raw input -----
    dataset: List[Dict]
    dataset_sample: List[Dict]

    # Working frame that evolves through preprocessing / feature engineering.
    # `dataset` is kept pristine; executors read/write `processed_dataset`.
    processed_dataset: List[Dict]

    # ----- Problem definition -----
    business_problem: Optional[str]
    target_column: Optional[str]
    problem_type: Optional[str]
    problem_definition_candidates: List[Dict]
    problem_definition_reason: Optional[str]

    # ----- Feature analysis (input preparation) -----
    feature_metadata: Dict[str, str]
    feature_stats: Dict[str, Dict]

    # ----- Per-phase generated code (latest LLM proposal for each phase) -----
    preprocessing_code: Optional[str]
    eda_code: Optional[str]
    feature_engineering_code: Optional[str]
    modeling_code: Optional[str]

    # Append-only log of every approved code block that was executed.
    transformation_history: Annotated[List[str], append_list]

    # ----- EDA outputs -----
    eda_insights: Annotated[List[str], append_list]
    eda_artifacts: Annotated[List[Dict], append_list]

    # ----- Modeling outputs -----
    model_results: Annotated[List[Dict], append_list]
    best_model: Optional[Dict]

    # ----- Execution tracking (drives the self-healing retry loop) -----
    last_executed_code: Optional[str]
    last_error: Optional[str]

    # ----- Logging -----
    event_logs: Annotated[List[Dict], append_list]
    execution_logs: Annotated[List[Dict], append_list]

    # Console stream: one entry per code execution (stdout + full traceback),
    # for a console-style view in the UI.
    console_output: Annotated[List[Dict], append_list]
