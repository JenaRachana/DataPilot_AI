"""
Headless end-to-end test of the full pipeline.

Drives the graph from start to finish, auto-approving the generated code at each
human-review interrupt (mirroring what the UI does on "Approve"). Requires a
GROQ_API_KEY; skipped automatically when absent. Can also be run directly:

    uv run python -m tests.test_pipeline
"""

import os
import uuid
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langgraph.types import Command

# Verify TLS against the OS trust store (portable; see utils/ssl_bootstrap.py).
from utils.ssl_bootstrap import install as _install_ssl_trust

_install_ssl_trust()

from workflows.graph import graph

load_dotenv()

DATA_PATH = Path(__file__).resolve().parents[1] / "Housing.csv"
MAX_STEPS = 40  # safety cap against an unproductive self-healing loop


def _auto_resume_value(payload: dict):
    """Return the resume value the UI would send when approving as-is."""
    phase = payload.get("phase")
    if phase == "problem_definition":
        return payload["recommended"]
    if payload.get("code") is not None:
        return {"action": "approve", "code": payload["code"]}
    return {"action": "approve", "code": None}


def run_pipeline(business_problem: str = "Predict house price") -> dict:
    df = pd.read_csv(DATA_PATH)
    config = {"configurable": {"thread_id": f"e2e-{uuid.uuid4()}"}}

    initial_state = {
        "dataset": df.to_dict(orient="records"),
        "business_problem": business_problem,
        "dataset_sample": [],
        "processed_dataset": [],
        "feature_metadata": {},
        "feature_stats": {},
        "problem_definition_candidates": [],
        "problem_definition_reason": None,
        "target_column": None,
        "problem_type": None,
        "event_logs": [],
        "execution_logs": [],
    }

    graph.invoke(initial_state, config=config)

    for _ in range(MAX_STEPS):
        snapshot = graph.get_state(config)
        if snapshot.interrupts:
            payload = snapshot.interrupts[0].value
            resume = _auto_resume_value(payload)
            print(f"[resume] phase={payload.get('phase')}")
            graph.invoke(Command(resume=resume), config=config)
        elif not snapshot.next:
            break
        else:
            graph.invoke(None, config=config)

    return graph.get_state(config).values


def test_full_pipeline_reaches_a_trained_model():
    if not os.getenv("GROQ_API_KEY"):
        import pytest

        pytest.skip("GROQ_API_KEY not set")

    final = run_pipeline()

    assert final.get("target_column"), "problem definition did not set a target"
    assert final.get("best_model"), "no model was trained"
    assert final.get("evaluation_report"), "evaluation report is empty"
    assert final.get("deployment_artifacts"), "no deployment artifacts produced"


if __name__ == "__main__":
    result = run_pipeline()
    print("\n===== PIPELINE RESULT =====")
    print("target_column     :", result.get("target_column"))
    print("problem_type      :", result.get("problem_type"))
    print("best_model        :", (result.get("best_model") or {}).get("name"))
    print("model metrics     :", (result.get("best_model") or {}).get("metrics"))
    print("evaluation_report :", result.get("evaluation_report"))
    print("eda_artifacts     :", len(result.get("eda_artifacts", [])), "figures")
    print("deployment        :", [a["name"] for a in result.get("deployment_artifacts", [])])
