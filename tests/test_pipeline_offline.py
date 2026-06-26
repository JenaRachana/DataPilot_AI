"""
Offline end-to-end orchestration test (no network / no LLM).

Stubs the LLM agents with deterministic canned code so the full graph —
interrupts, resume, code execution, output mapping, artifacts, and completion —
is exercised without any API calls. This is the regression guard for the wiring
that previously broke. The live-LLM counterpart is tests/test_pipeline.py.
"""

import uuid

import pandas as pd
import pytest
from langgraph.types import Command

import workflows.nodes.problem_definition_node as pd_node
import workflows.nodes.phase_base as phase_base
from workflows.graph import graph


# --- Canned code per phase (selected by keywords in the task instructions) --- #
_PREPROCESSING = """
df = df.dropna()
df = pd.get_dummies(df, drop_first=True)
for c in df.columns:
    if df[c].dtype == bool:
        df[c] = df[c].astype(int)
outputs['preprocessing_summary'] = ['dropna', 'one-hot encode']
"""

_EDA = """
fig, ax = plt.subplots()
ax.hist(df[target_column])
ax.set_title('Target distribution')
outputs['figures'] = [('Target distribution', fig)]
outputs['insights'] = ['Target column shows a right-skewed distribution.']
"""

_FE = """
outputs['engineered_features'] = []
"""

_MODELING = """
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
X = df.drop(columns=[target_column])
y = df[target_column]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
results, best, best_score, best_name = [], None, float('-inf'), None
for name, model in [('LinearRegression', LinearRegression()),
                    ('RandomForest', RandomForestRegressor(random_state=42))]:
    model.fit(Xtr, ytr)
    score = float(r2_score(yte, model.predict(Xte)))
    results.append({'name': name, 'metrics': {'r2': score}})
    if score > best_score:
        best_score, best, best_name = score, model, name
outputs['model_results'] = results
outputs['model'] = best
outputs['best_model_name'] = best_name
outputs['feature_columns'] = list(X.columns)
"""

_EVALUATION = """
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
X = df.drop(columns=[target_column])
y = df[target_column]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(Xtr, ytr)
pred = model.predict(Xte)
outputs['evaluation_report'] = {
    'r2': float(r2_score(yte, pred)),
    'rmse': float(mean_squared_error(yte, pred) ** 0.5),
}
fig, ax = plt.subplots()
ax.scatter(yte, pred)
ax.set_title('Predicted vs actual')
outputs['figures'] = [('Predicted vs actual', fig)]
"""


def _fake_generate_phase_code(
    task_instructions, context, last_error=None, api_hint=None, user_comments=None, attempt=0
):
    t = task_instructions.lower()
    if "preprocess" in t:
        code = _PREPROCESSING
    elif "exploratory" in t:
        code = _EDA
    elif "engineer features" in t:
        code = _FE
    elif "train and compare" in t:
        code = _MODELING
    else:  # evaluation
        code = _EVALUATION
    return {"code": code, "explanation": "canned test code"}


def _fake_define_problem(inputs):
    return {
        "candidates": [
            {"target_column": "price", "problem_type": "regression", "reason": "numeric target"},
        ],
        "recommended": {
            "target_column": "price",
            "problem_type": "regression",
            "reason": "price is the continuous outcome to predict",
        },
    }


def test_offline_pipeline_completes(monkeypatch):
    monkeypatch.setattr(phase_base, "generate_phase_code", _fake_generate_phase_code)
    monkeypatch.setattr(pd_node, "define_problem", _fake_define_problem)

    df = pd.read_csv("Housing.csv")
    config = {"configurable": {"thread_id": f"offline-{uuid.uuid4()}"}}
    initial_state = {
        "dataset": df.to_dict(orient="records"),
        "business_problem": "Predict house price",
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

    for _ in range(40):
        snap = graph.get_state(config)
        if snap.interrupts:
            payload = snap.interrupts[0].value
            if payload.get("phase") == "problem_definition":
                resume = payload["recommended"]
            elif payload.get("code") is not None:
                resume = {"action": "approve", "code": payload["code"]}
            else:
                resume = {"action": "approve", "code": None}
            graph.invoke(Command(resume=resume), config=config)
        elif not snap.next:
            break
        else:
            graph.invoke(None, config=config)

    final = graph.get_state(config).values
    assert final.get("target_column") == "price"
    assert final.get("best_model"), "no model trained"
    assert final["best_model"].get("name") in {"LinearRegression", "RandomForest"}
    assert final.get("evaluation_report"), "evaluation report empty"
    assert final.get("eda_artifacts"), "no figures produced"
    assert final.get("deployment_artifacts"), "no deployment artifacts"


def test_offline_pipeline_handles_regenerate(monkeypatch):
    # The reviewer asks to regenerate once (with comments) at the first code
    # review, then approves everything. The graph must still complete.
    monkeypatch.setattr(phase_base, "generate_phase_code", _fake_generate_phase_code)
    monkeypatch.setattr(pd_node, "define_problem", _fake_define_problem)

    df = pd.read_csv("Housing.csv")
    config = {"configurable": {"thread_id": f"offline-regen-{uuid.uuid4()}"}}
    initial_state = {
        "dataset": df.to_dict(orient="records"),
        "business_problem": "Predict house price",
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

    regenerated = False
    for _ in range(45):
        snap = graph.get_state(config)
        if snap.interrupts:
            payload = snap.interrupts[0].value
            if payload.get("phase") == "problem_definition":
                resume = payload["recommended"]
            elif payload.get("code") is not None:
                if payload.get("phase") == "preprocessing" and not regenerated:
                    regenerated = True
                    resume = {"action": "regenerate", "comments": "use a simpler approach"}
                else:
                    resume = {"action": "approve", "code": payload["code"]}
            else:
                resume = {"action": "approve", "code": None}
            graph.invoke(Command(resume=resume), config=config)
        elif not snap.next:
            break
        else:
            graph.invoke(None, config=config)

    assert regenerated, "the regenerate branch was never exercised"
    final = graph.get_state(config).values
    assert final.get("best_model"), "pipeline did not complete after a regenerate"
