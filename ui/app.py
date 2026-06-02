import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Verify TLS against the OS trust store (portable; see utils/ssl_bootstrap.py).
from utils.ssl_bootstrap import install as _install_ssl_trust

_install_ssl_trust()

import hashlib
import io
import uuid
import zipfile
from pathlib import Path

import streamlit as st
import pandas as pd

from workflows.graph import graph
from langgraph.types import Command
from utils.llm import available_models, set_model

st.set_page_config(page_title="DataPilot AI", layout="wide")

# Pipeline phases, in order, for the progress stepper. Keys match interrupt
# payload `phase` values and node-name prefixes.
PHASES = [
    ("input_preparation", "Input"),
    ("problem_definition", "Problem"),
    ("preprocessing", "Preprocess"),
    ("eda", "EDA"),
    ("feature_engineering", "Features"),
    ("modeling", "Modeling"),
    ("evaluation", "Evaluation"),
    ("deployment", "Deploy"),
]

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "started" not in st.session_state:
    st.session_state.started = False


def get_config():
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def reset_session():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.started = False


def _safe_invoke(graph_input) -> bool:
    """Run the graph, surfacing rate-limit / API errors as a friendly message
    instead of crashing the app. Returns True on success."""
    try:
        with st.spinner("Running workflow..."):
            graph.invoke(graph_input, config=get_config())
        return True
    except Exception as exc:  # noqa: BLE001 - show any backend error to the user
        msg = str(exc)
        low = msg.lower()
        if any(k in low for k in ("connection", "connect", "11434", "ollama", "max retries")):
            st.error(
                "**Couldn't reach the local model (Ollama).** Make sure Ollama is "
                "running and the model is pulled:\n\n"
                "```\nollama serve\nollama pull qwen2.5-coder:7b\n```\n"
                "Or pick another model in the sidebar."
            )
        elif "rate" in low or "429" in low or "quota" in low or "tokens per" in low:
            st.error(
                "**Rate limit reached.** Wait a minute and click again — your "
                "progress is saved, so it resumes from here. Or switch models in "
                "the sidebar; you may have hit a daily cap."
            )
        else:
            st.error("The workflow hit an error. You can click the button again to retry.")
        with st.expander("Error details"):
            st.code(msg)
        return False


def _resume(resume_value):
    """Resume the graph from an interrupt and advance to the next pause/end."""
    if _safe_invoke(Command(resume=resume_value)):
        st.rerun()


# --------------------------------------------------------------------------- #
# Progress stepper
# --------------------------------------------------------------------------- #
def _current_phase_key(snapshot):
    """Return the active phase key, or '__done__' when finished, else None."""
    if snapshot.interrupts:
        return snapshot.interrupts[0].value.get("phase")
    if not snapshot.next:
        return "__done__"
    nxt = snapshot.next[0] if snapshot.next else ""
    for key, _ in PHASES:
        if nxt.startswith(key):
            return key
    return None


def render_stepper(snapshot):
    keys = [k for k, _ in PHASES]
    cur = _current_phase_key(snapshot)
    done = cur == "__done__"
    cur_idx = len(keys) if done else (keys.index(cur) if cur in keys else 0)

    parts = []
    for i, (_, label) in enumerate(PHASES):
        if done or i < cur_idx:
            parts.append(f"✅ {label}")
        elif i == cur_idx:
            parts.append(f"🔵 **{label}**")
        else:
            parts.append(f"⚪ {label}")
    st.markdown("&nbsp;&nbsp;›&nbsp;&nbsp;".join(parts))


# --------------------------------------------------------------------------- #
# Inspector tabs (right pane)
# --------------------------------------------------------------------------- #
def render_dataset_tab(values):
    processed = values.get("processed_dataset")
    original = pd.DataFrame(values.get("dataset", []))

    # 1. The dataset itself — working frame if preprocessing has run, else the input sample.
    if processed:
        working = pd.DataFrame(processed)
        o_rows, o_cols = original.shape if not original.empty else (0, 0)
        st.caption(
            f"Working dataset — original {o_rows}×{o_cols} → now "
            f"{working.shape[0]}×{working.shape[1]}"
        )
        if not original.empty:
            added = [c for c in working.columns if c not in original.columns]
            removed = [c for c in original.columns if c not in working.columns]
            if added:
                st.caption(f"➕ Added: {', '.join(map(str, added))}")
            if removed:
                st.caption(f"➖ Removed: {', '.join(map(str, removed))}")
        st.dataframe(working.head(30), width="stretch")
    else:
        sample = values.get("dataset_sample")
        if sample:
            st.caption("Input sample (preprocessing not run yet)")
            st.dataframe(pd.DataFrame(sample), width="stretch")
        else:
            st.info("The dataset will appear here once the workflow starts.")

    # 2. Input-prep feature analysis (computed on the original dataset) — always shown.
    meta = values.get("feature_metadata", {})
    stats = values.get("feature_stats", {})

    if meta:
        with st.expander("Feature types", expanded=True):
            st.dataframe(
                pd.DataFrame([{"Column": c, "Type": t} for c, t in meta.items()]),
                width="stretch",
            )

    if stats:
        with st.expander("Feature statistics", expanded=False):
            numeric_rows, cat_rows = [], []
            for col, s in stats.items():
                if meta.get(col) == "numeric":
                    numeric_rows.append({
                        "Feature": col, "Mean": s.get("mean"), "Median": s.get("median"),
                        "Std": s.get("std"), "Min": s.get("min"), "Max": s.get("max"),
                        "Missing": s.get("missing"),
                    })
                else:
                    cat_rows.append({
                        "Feature": col, "Unique": s.get("unique_count"),
                        "Mode": s.get("mode"), "Missing": s.get("missing"),
                    })
            if numeric_rows:
                st.caption("Numeric features")
                st.dataframe(pd.DataFrame(numeric_rows), width="stretch")
            if cat_rows:
                st.caption("Categorical features")
                st.dataframe(pd.DataFrame(cat_rows), width="stretch")


def render_charts_tab(values):
    insights = values.get("eda_insights", [])
    artifacts = values.get("eda_artifacts", [])
    if not insights and not artifacts:
        st.info("EDA and evaluation charts will appear here as they're produced.")
        return
    for insight in insights:
        st.markdown(f"- {insight}")
    for art in artifacts:
        path = art.get("path")
        if path and Path(path).exists():
            caption = art.get("title", "")
            phase = art.get("phase")
            if phase:
                caption = f"{caption} ({phase})"
            st.image(path, caption=caption, width="stretch")


def render_console_tab(values):
    entries = values.get("console_output", [])
    if not entries:
        st.info("Code execution output (stdout + tracebacks) will appear here.")
        return

    st.caption(f"{len(entries)} execution(s) — oldest first. Click a row to expand.")
    for i, entry in enumerate(entries, start=1):
        ok = entry.get("ok")
        phase = entry.get("phase", "?")
        stdout = (entry.get("stdout") or "").strip()
        error = (entry.get("error") or "").strip()

        if ok:
            summary = "ran successfully"
        else:
            last_line = error.splitlines()[-1] if error else "error"
            summary = last_line[:70] + ("…" if len(last_line) > 70 else "")
        icon = "✅" if ok else "❌"

        # Collapsed by default so a huge traceback never floods the pane.
        with st.expander(f"{icon} {i}. {phase} — {summary}", expanded=False):
            if stdout:
                st.caption("stdout")
                st.code(stdout, language="text")
            if error:
                st.caption("traceback")
                st.code(error, language="text")
            if not stdout and not error:
                st.caption("(no output)")


def render_inspector(values):
    tab_data, tab_charts, tab_console = st.tabs(["📊 Dataset", "📈 Charts", "🖥 Console"])
    with tab_data:
        render_dataset_tab(values)
    with tab_charts:
        render_charts_tab(values)
    with tab_console:
        render_console_tab(values)


# --------------------------------------------------------------------------- #
# Action pane (left): reviews + results
# --------------------------------------------------------------------------- #
def render_problem_definition_review(payload):
    recommended = payload["recommended"]
    candidates = payload["candidates"]

    st.subheader("Human Review — Problem Definition")
    st.info(payload["message"])

    with st.container(border=True):
        st.markdown("**Recommended**")
        st.write(f"Target column: `{recommended['target_column']}`")
        st.write(f"Problem type: `{recommended['problem_type']}`")
        st.write(f"Reason: {recommended.get('reason', '')}")

    if st.button("Approve recommendation", type="primary"):
        _resume(recommended)

    with st.expander("Pick a different candidate"):
        targets = [c["target_column"] for c in candidates]
        types = sorted({c["problem_type"] for c in candidates})
        sel_target = st.selectbox("Target column", targets, key="pd_target")
        sel_type = st.selectbox("Problem type", types, key="pd_type")
        if st.button("Submit custom selection"):
            _resume(
                {
                    "target_column": sel_target,
                    "problem_type": sel_type,
                    "reason": "User override",
                }
            )


def render_code_review(payload):
    phase = payload["phase"]
    code = payload.get("code", "")
    st.subheader(f"Human Review — {phase.replace('_', ' ').title()}")
    st.info(payload["message"])
    if payload.get("explanation"):
        with st.expander("What this code does", expanded=True):
            st.markdown(payload["explanation"])

    previous_error = payload.get("previous_error")
    if previous_error:
        st.error(
            "The previous attempt **failed** and was automatically regenerated. "
            "This is a new attempt — review it before running."
        )
        with st.expander("Previous error"):
            st.code(previous_error, language="text")

    for warning in payload.get("warnings", []):
        st.warning(warning)

    # Key widgets on the code content so a regenerated version produces a FRESH
    # widget (Streamlit ignores `value=` for an existing key).
    token = hashlib.md5(code.encode("utf-8")).hexdigest()[:8]

    st.markdown("**Generated code**")
    st.code(code, language="python")

    comments = st.text_area(
        "Comments / instructions for the AI (optional)",
        height=80,
        key=f"comments_{phase}_{token}",
        placeholder="e.g., use OrdinalEncoder instead of one-hot. Sent to the AI on Regenerate.",
    )

    col_run, col_regen = st.columns(2)
    with col_run:
        if st.button("✅ Approve & run", type="primary", key=f"approve_{phase}_{token}"):
            _resume({"action": "approve", "code": code, "comments": comments})
    with col_regen:
        if st.button("🔄 Regenerate", key=f"regen_{phase}_{token}"):
            _resume({"action": "regenerate", "comments": comments})


def render_plan_review(payload):
    phase = payload["phase"]
    st.subheader(f"Human Review — {phase.replace('_', ' ').title()}")
    st.info(payload["message"])
    if payload.get("explanation"):
        st.write(payload["explanation"])
    if st.button("Approve", type="primary", key=f"approve_{phase}"):
        _resume({"action": "approve", "code": None})


def render_review(payload):
    phase = payload.get("phase")
    if phase == "problem_definition":
        render_problem_definition_review(payload)
    elif payload.get("code") is not None:
        render_code_review(payload)
    else:
        render_plan_review(payload)


def render_results(values):
    st.success("✅ Workflow complete.")

    with st.expander("Problem Definition", expanded=True):
        st.write(f"Target column: `{values.get('target_column')}`")
        st.write(f"Problem type: `{values.get('problem_type')}`")
        st.write(values.get("problem_definition_reason", ""))

    results = values.get("model_results", [])
    best = values.get("best_model") or {}
    if results or best:
        with st.expander("Modeling", expanded=True):
            if best:
                st.markdown(f"**Best model:** `{best.get('name')}`")
                st.json(best.get("metrics", {}))
                model_path = best.get("path")
                if model_path and Path(model_path).exists():
                    st.download_button(
                        "Download trained model (.pkl)",
                        data=Path(model_path).read_bytes(),
                        file_name="model.pkl",
                        key="dl_model",
                    )
            if results:
                rows = [{"Model": r.get("name"), **r.get("metrics", {})} for r in results]
                st.dataframe(pd.DataFrame(rows), width="stretch")

    report = values.get("evaluation_report")
    if report:
        with st.expander("Evaluation", expanded=True):
            st.json(report)

    deployment = values.get("deployment_artifacts", [])
    if deployment:
        with st.expander("Deployment Artifacts", expanded=True):
            present = [a for a in deployment if a.get("path") and Path(a["path"]).exists()]
            if present:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for art in present:
                        zf.write(art["path"], arcname=art["name"])
                st.download_button(
                    "⬇️ Download all (.zip)",
                    data=zip_buffer.getvalue(),
                    file_name="datapilot_pipeline.zip",
                    mime="application/zip",
                    type="primary",
                    key="dl_all_zip",
                )
            for art in present:
                st.download_button(
                    f"Download {art['name']}",
                    data=Path(art["path"]).read_bytes(),
                    file_name=art["name"],
                    key=f"dl_{art['name']}",
                )
            st.markdown("---")
            st.markdown("**How to run this pipeline on a new dataset**")
            st.markdown(
                "1. Download all three files into the **same folder**.\n"
                "2. Install deps and run on a CSV with the **same columns** as training:"
            )
            st.code("pip install -r requirements.txt\npython pipeline.py new_data.csv", language="bash")
            st.caption(
                "`pipeline.py` loads `model.pkl`, applies the preprocessing + "
                "feature-engineering steps, and writes your rows plus a `prediction` "
                "column to `<input>_predictions.csv`."
            )


# --------------------------------------------------------------------------- #
# Sidebar log feeds (collapsed)
# --------------------------------------------------------------------------- #
def render_sidebar_logs(values):
    event_logs = values.get("event_logs", [])
    exec_logs = values.get("execution_logs", [])
    with st.sidebar.expander("Workflow log", expanded=False):
        for log in event_logs:
            line = f"[{log['step']}] {log['message']}"
            if log.get("level") == "ERROR":
                st.error(line)
            elif log.get("level") == "WARNING":
                st.warning(line)
            else:
                st.write(line)
    with st.sidebar.expander("Execution history", expanded=False):
        for log in exec_logs:
            st.caption(f"[{log['step']}] {log['message']}")


# --------------------------------------------------------------------------- #
# Sidebar inputs
# --------------------------------------------------------------------------- #
st.title("DataPilot AI")

st.sidebar.header("Inputs")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
business_problem = st.sidebar.text_input(
    "Business problem", placeholder="e.g., Predict house price"
)

_models = available_models()
selected_model = st.sidebar.selectbox(
    "LLM model",
    list(_models.keys()),
    format_func=lambda mid: _models[mid],
    help="Used for all reasoning / code-generation steps. Changeable between runs.",
)
set_model(selected_model)

run_clicked = st.sidebar.button("▶ Run workflow", type="primary")
if st.sidebar.button("↺ Reset"):
    reset_session()
    st.rerun()

# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #
if uploaded_file is None:
    st.info("Upload a CSV file in the sidebar to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

# Start a new run (only on the explicit Run click).
if run_clicked and not st.session_state.started:
    st.session_state.started = True
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
    if _safe_invoke(initial_state):
        st.rerun()

if not st.session_state.started:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), width="stretch")
    st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.info("Pick your model in the sidebar and click **Run workflow**.")
    st.stop()

# ---- Cockpit: stepper on top, action left, inspector right ----
snapshot = graph.get_state(get_config())
values = snapshot.values

render_stepper(snapshot)
st.divider()

action_col, inspector_col = st.columns([3, 2], gap="large")

with action_col:
    if snapshot.interrupts:
        render_review(snapshot.interrupts[0].value)
    elif not snapshot.next:
        render_results(values)
    else:
        st.warning("The workflow paused before finishing (possibly a rate limit).")
        if st.button("Resume / retry", type="primary"):
            if _safe_invoke(None):
                st.rerun()

with inspector_col:
    render_inspector(values)

render_sidebar_logs(values)
