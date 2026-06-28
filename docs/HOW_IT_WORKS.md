# How DataPilot AI Works — A Guide for Complete Beginners

This document assumes **no prior knowledge** of this project, LangGraph, or
agentic AI. By the end you'll understand what the app does, the ideas behind it,
and how to read the code yourself.

---

## Part 1 — What is this app, in plain words?

Imagine hiring a junior data scientist who:
1. Looks at your spreadsheet,
2. Figures out what you're trying to predict,
3. Cleans the data, explores it, builds features, trains models, evaluates them,
   and packages the result —
4. **but checks in with you before doing each step**, showing you exactly the
   code they're about to run so you can approve or change it.

That junior data scientist is **DataPilot AI**. You upload a CSV, type your goal
in plain English ("predict house price"), and it walks the whole machine-learning
pipeline with you in control at every step.

---

## Part 2 — The five big ideas

You only need five concepts to understand the whole system.

### Idea 1: The pipeline is a series of *phases*
A data-science project has natural stages. We model each as a **phase**:

```
Input prep → Problem definition → Preprocessing → EDA →
Feature engineering → Modeling → Evaluation → Deployment
```

### Idea 2: A *graph* runs the phases in order
We use a library called **LangGraph**. Think of it as a flowchart engine: each
box is a step ("node"), and arrows connect them. LangGraph runs node 1, passes
the results to node 2, and so on. Our flowchart lives in
[`workflows/graph.py`](../workflows/graph.py).

### Idea 3: *State* is the shared notebook
As the graph runs, every node reads from and writes to one shared dictionary
called the **state** (`GraphState` in [`workflows/state.py`](../workflows/state.py)).
It holds the dataset, the chosen target column, the trained model, the logs —
everything. Each node returns a small update; LangGraph merges it in.

### Idea 4: An *LLM agent* writes the code; an *executor* runs it
For the thinking steps, we ask a large language model to **write Python code** for
that phase. That's the **agent**. Then a separate piece of code actually **runs**
that Python — that's the **executor** ([`utils/code_exec.py`](../utils/code_exec.py)).
The model is configured in one place, [`utils/llm.py`](../utils/llm.py) — a local
Ollama model (`qwen2.5-coder:7b`) by default, or a cloud model (Gemini or Groq).

> So the AI doesn't magically transform your data — it *writes a small program*,
> and we run that program. You get to see and edit the program first.

### Idea 5: *Human-in-the-loop* (HITL) — the pause for approval
Before any AI-written code runs, the graph **pauses** and shows it to you. This
pause is done with LangGraph's `interrupt()`. You **Approve** (or edit) and the
graph **resumes**. This is the safety mechanism and the heart of the app.

---

## Part 3 — Following one phase end-to-end

Let's trace the **Preprocessing** phase. Every code phase works identically, so
once you understand this one, you understand them all.

```
                 ┌──────────────────────┐
   (1) PLANNER    │  preprocessing_node  │   asks the LLM to write cleaning code
                 └──────────┬───────────┘
                            │  interrupt() — PAUSE
                            ▼
                 ┌──────────────────────┐
   (2) YOU        │   Human review (UI)  │   read / edit / Approve the code
                 └──────────┬───────────┘
                            │  resume with approved code
                            ▼
                 ┌──────────────────────────────┐
   (3) EXECUTOR   │ preprocessing_execution_node │  runs the code on the data
                 └──────────┬───────────────────┘
                            │
              error? ───────┤── yes ──► back to (1) to regenerate (self-healing)
                            │
                            ▼ no
                      next phase (EDA)
```

1. **Planner** ([`preprocessing_node.py`](../workflows/nodes/preprocessing_node.py)):
   calls the LLM with instructions ("clean this data"), gets back Python code,
   and calls `request_review(...)` which **pauses** the graph.
2. **You**: the Streamlit UI shows the code; you approve or edit it.
3. **Executor** ([`preprocessing_execution_node.py`](../workflows/nodes/preprocessing_execution_node.py)):
   runs the approved code on the dataset and saves the cleaned data back into the
   state. If the code crashed, it records the error and the graph loops back to
   the planner so the AI can fix it — then you review again.

The same two-node pattern (planner + executor) repeats for EDA, feature
engineering, modeling, and evaluation. This repetition is intentional — it's
**the standard** every phase follows.

---

## Part 4 — Reading the code (a guided tour)

Here's the order to read files so the project unfolds logically.

### Start here — the shape of the data
- [`workflows/state.py`](../workflows/state.py) — the shared "notebook"
  (`GraphState`). Skim the field names; they're the vocabulary of the whole app.

### The map
- [`workflows/graph.py`](../workflows/graph.py) — how the phases connect. Read the
  edges top to bottom; it's literally the flowchart. Note the
  `add_conditional_edges(...)` calls — those are the "retry on error" arrows.

### The reusable machinery (read this and you've read 80% of the logic)
- [`workflows/nodes/phase_base.py`](../workflows/nodes/phase_base.py) — the
  factories `make_planner_node` and `make_execution_node`. **Every code phase is
  built from these.** Understand these two functions and every phase file becomes
  obvious.
- [`utils/hitl.py`](../utils/hitl.py) — `request_review()`, the one function that
  pauses the graph for your approval. Defines the exact "payload" the UI receives.
- [`utils/code_exec.py`](../utils/code_exec.py) — `run_code()` actually executes
  the approved Python; `scan_code()` flags risky lines for the warning banner.
- [`agents/code_agent.py`](../agents/code_agent.py) — `generate_phase_code()`: the
  shared prompt that asks the LLM for `{code, explanation}`. The "brain."
- [`utils/llm.py`](../utils/llm.py) — the single place the model is configured
  (`get_llm()`): a local Ollama model by default, or a cloud model (Gemini /
  Groq). Also enables an in-process response cache so re-runs on resume don't
  re-call the model.

### A phase, concretely
Open one planner + one executor and see how thin they are:
- [`agents/preprocessing_agent.py`](../agents/preprocessing_agent.py) — just the
  task instructions (the phase-specific part of the prompt).
- [`workflows/nodes/preprocessing_node.py`](../workflows/nodes/preprocessing_node.py)
  — one call to `make_planner_node`.
- [`workflows/nodes/preprocessing_execution_node.py`](../workflows/nodes/preprocessing_execution_node.py)
  — one `apply_outputs` function + one call to `make_execution_node`.

### The two special cases
- [`workflows/nodes/input_preparation_node.py`](../workflows/nodes/input_preparation_node.py)
  — pure pandas, **no LLM and no pause** (there's nothing to approve). It just
  samples the data and computes column stats.
- [`workflows/nodes/problem_definition_node.py`](../workflows/nodes/problem_definition_node.py)
  — pauses for a **choice** (which target column) rather than for code.
- [`workflows/nodes/deployment_node.py`](../workflows/nodes/deployment_node.py) —
  pauses for a simple **confirmation**, then writes `pipeline.py`,
  `requirements.txt`, and references the saved `model.pkl`.

### The face of it
- [`ui/app.py`](../ui/app.py) — the Streamlit app. The important idea: the UI is
  driven entirely by the **saved graph state**, not by button clicks. It reads the
  current state, and if the graph is paused it renders **one generic review
  screen** that works for every phase (because every phase uses the same review
  payload from `request_review`).

### Proof it works
- [`tests/test_code_exec.py`](../tests/test_code_exec.py) — small unit tests for
  the executor.
- [`tests/test_pipeline_offline.py`](../tests/test_pipeline_offline.py) — runs the
  **entire** pipeline with the LLM replaced by canned code, so it needs no
  network. Best single file to read to see the whole flow exercised.
- [`tests/test_pipeline.py`](../tests/test_pipeline.py) — the same, but against the
  real configured LLM (Ollama, Gemini, or Groq).

---

## Part 5 — How to add a new phase (the payoff of the standard)

Because every phase follows the same pattern, adding one is mechanical:

1. **Agent**: create `agents/<phase>_agent.py` with a `TASK_INSTRUCTIONS` string
   describing what the LLM should write.
2. **Planner node**: create `workflows/nodes/<phase>_node.py` calling
   `make_planner_node(phase=..., code_field=..., task_instructions=...)`.
3. **Executor node**: create `workflows/nodes/<phase>_execution_node.py` with an
   `apply_outputs(result, state, config)` that folds the code's `outputs` back
   into the state, then call `make_execution_node(...)`.
4. **State**: add any new fields to `GraphState` in `workflows/state.py`.
5. **Wire it**: in `workflows/graph.py`, add the two nodes and the edges
   (planner → executor, and the conditional retry edge).
6. The **UI needs no changes** — the generic review renderer already handles it.

---

## Part 6 — Mini-glossary

| Term | Plain meaning |
|------|---------------|
| **LangGraph** | The engine that runs our steps in order, like a flowchart. |
| **Node** | One step/box in the flowchart (a Python function). |
| **State (`GraphState`)** | The shared dictionary all steps read from and write to. |
| **Agent** | The part that asks the LLM to write code for a phase. |
| **Executor** | The part that actually runs the AI-written code. |
| **LLM** | Large Language Model (here, local Qwen2.5-Coder via Ollama, or a cloud model like Gemini / Groq) — it writes the code. |
| **HITL / `interrupt()`** | Human-in-the-loop: the graph pauses for your approval. |
| **`Command(resume=...)`** | How the app un-pauses the graph after you approve. |
| **Checkpointer** | Saves the graph's state so it can pause and resume. |
| **Artifacts** | Files the run produces: charts, the `.pkl` model, the pipeline script. |

---

## Part 7 — One paragraph to remember

DataPilot AI is a **flowchart of data-science steps** (LangGraph) sharing one
**notebook of facts** (the state). The interesting steps ask an **LLM to write a
small Python program**, then **pause and show it to you** (human-in-the-loop)
before an **executor runs it** on your data. The same planner→approve→execute
pattern repeats for every phase, which is why the code is small and adding new
capabilities is easy.
