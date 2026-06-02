# DataPilot AI

An agentic data-science assistant built with **LangGraph** + **Streamlit**. It
walks a dataset through an end-to-end ML pipeline, generating the Python for each
phase and **pausing for human approval** before anything runs.

Status: foundation — input preparation and LLM problem definition, on a reusable
human-in-the-loop (HITL) node framework.

```bash
uv run streamlit run ui/app.py
```
