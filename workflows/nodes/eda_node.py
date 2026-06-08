"""EDA planner node: generate analysis code, pause for approval."""

from agents.eda_agent import TASK_INSTRUCTIONS
from workflows.nodes.phase_base import make_planner_node

eda_node = make_planner_node(
    phase="eda",
    title="Review the EDA code",
    message="Approve or edit the exploratory-analysis code before it runs.",
    task_instructions=TASK_INSTRUCTIONS,
    code_field="eda_code",
)
