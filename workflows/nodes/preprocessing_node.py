"""Preprocessing planner node: generate cleaning code, pause for approval."""

from agents.preprocessing_agent import TASK_INSTRUCTIONS
from workflows.nodes.phase_base import make_planner_node

preprocessing_node = make_planner_node(
    phase="preprocessing",
    title="Review the preprocessing code",
    message="Approve or edit the data-cleaning code before it runs.",
    task_instructions=TASK_INSTRUCTIONS,
    code_field="preprocessing_code",
)
