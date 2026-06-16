"""Modeling planner node: generate model-training code, pause for approval."""

from agents.modeling_agent import TASK_INSTRUCTIONS
from workflows.nodes.phase_base import make_planner_node

modeling_node = make_planner_node(
    phase="modeling",
    title="Review the modeling code",
    message="Approve or edit the model-training code before it runs.",
    task_instructions=TASK_INSTRUCTIONS,
    code_field="modeling_code",
)
