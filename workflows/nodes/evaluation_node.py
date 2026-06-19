"""Evaluation planner node: generate evaluation code, pause for approval."""

from agents.evaluation_agent import TASK_INSTRUCTIONS
from workflows.nodes.phase_base import make_planner_node

evaluation_node = make_planner_node(
    phase="evaluation",
    title="Review the evaluation code",
    message="Approve or edit the model-evaluation code before it runs.",
    task_instructions=TASK_INSTRUCTIONS,
    code_field="evaluation_code",
)
