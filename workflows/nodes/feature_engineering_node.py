"""Feature engineering planner node: generate FE code, pause for approval."""

from agents.feature_engineering_agent import TASK_INSTRUCTIONS
from workflows.nodes.phase_base import make_planner_node

feature_engineering_node = make_planner_node(
    phase="feature_engineering",
    title="Review the feature engineering code",
    message="Approve or edit the feature-engineering code before it runs.",
    task_instructions=TASK_INSTRUCTIONS,
    code_field="feature_engineering_code",
)
