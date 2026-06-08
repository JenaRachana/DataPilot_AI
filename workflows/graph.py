from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from workflows.state import GraphState

from workflows.nodes.input_preparation_node import input_preparation_node
from workflows.nodes.problem_definition_node import problem_definition_node

from workflows.nodes.preprocessing_node import preprocessing_node
from workflows.nodes.preprocessing_execution_node import preprocessing_execution_node

from workflows.nodes.eda_node import eda_node
from workflows.nodes.eda_execution_node import eda_execution_node

from workflows.nodes.phase_base import make_error_router


builder = StateGraph(GraphState)

# ----- Nodes -----
builder.add_node("input_preparation_node", input_preparation_node)
builder.add_node("problem_definition_node", problem_definition_node)
builder.add_node("preprocessing_node", preprocessing_node)
builder.add_node("preprocessing_execution_node", preprocessing_execution_node)
builder.add_node("eda_node", eda_node)
builder.add_node("eda_execution_node", eda_execution_node)

# ----- Edges -----
builder.add_edge(START, "input_preparation_node")
builder.add_edge("input_preparation_node", "problem_definition_node")
builder.add_edge("problem_definition_node", "preprocessing_node")
builder.add_edge("preprocessing_node", "preprocessing_execution_node")
builder.add_edge("eda_node", "eda_execution_node")

# Self-healing: re-plan on execution error, else advance.
builder.add_conditional_edges(
    "preprocessing_execution_node",
    make_error_router("preprocessing_node", "eda_node"),
    ["preprocessing_node", "eda_node"],
)
builder.add_conditional_edges(
    "eda_execution_node",
    make_error_router("eda_node", END),
    ["eda_node", END],
)

# In-memory checkpointing. To persist across restarts, swap for SqliteSaver:
#   from langgraph.checkpoint.sqlite import SqliteSaver
#   checkpointer = SqliteSaver.from_conn_string("checkpoints.sqlite")
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
