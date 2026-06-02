from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from workflows.state import GraphState

from workflows.nodes.input_preparation_node import input_preparation_node
from workflows.nodes.problem_definition_node import problem_definition_node


builder = StateGraph(GraphState)

# ----- Nodes -----
builder.add_node("input_preparation_node", input_preparation_node)
builder.add_node("problem_definition_node", problem_definition_node)

# ----- Edges -----
builder.add_edge(START, "input_preparation_node")
builder.add_edge("input_preparation_node", "problem_definition_node")
builder.add_edge("problem_definition_node", END)

# In-memory checkpointing. To persist across restarts, swap for SqliteSaver:
#   from langgraph.checkpoint.sqlite import SqliteSaver
#   checkpointer = SqliteSaver.from_conn_string("checkpoints.sqlite")
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
