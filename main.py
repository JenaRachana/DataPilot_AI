import pandas as pd

from workflows.graph import graph

df = pd.read_csv("Housing.csv")

initial_state = {
    "dataset": df.to_dict(orient="records"),
    "business_problem": "Predict house price",

    "dataset_sample": [],
    "feature_metadata": {},
    "feature_stats": {},

    "problem_definition_candidates": [],
    "problem_definition_reason": None,

    "target_column": None,
    "problem_type": None,

    "event_logs": []
}

result = graph.invoke(initial_state, config={"configurable": {"thread_id": "cli"}})

print(result)
