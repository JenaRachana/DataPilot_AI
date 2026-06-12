"""Feature engineering phase agent: task instructions for the shared code agent."""

TASK_INSTRUCTIONS = """
Engineer features that should improve model performance for this problem:
interaction terms, polynomial features, date/time parts, binning, ratios, or
aggregations where they make sense for this dataset.
- Avoid target leakage: never build features from the target column.
- Keep the target column intact and keep `df` fully numeric and model-ready.
- Any indicator/dummy columns must be integer 0/1, not boolean dtype
  (`pd.get_dummies(..., dtype=int)` or `.astype(int)`) — bool columns get
  excluded from modelling downstream.
- Reassign the enhanced DataFrame back to `df`.
- If you rebuild `df` from transformed parts, save the target to a variable first
  and reattach it with a matching index; never read the target from a DataFrame
  you dropped it from.
Set outputs['engineered_features'] to a list of the new column names you added.
"""
