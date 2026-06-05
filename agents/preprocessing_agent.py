"""Preprocessing phase agent: task instructions for the shared code agent."""

TASK_INSTRUCTIONS = """
Clean and preprocess the dataset so it is model-ready.
- Handle missing values appropriately (impute or drop with justification).
- Encode categorical columns into numeric form (one-hot or ordinal as suitable).
  When one-hot encoding, produce integer 0/1 indicator columns, NOT booleans —
  use `pd.get_dummies(..., dtype=int)` (or cast the result with `.astype(int)`).
  Boolean dtype columns get excluded from modelling downstream, so keep them int.
- Optionally scale/normalise numeric features where it helps modelling.
- Keep the target column intact in `df` (do not drop or encode it away).
- Reassign the cleaned, fully-numeric DataFrame back to `df`.
- IMPORTANT — preserving the target safely: if you transform the features
  separately from the target, FIRST save the target to a variable
  (e.g. `target = df[TARGET].reset_index(drop=True)`), build the transformed
  feature DataFrame, then reattach the target to it
  (`features = features.reset_index(drop=True); features[TARGET] = target; df = features`).
  Never read the target column from a DataFrame you have already dropped it from.
Set outputs['preprocessing_summary'] to a short list of the steps you applied.
"""
