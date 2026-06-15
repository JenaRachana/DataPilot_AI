"""Modeling phase agent: task instructions for the shared code agent."""

TASK_INSTRUCTIONS = """
Train and compare candidate scikit-learn models for this problem.
- Separate the target column from the features in `df`.
- Use train_test_split(..., random_state=42) to make a hold-out test set.
- Train 2-4 appropriate scikit-learn models for the stated problem type
  (e.g. LinearRegression/Ridge/RandomForestRegressor/GradientBoostingRegressor
  for regression; LogisticRegression/RandomForestClassifier/
  GradientBoostingClassifier for classification). Use ONLY scikit-learn models.
- Evaluate each model on the test set with suitable metrics.

Populate outputs as follows:
- outputs['model_results']: list of dicts, each {'name': str, 'metrics': {metric: value}}.
- outputs['model']: the single best trained estimator object.
- outputs['best_model_name']: the name of that best model.
- outputs['feature_columns']: list of feature column names used (excluding target).
Do not save anything to disk; the pipeline persists the model for you.
"""
