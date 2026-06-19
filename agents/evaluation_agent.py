"""Evaluation phase agent: task instructions for the shared code agent."""

TASK_INSTRUCTIONS = """
Produce an in-depth evaluation of the best model identified earlier (see "Best
model so far" in the context).
- Re-create the same train/test split (train_test_split(..., random_state=42))
  and retrain the best model type on `df` for a consistent evaluation.
- Compute a thorough metrics report:
    * classification -> accuracy, precision, recall, f1, confusion_matrix, and
      roc_auc when binary;
    * regression -> rmse, mae, r2.
- Build helpful figures (e.g. confusion matrix heatmap, ROC curve, residual
  plot, and a feature-importance bar chart when available) and append each as a
  (title:str, fig) tuple to outputs['figures'].
Set outputs['evaluation_report'] to a dict of the computed metrics (JSON-friendly
numbers, not numpy types).
"""
