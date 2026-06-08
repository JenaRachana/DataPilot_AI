"""EDA phase agent: task instructions for the shared code agent."""

TASK_INSTRUCTIONS = """
Perform exploratory data analysis on `df`. Produce a handful (3-6) of informative
figures, such as: target distribution, correlation heatmap, missingness overview,
key feature distributions, and class balance (if classification).
- Build each figure with matplotlib/seaborn using explicit Figure objects.
- Append a (title:str, fig) tuple for EACH figure to the list outputs['figures'].
- Do NOT call plt.show(). Do NOT modify `df`.
Also set outputs['insights'] to a list of short, plain-language findings a
stakeholder would care about.
"""
