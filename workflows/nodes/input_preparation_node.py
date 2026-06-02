import pandas as pd
from workflows.state import GraphState
from utils.logging_utils import create_log

def input_preparation_node(state: GraphState):
    """
    This node prepares the dataset for all downstream agents.

    It does NOT use any LLM.
    It simply extracts useful information from the dataset.

    Input:
        state["dataset"] → pandas DataFrame

    Output:
        dataset_sample → small subset for LLM reasoning
        feature_metadata → column type info (numeric/categorical)
        feature_stats → basic statistics for each column
    """

    logs = []
    logs.append(
        create_log(
            step="input_preparation_node",
            message="Started input preparation"
        )
    )
    # Get Dataset
    df = pd.DataFrame(state["dataset"])

    # Create data sample
    sample_df = df.sample(n=min(20, len(df)), random_state=42)

    dataset_sample = sample_df.to_dict(orient="records")

    logs.append(
        create_log(
            step="input_preparation_node",
            message="Generated dataset sample"
        )
    )

    # Feature Metadata
    feature_metadata = {}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_metadata[col] = "numeric"
        else:
            feature_metadata[col] = "categorical"
    
    logs.append(
        create_log(
            step="input_preparation_node",
            message="Computed feature metadata"
        )
    )

    # Feature Stats
    feature_stats = {}

    for col in df.columns:
        col_data = df[col]

        if feature_metadata[col] == "numeric":
            feature_stats[col] = {
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "missing": int(col_data.isna().sum())
            }
        else:
            mode_value = None
            if not col_data.mode().empty:
                mode_value = col_data.mode().iloc[0]

            feature_stats[col] = {
                "unique_count": int(col_data.nunique()),
                "top_values": col_data.value_counts().head(5).index.to_list(),
                "mode": mode_value,
                "missing": int(col_data.isna().sum())
            }
    
    logs.append(
        create_log(
            step="input_preparation_node",
            message="Computed feature statistics"
        )
    )


    return {
        "dataset_sample": dataset_sample,
        "feature_metadata": feature_metadata,
        "feature_stats": feature_stats,
        "event_logs": [
            create_log(
                step="input_preparation_node",
                message="Completed input preparation"
            )
        ],
        "execution_logs": logs
    }