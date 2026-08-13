from typing import Dict, Optional, Tuple, Union

import pandas as pd

from .utils import resample_by_time


def get_cold(
    data: pd.DataFrame, reference_data: pd.DataFrame, col: str = "user_id"
) -> pd.DataFrame:
    """
    Mark entries in data as 'cold' if their ID is not present in reference_data.

    Args:
        data: Target DataFrame to evaluate for cold entries.
        reference_data: Reference DataFrame with known IDs.
        col: Column name to check for coldness (e.g., 'user_id' or 'item_id').

    Returns:
        A copy of data with an added boolean 'is_cold' column.
    """
    # Get warm values from reference_data
    warm = reference_data[col].unique()

    # Mark entries not in warm set as cold
    final_df = data.copy()
    final_df["is_cold"] = ~final_df[col].isin(warm)

    return final_df


def share_of_cold(
    data: pd.DataFrame, reference_data: pd.DataFrame, col: str = "user_id"
) -> Tuple[int, float, float]:
    """
    Calculates the share and count of cold entities and interactions.

    Args:
        data (pd.DataFrame): Target DataFrame to evaluate for cold entries.
        reference_data (pd.DataFrame): Reference DataFrame containing known entities.
        col (str): Column name to check for coldness (e.g., 'user_id' or 'item_id').

    Returns:
        Tuple[int, float, float]:
            - Number of cold entities.
            - Share of cold entities (by unique count).
            - Share of cold interactions (by total interactions).
    """
    cold_df = get_cold(data, reference_data, col)

    # Number of unique cold entities
    col_num = cold_df[cold_df["is_cold"]][col].nunique()

    # Share of cold entities in total count (e.g., share of cold users in all users)
    per_col = col_num / cold_df[col].nunique()

    # Share of cold intercations
    per_inter = cold_df["is_cold"].mean()

    return col_num, per_col, per_inter


def cold_stats(data: pd.DataFrame, reference_data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cold-start statistics for users and items.

    Args:
        data (pd.DataFrame): Target DataFrame to evaluate for cold entries.
        reference_data (pd.DataFrame): Reference DataFrame containing known entities.

    Returns:
        pd.DataFrame: A DataFrame summarizing cold-start metrics for users and items.
    """
    cold_user, cold_user_per_user, cold_user_per_inter = share_of_cold(
        data, reference_data, "user_id"
    )
    cold_item, cold_item_per_item, cold_item_per_inter = share_of_cold(
        data, reference_data, "item_id"
    )

    data = [
        [cold_user, cold_user_per_user, cold_user_per_inter],
        [cold_item, cold_item_per_item, cold_item_per_inter],
    ]

    metrics_df = pd.DataFrame(
        data,
        index=["Cold Users", "Cold Items"],
        columns=["Number", "Share (by count)", "Share (by interactions)"],
    )

    return metrics_df


def cold_counts(
    data: pd.DataFrame,
    reference_data: pd.DataFrame,
    col: str = "user_id",
    granularity: Optional[str] = None,
) -> Dict[str, Union[pd.Series, float]]:
    """
    Computes cold interaction counts over time.

    Args:
        data (pd.DataFrame): Target interactions DataFrame.
        reference_data (pd.DataFrame): Reference DataFrame containing known entities.
        col (str): Column name to check for coldness (e.g., 'user_id').
        granularity (Optional[str]): Time-based resampling granularity (e.g., 'D', 'W') from pandas.

    Returns:
        Dict[str, Union[pd.Series, float]]: Dictionary with total, cold interaction counts, and share.
    """
    df = get_cold(data, reference_data, col)

    if granularity:
        # Convert timestamps and set as index for resampling
        df = resample_by_time(df, granularity)

    # Calculate cold interaction counts
    cold_counts = df["is_cold"].sum()
    total_counts = df["item_id"].count()

    result = {
        "total_interactions": total_counts,
        "cold_interactions": cold_counts,
        "cold_share": cold_counts / total_counts,
    }

    return result
