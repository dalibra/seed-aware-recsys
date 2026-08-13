from typing import Dict, Optional, Union

import pandas as pd

from .utils import resample_by_time


def get_leaks(data: pd.DataFrame, reference_data: pd.DataFrame, convert_timestamp: bool = False) -> pd.DataFrame:
    """
    Detects data leaks by comparing timestamps against reference data.

    A leak is defined as an interaction that occurred on or before the latest
    timestamp for the same item_id in the reference_data (typically training subset).

    Args:
        data (pd.DataFrame): Target DataFrame to check for leaks.
        reference_data (pd.DataFrame): Reference DataFrame providing the max timestamp per 'item_id'.
        convert_timestamp (bool): Whether to convert the 'timestamp' column from Unix time to datetime. Defaults to False.

    Returns:
        pd.DataFrame: A copy of data with two additional columns:
                      - 'timestamp_ref_max': the max timestamp from 'reference_data' per item.
                      - 'is_leak': boolean flag indicating whether the entry is a leak.
    """
    final_df = data.copy()
    ref_df = reference_data.copy()

    final_df["timestamp"] = pd.to_datetime(final_df["timestamp"], unit="s")
    ref_df["timestamp"] = pd.to_datetime(ref_df["timestamp"], unit="s")

    ref_max_timestamp = ref_df.groupby("item_id")["timestamp"].max()

    final_df["timestamp_ref_max"] = final_df["item_id"].map(ref_max_timestamp)
    final_df["is_leak"] = final_df["timestamp"] <= final_df["timestamp_ref_max"]

    if not convert_timestamp:
        final_df["timestamp_ref_max"] = final_df["timestamp_ref_max"].astype(int) / 1e9
        final_df["timestamp"] = final_df["timestamp"].astype(int) / 1e9

    return final_df


def leak_counts(
    data: pd.DataFrame, reference_data: pd.DataFrame, granularity: Optional[str] = None
) -> Dict[str, Union[pd.Series, float]]:
    """
    Computes leak interactions count over time.

    Args:
        data (pd.DataFrame): Target interactions DataFrame.
        reference_data (pd.DataFrame): Reference DataFrame used to determine leak status.
        granularity (Optional[str]): Time-based resampling granularity (e.g., 'D', 'W', 'M') from pandas.

    Returns:
        Dict[str, Union[pd.Series, float]]: Dictionary with:
            - 'total_interactions': total number of interactions per time unit (or overall).
            - 'leak_interactions': number of leak interactions per time unit (or overall).
            - 'leak_share': proportion of leak interactions.
    """

    df = get_leaks(data, reference_data)

    if granularity:
        # Convert timestamps and set as index for resampling
        df = resample_by_time(df, granularity)

    # Calculate leak interaction counts
    leak_counts = df["is_leak"].sum()
    total_counts = df["item_id"].count()

    result = {
        "total_interactions": total_counts,
        "leak_interactions": leak_counts,
        "leak_share": leak_counts / total_counts,
    }

    return result

def temporal_overlap(data: pd.DataFrame, reference_data: pd.DataFrame):
    reference_data['timestamp_dt'] = pd.to_datetime(reference_data['timestamp'], unit='s')
    data['timestamp_dt'] = pd.to_datetime(data['timestamp'], unit='s')

    base_start, base_end = reference_data['timestamp_dt'].min(), reference_data['timestamp_dt'].max()
    new_start, new_end = data['timestamp_dt'].min(), data['timestamp_dt'].max()

    overlap_start = max(base_start, new_start)
    overlap_end = min(base_end, new_end)

    if overlap_start < overlap_end:
        overlap_duration = (overlap_end - overlap_start).total_seconds()
        total_base = (base_end - base_start).total_seconds()
        total_new = (new_end - new_start).total_seconds()

        overlap_share_ref = overlap_duration / total_base if total_base > 0 else 0
        overlap_share_tgt = overlap_duration / total_new if total_new > 0 else 0
    else:
        overlap_duration = 0
        overlap_start = overlap_end = pd.NaT
        overlap_share_ref = 0
        overlap_share_tgt = 0
    
    summary = pd.DataFrame([{
        'reference_start': base_start,
        'reference_end': base_end,
        'target_start': new_start,
        'target_end': new_end,
        'overlap_start': overlap_start,
        'overlap_end': overlap_end,
        'overlap_duration_sec': overlap_duration,
        'overlap_share_reference': overlap_share_ref,
        'overlap_share_target': overlap_share_tgt
    }])

    return summary

def find_shared_interactions(data: pd.DataFrame, reference_data: pd.DataFrame):
    """
    Find overlapping interaction records between two DataFrames.
    
    Both DataFrames must have columns: ['timestamp', 'user_id', 'item_id'].
    Overlap is defined as having the same (timestamp, user_id, item_id) combination in both.
    
    Returns:
        A DataFrame of overlapping interactions with info from both sources.
    """
    
    overlaps = pd.merge(
        data, reference_data,
        on=['timestamp', 'user_id', 'item_id'],
        suffixes=('_target', '_reference'),
        how='inner'
    )
    
    return overlaps
