from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt

from .base import calculate_sequence_stats
from .temporal import inters_per_period


def input2dict(
    data: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
    labels: Optional[List[str]] = None,
    col: Optional[str] = None,
) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """Normalize input data into a labeled dictionary of DataFrames/Series.

    Args:
        data: DataFrame, list of DataFrames, or dict of DataFrames.
        labels: Optional list of labels (applies to list/DataFrame inputs).
        col: Optional column name to filter (returns Series if specified).

    Returns:
        Dict of {label: DataFrame/Series}.
    """
    labels = labels or []
    if isinstance(data, list):
        keys = [
            labels[i] if labels and i < len(labels) else f"Subset {i + 1}"
            for i in range(len(data))
        ]
        data_dict = {key: df for key, df in zip(keys, data)}
    elif isinstance(data, pd.DataFrame):
        key = labels[0] if labels else "Data"
        data_dict = {key: data}
    elif isinstance(data, dict):
        data_dict = data
    else:
        raise ValueError(
            "`data` must be a DataFrame, list of DataFrames, or dict of DataFrames."
        )

    if col is not None:
        data_dict = {key: df[col] for key, df in data_dict.items()}

    return data_dict


def plot_hist(
    data: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame], pd.Series],
    col: str,
    log: bool = False,
    labels: Optional[List[str]] = None,
    **kwargs,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Plot a histogram using matplotlib and return sequence statistics.

    Args:
        data: Single DataFrame/Series or list/dict of DataFrames.
        col: Column name to plot.
        log: Whether to apply log1p transformation to values.
        labels: Optional list of labels for subsets.
        **kwargs: Additional keyword arguments for `plt.hist`.

    Returns:
        A tuple containing:
        - matplotlib Figure object
        - DataFrame with statistics of selected column for each subset
    """
    # Normalize input to dictionary
    data_dict = input2dict(data, labels, col)
    stats_dict = {
        key: calculate_sequence_stats(series) for key, series in data_dict.items()
    }

    if log:
        data_dict = {label: np.log1p(df) for label, df in data_dict.items()}

    fig, ax = plt.subplots()

    ax.hist(data_dict.values(), label=data_dict.keys(), **kwargs)
    ax.legend()
    ax.set_xlabel(f"{col} {'(log scale)' if log else ''}")
    ax.grid(alpha=0.3)

    return fig, pd.DataFrame(stats_dict).T


def plot_hist_px(
    data: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame], pd.Series],
    col: str,
    log: bool = False,
    labels: Optional[List[str]] = None,
    **kwargs,
) -> Tuple[px.histogram, pd.DataFrame]:
    """
    Plot a histogram using Plotly Express and return sequence statistics.

    Args:
        data: Single DataFrame/Series or list/dict of DataFrames.
        col: Column name to plot.
        log: Whether to apply log1p transformation to values.
        labels: Optional list of labels for subsets.
        **kwargs: Additional keyword arguments for `px.histogram`.

    Returns:
        A tuple containing:
        - plotly histogram figure object
        - DataFrame with statistics of selected column for each subset
    """
    # Normalize input to dictionary
    data_dict = input2dict(data, labels, col)
    stats_dict = {
        key: calculate_sequence_stats(series) for key, series in data_dict.items()
    }

    if log:
        data_dict = {label: np.log1p(df) for label, df in data_dict.items()}

    return px.histogram(data_dict, **kwargs), pd.DataFrame(stats_dict).T


def plot_inters(
    data: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    granularity: str = "h",
    labels: Optional[List[str]] = None,
    figsize: tuple = (10, 5),
    title: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot number of interactions over time using seaborn with grouping via hue.

    Args:
        data: A single DataFrame, or a list/dict of DataFrames.
        start_date (str, optional): Start date for filtering (inclusive) in DD/MM/YYYY format.
        end_date (str, optional): End date for filtering (inclusive) in DD/MM/YYYY format.
        granularity: Time aggregation level ('h', 'd', etc.).
        labels: Optional list of labels for subsets.
        figsize: Size of the matplotlib figure.
        title: Optional title for the plot.
        **kwargs: Additional arguments passed to `sns.barplot`.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    # Normalize input to dictionary
    data_dict = input2dict(data, labels)

    # Resample interaction counts
    records = []
    for key, df in data_dict.items():
        resampled = inters_per_period(df, start_date, end_date, granularity)
        resampled["label"] = key
        records.append(resampled)

    combined_df = pd.concat(records, ignore_index=True)

    # Convert timestamp to string or rounded datetime for clearer x-axis
    if granularity == "h":
        combined_df["timestamp"] = combined_df["timestamp"].dt.floor("h")
    elif granularity == "d":
        combined_df["timestamp"] = combined_df["timestamp"].dt.floor("d")
    elif granularity == "w":
        combined_df["timestamp"] = (
            combined_df["timestamp"].dt.to_period("W").astype("datetime64[ns]")
        )
    elif granularity == "m":
        combined_df["timestamp"] = (
            combined_df["timestamp"].dt.to_period("M").astype("datetime64[ns]")
        )

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=combined_df, x="timestamp", y="n_inters", hue="label", **kwargs)

    if title:
        ax.set_title(title)

    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Number of Interactions")
    ax.legend(title="Subset")
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    return fig


def plot_inters_px(
    data: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    granularity: str = "h",
    labels: Optional[List[str]] = None,
    **kwargs,
) -> px.bar:
    """
    Plot number of interactions over time with group differentiation using Plotly.

    Args:
        data: A single DataFrame, or a list/dict of DataFrames.
        start_date (str, optional): Start date for filtering (inclusive) in DD/MM/YYYY format.
        end_date (str, optional): End date for filtering (inclusive) in DD/MM/YYYY format.
        granularity: Time aggregation level ('h', 'd', etc.).
        labels: Optional list of labels for subsets.
        **kwargs: Additional arguments to pass to Plotly.

    Returns:
        Plotly bar chart figure.
    """
    # Normalize input to dictionary
    data_dict = input2dict(data, labels)

    # Get resampled interactions for each subset
    stats_dict = {
        key: inters_per_period(df, start_date, end_date, granularity)
        for key, df in data_dict.items()
    }
    df_combined = pd.concat([df.assign(subset=key) for key, df in stats_dict.items()])

    return px.bar(
        df_combined,
        y="n_inters",
        x="timestamp",
        color="subset",
        **kwargs,
    )


def plot_bars(
    data: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
    x: str,
    y: str,
    labels: Optional[List[str]] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot bar chart from DataFrame columns using seaborn.

    Args:
        data: A single DataFrame, or a list/dict of DataFrames.
        x: Column name for x-axis (typically categorical).
        y: Column name for y-axis (bar height).
        labels: Optional list of labels for subsets.
        **kwargs: Additional arguments passed to `sns.barplot`.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    data_dict = input2dict(data, labels)

    df_combined = pd.concat([df.assign(subset=key) for key, df in data_dict.items()])

    fig, ax = plt.subplots()
    sns.barplot(data=df_combined, x=x, y=y, hue="subset" if len(data_dict) > 1 else None, ax=ax, **kwargs)
    ax.set_ylabel(y)
    ax.set_xlabel(x)
    ax.grid(alpha=0.3)
    ax.legend(title="Subset")
    fig.tight_layout()

    return fig


def plot_bars_px(
    data: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
    x: str,
    y: str,
    labels: Optional[List[str]] = None,
    **kwargs,
) -> px.bar:
    """
    Plot bar chart from DataFrame columns using seaborn using Plotly.

    Args:
        data: A single DataFrame, or a list/dict of DataFrames.
        x: Column name for x-axis (typically categorical).
        y: Column name for y-axis (bar height).
        labels: Optional list of labels for subsets.
        **kwargs: Additional arguments passed to `px.bar`.

    Returns:
        Plotly bar chart figure.
    """
    # Normalize input to dictionary
    data_dict = input2dict(data, labels)

    df_combined = pd.concat([df.assign(subset=key) for key, df in data_dict.items()])

    return px.bar(
        df_combined,
        y=y,
        x=x,
        color="subset",
        **kwargs,
    )

def plot_inters_scatter(
    data: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
    labels: Optional[List[str]] = None,
    users: Optional[Union[int, List[int]]] = None,
    random_state: Optional[int] = None,
    figsize: tuple = (25, 12),
    point_size: int = 64,
    **kwargs,
) -> plt.Figure:
    """
    Plot scatter plot showing data splits by user_id and timestamp.

    Args:
        data: DataFrame, list of DataFrames, or dict of DataFrames.
              Each subset should contain columns ["user_id", "timestamp"].
        labels: Optional labels for subsets (applies to list/DataFrame inputs).
        users: Number of unique users to sample for display. If None, shows all users.
        random_state: Seed for reproducible random sampling.
        figsize: Figure size (width, height). Defaults to (25, 12).
        point_size: Scatter point size. Defaults to 64.
        **kwargs: Additional arguments passed to `sns.scatterplot`.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    # Normalize input
    data_dict = input2dict(data, labels)

    # Add split labels and combine
    pd_for_plot = []
    for split_label, df in data_dict.items():
        tmp = df.copy()
        tmp["split"] = split_label
        pd_for_plot.append(tmp)

    pd_for_plot = pd.concat(pd_for_plot, axis=0)
    pd_for_plot["user_id"] = pd_for_plot["user_id"].astype(str)

    # Optionally subsample users
    if users is not None:
        if isinstance(users, (int, float)):
            unique_users = pd_for_plot["user_id"].unique()
            
            if random_state is not None:
                np.random.seed(random_state)

            sampled_users = np.random.choice(unique_users, size=users, replace=False)

        elif isinstance(users, (list, np.ndarray)):
            sampled_users = np.array(users)
        else:
            raise ValueError("`users` must be int, list, np.ndarray, or None.")
        
        pd_for_plot = pd_for_plot[pd_for_plot["user_id"].isin(sampled_users.astype(str))]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=pd_for_plot,
        x="timestamp",
        y="user_id",
        hue="split",
        s=point_size,
        ax=ax,
        **kwargs,
    )

    # Style
    plt.autoscale(enable=True, axis="x")
    ax.grid(False)
    ax.set_xlabel("timestamp")
    ax.set_ylabel("user_id")
    ax.legend()
    plt.grid(axis="y", alpha=0.2)

    return fig