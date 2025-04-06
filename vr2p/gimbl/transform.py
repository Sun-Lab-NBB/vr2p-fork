"""Utilities for transforming and merging Gimbl log data frames.

This module provides functions for assigning frame information to Gimbl log data,
filling missing frame data, and adding timestamp values to data frames based on
time matching criteria.
"""

from typing import Optional

import numpy as np
import pandas as pd


def assign_frame_info(
    data: pd.DataFrame,
    frames: pd.DataFrame,
    remove_nan: bool = True
) -> pd.DataFrame:
    """Joins indexed Gimbl log data with frame info to assign frame numbers.

    This function merges the provided 'data' DataFrame with 'frames' based on their time indices,
    ensuring each row in 'data' is assigned the corresponding microscopic 'frame' value.

    Args:
        data (pd.DataFrame):
            Gimbl log data to assign frame info to. Must contain a "time" column.
        frames (pd.DataFrame):
            Parsed frame info with "time" and "frame" columns. 
            The "time" column must be a datetime-like or timedelta type.
        remove_nan (bool, optional):
            Whether to remove entries outside the frame range (timesteps before the first frame or after the last frame).
            Defaults to True.

    Returns:
        pd.DataFrame:
            A new DataFrame with an added "frame" column, representing the microscope frame index.
            Rows outside of the valid range are either dropped (if remove_nan is True)
            or assigned frame = -1 (if remove_nan is False).
    """
    # Combine frames and data on the time index, then sort
    frame_data = pd.concat(
        [frames.reset_index().set_index("time"), data.reset_index().set_index("time")]
    ).sort_index()

    # Identify rows originally from 'data' (these lack frame info)
    idx = frame_data["frame"].isnull()

    # Forward-fill from frames to those rows
    frame_data["frame"] = frame_data["frame"].ffill()

    # Keep only rows that were originally from the 'data'
    frame_data = frame_data[idx]

    # Drop any leftover "index" column if present
    if "index" in frame_data.columns:
        frame_data = frame_data.drop(columns="index")

    # Remove rows beyond the valid frame limits
    if remove_nan:
        frame_data = frame_data.dropna(subset=["frame"])
    else:
        # Assign -1 if the index is outside frame range
        frame_data["frame"] = frame_data["frame"].fillna(value=-1)
        if "level_0" in frame_data.columns:
            frame_data = frame_data.drop(columns="level_0")

    frame_data["frame"] = frame_data["frame"].astype("int32")
    frame_data = frame_data.reset_index()
    return frame_data


def ffill_missing_frame_info(
    frame_data: pd.DataFrame,
    frames: pd.DataFrame,
    nan_fill: bool = False,
    subset_columns: Optional[list[str]] = None
) -> pd.DataFrame:
    """Forward-fill missing frames by copying info from previous frames.

    For each actor (identified by 'name'), rows are expanded to cover all frames
    in the session. Missing data is then either forward-filled (the default) or
    left as None, depending on 'nan_fill'.

    Args:
        frame_data (pd.DataFrame):
            Gimbl log data with a "frame" column in its MultiIndex or columns.
            Typically, this DataFrame is indexed by ["frame", "name"] or can be reset to that state.
        frames (pd.DataFrame):
            Parsed frame information with a known number of frames and associated times.
        nan_fill (bool, optional):
            If True, fill missing values with None instead of forward-filling. Defaults to False.
        subset_columns (List[str], optional):
            Columns to limit the fill operation. Defaults to None (all columns).

    Returns:
        pd.DataFrame:
            Same columns as frame_data, potentially with extra entries to fill in frame gaps.
            The MultiIndex ["frame", "name"] is preserved or recreated, and data is sorted by this index.
    """
    if not subset_columns:
        subset_columns = frame_data.columns

    num_frames = frames.shape[0]
    # Ensure 'frame_data' is multi-indexed consistently
    # for iteration over each actor's data subset
    names = frame_data.reset_index()["name"].unique()

    for name in names:
        # Slice by current actor, reindex by 'frame'
        df = frame_data.loc[(slice(None), name), :].copy()
        df = df.reset_index().set_index("frame")

        # Create a template with all frames and associated times for this actor
        template = pd.DataFrame({
            "frame": range(num_frames),
            "name": name,
            "time": frames["time"].to_numpy()
        }).set_index("frame")

        # Merge actor data with the template across all frames
        df = df.combine_first(template).sort_index()

        # Forward-fill or leave NaN
        if not nan_fill:
            df[subset_columns] = df[subset_columns].ffill(axis=0)

        # Convert back to multi-index ["frame", "name"]
        df = df.reset_index().set_index(["frame", "name"])
        # Combine final results back into the main DataFrame
        frame_data = frame_data.combine_first(df)

    return frame_data.sort_index()


def add_timestamp_values(
    df: pd.DataFrame,
    timestamp_df: pd.DataFrame,
    fields: list[str]
) -> pd.DataFrame:
    """Forward-fill values from a timestamped DataFrame to align with 'df' based on matching or preceding time.

    Unlike add_ranged_timestamp_values (which uses time ranges), this routine picks
    the closest preceding timestamp from 'timestamp_df' for each row in 'df'.

    Args:
        df (pd.DataFrame):
            DataFrame to which new values will be added (must have a "time" column).
        timestamp_df (pd.DataFrame):
            Timestamped DataFrame with fields to assign (must have a "time" column, sorted ascending).
        fields (List[str]):
            Columns in 'timestamp_df' to add to 'df'.

    Returns:
        pd.DataFrame:
            A copy of 'df' with the specified fields added or expanded, assigned from the nearest 
            earlier (or equal) timestamp in 'timestamp_df'. Rows with no suitable preceding timestamp 
            remain None for those columns.
    """
    df = df.copy()

    # Make sure both DataFrames are sorted by time
    timestamp_df = timestamp_df.sort_values("time")
    df = df.sort_values("time")

    # Initialize columns in 'df' to None
    for field in fields:
        df[field] = None

    # For each row in 'df', find the latest row in timestamp_df whose time <= row's time
    ind = np.array(timestamp_df["time"].searchsorted(df["time"]) - 1)

    # Mark rows where no preceding timestamp is available
    mask = ind < 0
    ind[mask] = 0

    # Assign fields from the matched timestamps
    df[fields] = timestamp_df.iloc[ind][fields].reset_index(drop=True).to_numpy()

    # Rows with no preceding timestamp remain None
    df.loc[mask, fields] = None

    return df


def add_ranged_timestamp_values(
    df: pd.DataFrame,
    timestamp_df: pd.DataFrame,
    fields: list[str]
) -> pd.DataFrame:
    """Assign values to rows in 'df' based on whether the row's time falls within time ranges in
    'timestamp_df'.

    If 'timestamp_df' has columns "time_start" and "time_end", each row in 'df' whose time
    is in [time_start, time_end] receives the corresponding fields from that row of 'timestamp_df'.

    Args:
        df (pd.DataFrame):
            The DataFrame to be enriched, must have a "time" column.
        timestamp_df (pd.DataFrame):
            A DataFrame containing columns "time_start" and "time_end", which define the time
            ranges.
        fields (List[str]):
            Columns from 'timestamp_df' to merge into 'df' for rows whose 'time' falls in
            the specified [time_start, time_end] range.

    Returns:
        pd.DataFrame:
            The original 'df' with the specified fields populated based on matching time ranges.
            Columns are appended in the order encountered. Overlapping ranges in 'timestamp_df'
            can result in overwrites based on the later range in code iteration.
    """
    df = df.copy()
    # Remember original column order
    original_columns = df.columns.to_list()

    # Verify that requested 'fields' exist in 'timestamp_df'
    if not set(fields).issubset(timestamp_df.columns):
        msg = "Requested fields are not present in the supplied 'timestamp_df'."
        raise NameError(msg)

    # Add any missing columns to 'df'
    missing_columns = list(set(fields).difference(set(df.columns)))
    for field in missing_columns:
        df[field] = None

    # Extract time arrays for interval checks
    time = df["time"].values
    start_times = timestamp_df["time_start"].values
    end_times = timestamp_df["time_end"].values

    # For each row in 'timestamp_df', assign the relevant fields 
    # to all rows in 'df' falling within [time_start, time_end]
    for i in range(start_times.size):
        in_range = (time >= start_times[i]) & (time <= end_times[i])
        df.loc[in_range, fields] = timestamp_df.loc[i, fields].values

    # Reorder columns to keep the newly added fields at the end
    fields_in_order = [col for col in fields if col in missing_columns]
    return df[original_columns + fields_in_order]
