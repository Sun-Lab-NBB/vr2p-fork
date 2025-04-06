"""Utilities for extracting and analyzing movement data from Gimbl logs.

This module provides functions to calculate movement speed and other metrics
from position data stored in DataFrames, supporting analysis of animal
movement patterns in virtual reality experiments.
"""

import numpy as np
import pandas as pd


def movement_speed(
    df: pd.DataFrame,
    window_size: int = 100,
    ignore_threshold: float = 20
) -> np.ndarray:
    """Calculate rolling average movement speed in centimeters per second (cm/s) from a DataFrame.

    The DataFrame must contain either:
      1. Columns named "x", "y", "z", "time", or
      2. Columns named "position", "path", "time".

    If "x", "y", and "z" are present, the function automatically calculates
    the distance in 3D space for each frame and assigns them to a temporary
    path "test". Otherwise, if "path" is present, it processes each path independently.

    Teleport artifacts or extremely large movements beyond the specified threshold
    are set to zero before rolling computation. The rolling window is indexed by the
    time column, which must be a pandas DateTime-like column.

    Args:
        df (pd.DataFrame): A DataFrame containing the requisite columns ("x", "y", "z", "time",
            or "position", "path", "time").
        window_size (int, optional): The size of the rolling average window in milliseconds.
            Defaults to 100.
        ignore_threshold (float, optional): Distance threshold, above which the movement is
            assumed to be a teleport artifact and is set to zero. Defaults to 20.

    Returns:
        np.ndarray: A 1D NumPy array of speed values (in cm/s) aligned with the input DataFrame.

    Raises:
        KeyError: If the required columns ("time" and either "path" or "x", "y", "z") are missing
            from the DataFrame.

    Notes:
        - Speed calculation is performed by computing the distance traveled between consecutive
          frames, then taking the rolling mean of these distances over the specified time window.
    """
    # Copy the DataFrame to avoid side effects
    df = df.copy()

    # If x, y, z columns exist, compute the 3D distance
    if ("x" in df) and ("y" in df) and ("z" in df):
        dist_3d = df[["x", "y", "z"]].diff()
        df["dist"] = np.sqrt(dist_3d.x ** 2 + dist_3d.y ** 2 + dist_3d.z ** 2).abs()
        # Assign a generic path for all points
        df["path"] = "test"

    # If a "path" column exists, process each path individually
    if "path" in df:
        # For each path, compute movement distance and filter out teleports or large jumps
        for path in df["path"].unique():
            # Block out values not on the current path
            pos = df["position"].copy() if "position" in df else None
            in_current_path = df["path"] == path
            if pos is not None:
                pos[~in_current_path] = np.nan
                # Calculate moved distance only for values on the current path
                df.loc[in_current_path, "dist"] = pos.diff().abs()

        # For each path, remove teleports and calculate rolling speed
        for path in df["path"].unique():
            path_df = df.copy()
            in_current_path = df["path"] == path
            # Mark distances as NaN if not on this path
            path_df.loc[~in_current_path, "dist"] = np.nan
            # Filter out high teleport distances
            path_df.loc[path_df["dist"] > ignore_threshold, "dist"] = 0
            # Fill missing distances at the start of the path
            path_df.loc[in_current_path, "dist"] = path_df.loc[in_current_path, "dist"].fillna(0)
            # Convert time deltas to seconds, then compute instantaneous speed
            path_df["speed"] = path_df["dist"] / path_df["time"].dt.total_seconds().diff()
            path_df["speed"] = path_df["speed"].fillna(0)
            # Compute rolling speed based on the specified time window
            path_df = path_df.reset_index().set_index("time")
            path_df["speed"] = path_df["speed"].rolling(f"{window_size}ms", min_periods=1).mean()
            path_df = path_df.reset_index(drop=True)

            # Update the main DataFrame with computed speeds
            in_current_path = in_current_path.to_numpy()  # Prevents dtype mismatch error
            df.loc[in_current_path, "speed"] = path_df.loc[in_current_path, "speed"].to_numpy()

    else:
        raise KeyError(
            "DataFrame must contain a 'path' column or 'x', 'y', 'z' columns along with 'time'."
        )

    return df["speed"].to_numpy()
