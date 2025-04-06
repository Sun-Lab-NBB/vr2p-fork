import numpy as np
import pandas as pd

from vr2p.gimbl.parse import parse_custom_msg


def get_lap_info(
    log_df: pd.DataFrame,
    fields: list[str],
    path_time_df: pd.DataFrame
) -> pd.DataFrame:
    """Parse lap information from a Gimbl log.

    Uses actual position on the path to increase precision. The function assumes
    "Lap" messages with lap == 0 can be ignored, and that laps last at least 3 seconds.

    Args:
        log_df (pd.DataFrame):
            The complete DataFrame from the Gimbl log. Must contain columns:
            - "time": time since start in milliseconds.
            - "msg": identifier for Lap message.
            - "data.[fields]": requested data field.
        fields (List[str]):
            The Lap message fields to read. Must include "lap".
        path_time_df (pd.DataFrame):
            Path information from "parse_path". Used for extracting time ranges
            that refine lap detection.

    Returns:
        pd.DataFrame:
            A DataFrame of parsed Lap information:
            Index:
                Automatic integer index.
            Columns:
                - "time": The time at which the lap is detected (ms).
                - "lap": Lap number decremented by 1 (starting from zero).
                - "time_start": The computed start time of the lap (as timedelta).
                - "time_end": The computed end time of the lap (as timedelta).

    Raises:
        NameError:
            If fewer track periods are detected than the logged number of laps.
    """
    data = parse_custom_msg(log_df, "Lap", fields).drop(columns="frame")

    # Use path_time_df to locate actual start and end times for laps
    time_range = get_lap_from_path(path_time_df)

    # Remove entries at start of new environment (lap == 0)
    data = data.drop(data[data["lap"] == 0].index)
    data = data.reset_index(drop=True)
    data["lap"] = data["lap"] - 1

    # Ensure we have enough detected periods for all logged laps
    if time_range.shape[0] < data.shape[0]:
        raise NameError(
            f"Fewer periods ({time_range.size}) detected than logged laps ({data.size})."
        )

    # For each lap, pick the closest possible end time not yet assigned
    assigned = np.zeros(time_range.shape[0], dtype=bool)
    indices = []

    for time_val in data["time"].dt.total_seconds():
        offset = np.abs(time_range[:, 1] - time_val)
        offset[assigned] = np.inf
        argmin_idx = offset.argmin()
        assigned[argmin_idx] = True
        indices.append(argmin_idx)

    indices = np.array(indices)

    # Attach matching start/end times to the DataFrame
    data["time_start"] = pd.to_timedelta(time_range[indices, 0], unit="seconds")
    data["time_end"] = pd.to_timedelta(time_range[indices, 1], unit="seconds")

    return data


def get_lap_from_path(path_time_df: pd.DataFrame) -> np.ndarray:
    """Find possible lap periods based on position information.

    This function identifies potential laps on each path by detecting
    segments that begin and end at low/high thresholds, ensuring
    that the entire path is visited within the segment.

    Args:
        path_time_df (pd.DataFrame):
            The DataFrame containing parsed path information. Must have columns:
            - "position" (path position values, in cm).
            - "time"     (time since start as pd.Timedelta).
            - "path"     (path name/category).

    Returns:
        np.ndarray:
            A 2D NumPy array of shape (<num_periods>, 2), representing
            the start and end times (in seconds) of each detected lap.
    """
    position = path_time_df["position"].to_numpy()
    timestamps = path_time_df["time"].dt.total_seconds().to_numpy()
    paths = path_time_df["path"].to_numpy()

    # Collect all potential start and end points per path
    start_points = []
    end_points = []

    for path in np.unique(paths):
        path_idx = (paths == path)
        path_positions = position[path_idx]

        # 10% threshold for start
        threshold_start = path_positions.min() + (
            (path_positions.max() - path_positions.min()) * 0.1
        )
        temp_start = (position < threshold_start) & path_idx
        start_array = np.convolve(temp_start.astype(int), [0, 1, -1], mode="same")
        start_points.append(np.argwhere(start_array == 1).flatten())

        # 10% threshold for end
        threshold_end = path_positions.min() + (
            (path_positions.max() - path_positions.min()) * 0.1
        )
        temp_end = (position > threshold_end) & path_idx
        end_array = np.convolve(temp_end.astype(int), [-1, 1, 0], mode="same")
        end_points.append(np.argwhere(end_array == 1).flatten())

    start_points = np.sort(np.hstack(np.array(start_points, dtype=object)))
    end_points = np.sort(np.hstack(np.array(end_points, dtype=object)))

    # Identify valid periods
    time_range = []

    for i, start_idx in enumerate(start_points[:-1]):
        next_start_idx = start_points[i + 1]

        # Only proceed if the next start is on the same path
        if paths[start_idx] == paths[next_start_idx]:
            # Index of end points between start_idx and next_start_idx
            end_indices = np.argwhere((end_points > start_idx) & (end_points < next_start_idx))

            if end_indices.size != 0:
                # Get the last valid end index within that range
                match_end_idx = end_points[end_indices[-1]][0]

                # Check coverage of the path segment
                segment_values = position[start_idx:match_end_idx]
                counts, _ = np.histogram(segment_values, bins=5)
                if all(counts > 0):
                    time_range.append([timestamps[start_idx], timestamps[match_end_idx]])

    return np.array(time_range)
