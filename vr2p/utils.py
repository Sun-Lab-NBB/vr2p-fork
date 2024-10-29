from suite2p.io import compute_dydx, BinaryFileCombined
from pathlib import Path
import numpy as np
import pandas as pd


def memory_usage(dataframe: pd.DataFrame | pd.Series, verbose: bool = True) -> float:
    """Calculates memory usage of a pandas DataFrame or Series.

    Computes the total memory usage of a Pandas' object, accounting for all data types and memory optimizations.
    
    Args:
        dataframe: Pandas DataFrame or Series to analyze.
        verbose: If True, prints the memory usage.

     Returns:
        float: Memory usage in megabytes.
    """

    # Calculates the total memory usage of the DataFrame or Series (if one-dimensional). Sums up the memory used by
    # each data column for Dataframes, then converts the memory to Megabytes and rounds to 2 SFs.
    value = round((dataframe.memory_usage(deep=True).sum() if dataframe.ndim > 1 else dataframe.memory_usage(
        deep=True)) / 1024 ** 2, 2)

    if verbose:
        print('Memory used:', value, 'Mb')

    return value


def read_raw_frames(plane_folder: Path | str, frames: np.ndarray) -> np.ndarray:
    """Reads and combines raw imaging frames from multiple binary files.

    Reads specified frames from Suite2p binary files across multiple imaging planes and combines them into a single
    array. Handles proper alignment of different planes based on their spatial offsets.

    Args:
        plane_folder: Path to the Suite2p folder containing imaging plane folders.
        frames: NumPy array specifying which frames to read (e.g., np.arange(0,100)).

    Returns:
        np.ndarray: Combined imaging frame data with shape (n_frames, Lx, Ly).

    Raises:
        NameError: If no plane folders are found in the specified directory.
        FileNotFoundError: If ops.npy or data.bin files are missing.
    """
    plane_folder = Path(plane_folder)  # Convert to the Path object if input is originally string

    # Finds all plane folders sorted numerically
    plane_folders = sorted(plane_folder.glob('plane*/'))
    if not plane_folders:
        raise NameError(f"Unable to find plane folders in {plane_folder}.")

    # Loads 'ops' files for all planes as numpy arrays
    try:
        ops1 = [np.load(f / 'ops.npy', allow_pickle=True).item() for f in plane_folders]
    except FileNotFoundError:
        raise FileNotFoundError(f"Unable to find one of the ops.npy files.")

    # Gets registered binary file locations
    reg_loc = [plane_directory / 'data.bin' for plane_directory in plane_folders]
    if not all(path.exists() for path in reg_loc):
        raise FileNotFoundError("Unable to find one of the data.bin files.")

    # Computes x and y anchor coordinates of each plane ROI
    y_anchors, x_anchors = compute_dydx(ops1)

    roi_heights = np.array([ops['Ly'] for ops in ops1])  # Computes the height of each ROI
    roi_widths = np.array([ops['Lx'] for ops in ops1])  # Computes the width of each ROI

    # Calculates the maximum x and y coordinates for the ROIs
    roi_y_max = int(np.amax(y_anchors + roi_heights))
    roi_x_max = int(np.amax(x_anchors + roi_widths))

    # Combines individual planes into a virtual stack with Suit2P and packages the results into a numpy array
    # noinspection PyTypeChecker
    with BinaryFileCombined(roi_y_max, roi_x_max, roi_heights, roi_widths, y_anchors, x_anchors, reg_loc) as f:
        return f[frames]
