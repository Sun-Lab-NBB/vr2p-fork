from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
from suite2p.io import BinaryFileCombined, compute_dydx
from numpy.typing import NDArray


def memory_usage(dataframe: pd.DataFrame | pd.Series, verbose: bool = True) -> float:
    """Calculates memory usage for a pandas DataFrame or Series.
    
    Args:
        dataframe: Pandas DataFrame or Series to analyze.
        verbose: If True, prints the memory usage.

    Returns:
        Memory usage in megabytes, rounded to 2 significant figures.
    """
    # Calculates the total memory usage of the DataFrame or Series (if one-dimensional). Sums up the memory used by
    # each data column for Dataframes, then converts the memory to Megabytes and rounds to 2 SFs.
    value = round((dataframe.memory_usage(deep=True).sum() if dataframe.ndim > 1 else dataframe.memory_usage(
        deep=True)) / 1024 ** 2, 2)

    if verbose:
        print("Memory used:", value, "Mb")

    return value


def read_raw_frames(bin_folder: Path | str, frames: NDArray[Any]) -> NDArray[Any]:
    """Reads and combines raw imaging frames from multiple binary files.

    Reads the requested number of frames from Suite2p binary files across multiple imaging planes and combines them
    into a single array. Aligns the planes into a virtual stack based on their spatial offsets.

    Args:
        bin_folder: Path to the Suite2p folder containing imaging plane data.
        frames: NumPy array specifying which frames to read (e.g., np.arange(0,100)).

    Returns:
        Combined imaging frame data packaged into a numpy array with shape n_frames x roi width (Lx) x roi height (Ly).

    Raises:
        NameError: If no plane folders are found in the specified directory.
        FileNotFoundError: If ops.npy or data.bin files are missing.
    """
    bin_folder = Path(bin_folder)  # Convert to the Path object if input is originally string

    # Finds all plane folders sorted numerically
    plane_folders = sorted(bin_folder.glob("plane*/"))
    if not plane_folders:
        raise NameError(f"Unable to find plane folders in {bin_folder}.")

    # Loads 'ops' files for all planes as numpy arrays
    try:
        ops1 = [np.load(file / "ops.npy", allow_pickle=True).item() for file in plane_folders]
    except FileNotFoundError:
        raise FileNotFoundError("Unable to find one of the ops.npy files.")

    # Gets registered binary file locations
    registration_location = [plane_directory / "data.bin" for plane_directory in plane_folders]
    if not all(path.exists() for path in registration_location):
        raise FileNotFoundError("Unable to find one of the data.bin files.")

    # Computes x and y anchor coordinates of each frame
    y_anchors, x_anchors = compute_dydx(ops1)

    frame_heights = np.array([ops["Ly"] for ops in ops1])  # Computes the height of each frame
    frame_widths = np.array([ops["Lx"] for ops in ops1])  # Computes the width of each frame

    # Determines the width and height of the overall FOV. Since the resultant stack includes multiple frames, they
    # may not be 100% aligned, resulting in the overall FOV being larger than any given frame.
    fov_height = int(np.amax(y_anchors + frame_heights))
    fov_width = int(np.amax(x_anchors + frame_widths))

    # Combines individual planes into a virtual stack with Suit2P and packages the results into a numpy array
    # noinspection PyTypeChecker
    with BinaryFileCombined(fov_height, fov_width, frame_heights, frame_widths, y_anchors, x_anchors,
                            registration_location) as f:
        return f[frames]
