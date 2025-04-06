import re
import shutil
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

import zarr
import numpy as np
from numpy.typing import NDArray
from natsort import natsorted
import numcodecs
from tqdm import tqdm
from suite2p.extraction import dcnv

from vr2p.signal_processing import demix_traces
from vr2p.gimbl.parse import parse_gimbl_log


class LogFile:
    """Container for storing DataFrame logs in zarr format.

    This class provides a simple wrapper to store pandas DataFrames
    in the zarr file structure.

    Attributes:
        value (pandas.DataFrame): The stored DataFrame.
    """

    def __init__(self, value: Any) -> None:
        """Initialize the LogFile with a DataFrame.

        Args:
            value (pandas.DataFrame): DataFrame containing log data.
        """
        self.value = value


def _prepare_data_paths(data_info: dict[str, Any], info: dict[str, Any]) -> list[str]:
    """Prepare and validate data paths for processing.

    Args:
        data_info (dict): Information about data locations.
        info (dict): Processed information about sessions.

    Returns:
        list[str]: Sorted list of validated data paths.

    Raises:
        ValueError: If session paths are invalid or duplicated.
    """
    data_paths = info["data_paths"].copy()

    if data_info["data"]["individual_sessions"]:
        for date in data_info["data"]["individual_sessions"]:
            # Check format
            if not re.match(r"[0-9]{4}_[0-9]{2}_[0-9]{2}/[0-9]", date):
                raise ValueError(
                    f"Requested individual_session: {date}, does not match the "
                    "expected format (e.g. 2020_01_01/1)"
                )

            # Check for duplicates
            if date in data_paths:
                raise ValueError(
                    f"{date} was requested to be added as an individual (non-aligned) "
                    "session but was part of the multiday registration process."
                )

        data_paths.extend(data_info["data"]["individual_sessions"])

    return natsorted(data_paths)


def _add_imaging_data(
    zarr_file: zarr.Group,
    folder_path: Path,
    group: str,
    stat: NDArray[Any],
    counter: int,
    selected_cells: Optional[list[bool]] = None,
    settings: Optional[dict[str, Any]] = None,
) -> None:
    """Load, process, and add imaging data to the zarr file.

    Args:
        zarr_file (zarr.Group): Open zarr file for writing.
        folder_path (Path): Path to the data folder.
        group (str): Group name for data organization ('single_session' or 'multi_session').
        stat (NDArray[Any]): Cell statistics data.
        counter (int): Current session index.
        selected_cells (list[bool], optional): Boolean mask for selecting cells. Defaults to None.
        settings (dict, optional): Processing settings required for demixing. Defaults to None.
    """
    # Load fluorescence data
    if selected_cells is not None:
        F = np.load(folder_path / "F.npy")[selected_cells, :]
        Fneu = np.load(folder_path / "Fneu.npy")[selected_cells, :]
    else:
        F = np.load(folder_path / "F.npy")
        Fneu = np.load(folder_path / "Fneu.npy")

    # Store raw fluorescence data
    zarr_file.require_group(f"{group}/F").create_dataset(
        str(counter),
        data=F,
        chunks=(1000, 10000)
    )

    zarr_file.require_group(f"{group}/Fneu").create_dataset(
        str(counter),
        data=Fneu,
        chunks=(1000, 10000)
    )

    # If we're processing single session data with settings
    if settings is not None:
        # Calculate neuropil-subtracted signal
        Fns = F - settings["demix"]["neucoeff"] * Fneu
        zarr_file.require_group(f"{group}/Fns").create_dataset(
            str(counter),
            data=Fns,
            chunks=(1000, 10000)
        )

        # Calculate demixed signal
        Fdemix, _, _, _ = demix_traces(F, Fneu, stat, settings["demix"])
        zarr_file.require_group(f"{group}/Fdemix").create_dataset(
            str(counter),
            data=Fdemix,
            chunks=(1000, 10000)
        )

        # Calculate spike inference
        ops_file = folder_path / "ops.npy"
        ops = np.load(ops_file, allow_pickle=True).item()

        spks = dcnv.oasis(Fdemix, ops["batch_size"], ops["tau"], ops["fs"])
        zarr_file.require_group(f"{group}/spks").create_dataset(
            str(counter),
            data=spks,
            chunks=(1000, 10000)
        )
    else:
        # For multi-session data, load pre-computed values
        for signal_type in ["Fns", "Fdemix", "spks"]:
            signal_data = np.load(folder_path / f"{signal_type}.npy")
            zarr_file.require_group(f"{group}/{signal_type}").create_dataset(
                str(counter),
                data=signal_data,
                chunks=(1000, 10000)
            )


def _process_vr_data(
    zarr_file: zarr.Group,
    data_info: Dict[str, Any],
    data_path: str,
    counter: int
) -> None:
    """Process VR data and store it in a zarr file.

    Args:
        zarr_file (zarr.Group): Open zarr file for writing.
        data_info (dict): Information about data locations.
        data_path (str): Path to the current session's data.
        counter (int): Current session index.

    Raises:
        NameError: If the Gimbl log file cannot be found or multiple log files are detected.
    """
    log_file = find_gimbl_log(Path(data_info["data"]["local_processed_root"]) / data_path)
    df, vr_info = parse_gimbl_log(log_file)

    zarr_file.require_group("gimbl/vr").create_dataset(
        str(counter),
        data=vr_info,
        dtype=object,
        object_codec=numcodecs.Pickle()
    )

    zarr_file.require_group("gimbl/log").create_dataset(
        str(counter),
        data=LogFile(df),
        dtype=object,
        object_codec=numcodecs.Pickle()
    )


def _process_single_session(
    zarr_file: zarr.Group,
    session_path: Path,
    settings: Dict[str, Any],
    counter: int
) -> None:
    """Process single session data and store it in a zarr file.

    Args:
        zarr_file (zarr.Group): Open zarr file for writing.
        session_path (Path): Path to the session data.
        settings (dict): Processing settings.
        counter (int): Current session index.
    """
    # Read cell info
    stat = np.load(session_path / "stat.npy", allow_pickle=True)
    iscell = np.load(session_path / "iscell.npy", allow_pickle=True)

    # Select valid cells
    selected_cells = [
        (iscell[icell, 1] > settings["cell_detection"]["prob_threshold"]) and
        (mask["npix"] < settings["cell_detection"]["max_size"])
        for icell, mask in enumerate(stat)
    ]

    # Store cell info
    zarr_file.require_group("cells/single_session").create_dataset(
        str(counter),
        data=stat[selected_cells],
        dtype=object,
        object_codec=numcodecs.Pickle()
    )

    # Process and store imaging data
    _add_imaging_data(
        zarr_file,
        session_path,
        "single_session",
        stat[selected_cells],
        counter,
        selected_cells,
        settings
    )


def _process_multi_session(
    zarr_file: zarr.Group,
    multiday_folder: Path,
    data_path: str,
    multi_index: np.ndarray,
    backwards_deformed: np.ndarray,
    trans_images: np.ndarray,
    original_images: np.ndarray,
    counter: int,
    settings: Dict[str, Any]
) -> None:
    """Process multi-session data and store it in a zarr file.

    Args:
        zarr_file (zarr.Group): Open zarr file for writing.
        multiday_folder (Path): Path to the multiday data folder.
        data_path (str): Path to the current session's data.
        multi_index (np.ndarray): Index of this session in the multi-session structure.
        backwards_deformed (np.ndarray): Backwards deformed cell masks.
        trans_images (np.ndarray): Transformed images.
        original_images (np.ndarray): Original images.
        counter (int): Current session index.
        settings (dict): Processing settings.
    """
    session_path = multiday_folder / "sessions" / data_path
    stat = backwards_deformed[multi_index]

    # Store fluorescence data
    _add_imaging_data(zarr_file, session_path, "multi_session", stat, counter, None, settings)

    # Store cell masks and images
    zarr_file.require_group("cells/multi_session/original").create_dataset(
        str(counter),
        data=stat,
        dtype=object,
        object_codec=numcodecs.Pickle()
    )

    zarr_file.require_group("images/registered").create_dataset(
        str(counter),
        data=trans_images[multi_index],
        dtype=object,
        object_codec=numcodecs.Pickle()
    )

    zarr_file.require_group("images/original").create_dataset(
        str(counter),
        data=original_images[multi_index],
        dtype=object,
        object_codec=numcodecs.Pickle()
    )


def _process_individual_session(
    zarr_file: zarr.Group,
    session_path: Path,
    cell_templates: np.ndarray,
    counter: int
) -> None:
    """Process individual session data (non-aligned) and store it in a zarr file.

    Args:
        zarr_file (zarr.Group): Open zarr file for writing.
        session_path (Path): Path to the session data.
        cell_templates (np.ndarray): Cell templates for reference.
        counter (int): Current session index.
    """
    # Read ops
    ops = np.load(session_path / "ops.npy", allow_pickle=True).item()

    # Create empty stat structure
    empty_stat = {}
    for key in ops['stat'][0].keys() if 'stat' in ops else {'xpix', 'ypix', 'lam', 'npix'}:
        empty_stat[key] = None

    zarr_file.require_group("cells/multi_session/original").create_dataset(
        str(counter),
        data=[empty_stat],
        dtype=object,
        object_codec=numcodecs.Pickle()
    )

    # Create empty fluorescence data
    vr_group = zarr_file["gimbl/vr"]
    vr_info = vr_group[str(counter)][()]

    num_frames = vr_info.position.frame.reset_index()["frame"].max() if hasattr(vr_info, 'position') else 0
    num_cells = len(cell_templates)

    empty_array = np.empty((num_cells, num_frames))
    empty_array[:] = np.nan

    for field in ["F", "Fneu", "Fns", "Fdemix", "spks"]:
        zarr_file.require_group(f"multi_session/{field}").create_dataset(
            str(counter),
            data=empty_array,
            chunks=(1000, 10000)
        )

    # Store images
    imgs = {
        "mean_img": ops["meanImg"],
        "enhanced_img": ops["meanImgE"],
        "max_img": ops["max_proj"]
    }

    zarr_file.require_group("images/original").create_dataset(
        str(counter),
        data=imgs,
        dtype=object,
        object_codec=numcodecs.Pickle()
    )

    # Create empty registered images
    empty_img = np.zeros(ops["meanImg"].shape)
    empty_imgs = {key: empty_img for key in ["mean_img", "enhanced_img", "max_img"]}

    zarr_file.require_group("images/registered").create_dataset(
        str(counter),
        data=empty_imgs,
        dtype=object,
        object_codec=numcodecs.Pickle()
    )


def process_session_data(data_info: Dict[str, Any], settings: Dict[str, Any]) -> None:
    """Process and aggregate multi-day experimental data.

    This function takes data information and settings, processes all sessions, and 
    stores the results in a standardized zarr format.

    Args:
        data_info (dict): Information about data locations and parameters.
        settings (dict): Processing settings generated by parse_settings.

    Raises:
        NameError: If session paths are invalid or duplicated.
        FileNotFoundError: If required files cannot be found.

    Notes:
        Zarr organization structure:

        Signal data:
            - single_session/(F, Fdemix, Fneu, Fns, spks)/0   # Single session data
            - multi_session/(F, Fdemix, Fneu, Fns, spks)/0    # Multi-session aligned data

        Cell masks:
            - cells/single_session/0             # Cell data of original sessions
            - cells/multi_session/original/0     # Multi-session cell masks in original coordinates
            - cells/multi_session/registered/0   # Multi-session cell masks in registered coords

        Images:
            - images/original/0                  # Original session images
            - images/registered/0                # Registered session images

        VR Info:
            - gimbl/log/0                        # Raw pandas VR log
            - gimbl/vr/0                         # Processed VR data
    """
    # Load info and set up directories
    multiday_folder = Path(data_info["data"]["local_processed_root"]) / data_info["data"]["output_folder"]
    info = np.load(multiday_folder / "info.npy", allow_pickle=True).item()

    # Create output directory
    save_folder = multiday_folder
    save_folder.mkdir(parents=True, exist_ok=True)

   # Get imaging information
    ops_file = multiday_folder / "sessions" / info["data_paths"][0] / "ops.npy"
    ops = np.load(ops_file, allow_pickle=True).item()

    # Update settings with imaging info
    settings["imaging"] = {
        "frame_rate": ops["fs"],
        "num_planes": ops["nplanes"]
    }

    # Copy necessary settings from ops
    for field in ["fs", "Lx", "Ly"]:
        settings["demix"][field] = ops[field]

    settings["animal"] = data_info["animal"]
    settings["individual_sessions"] = data_info["data"]["individual_sessions"]

    # Initialize zarr storage
    zarr_folder = save_folder / "vr2p.zarr"
    if zarr_folder.is_dir():
        print(f"Removing previous data at {zarr_folder}")
        shutil.rmtree(zarr_folder, ignore_errors=True)

    # Load registration data
    backwards_deformed = np.load(multiday_folder / "backwards_deformed_cell_masks.npy", allow_pickle=True)
    cell_templates = np.load(multiday_folder / "cell_templates.npy", allow_pickle=True)
    trans_images = np.load(multiday_folder / "trans_images.npy", allow_pickle=True)
    original_images = np.load(multiday_folder / "original_images.npy", allow_pickle=True)

    # Process data paths
    data_paths = _prepare_data_paths(data_info, info)

    # Create and populate zarr store
    with zarr.open(zarr_folder.as_posix(), mode="w") as f:
        # Store settings and cell templates
        f.create_dataset("meta", data=settings, dtype=object, object_codec=numcodecs.Pickle())
        f.create_dataset("data_paths", data=data_paths)
        f.require_group("cells/multi_session").create_dataset(
            "registered",
            data=cell_templates,
            dtype=object,
            object_codec=numcodecs.Pickle()
        )

        # Process each session
        for counter, data_path in tqdm(enumerate(data_paths), desc="Processing session", total=len(data_paths)):
            multi_index = np.argwhere(np.array(info["data_paths"]) == data_path).squeeze()

            # Process VR data
            _process_vr_data(f, data_info, data_path, counter)

            # Process single session data
            session_path = Path(data_info["data"]["local_processed_root"]) / data_path / data_info["data"]["suite2p_folder"] / "combined"
            _process_single_session(f, session_path, settings, counter)

            # Process multi-session data if available
            if multi_index.size != 0:
                _process_multi_session(
                    f, multiday_folder, data_path, multi_index,
                    backwards_deformed, trans_images, original_images, counter,
                    settings
                )
            else:
                _process_individual_session(
                    f, session_path, cell_templates, counter
                )


def find_gimbl_log(folder_path):
    """Find Gimbl log json file in folder. Note: assumes word Log/log is in title.

    Args:
        folder_path (Path): Folder where to look for log
    """
    log_file = list(folder_path.glob("*Log*.json"))
    if not log_file:
        raise NameError(f"Could not find Gimbl Log in {folder_path}\nDoes file name include 'L(l)og' in its name?")
    if len(log_file)>1:
        raise NameError(f"Found multiple possible Gimbl log files in {folder_path}\nDo multiple json files include 'L(l)og' in their name?")
    log_file= log_file[0]
    return log_file
