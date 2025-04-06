"""Data structures and utilities for working with Gimbl virtual reality experiment data.

This module provides classes for representing, processing and analyzing virtual reality
data collected from Gimbl VR systems, including position tracking, path information,
controller data, and other experimental metrics.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import Rbf, splev, splprep
from scipy.spatial.distance import cdist

from vr2p.gimbl.extract import movement_speed
from vr2p.gimbl.transform import add_ranged_timestamp_values


class IdleData:
    """Represents idle state data, such as times when no movement or interaction occurs.

    Attributes:
        sound (Optional[str]): Placeholder for any sound or audio cue during idle states.
    """

    def __init__(self) -> None:
        self.sound: Optional[str] = None


class TimeData:
    """Represents time-based data, including timestamps and frame data.

    Attributes:
        time (Optional[np.ndarray]): Array of time points for the session.
        frame (Optional[pd.DataFrame]): DataFrame describing each frame (e.g., positions, states).
    """

    def __init__(self) -> None:
        self.time: Optional[np.ndarray] = None
        self.frame: Optional[pd.DataFrame] = None


class ControllerData:
    """Represents data from a VR controller, including settings and frame references.

    Attributes:
        settings (Optional[dict]): Controller settings or configuration parameters.
        time (Optional[np.ndarray]): Time array for the controller updates.
        frame (Optional[pd.DataFrame]): DataFrame capturing controller states per frame.
    """

    def __init__(self) -> None:
        self.settings: Optional[dict] = None
        self.time: Optional[np.ndarray] = None
        self.frame: Optional[pd.DataFrame] = None


class GimblData:
    """Container class for VR Gimbl data, including positional, path, and controller information.

    This class holds multiple nested classes for specialized data types (e.g., time-related
    or controller-related). It also provides methods for converting between path coordinates
    and XYZ coordinates.

    Attributes:
        time (Optional[np.ndarray]): Global time array for the session.
        info (Optional[dict]): General information or metadata for the session.
        frames (Optional[pd.DataFrame]): Table of frame-based data (e.g., frame indices,
            timestamps).
        position (TimeData): Time-based position data containing coordinates over time.
        path (TimeData): Time-based path data containing path names or positions.
        camera (Optional[dict]): Camera-related information (e.g., camera parameters or
            transforms).
        reward (Optional[dict]): Reward-related data (e.g., reward timings or amounts).
        lick (Optional[dict]): Lick-related data (e.g., lick detection or timing).
        idle (IdleData): Idle-related data containing placeholders for non-active states.
        linear_controller (ControllerData): Data from a linear VR controller with settings and
            frames.
        spherical_controller (ControllerData): Data from a spherical VR controller with settings
            and frames.
    """

    def __init__(self):
        self.time = None
        self.info = None
        self.frames = None
        self.position = TimeData()
        self.path = TimeData()
        self.camera = None
        self.reward = None
        self.lick = None
        self.idle = IdleData()
        self.linear_controller = ControllerData()
        self.spherical_controller = ControllerData()

    def path_to_xyz(self, values: np.ndarray, path: str) -> np.ndarray:
        """Interpolate from path positions (1D) to XYZ coordinates using a B-spline fit.

        Args:
            values (np.ndarray): Array of path positions (shape: [num_positions]).
            path (str): Name of the path to interpolate (must exist in self.path.frame).

        Returns:
            np.ndarray: Interpolated XYZ values (shape: [num_positions, 3]).

        Raises:
            NameError: If the specified path is not found in self.path.frame.
        """
        # get indices that are on the requested path.
        ind = self.path.frame["path"] == path
        if sum(ind) == 0:
            raise NameError(f"Could not find path with name {path}")

        df = self.position.frame.loc[ind, ["x", "y", "z"]].copy()
        df["path"] = self.path.frame.loc[ind, "position"]

        # Sort by path position and remove duplicates
        df = df.sort_values(by=["path"])
        df["path_r"] = df["path"].round(0)
        df = df.drop_duplicates(subset="path_r")

        # Fit a B-spline over the x, y, z coordinates
        tck, _ = splprep([df["x"], df["y"], df["z"]], u=df["path"], s=0.01)
        xi, yi, zi = splev(values, tck)

        result = np.column_stack((xi, yi, zi))
        if result.shape[0] == 1:
            return result.flatten()
        return result

    def xyz_to_path(self, values: np.ndarray) -> pd.DataFrame:
        """Interpolate from XYZ coordinates to path positions.

        This method determines the closest path among all available paths, then
        evaluates a radial basis function (RBF) fit to compute the corresponding
        path position for each 3D coordinate.

        Args:
            values (np.ndarray): Requested XYZ positions (shape: [num_positions x 3]).

        Returns:
            pd.DataFrame:
                A DataFrame containing the inferred path positions. It has two columns:
                - "position": The path position (float).
                - "path": The path name (string).
        """
        # create fit for all paths.
        fits = []
        path_names = self.path.frame["path"].unique()
        for path in path_names:
            # get data of path.
            ind = self.path.frame["path"]==path
            df = self.position.frame.loc[ind,["x","y","z"]]
            df["path"] = self.path.frame.loc[ind,"position"]
            # sort.
            df= df.sort_values(by=["path"])
            # drop duplicates
            df["path_r"] = df["path"].round(0)
            df = df.drop_duplicates(subset="path_r")
            # fit
            fits.append(Rbf(df["x"], df["y"], df["z"], df["path"],smooth=0.01))

        # find closest path each point and evaluate
        obs = self.position.frame.loc[:,["x","y","z"]].to_numpy() # observed
        dist = cdist(values,obs)
        result = []
        for i, value in enumerate(values):
            ind = np.argmin(dist[i,:])
            ind = np.argwhere(path_names == self.path.frame.loc[ind,"path"].item())[0][0]
            pos = fits[ind](value[0],value[1],value[2])
            result.append({"position":pos,"path":path_names[ind]})

        return pd.DataFrame(result)

@pd.api.extensions.register_dataframe_accessor("vr2p")
class Vr2pAccessor:
    """A pandas DataFrame accessor that provides VR2P-specific methods for analyzing and
    transforming virtual reality data.

    This accessor allows rolling speed calculations and timed value assignments in a pandas
    DataFrame, ensuring the necessary columns and formats are present.
    """
    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        """Initialize the Vr2pAccessor with a pandas DataFrame.

        Args:
            pandas_obj (pd.DataFrame): The DataFrame to access.
        """
        # Check if the object is a DataFrame
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: pd.DataFrame) -> None:
        """Validate that the DataFrame has the necessary columns.

        Args:
            obj (pd.DataFrame): The DataFrame to validate.

        Raises:
            AttributeError: If required columns are missing.
        """
        # verify there is a column time.
        if "time" not in obj.columns:
            raise AttributeError("Must have 'time'.")

    def rolling_speed(self, window_size: int, ignore_threshold: int = 50) -> pd.Series:
        """Calculate rolling movement speed over a specific time window.

        Args:
            window_size (int): Rolling window size in milliseconds.
            ignore_threshold (int, optional): Speed threshold above which
                measurements are considered invalid (teleport artifacts). Defaults to 50.

        Returns:
            pd.Series: The calculated rolling movement speed.
        """
        return movement_speed(self._obj,window_size=window_size,ignore_threshold = ignore_threshold )

    def ranged_values(self, df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
        """Add columns to DataFrame entries based on their timestamps.

        Uses the provided `df` to look up a time range (columns "time_start"
        and "time_end"), then merges specified columns into the accessor's
        DataFrame.

        Args:
            df (pd.DataFrame): A DataFrame with time ranges ("time_start" and "time_end")
                and the fields to add.
            fields (list[str]): Columns from `df` to merge into the accessor's DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with the requested columns added, populated by
            matching timestamp ranges.
        """
        return add_ranged_timestamp_values(self._obj,df,fields)

class FieldTypes:
    """Registry of field data types used in GIMBL data processing.

    This class provides a centralized mapping of field names to their corresponding 
    pandas data types to ensure consistent data type conversion when loading or 
    processing data.

    Attributes:
        fields (Dict[str, str]): Dictionary mapping field names to pandas dtypes.
    """

    fields: dict[str, str] = {
        # Metadata fields
        "data.name": "category",
        "data.isActive": "bool",
        "data.time": "category",
        "data.project": "category",
        "data.scene": "category",
        "data.source": "category",
        "data.loopPath": "bool",
        "data.environment": "category",
        
        # Message fields
        "msg": "category",
        "data.msg": "category",
        "data.msg.event": "category",
        "data.msg.id": "category",
        "data.msg.action": "category",
        "data.msg.type": "category",
        "data.msg.withSound": "bool",
        "data.msg.frequency": "category",
        
        # Path and navigation fields
        "data.pathName": "category",
        "data.duration": "category",
        "data.distance": "int",
        
        # Session state fields
        "data.level": "int",
        "data.epoch": "int",
        "data.lap": "int",
        "data.success": "bool",
        
        # Control fields
        "data.gain.forward": "category",
        "data.gain.backward": "category",
        "data.inputSmooth": "category"
    }
