"""VR2P: A library for analyzing virtual reality experimental data with
longitudinal two-photon imaging."""

import os
from pathlib import Path
from typing import Any, Dict, List, Union, Optional, Iterator, Tuple

import zarr
import gcsfs
import s3fs
import numpy as np
from numpy.typing import NDArray

import vr2p


class ExperimentData:
    """Container for all experimental data from an animal.

    This class organizes and provides access to virtual reality data,
    imaging signals, cell masks, and metadata across multiple sessions.

    Parameters
    ----------
    file : str or Path
        Path to the data directory or cloud storage location

    Attributes
    ----------
    vr : ObjectSessionData
        Virtual reality session data
    log : LogSessionData
        Raw pandas VR log data
    signals : SignalCollection
        Neural activity signals (F, Fneu, Fdemix, spks)
    images : ImageCollection
        Original and registered images
    cells : CellCollection
        Cell mask information for both single and multi-session data
    meta : dict
        Processing settings and animal metadata
    data_paths : dict
        Paths to session data files

    Notes
    -----
    Data organization structure:
    - self.vr[session] : VR info
    - self.log[session] : Raw pandas VR log
    - self.signals.single_session.F[session][cell,frame] : Within-session cell signals
    - self.signals.multi_session.F[session][cell,frame] : Registered cell signals
    - self.images.original[session][key] : Original, non-registered images
    - self.images.registered[session][key] : Registered, transformed images
    - self.cells.single_session[session][cell][key] : Single-session cell mask info
    - self.cells.multi_session.original[session][cell][key] : Original coordinates
    - self.cells.multi_session.registered[session][cell][key] : Transformed coordinates
    - self.meta[key] : Processing settings and animal metadata
    - self.data_paths : Session data paths
    """

    def __init__(self, file: Union[str, Path]) -> None:
        # Check if cloud path
        is_gcp: bool = isinstance(file, str) and file.lower().startswith("gs://")
        is_aws: bool = isinstance(file, str) and file.lower().startswith("s3://")

        if not (is_gcp or is_aws):
            file = Path(file)
            if not file.is_dir():
                raise FileNotFoundError(f"Could not find directory: {file}")
            file = zarr.open(str(file), mode="r")
        elif is_gcp:
            gcsmap = gcsfs.mapping.GCSMap(file)
            file = zarr.open(gcsmap, mode="r")
        elif is_aws:
            s3map = s3fs.S3Map(file)
            file = zarr.open(s3map, mode="r")

        self._file = file
        self.vr = ExperimentData.ObjectSessionData(self._file, "gimbl/vr")
        self.log = ExperimentData.LogSessionData(self._file, "gimbl/log")
        self.signals = self.SignalCollection(file)
        self.images = self.ImageCollection(file)
        self.cells = self.CellCollection(file)
        self.meta = file["meta"][()]
        self.data_paths = file["data_paths"][()]

    class SignalCollection:
        """Collection of neural activity signals across sessions.

        Parameters
        ----------
        file : zarr.Group
            Zarr group containing signal data

        Attributes
        ----------
        single_session : SignalCollectionData
            Single session data without alignment
        multi_session : SignalCollectionData
            Multi-session data with alignment between sessions
        """

        def __init__(self, file: zarr.Group) -> None:
            self._file = file
            self.single_session = self.SignalCollectionData(file, "single_session")
            self.multi_session = self.SignalCollectionData(file, "multi_session")

        class SignalCollectionData:
            """Container for session-related neural signal data.

            Holds all signal types (raw, neuropil, etc.) for either 
            single-session or multi-session aligned data.

            Parameters
            ----------
            file : zarr.Group
                Zarr group containing signal data
            field : str
                Either "single_session" or "multi_session" to specify data type

            Attributes
            ----------
            F : SessionSignalData
                Raw cell fluorescence signal
            Fneu : SessionSignalData
                Raw neuropil fluorescence signal
            Fns : SessionSignalData
                Neuropil-subtracted fluorescence signal
            Fdemix : SessionSignalData
                Demixed, neuropil and baseline subtracted signal
            spks : SessionSignalData
                Inferred spike signal (based on Fdemix)
            """

            def __init__(self, file: zarr.Group, field: str) -> None:
                self._file = file
                self._field = field
                self.F = self.SessionSignalData(self._file, f"{self._field}/F")
                self.Fneu = self.SessionSignalData(self._file, f"{self._field}/Fneu")
                self.Fns = self.SessionSignalData(self._file, f"{self._field}/Fns")
                self.Fdemix = self.SessionSignalData(self._file, f"{self._field}/Fdemix")
                self.spks = self.SessionSignalData(self._file, f"{self._field}/spks")

            class SessionSignalData:
                """Provides access to signal data for specific sessions.

                Parameters
                ----------
                file : zarr.Group
                    Zarr group containing signal data
                field : str
                    Path to signal data within the zarr file
                """

                class SignalData:
                    """Handles direct access to signal data arrays.

                    Parameters
                    ----------
                    file : zarr.Group
                        Zarr group containing signal data
                    field : str
                        Complete path to specific signal data
                    """

                    def __init__(self, file: zarr.Group, field: str) -> None:
                        self._file = file
                        self._field = field

                    def __getitem__(self, indices: Union[int, Tuple[int, int]]) -> Any:
                        return self._file[f"{self._field}"].oindex[indices]

                    def __getattr__(self, name: str) -> Any:
                        return getattr(self._file[f"{self._field}"], name)

                    @property
                    def array(self) -> NDArray:
                        """Get the full underlying array."""
                        return self._file[f"{self._field}"]

                def __init__(self, file: zarr.Group, field: str) -> None:
                    self._file = file
                    self._field = field

                def __getitem__(self, index: int) -> SignalData:
                    if not isinstance(index, int):
                        raise ValueError(
                            "vr2p: Can only access one session at a time "
                            "(Do you remember the session index?)"
                        )
                    return self.SignalData(self._file, f"{self._field}/{index}")

                def __len__(self) -> int:
                    return len(self._file[f"{self._field}"])
    class LogSessionData:
        """Handles access to pickled session information from zarr file.

        Parameters
        ----------
        file : zarr.Group
            Zarr group containing log data
        field : str
            Path to log data within the zarr file
        """

        def __init__(self, file: zarr.Group, field: str) -> None:
            self._file = file
            self._field = field

        def __getitem__(self, index: int) -> object:
            if not isinstance(index, int):
                raise ValueError("can only access one session at a time")
            return self._file[f"{self._field}/{index}"][()].value

    class ObjectSessionData:
        """Handles access to pickled session objects from zarr file.

        Provides iteration capabilities over sessions.

        Parameters
        ----------
        file : zarr.Group
            Zarr group containing session data
        field : str
            Path to session data within the zarr file
        """

        def __init__(self, file: zarr.Group, field: str) -> None:
            self._file = file
            self._field = field
            self.index = -1

        def __getitem__(self, index: Union[int, np.int32]) -> Any:
            if not (isinstance(index, int) or isinstance(index, np.int32)):
                raise ValueError("can only access one session at a time")
            return self._file[f"{self._field}/{index}"][()]

        def __len__(self) -> int:
            return self._file[f"{self._field}"].__len__()

        def __iter__(self) -> 'ExperimentData.ObjectSessionData':
            self.index = -1
            return self

        def __next__(self) -> Any:
            self.index += 1
            # Stop iteration if limit is reached
            if self.index == len(self):
                raise StopIteration
            return self.__getitem__(self.index)

    class ImageCollection:
        """
        Contains original and registered imaging data.
        
        Parameters
        ----------
        file : zarr.Group
            Zarr group containing image data
            
        Attributes
        ----------
        original : ObjectSessionData
            Original, non-registered images
        registered : ObjectSessionData
            Registered, transformed images
        """
        
        def __init__(self, file: zarr.Group) -> None:
            self._file = file
            self.original = ExperimentData.ObjectSessionData(self._file, "images/original")
            self.registered = ExperimentData.ObjectSessionData(self._file, "images/registered")

    class CellCollection:
        """Contains cell masks from single and multi-session experiments.

        Parameters
        ----------
        file : zarr.Group
            Zarr group containing cell data

        Attributes
        ----------
        single_session : ObjectSessionData
            Cell masks from single-session data
        multi_session : MultiSessionCells
            Cell masks from multi-session data in both original and registered coordinates
        """

        def __init__(self, file: zarr.Group) -> None:
            self._file = file
            self.single_session = ExperimentData.ObjectSessionData(self._file, "cells/single_session")
            self.multi_session = self.MultiSessionCells(file)

        class MultiSessionCells:
            """Container for multi-session cell data in original and registered formats.

            Parameters
            ----------
            file : zarr.Group
                Zarr group containing multi-session cell data

            Attributes
            ----------
            original : ObjectSessionData
                Cell masks in original coordinates
            registered : RegisteredCells
                Cell masks in registered, transformed coordinates
            """

            def __init__(self, file: zarr.Group) -> None:
                self._file = file
                self.original = ExperimentData.ObjectSessionData(self._file, "cells/multi_session/original")
                self.registered = self.RegisteredCells(file)

            class RegisteredCells:
                """Provides access to registered cell data with iteration capability.

                Parameters
                ----------
                file : zarr.Group
                    Zarr group containing registered cell data
                """

                def __init__(self, file: zarr.Group) -> None:
                    self._file = file
                    self._index = -1
                    self._values = None

                def __getitem__(self, indices: Any) -> Any:
                    return self._file["cells/multi_session/registered"][indices]

                def __len__(self) -> int:
                    return self._file["cells/multi_session/registered"].__len__()

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._file["cells/multi_session/registered"], name)

                def __iter__(self) -> 'ExperimentData.CellCollection.MultiSessionCells.RegisteredCells':
                    self._index = -1
                    self._values = self._file["cells/multi_session/registered"][()]
                    return self

                def __next__(self) -> Any:
                    self._index += 1
                    # Stop iteration if limit is reached
                    if self._index == len(self._values):
                        raise StopIteration
                    return self._values[self._index]

def styles(name: str) -> str:
    """Get the path to a matplotlib style file.

    Parameters
    ----------
    name : str
        Name of the style file (without extension)

    Returns
    -------
    str
        Full path to the requested style file
    """
    file = os.path.join(os.path.dirname(vr2p.__file__), f"styles/{name}.mplstyle")
    return file


def test() -> None:
    """Simple test function to verify the library is working."""
    print("Hello World!")
