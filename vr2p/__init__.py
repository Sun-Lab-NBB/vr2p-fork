import os
import json
import pickle
from pathlib import Path

import h5py
import zarr
import gcsfs
import numpy as np
import pandas as pd

import vr2p
from vr2p.gimbl.data import GimblData


class ExperimentData:
    """Holds all experimental data from one animal.
    General Organization:
    self.vr[session] # vr info.
    self.log[session] raw pandas vr log.
    self.signals.single_session.F[session][cell,frame] within session cell masks (also has F, Fneu, Fns, Fdemix, and spks)
    self.signals.multi_session.F[session][cell,frame] registered cell masks (consistent between sessions)
    self.images.original[session][key] original, non registered images
    self.images.registered[session][key] registered, transformed images
    self.cells.single_session[session][cell][key] within single session cel masks info.
    self.cells.multi_session.original[session][cell][key] cell masks info in original, untransformed coordinates.
    self.cells.multi_session.registered[session][cell][key] cell masks info in registered, transformed coordinates.
    self.meta[key] processing settings and animal meta info
    self.data_paths data paths of sessions.
    """

    def __init__ (self, file):
        class SignalData:
            """Actually temporarily opens the H5 file to read the data.
            """
            def __init__(self,file,field):
                self._file = file
                self._field = field
            def  __getattribute__(self,name):
                print(name)
        # check if cloud path:
        is_gcp = (file[:5].lower()=="gs://")
        is_aws = (file[:5].lower()=="s3://")
        if (not is_gcp) & (not is_aws):
            file = Path(file)
            if not file.is_dir():
                raise NameError(f"Could not find file: {file.as_posix()}")
            file = zarr.open(file,mode="r")
        if is_gcp:
            gcsmap = gcsfs.mapping.GCSMap(file)
            file = zarr.open(gcsmap,mode="r")

        self._file = file
        self.vr = ExperimentData.ObjectSessionData(self._file,"gimbl/vr")
        self.log = ExperimentData.LogSessionData(self._file,"gimbl/log")
        self.signals = self.SignalCollection(file)
        self.images = self.ImageCollection(file)
        self.cells = self.CellCollection(file)
        self.meta = file["meta"][()]
        self.data_paths = file["data_paths"][()]

    class SignalCollection:
        def __init__(self,file):
            self._file = file
            self.single_session = self.SignalCollectionData(file, "single")
            self.multi_session = self.SignalCollectionData(file, "multi")
        class SignalCollectionData:
            """Main class that holds all sessions related data (single or multi-aligned).
            """
            def __init__(self,file,field):
                self._file = file
                self._field = field
                self.F = self.SessionSignalData(self._file, f"{self._field}/F")            # raw cell signal
                self.Fneu = self.SessionSignalData(self._file,f"{self._field}/Fneu")       # raw neuropil signal
                self.Fns = self.SessionSignalData(self._file, f"{self._field}/Fns")        # neuropil subtracted signal.
                self.Fdemix = self.SessionSignalData(self._file, f"{self._field}/Fdemix")  # demixed, neuropil, and baseline subtracted signal
                self.spks = self.SessionSignalData(self._file, f"{self._field}/spks")      # spike signal (based on Fdemix)

            class SessionSignalData:
                """Responsible for supplying data from the right session.
                """
                class SignalData:
                    """Actually temporarily opens the H5 file to read the data.
                    """
                    def __init__(self,file,field):
                        self._file = file
                        self._field = field
                    def __getitem__(self,indices):
                        return self._file[f"{self._field}"].oindex[indices]
                    def __getattr__(self,name):
                        return getattr(self._file[f"{self._field}"],name)
                    @property
                    def array(self):
                        return self._file[f"{self._field}"]

                def __init__(self,file,field):
                    self._file = file
                    self._field = field
                def __getitem__(self,index):
                    if not isinstance(index,int):
                        raise ValueError("vr2p: Can only access one session at a time (did you remember the session index?)")
                    return self.SignalData(self._file, f"{self._field}/{index}")
                def __len__(self):
                    return len(self._file[f"{self._field}"])
    class LogSessionData:
        """General class that handles reading pickled session info from h5 file.
        """
        def __init__(self,file,field):
            self._file = file
            self._field = field
        def __getitem__(self,index):
            if not isinstance(index,int):
                raise ValueError("can only access one session at a time")
            return self._file[f"{self._field}/{index}"][()].value
    class ObjectSessionData:
        """General class that handles reading pickled session info from h5 file.
        """
        def __init__(self,file,field):
            self._file = file
            self._field = field
        def __getitem__(self,index):
            if not (isinstance(index,int) | isinstance(index,np.int32)):
                raise ValueError("can only access one session at a time")
            return self._file[f"{self._field}/{index}"][()]
        def __len__(self):
            return self._file[f"{self._field}"].__len__()
        def __iter__(self):
            self.index = -1
            return self
        def __next__(self):
            self.index+=1
            # Stop iteration if limit is reached
            if self.index == len(self):
                raise StopIteration
            return self.__getitem__(self.index)
    class ImageCollection:
        """Contains original and registered images
        """
        def __init__(self,file):
            self._file = file
            self.original = ExperimentData.ObjectSessionData(self._file,"images/original")
            self.registered = ExperimentData.ObjectSessionData(self._file,"images/registered")

    class CellCollection:
        """Contains cell masks from single session, and original+registered multi sessions
        """
        class MultiSessionCells:
            class RegisteredCells:
                def __init__(self,file):
                    self._file = file
                def __getitem__(self,indices):
                    return self._file["cells/multi/registered"][indices]
                def __len__(self):
                    return self._file["cells/multi/registered"].__len__()
                def __getattr__(self,name):
                    return getattr(self._file["cells/multi/registered"],name)
                def __iter__(self):
                    self._index = -1
                    self._values = self._file["cells/multi/registered"][()]
                    return self
                def __next__(self):
                    self._index+=1
                    # Stop iteration if limit is reached
                    if self._index == len(self._values):
                        raise StopIteration
                    return self._values[self._index]
            def __init__(self,file):
                self._file = file
                self.original = ExperimentData.ObjectSessionData(self._file,"cells/multi/original")
                self.registered = self.RegisteredCells(file)

        def __init__(self,file):
            self._file = file
            self.single_session = ExperimentData.ObjectSessionData(self._file,"cells/single")
            self.multi_session = self.MultiSessionCells(file)

def styles(name):
    file = os.path.join(os.path.dirname(vr2p.__file__),f"styles/{name}.mplstyle")
    return file

def test():
    print("Hello World!")
