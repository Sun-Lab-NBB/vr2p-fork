import numpy as np
import pandas as pd
from scipy.interpolate import Rbf, splev, splprep
from scipy.spatial.distance import cdist

from vr2p.gimbl.extract import movement_speed
from vr2p.gimbl.transform import add_ranged_timestamp_values


class GimblData:
    class IdleData:
        def __init__(self):
            self.sound = None
    class TimeData:
        def __init__(self):
            self.time = None
            self.frame = None
    class ControllerData:
        def __init__(self):
            self.settings = None
            self.time = None
            self.frame = None
    def __init__(self):
        self.time = None
        self.info = None
        self.frames = None
        self.position = self.TimeData()
        self.path = self.TimeData()
        self.camera = None
        self.reward = None
        self.lick = None
        self.idle = self.IdleData()
        self.linear_controller = self.ControllerData()
        self.spherical_controller = self.ControllerData()

    def path_to_xyz(self,values,path):
        """Interpolates from path positions (1d) to xyz coordinates.

        Arguments:
            values {1d numpy array} -- path positions
            path {[type]} -- Name of path.

        Returns:
            numpy array -- Interpolated values (size: [num_values x 3])
        """
        # get indices that are on the requested path.
        ind = self.path.frame["path"]==path
        if sum(ind)==0:
            raise NameError(f"could not find path with name {path}")
        df = self.position.frame.loc[ind,["x","y","z"]]
        df["path"] = self.path.frame.loc[ind,"position"]
        # sort.
        df= df.sort_values(by=["path"])
        #drop duplicates
        df["path_r"] = df["path"].round(0)
        df = df.drop_duplicates(subset="path_r")
        # b spline fit
        tck, _= splprep([df["x"],df["y"],df["z"] ],u=df["path"], s=0.01)
        xi, yi, zi = splev(values, tck)
        result = np.c_[xi,yi,zi]
        if result.shape[0]==1:
            result=result.flatten()
        return result

    def xyz_to_path(self,values):
        """Interpolates from xyz to path position based on observed values.

        Arguments:
            values {numpy array} -- requested xyz positions (size: num_positions x 3)

        Returns:
            DataFrame -- found positions
                            columns:
                                "position"  -- path position.
                                "path"      -- path name.
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
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # verify there is a column time.
        if "time" not in obj.columns:
            raise AttributeError("Must have 'time'.")

    def rolling_speed(self, window_size, ignore_threshold = 50):
        """Calculates rolling movement speed. Accessible from vr2p object.

        Arguments:
            window_size {int}       -- Rolling window size (ms.)

        Keyword Arguments:
            ignore_threshold {int} -- Movement speed higher then this is ignored (teleport) (default: {50})

        Returns:
            [DataSeries]           -- Calculated movement speed.
        """
        return movement_speed(self._obj,window_size=window_size,ignore_threshold = ignore_threshold )

    def ranged_values(self,df,fields):
        """Adds columns to entries of dataframe with values base on its timestamp.
        supplied dataframe gives a range of timestamp values and related values.

        Arguments:
            df {dataframe} -- timestamped dataframe with values to assign (must have "time_start" and "time_end" column).
            fields {list of strings} -- requested columns in suplied df to add.

        Returns:
            dataframe -- dataframe with additional columns assigned based on timestamps.
        """
        return add_ranged_timestamp_values(self._obj,df,fields)

class FieldTypes:
        fields = { "data.name": "category",
        "data.isActive": "bool",
        "msg":"category",
        "data.loopPath":"bool",
        "data.time":"category",
        "data.project":"category",
        "data.scene":"category",
        "data.source":"category",
        "data.msg.event":"category",
        "data.msg.id":"category",
        "data.duration":"category",
        "data.pathName":"category",
        "data.msg":"category",
        "data.environment":"category",
        "data.level":"int",
        "data.epoch":"int",
        "data.lap":"int",
        "data.success":"bool",
        "data.msg.action":"category",
        "data.msg.type":"category",
        "data.msg.withSound":"bool",
        "data.msg.frequency":"category",
        "data.distance":"int",
        "data.gain.forward":"category",
        "data.gain.backward":"category",
        "data.inputSmooth":"category"}
