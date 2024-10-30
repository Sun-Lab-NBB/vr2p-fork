import numpy as np


def movement_speed(df, window_size=100, ignore_threshold = 20):
    """Calculates rolling average movement speed in cm/s.
    Speed is absolute (no directionality)
    
    Arguments:
        df {dataframe} -- Must contain "position" and 'path', or "x", "y", "z", and "time" column
    
    Keyword Arguments:
        window_size {int} -- Size of rolling average window in ms. (default: {100})
        ignore_threshold {int} -- set delta frame distance movement above this threshold to zero (due to teleport etc.) (default: {20})
    
    Returns:
        [numpy array] -- calculated movement speed.
    """
    df = df.copy()
    # calculate moved distance.
    if ("x" in df) & ("y" in df) & ("z" in df):
        dist = df.loc[:,["x","y","z"]].diff()
        df["dist"] = abs(np.sqrt(dist.x**2 + dist.y**2 + dist.z**2))
        # Set fake path to process all values together.
        # Could be expanded to have some "environment" variable
        # to deal with teleports.
        df["path"] = "test"
    if ("path" in df):
        for path in df["path"].unique():
            # block out values not on current path.
            pos = df["position"].copy()
            ind = df["path"]==path
            pos[~ind] = np.nan
            # moved distance
            df.loc[ind,"dist"] = abs(pos.diff())
    for path in df["path"].unique():
        pos = df.copy()
        ind = df["path"]==path
        pos.loc[~ind,"dist"] = np.nan
        # filter unreasonable high number (teleport)
        pos.loc[pos["dist"]>ignore_threshold,"dist"] = 0
        # set moved to zero in case of missing values (start path etc.)
        pos.loc[ind,"dist"] = pos.loc[ind,"dist"].fillna(0)
        pos["speed"] = pos["dist"]/pos["time"].dt.total_seconds().diff()
        pos["speed"] = pos["speed"].fillna(0)
        # index by deltatime stamp for rolling window.
        pos=pos.reset_index().set_index("time")
        pos["speed"] = pos["speed"].rolling(f"{window_size}ms",min_periods=1).mean()
        pos = pos.reset_index(drop=True)
        # insert in main dataframe
        ind = ind.to_numpy() # This prevents a random error 'Buffer dtype mismatch, expected 'Python object' but got 'long''
        df.loc[ind,"speed"] = pos.loc[ind,"speed"].to_numpy() # also necessary for writing.
    return df["speed"].to_numpy()
