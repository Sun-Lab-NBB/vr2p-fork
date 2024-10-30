import numpy as np
import pandas as pd


def assign_frame_info(data, frames,remove_nan=True):
    """Joins indexed gimbl log data with frame info to assign frame numbers.
    
    Arguments:
        data {dataframe}    -- gimbl log data to assign frame info to 
                                (note:requires "time" column, denoting time since start in ms). 
        frames {[type]}     -- parsed frame info.

    Keyword Arguments:
        remove_nan {bool}   -- Removes entries outside of microscope frame range (at start) (default: {True})    
    
    Returns:
        [dataframe]         -- joined dataframe (added "frame" column)
    """
    frame_data = pd.concat([frames.reset_index().set_index("time"),data.reset_index().set_index("time")]).sort_index()
    idx = frame_data["frame"].isnull()
    frame_data["frame"] = frame_data["frame"].fillna(method="ffill")
    # remove frame info.
    frame_data = frame_data[idx]
    frame_data = frame_data.drop(columns="index")
    # remove outside frame start data.
    if remove_nan:
        frame_data = frame_data.dropna(subset=["frame"])
    else:
        frame_data["frame"] = frame_data["frame"].fillna(value=-1)
        if "level_0" in frame_data:
            frame_data = frame_data.drop(columns="level_0")
    frame_data["frame"] = frame_data["frame"].astype("int32")
    frame_data=frame_data.reset_index()
    return frame_data

def ffill_missing_frame_info(frame_data, frames, nan_fill=False,subset_columns=None):
    """Forward filling of missing frames. Copies info in previous frame.

    Arguments:
        frame_data {dataframe}  -- gimbl log data with "frame" column
        frames {dataframe}      -- parsed frame info.

    Keyword Arguments:
        verbose {bool}          -- Print message if missing frame info is found (default: {False})
        nan_fill {bool}         -- instead of value in previous frame fill in None.
        subset_columns {list}   -- Apply only to subset of columns (all if left empty)

    Returns:
        [dataframe]             -- Same columns as frame_data with possible extra entries.
    """
    if not subset_columns:
        subset_columns = frame_data.columns
    # get info.
    num_frames = frames.shape[0]
    names = frame_data.reset_index()["name"].unique()
    for name in names:
        # select info for actor and fill in missing values from template dataframe with timestamps for all frames
        df = frame_data.loc[(slice(None),name),:].copy().reset_index().set_index("frame")
        template = pd.DataFrame({"frame" : range(num_frames),"name": name,"time":frames["time"].to_numpy()}).set_index("frame")
        df = df.combine_first(template).sort_index()
        # ffil.
        if not nan_fill:
            df[subset_columns] = df[subset_columns].fillna(method="ffill", axis=0)
        # add.
        df = df.reset_index().set_index(["frame","name"])
        frame_data = frame_data.combine_first(df)
    return frame_data.sort_index()

def add_timestamp_values(df, timestamp_df, fields):
    """Takes two dataframes with time column and adds extra columns to excisting entries in first dataframe based on its timestamp.
    This is performed in a 'forward-filling' way (no edges only starting time points unlike add_ranged_timestamp_values).
    
    Arguments:
        df {dataframe}              -- dataframe to add values to (must have "time" column).
        timestamp_df {dataframe}    -- timestamped dataframe with values to assign (must have "time" column).
        fields {list of strings}    -- requested columns in timestamp_df to add to df.
    
    Returns:
        [dataframe]                 -- copy of df with additional columns assigned based on timestamp_df. [rows_df] x [columns_df + extra_columns_timestamp_df]
    """
    df = df.copy()
    # sort time stamps.
    timestamp_df = timestamp_df.sort_values("time")
    df = df.sort_values("time")
    # Set default value.
    for field in fields:
        df[field] = None
    # search for matching indices.
    ind = np.array(timestamp_df["time"].searchsorted(df["time"])-1)
    # store indices without timestamp
    mask = ind<0
    ind[mask] = 0
    # assing values
    df[fields]= timestamp_df.iloc[ind][fields].reset_index(drop=True).to_numpy()
    # set missing to none
    df.loc[mask,fields]=None
    return df

def add_ranged_timestamp_values(df, timestamp_df,fields):
    """Adds columns to entries of dataframe with values base on its timestamp
    Timestamp_df gives a range of timestamp values and the related values.
    (this is unlike add_timestamp_values which does forward-filling based on start times only)
    
    Arguments:
        df {dataframe}              -- dataframe to add values to (must have "time" column).
        timestamp_df {dataframe}    -- timestamped dataframe with values to assign (must have "time_start" and "time_end" column).
        fields {list of strings}    -- requested columns in timestamp_df to add to df.

    Returns:
        [dataframe]                 -- copy of df with additional columns assigned based on timestamp_df. [rows_df] x [columns_df + extra_columns_timestamp_df]
    """
    df = df.copy()
    # get column list. (for fixing the order)
    col = df.columns.to_list()
    # check if all requested columns present.
    if (set(fields).issubset(timestamp_df.columns)==False):
        raise NameError("requested fields not present in supplied dataframe")
    # add missing columns to target
    missing = list(set(fields).difference(set(df.columns)))
    for field in missing:
        df[field] = None
    # get indexes of each entry
    time = df["time"].values
    start = timestamp_df["time_start"].values
    end = timestamp_df["time_end"].values
    for i in range(start.size):
        ind = (time>=start[i]) & (time<=end[i])
        df.loc[(ind),fields] = timestamp_df.loc[(i),fields].values
    # reorder.
    missing = [field for field in fields if field in missing]
    df=df[col+missing]
    return df
