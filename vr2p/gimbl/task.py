import numpy as np
import pandas as pd

from vr2p.gimbl.parse import parse_custom_msg


def get_lap_info(log_df, fields, path_time_df):
    """Parses lap information from log file.
    Uses actual position on path to increase precision. 
    The function assumes "lap" messages with lap#==0 can be ignored. 
    Also, assumes laps are take atleast 3 seconds.

    Arguments:
        log_df {dataframe}  --  complete dataframe from gimbl log
                                must contain columns:
                                    "time"  -- time since start in ms.
                                    "msg"   -- indentifier for Lap message.
                                    "data.[field]" -- requested data field
        fields {[list]}     -- Info to read from lap message. Must include 'lap'.
        path_time_df {dataframe} -- Info from "parse_path"

    Returns:
        dataframe -- parsed Lap information.
                index: 
                    "index" -- number of parsed lap
                columns:
                    "time"          -- time since start in ms.
                    "time_start"    -- start time of lap.
                    "time_start"    -- end time of lap.
    """
    data = parse_custom_msg(log_df,"Lap",fields).drop(columns="frame")
    # get time ranges based on position (more accurate)
    time_range = get_lap_from_path(path_time_df)
    #remove entires at start of new environment.
    data = data.drop(data[data["lap"]==0].index)
    data = data.reset_index(drop=True)
    data["lap"] = data["lap"]-1
    # find matching entries.
    if (time_range.shape[0]<data.shape[0]):
        raise NameError(f"Fewer periods ({time_range.size}) detected then logged laps({data.size}).")
    ind = []
    assigned = np.zeros(time_range.shape[0],np.bool)
    for time in data["time"].dt.total_seconds():
        offset = np.abs(time_range[:,1] - time)
        offset[assigned] = np.inf
        assigned[offset.argmin()] = True
        ind.append(offset.argmin())
    ind = np.array(ind)
    # add info.
    data["time_start"] = pd.to_timedelta(time_range[ind,0],unit="seconds")
    data["time_end"] = pd.to_timedelta(time_range[ind,1],unit="seconds")
    return data

def get_lap_from_path(path_time_df):
    """Finds possible lap periods based on position info

    Arguments:
        path_time_df {dataframe} -- Info from "parse_path"

    Returns:
        numpy array -- detected start and end times. shape: <num periods x 2>
    """
    # get data.
    position = path_time_df["position"].to_numpy()
    timestamps = path_time_df["time"].dt.total_seconds().to_numpy()
    paths = path_time_df["path"].to_numpy()
    # get possible start and end points.
    start_points =[]
    end_points = []
    for path in np.unique(paths):
        threshold = position[paths==path].min() + ((position[paths==path].max() - position[paths==path].min())*0.1)
        temp_start = (position<threshold) & (paths==path)
        start_points.append(np.argwhere(np.convolve( temp_start,[0,1,-1],"same")==1).flatten())
        threshold = position[paths==path].min() + ((position[paths==path].max() - position[paths==path].min())*0.1)
        temp_end =  (position>threshold) & (paths==path)
        end_points.append(np.argwhere(np.convolve(temp_end,[-1,1,0],"same")==1).flatten())
    start_points = np.sort(np.hstack(np.array(start_points)))
    end_points = np.sort(np.hstack(np.array(end_points)))
    # find periods.
    time_range = []
    for i, start in enumerate(start_points[:-1]):
        next_start = start_points[i+1]
        #next start on same path.
        if (paths[start] == paths[next_start]):
            #find matching end
            ind = np.argwhere((end_points>start) & (end_points<next_start))
            # check if next start isnt before end point.
            if ind.size != 0:
                # get latest possible end points.
                match_end = end_points[ind[-1]][0]
                # check that all sections of track were visited.
                values = position[start:match_end]
                count,_ = np.histogram(values,bins=5)
                if all(count>0):
                    time_range.append([timestamps[start],timestamps[match_end]])
    time_range = np.array(time_range)
    return time_range


