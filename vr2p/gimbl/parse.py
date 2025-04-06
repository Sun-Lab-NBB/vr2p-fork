import json
import numpy as np
import pandas as pd

from vr2p.gimbl.data import GimblData, FieldTypes
from vr2p.gimbl.transform import assign_frame_info, ffill_missing_frame_info


def parse_gimbl_log(file_loc, verbose=False):
    data = GimblData()
    # load file.
    with open(file_loc) as data_file:
        file_data = json.load(data_file)
    df = pd.json_normalize(file_data)
    df = set_data_types(df)
    # get session info.
    data.info = parse_session_info(df)
    # get frame info.
    frames = parse_frames(df)
    data.frames = frames
    # get absolute position info
    data.position.time = parse_position(df)
    data.position.frame = get_position_per_frame(data.position.time, frames)
    #get path position.
    data.path.time = parse_path(df)
    data.path.frame = get_path_position_per_frame(df, frames)
    # camera info.
    data.camera = parse_camera(df, frames)
    # reward info.
    data.reward = parse_reward(df, frames)
    # lick info.
    data.lick = parse_custom_msg(df, "Lick", [], frames=frames)
    # idle info
    data.idle.sound = parse_idle_sound(df, frames)
    # controller info
    data.linear_controller.settings = parse_linear_settings(df, frames)
    data.spherical_controller_settings = parse_spherical_settings(df, frames)
    # controller data.
    data.linear_controller.time = parse_linear_data(df)
    data.spherical_controller.time = parse_spherical_data(df)
    data.linear_controller.frame = get_linear_data_per_frame(df, frames)
    data.spherical_controller.frame = get_spherical_data_per_frame(df, frames)
    return df, data

def set_data_types(df):
    fields = FieldTypes.fields
    for key in fields.keys():
        if key in df:
            if fields[key] == "int":
                df[key] = df[key].fillna(0)
            df[key] = df[key].astype(fields[key])
    df["time"] = pd.to_timedelta(df["time"], unit="ms")
    return df

def parse_custom_msg(df, msg, fields, frames=pd.DataFrame(), rename_columns={}, msg_field="msg", data_field="data", remove_nan=False):
    """Main parsing function. Looks for message in dataframe of log and pulls out requested data.
    
    Arguments:
        df {dataframe}              -- complete dataframe from gimbl log
        msg {string}                -- expected identifier message string in log.
        fields {list}               -- list of strings of data fields to read.
    
    Keyword Arguments:
        frame {dataframe}          -- microscope frame information (default: {pd.DataFrame()})
        rename_columns {dict}       -- renames read in fields (default: {{}})
        msg_field {str}             -- field to search for msg identifier string (default: {"msg"})
        data_field {str}            -- field to read in data fields from(default: {"data"})
        remove_nan {bool}           -- if true: removes entries without associated frames. (default: {False})
    
    Returns:
        dataframe -- parsed dataframe. Will have index(0:num_messages) and time column + requested fields.
    """
    data = pd.DataFrame(columns=["index", "time", "frame"] + fields).set_index("index")
    if msg_field in df:
        idx = df[msg_field] == msg
        if any(idx):
            data_fields = ["time"] + [f"{data_field}.{field}" for field in fields]
            # check all columns are present.
            if ~set(data_fields).issubset(df.columns):
                for field in data_fields:
                    if field not in df:
                        raise NameError(f"{field} is not a column in given dataframe:")
            data = df.loc[idx, data_fields].reset_index().set_index("index")
            # rename
            for field in fields:
                data = data.rename(columns={f"{data_field}.{field}": field})
            #requested rename.
            for key in rename_columns.keys():
                data = data.rename(columns={f"{key}": rename_columns[key]})
            if not frames.empty:
                data = assign_frame_info(data, frames, remove_nan=remove_nan)
            else:
                data["frame"] = None
                data = data.reset_index(drop=True)
        data = data.reset_index().set_index("index")
    return data

def parse_frames(df):
    """Reads frame timestamps from gimbl log.
    
    Arguments:
        df {dataFrame} -- complete dataframe from gimbl log
    
    Returns:
        [dataFrame] -- parsed frame information <num_frames> x 1 column,
                index: 
                    "frame" 
                columns: 
                    "time"      -- time since start in ms,
    """
    frames = parse_custom_msg(df, "microscope frame", [], msg_field="data.msg")
    frames = frames.drop(columns="frame")
    frames = frames.rename_axis("frame")
    #if not frames.empty:
        #frames = frames.drop(0) # drops first signal (based on sync test.)
    return frames

def parse_position(df):
    """Parses position information from gimbl log dataframe
    
    Arguments:
        df {dataframe} -- complete dataframe from gimbl log
    
    Returns:
        dataframe -- parsed position information <num_monitor_frames>x6 columns,
                index: 
                    "index" -- occurence of position in log.
                    "name"  -- name of gimbl actor.
                columns:
                    "time"      -- time since start in ms.
                    "position"  -- 1x3 position vector xyz.
                    "heading"   -- heading of animal in degrees.
                    "x","y","z" -- position value in cms.
    """
    position = parse_custom_msg(df, "Position", ["name", "position", "heading"])
    # set name as category with only available actors.
    position["name"] = position["name"].astype("string")
    cat_type = pd.CategoricalDtype(categories=position.name.unique(), ordered=False)
    position["name"] = position["name"].astype(cat_type)
    # set index.
    position = position.reset_index().set_index(["index", "name"]).drop(columns="frame")
    # Convert position to cms.
    position["position"] = position["position"].apply(lambda x: np.asarray(x) / 100)
    position["x"] = position["position"].apply(lambda x: x[0])
    position["y"] = position["position"].apply(lambda x: x[1])
    position["z"] = position["position"].apply(lambda x: x[2])
    # Convert Y axis rotation to heading in degrees.
    position["heading"] = position["heading"].apply(lambda x: np.asarray(x[1]) / 1000)
    return position

def get_position_per_frame(position, frames):
    """Averages position information per frame and gimbl actor to give single position entry per frame.
    Frames without position info are forward-filled in.
    
    Arguments:
        position {dataframe}    -- Parsed position info.
        frames {dataframe}         -- Parsed frame info.
    
    Returns:
        [dataframe] -- Averages position information per frame and gimbl actor <num_microscope_frames> x 6 columns.
                index: 
                    "frame" -- microscope frame.
                    "name"  -- name of gimbl actor.
                columns:
                    "time"      -- time since start in ms.                
                    "heading"   -- heading of animal in degrees.
                    "x","y","z" -- position value in cms.                    
                    "position"  -- position value on path.          
    """
    frame_position = pd.DataFrame(columns=["frame", "name", "time", "heading", "x", "y", "z", "position"]).set_index(["frame", "name"])
    # remove names with none positions.
    if not frames.empty and not position.empty:
        frame_position = assign_frame_info(position, frames)
        # average per values per frame and actor.
        frame_position["time"] = frame_position["time"].dt.total_seconds()
        # groupby
        frame_position = frame_position.groupby(["frame", "name"], observed=True).mean()
        frame_position["time"] = pd.to_timedelta(frame_position["time"], unit="seconds")
        frame_position["position"] = frame_position[["x", "y", "z"]].to_numpy().tolist()
        # fill in frames with missing values.
        frame_position = ffill_missing_frame_info(frame_position, frames)
        frame_position = frame_position.ffill(axis=0)
        frame_position = frame_position[["time", "heading", "x", "y", "z", "position"]]
    return frame_position

def parse_path(df):
    """Parse path position information from gimbl log.
    
    Arguments:
        df {dataframe} -- complete dataframe from gimbl log
    
    Returns:
        dataframe -- parsed position on path information <num_monitor_frames> x 3 columns.
                index: 
                    "index" -- line in json file.
                    "name"  -- name of gimbl actor.
                columns:
                    "time"      -- time since start in ms.
                    "path"      -- path name in gimbl.
                    "position"  -- position value on path.    
    """
    path = parse_custom_msg(df, "Path Position", ["name", "pathName", "position"], rename_columns={"pathName": "path"})
    # set name as category with only available actors.
    path["name"] = path["name"].astype("string")
    cat_type = pd.CategoricalDtype(categories=path.name.unique(), ordered=False)
    path["name"] = path["name"].astype(cat_type)
    # structure.
    path = path.reset_index().set_index(["index", "name"]).drop(columns="frame")
    path["position"] = path["position"].apply(lambda x: x / 100)    # path to cms.
    return path

def get_path_position_per_frame(df, frames):
    """Position information per frame and gimbl actor to give single position entry per frame.
    To avoid unexpected values in looping paths, the first position value within a frame is used.
    Frames without position info are forward-filled in.
    
    Arguments:
        position {dataframe} -- Parsed position info.
        frames {[type]} -- Parsed frame info.
    
    Returns:
        [dataframe] -- Averages position information per frame and gimbl actor.<num_microscope_frames> x 3 columns
                index: 
                    "frame" -- microscope frame.
                    "name"  -- name of gimbl actor.
                columns:
                    "time"      -- time since start in ms.
                    "name"      -- path name in gimbl.
                    "position"  -- position value on path.            
    """
    path = parse_custom_msg(df, "Path Position", ["name", "pathName", "position"], rename_columns={"pathName": "path"}, remove_nan=True, frames=frames)
    if not path.empty:
        # set name as category with only available actors.
        path["name"] = path["name"].astype("string")
        cat_type = pd.CategoricalDtype(categories=path.name.unique(), ordered=False)
        path["name"] = path["name"].astype(cat_type)
        path["position"] = path["position"].apply(lambda x: x / 100)    # path to cms.
        # first value per frame and actor.
        path = path.groupby(["frame", "name", "path"], observed=True).first()
        path = path.reset_index().drop_duplicates(subset=["frame", "name"], keep="first")
        path = path.reset_index().set_index(["frame", "name"]).drop(columns="index")
        # fill in frames with missing values.
        path = ffill_missing_frame_info(path, frames)
        if path.empty == False:  # became an issue in pandas 1.1.2
            path = path.bfill(axis=0)
        path = path[["time", "path", "position"]]
    else:
        path = pd.DataFrame(columns=["time", "path", "position"])
    return path

def parse_camera(df, frames):
    """Parses camera frame information and aligns it to microscope frames.
    
    Arguments:
        df {dataframe} -- complete dataframe from gimbl log
        frames {dataframe} -- parsed frame info.
    
    Returns:
        [dataframe] -- parsed camera info <num_camera_frames> x 3 columns.
                index: 
                    "cam_frame" -- image frame from camera.
                    "id"        -- camera id.
                columns:
                    "frame"     -- microscope imaging frame.
                    "time"      -- time since start in ms.
         
    """
    msg = "Camera Frame"
    fields = ["id"]
    camera = parse_custom_msg(df, msg, fields, frames=frames, msg_field="data.msg.event", data_field="data.msg")
    camera = camera.rename_axis("cam_frame")
    camera["id"] = camera["id"].astype("int8")
    camera = camera.reset_index().set_index(["cam_frame", "id"])
    return camera

def parse_reward(df, frames):
    """Parse reward information from gimbl log
    
    Arguments:
        df {dataframe}      -- complete dataframe from gimbl log
        frames {dataframe}  -- parsed frame info.
    
    Returns:
        dataframe -- information on reward delivery
                index:
                    "index          -- reward number
                columns:
                    "time"          -- time since start in ms.
                    "frame"         -- microscope imaging frame.
                    "type"          -- category reward type (task/manual)
                    "amount"        -- amount of water in ul.
                    "valve_time"    -- open duration of valve in ms.
                    "sound_on"      -- if sound was played (bool)
                    "sound_freq"    -- frequeny of sound in hz (if played)
                    "sound_duration"-- duration of sound (in seconds; if played)
    """
    msg = "Reward Delivery"
    fields = ["type", "amount", "valveTime", "withSound", "frequency", "duration"]
    rename = {"valveTime": "valve_time", "withSound": "sound_on", "frequency": "sound_freq", "duration": "sound_duration"}
    reward = parse_custom_msg(df, msg, fields, rename_columns=rename, frames=frames,
        msg_field="data.msg.action", data_field="data.msg")
    reward["type"] = reward["type"].astype("category")
    return reward

def parse_idle_sound(df, frames):
    """Parse idle sound information from gimbl log
    
    Arguments:
        df {dataframe}      -- complete dataframe from gimbl log
        frames {dataframe}  -- parsed frame info.
    
    Returns:
        dataframe -- information on played idle sounds.
                index:          -- idle sound number
                columns:
                    "time"      -- time since start in ms.
                    "frame"     -- microscope imaging frame.
                    "type"      -- cause of sound to play (category:task/manual)
                    "duration"  -- duration of tone in seconds
                    "sound"     -- sound played (category:White Noise)
    """
    idle = parse_custom_msg(df, "Idle Sound", ["type", "duration", "sound"], frames=frames,
        msg_field="data.msg.action", data_field="data.msg")
    idle = idle.astype({"type": "category", "sound": "category"})
    return idle

def parse_session_info(df):
    """Parse session info in gimbl log.
    
    Arguments:
        df {dataframe} -- complete dataframe from gimbl log
    
    Returns:
        [dictionary] -- Read session information
                    keys:
                        "date_time"     -- YYYY-MM-DD HH:MM:SS date time string.
                        "project"       -- Unity project name.
                        "scene"         -- Unity scene name.
    """
    info = parse_custom_msg(df, "Info", ["time", "project", "scene"], rename_columns={"time": "date_time"}).drop(columns="frame")
    # this is kind of anoying because of two fields called time..
    if not info.empty:
        info = info.to_numpy().transpose()
        info = {"date_time": info[1].item(), "project": info[2].item(), "scene": info[3].item()}
    else:
        info = {"date_time": None, "project": None, "scene": None}
    return info

def parse_spherical_settings(df, frames):
    """Parse spherical controller settings from gimbl log.
        Changes to settings during session are shown as new entries.
    
    Arguments:
        df {dataframe}      -- complete dataframe from gimbl log
        frames {dataframe}  -- parsed frame info.
    
    Returns:
        dataframe -- information of linear controller settings
                index:
                    "index"         -- number of occurence changed setting this controller.
                    "name"          -- controller name.
                columns:
                    "time"          -- time since start in ms.
                    and spherical settings fields from Gimbl.
    """
    msg = "Spherical Controller Settings"
    fields = ["name", "isActive", "loopPath",
            "gain.forward", "gain.backward",
            "gain.strafeLeft", "gain.strafeRight",
            "gain.turnLeft", "gain.turnRight",
            "trajectory.maxRotPerSec", "trajectory.angleOffsetBias",
            "trajectory.minSpeed", "inputSmooth"]
    rename = {"isActive": "is_active",
    "gain.forward": "gain_forward", "gain.backward": "gain_backward",
    "gain.strafeLeft": "gain_strafe_left", "gain.strafeRight": "gain_strafe_right",
    "gain.turnLeft": "gain_turn_left", "gain.turnRight": "gain_turn_right",
    "trajectory.maxRotPerSec": "trajectory_max_rot_per_sec",
    "trajectory.angleOffsetBias": "trajectory_angle_offset_bias",
    "trajectory.minSpeed": "trajectory_min_speed",
    "inputSmooth": "input_smooth", "loopPath": "is_looping"}
    settings = parse_custom_msg(df, msg, fields, frames=frames, rename_columns=rename)
    settings["index"] = settings.groupby("name", observed=True).cumcount()
    settings = settings.set_index(["index", "name"])
    return settings

def parse_spherical_data(df):
    """Parses raw data from spherical controllers
    
    Arguments:
        df {dataframe} -- complete dataframe from gimbl log
    
    Returns:
        [dataframe] -- movement data from linear controllers [num_monitor_frames] x 2 columns
                index:
                    "index"         -- line number in gimbl log.
                    "name"          -- controller name.
                collumns:
                    "time"          -- time since start in ms.
                    "roll"          -- roll arc length in cm.
                    "yaw"           -- yaw in degrees.
                    "pitch"         -- pitch arc length in cm.
    """
    data = parse_custom_msg(df, "Spherical Controller", ["name", "roll", "yaw", "pitch"])
    data = data.reset_index().set_index(["index", "name"]).drop(columns="frame")
    data["roll"] = data["roll"] / 100 # convert to cm.
    data["pitch"] = data["pitch"] / 100
    data["yaw"] = data["yaw"] / 1000 # to degrees.
    return data

def parse_linear_settings(df, frames):
    """Parse linear controller settings from gimbl log.
        Changes to settings during session are shown as new entries.
    
    Arguments:
        df {dataframe}      -- complete dataframe from gimbl log
        frames {dataframe}  -- parsed frame info.
    
    Returns:
        dataframe -- information of linear controller settings
                index:
                    "index"         -- number of occurence changed setting this controller.
                    "name"          -- controller name.
                columns:
                    "time"          -- time since start in ms.
                    "frame"         -- microscope imaging frame (-1: before start microscope).
                    "is_active"     -- bool: is controller active
                    "is_looping"    -- bool: is controller looping on path.
                    "gain_forward   -- forward gain
                    "gain_backward" -- backward gain setting.
                    "input_smooth"  -- size input smoothing buffer (in ms.)
    """
    msg = "Linear Controller Settings"
    fields = ["name", "isActive", "loopPath", "gain.forward", "gain.backward", "inputSmooth"]
    rename = {"isActive": "is_active", "loopPath": "is_looping",
    "gain.forward": "gain_forward",
    "gain.backward": "gain_backward", "inputSmooth": "input_smooth"}
    settings = parse_custom_msg(df, msg, fields, frames=frames, rename_columns=rename)
    settings["index"] = settings.groupby("name", observed=True).cumcount()
    settings = settings.set_index(["index", "name"])
    return settings

def parse_linear_data(df):
    """Parses raw data from linear controllers
    
    Arguments:
        df {dataframe} -- complete dataframe from gimbl log
    
    Returns:
        [dataframe] -- movement data from linear controllers [num_monitor_frames] x 2 columns
                index:
                    "index"         -- line number in gimbl log.
                    "name"          -- controller name.
                collumns:
                    "time"          -- time since start in ms.
                    "move"          -- movement linear controller in cm.
    """
    data = parse_custom_msg(df, "Linear Controller", ["name", "move"])
    data = data.reset_index().set_index(["index", "name"]).drop(columns="frame")
    data["move"] = data["move"] / 100 # convert to cm.
    return data

def get_linear_data_per_frame(df, frames):
    """Sums movement from linear controller that falls within same frame.
    Returns dataframe with same number of rows as there are microscope frames. 
    Frames without data are set to None.
    
    Arguments:
        data {dataframe}            -- parsed linear controller data
        frames {dataframe}          -- parsed frame info.
    
    Returns:
        dataframe -- Linear movement data from controllers. [num_frames] x 2 columns
                index:
                    "frame"         -- microscope frame number.
                    "name"          -- controller name.
                columns:
                    "time"          -- - time since start in ms.
                    "move"          -- linear movement in cm

    """
    data = parse_custom_msg(df, "Linear Controller", ["name", "move"], frames=frames, remove_nan=True)
    data = data.reset_index().set_index(["index", "name"])
    if not data.empty:
        data["move"] = data["move"] / 100 # convert to cm.
        data = data.groupby(["frame", "name"]).agg({"move": "sum", "time": "first"})
        data = ffill_missing_frame_info(data, frames, nan_fill=True, subset_columns=["move"])
        data = ffill_missing_frame_info(data, frames, nan_fill=False, subset_columns=["move"])
    data = data[["time", "move"]]
    return data

def get_spherical_data_per_frame(df, frames):
    """Sums movement from spherical controller that falls within same frame.
    Returns dataframe with same number of rows as there are microscope frames. 
    Frames without data are set to None.
    
    Arguments:
        data {dataframe}            -- parsed linear controller data
        frames {dataframe}          -- parsed frame info.
    
    Returns:
        dataframe -- SPherical movement data from controllers. [num_frames] x 4 columns
                index:
                    "frame"         -- microscope frame number.
                    "name"          -- controller name.
                columns:
                    "time"          -- - time since start in ms.
                    "roll"          -- roll arc movement in cm
                    "yaw"           -- yaw movement in degrees
                    "pitch"         -- pitch arc movement in cm

    """
    data = parse_custom_msg(df, "Spherical Controller", ["name", "roll", "yaw", "pitch"], frames=frames, remove_nan=True)
    data = data.reset_index().set_index(["index", "name"])
    if not data.empty:
        data["roll"] = data["roll"] / 100 # convert to cms.
        data["yaw"] = data["yaw"] / 1000 # convert to degrees.
        data["pitch"] = data["pitch"] / 100 # convert to cms.
        data = data.groupby(["frame", "name"], observed=True).agg({"roll": "sum", "yaw": "sum", "pitch": "sum", "time": "first"})
        data = ffill_missing_frame_info(data, frames, nan_fill=True, subset_columns=["roll", "yaw", "pitch"])
        data = ffill_missing_frame_info(data, frames, nan_fill=False, subset_columns=["roll", "yaw", "pitch"])
    data = data[["time", "roll", "yaw", "pitch"]]
    return data
