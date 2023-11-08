from suite2p.io import compute_dydx, BinaryFileCombined
from pathlib import Path
import numpy as np
def memory_usage(df,verbose=True):
    """returns memory usage of dataframe or series
    
    Arguments:
        df {dataframe or series}    -- pandas object fo interest

    Keyword Arguments:
        verbose {bool}              -- print result (default: {True})            
    
    Returns:
        float -- Size in memory in mb.
    """
    if df.ndim>1:
        value = (round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2))
    else:
        value = (round(df.memory_usage(deep=True) / 1024 ** 2, 2))
    if verbose:
        print('Memory used:', value, 'Mb')
    return value

def read_raw_frames(bin_folder,frames):
    """Read raw imaging frames from binary

    Args:
        bin_folder (string): suite2p folder containing imaging plane folders
        frames (numpy array): frames to read (e.g. np.arange(0,100))

    Raises:
        NameError: Could not find plane folders

    Returns:
        [numpy array]: Read imaging frame data (shape: number of frames x Lx x Ly)
    """
    bin_folder = Path(bin_folder)
    # look for plane folders.
    plane_folders = list(bin_folder.glob('plane*/'))
    if not plane_folders:
        raise NameError(f"Could not find plane folders in {bin_folder}")
    # get ops info
    ops1 = [np.load(f/'ops.npy', allow_pickle=True).item() for f in plane_folders]
    # all the registered binaries
    reg_loc = [pdir/'data.bin' for pdir in plane_folders]
    # plane/ROI positions
    dy, dx = compute_dydx(ops1)

    # plane/ROI sizes
    Ly = np.array([ops['Ly'] for ops in ops1])
    Lx = np.array([ops['Lx'] for ops in ops1])
    LY = int(np.amax(dy + Ly))
    LX = int(np.amax(dx + Lx))

    with BinaryFileCombined(LY, LX, Ly, Lx, dy, dx, reg_loc) as f:   
        return f[frames]