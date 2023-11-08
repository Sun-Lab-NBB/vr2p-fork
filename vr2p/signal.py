from scipy.ndimage import filters
import scipy
import numpy as np
import pandas as pd
from numba import jit
from suite2p.extraction import preprocess

def demix_traces(F, Fneu, cell_masks, ops):
    """Demix activity from overlaping cells

    Args:
        F (numpy array): Raw fluoresence activity (size: num cells x num frames)
        Fneu ([type]): Raw neuropil activity
        cell_masks (list of dictionaries): description of cell masks (size: num cells)
                must contains keys "xpix", "ypix", "lam", and "overlap"
        ops (dictionary): Parameters for demixing (must contain "baseline",'"win_baseline", "sig_baseline", and "fs")
        l2_reg (float, optional): L2 regularization factor. Defaults to 0.01.

    Returns:
        [type]: [description]
    """
    # subtract neuropil signal and subtract baseline.
    Fcorr = F - ops['neucoeff']*Fneu
    Fbase = preprocess(Fcorr, ops['baseline'], ops['win_baseline'],
                       ops['sig_baseline'], ops['fs']) # baseline subtracted signal.
    #Collect mask information.
    num_cells = len(cell_masks) 
    Ly, Lx = ops['Ly'], ops['Lx']
    lammap = np.zeros((num_cells, Ly, Lx), np.float32) # weight mask for each mask
    Umap = np.zeros((num_cells, Ly, Lx), bool) # binarized weight masks
    covU = np.zeros((num_cells,num_cells), np.float32) # holds covariance matrix.
    for ni,mask in enumerate(cell_masks):
        ypix, xpix, lam = mask['ypix'], mask['xpix'], mask['lam']
        norm = lam.sum()
        Fbase[ni] *= norm
        lammap[ni,ypix,xpix] = lam
        Umap[ni,ypix,xpix] = True
        covU[ni,ni] = (lam**2).sum()
    #Create covariance matrix of the masks.
    for ni,mask in enumerate(cell_masks):
        if mask['overlap'].sum() > 0:
            ioverlap = mask['overlap']
            yp, xp, lam = mask['ypix'][ioverlap], mask['xpix'][ioverlap], mask['lam'][ioverlap]
            njs, ijs = np.nonzero(Umap[:, yp, xp])
            for nj in np.unique(njs):
                if nj!=ni:
                    inds = ijs[njs==nj]
                    covU[ni, nj] = (lammap[nj, yp[inds], xp[inds]] * lam[inds]).sum() #  each entry i,j is the sum of (weights in mask_i * weights in mask_j that overlap). this is an overlap score matrix in a sense
    #Solve for demixed traces of the cells. 
    #the equation we're solving is the movie M is a multiplication of the masks with the fluorescence V: M = U @ V.T . 
    #We have  U @ M.T = Fbase , and the solution for V with linear regression is np.linalg.solve(U @ U.T, U @ M.T) 
    #so we plug in for U @ M.T with Fbase and get the final equation
    l2 = np.diag(covU).mean() * ops['l2_reg']
    Fdemixed = np.linalg.solve(covU + l2*np.eye(num_cells), Fbase) 

    return Fdemixed, Fbase, covU, lammap

def bin_fluorescence_data(F, data, edges, method='mean',threshold=0):
    """bin fluorescene data according to some value by averaging.
    
    Arguments:
        F {numpy array}             -- Fluorescence data (size: num_cells x num_frames)
        data {dataseries}           -- Values by which to bin on (number rows must equal number of frame in F)
        edges {numpy array}         -- Bin edges according to pandas.cut
    
    Returns:
        numpy array                 -- binned fluorescent data (size: num_cells x num_bins)
        numpy array                 -- number of samples in each bin.
    """
    # return bin label (from 0 to edges.size-1) for each entry (nan if outside of edges).
    bins = pd.cut(data,edges, include_lowest=True,labels=False).to_numpy()
    # I do simple list comprehension here because I also want the values for missing bins.
    uni_bin_ids = np.arange(0,edges.size-1)
    count = np.array([sum(bins==cbin) for cbin in uni_bin_ids])
    if method=='mean':
        F = np.array([F[:,bins==cbin].mean(axis=1) if sum(bins==cbin)!=0 else np.full((F.shape[0]),np.nan) for cbin in uni_bin_ids]).T
    if method=='sum':
        F = np.array([F[:,bins==cbin].sum(axis=1) if sum(bins==cbin)!=0 else np.full((F.shape[0]),np.nan) for cbin in uni_bin_ids]).T
    if method == 'threshold sum':
        F = np.array([np.sum(F[:,bins==cbin]>threshold,axis=1) if sum(bins==cbin)!=0 else np.full((F.shape[0]),np.nan) for cbin in uni_bin_ids]).T
    return F, count

def df_over_f0(F, method_baseline = "maximin", subtract_min = False, **kwargs):
    """Calculate df_over_f0

    Arguments:
        F {numpy array}         -- Fluorescence data (num_cells x frames)

    Keyword Arguments:
        method_baseline {str}   -- method used to calculate the baseline (f0) (default: {"maximin"})
        subtract_min {bool}     -- Whether to subtract min. value from curve (avoid negative values)
        **kwargs                -- Optional parameters for baseline calculation.

    Returns:
        numpy array -- calculate df over f0.
    """
    if subtract_min:
        F = F-np.min(F,axis=1)[..., np.newaxis]
    f0 = baseline(F,method = method_baseline, **kwargs)
    dF = F-f0
    dF = np.divide(dF,f0)
    return dF,f0

def fold_change(F, method_baseline = "maximin", **kwargs):
    f0 = baseline(F,method = method_baseline, **kwargs)
    dF = np.divide(dF,f0)
    return dF,f0

def baseline(F, method="maximin",sigma_baseline=20, window_size = 600):
    """calculate the baseline of fluorescence data.

    Arguments:
        F {numpy array}         -- Fluorescence data (num_cells x frames)

    Keyword Arguments:
        method {str}            -- method used to calculate the baseline (default: {"maximin"})

        'maximin' options.
        sigma_baseline {int}    -- sigma of gaussian filter (default: {20})
        window_size {int}       -- window_size along which to calculate min/max (default: {600})

    Returns:
        numpy array             -- calculated baseline.
    """
    if method=="maximin":
        if (F.ndim)==2:
            Flow = filters.gaussian_filter(F,    [0., sigma_baseline])
        if (F.ndim)==1:
            Flow = filters.gaussian_filter(F,    [sigma_baseline])
        Flow = filters.minimum_filter1d(Flow,    window_size)
        Flow = filters.maximum_filter1d(Flow,    window_size)
    if method=='average':
        Flow = np.transpose(np.tile(np.mean(F,axis=1),(F.shape[1],1)))
    return Flow

def quantile_max_treshold(F, base_quantile = 0.25, threshold_factor = 0.25):
    """Thresholds fluorescence data with threshold based on percentage difference
    between base_quantile and max value.
    Threshold = base_quantile + ((max_value-base_quantile)*threshold_factor)

    Arguments:
        F {numpy array}             -- Fluorescence data (num_cells x frames)

    Keyword Arguments:
        base_quantile {float}       -- quantile value thats used for the baseline (default: {0.25})
        threshold_factor {float}    -- percentage of difference between max and baseline (default: {0.25})

    Returns:
        numpy_array (boolean)       -- boolean numpy array same size as F.
    """
    F = F.copy() # F gets adjusted later on (nan to -inf).
    # get max and baseline value (max peak and quantile value)
    max_val = np.nanmax(F,axis=1)
    quantile_val = np.nanquantile(F,base_quantile,axis=1)
    # get mean of lowest quantile bins.
    base_val = []
    for i in range(F.shape[0]):
        temp = F[i,~np.isnan(F[i,:])]
        base_val.append(np.nanmean(temp[temp<=quantile_val[i]]))
    # set threshold as percentage difference between baseline and max value.
    threshold = (base_val + ((max_val-base_val)*threshold_factor))[:,np.newaxis]
    threshold = np.tile(threshold,[1,F.shape[1]])
    F[np.isnan(F)]=-np.inf
    return F>threshold # nan_to_num prevents warning.

def find_calcium_events(dF, bin_size = 50, base_quantile = 0.5, onset_factor = 3, end_factor = 0.5):
    """Find signficant calcium events

    Fluorescence transients are identified as events that start when fluorescence
    deviated [onset_factor]σ from the corrected baseline, and ended when it returned to within [end_factor]σ of baseline.

    The baseline σ is calculated by binning the fluorescent trace in short periods of size [bin_size]
    and calculating the std. in each bin. Selected std. is based on the [base_quantile] percentile.

    Based on Tank 2010

    Args:
        dF ([type]): Fluorescence data (num_cells x frames). Must be centered around zero.
        bin_size (int, optional): Number of frames in each bin. Defaults to 50.
        base_quantile (float, optional): Quantile value of bins used to determine std.. Defaults to 0.5.
        onset_factor (int, optional): threshold deviation factor * std from baseline that determines onset of event. Defaults to 3.
        end_factor (float, optional): threshold deviation factor * std from baseline that determines end of event. Defaults to 0.5.

    Returns:
        [type]: [description]
    """
    # reshape traces into equal size bins.
    num_bins = int(np.floor(dF.shape[1]/bin_size))
    bin_dF = dF[:,0:(num_bins*bin_size)]
    bin_dF = np.reshape(bin_dF,[dF.shape[0],-1,bin_size])
    # calculate std. per bin.
    std_bin = np.std(bin_dF,axis=2)
    # get quantile.
    std_quant = np.nanquantile(std_bin,base_quantile,axis=1)
    event_mask = np.zeros(dF.shape,np.bool)
    # get start and stop of events.
    for icell in range(dF.shape[0]):
        onset_mask = scipy.signal.convolve(dF[icell,:]>=(std_quant[icell]*onset_factor),[1,-1],'same')==1
        end_mask = scipy.signal.convolve(dF[icell,:]<=(std_quant[icell]*end_factor),[1,-1],'same')==1
        event_mask = match_start_end(onset_mask,end_mask,event_mask,icell,dF.shape[1])
    return event_mask,std_quant

@jit(nopython=True)
def np_apply_along_axis(func1d, axis, arr):
  assert arr.ndim == 2
  assert axis in [0, 1]
  if axis == 0:
    result = np.empty(arr.shape[1])
    for i in range(len(result)):
      result[i] = func1d(arr[:, i])
  else:
    result = np.empty(arr.shape[0])
    for i in range(len(result)):
      result[i] = func1d(arr[i, :])
  return result

@jit(nopython=True)
def np_min(array, axis):
  return np_apply_along_axis(np.min, axis, array)

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def match_start_end(onset_mask,end_mask,event_mask,icell,num_frames):
    # get indices.
    onset_ind = np.argwhere(onset_mask)
    end_ind = np.argwhere(end_mask)
    # find minimum positive offset
    offset = np.transpose(end_ind)-onset_ind
    # remove smaller then 0
    for i in range(offset.shape[0]):
        for j in range(offset.shape[1]):
            if offset[i,j]<0:
                offset[i,j]=9999999
    matches = np_min(offset,axis=1)
    onset_ind = onset_ind.flatten()
    end = onset_ind+matches
    #if none then event lasts till end.
    end[end==9999999] = num_frames
    for i, start_ind in enumerate(onset_ind):
        event_mask[icell,start_ind:end[i]]=True
    return event_mask

class ChunkShuffler:
    def __init__(self,F, num_bins):
        step_size = np.ceil(F.shape[1]/num_bins)
        self._num_bins = num_bins
        self._F_split = np.split(F,
                np.arange(step_size,F.shape[1],step_size,np.uint16),axis=1)
    def shuffle(self):
        shuffle_ind = np.random.choice(np.arange(self._num_bins),self._num_bins,replace=False)
        return np.concatenate([self._F_split[i] for i in shuffle_ind],axis=1)
