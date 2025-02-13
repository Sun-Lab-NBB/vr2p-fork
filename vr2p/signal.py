from typing import Any
from numba import jit

import numpy as np
import scipy
import pandas as pd
from numpy.typing import NDArray
from scipy.ndimage import filters
from suite2p.extraction import preprocess


def demix_traces(raw_fluorescence: NDArray[Any], neuropil_fluorescence, cell_masks: list[dict[str, Any]],
                 ops: dict[str, Any]) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Demixes activity from overlapping cells using linear regression.

    Performs neuropil subtraction, baseline correction, and solves a linear system to separate overlapping cellular
    signals.

    Args:
        raw_fluorescence: Raw cell fluorescence activity matrix (n_cells × n_frames)
        neuropil_fluorescence: Raw neuropil fluorescence activity matrix (n_cells × n_frames)
        cell_masks: List of dictionaries containing cell mask properties:
            - xpix: X coordinates of mask pixels.
            - ypix: Y coordinates of mask pixels.
            - lam: Weights for each pixel.
            - overlap: Boolean mask indicating overlapping regions.
        ops: Dictionary of demixing parameters:
            - neucoeff: Neuropil subtraction coefficient.
            - baseline: Baseline calculation method.
            - win_baseline: Window size for baseline.
            - sig_baseline: Gaussian filter sigma for baseline.
            - fs: Sampling frequency.
            - Ly, Lx: Frame dimensions (height, width).
            - l2_reg: L2 regularization coefficient.

    Returns: A tuple containing 4 numpy arrays:
        - Demixed fluorescence traces (n_cells × n_frames).
        - Baseline-corrected traces (n_cells × n_frames).
        - Covariance matrix of cell masks (n_cells × n_cells).
        - Spatial weight maps for each cell (n_cells × Ly × Lx).
    """
    # Subtracts neuropil signal and performs baseline correction. This removes the neuropil contamination from the
    # signal and establishes df/f baseline.
    neuropil_subtracted = raw_fluorescence - ops["neucoeff"] * neuropil_fluorescence
    baseline_corrected = preprocess(
        neuropil_subtracted,
        ops["baseline"],
        ops["win_baseline"],
        ops["sig_baseline"],
        ops["fs"]
    )

    # Initializes arrays for mask processing.
    #   - weight_maps: stores intensity weights for each cell's pixels.
    #   - binary_masks: marks which pixels belong to each cell (for quick lookup).
    #   - mask_overlaps: tracks how much each cell overlaps with itself and others.
    num_cells = len(cell_masks)
    height, width = ops["Ly"], ops["Lx"]
    weight_maps = np.zeros((num_cells, height, width), dtype=np.float32)
    binary_masks = np.zeros((num_cells, height, width), dtype=bool)
    mask_overlaps = np.zeros((num_cells, num_cells), dtype=np.float32)

    # Processes each cell mask and computes diagonal elements of covariance matrix (self-overlap scores).
    # For each cell:
    # 1. Get its pixel coordinates and weights
    # 2. Scale its fluorescence by total weight
    # 3. Store its spatial mask information
    # 4. Calculate how much it overlaps with itself (diagonal of mask_overlaps)
    for cell, mask in enumerate(cell_masks):
        y_pixels, x_pixels, weights = mask["ypix"], mask["xpix"], mask["lam"]
        total_weight = np.sum(weights)
        baseline_corrected[cell] *= total_weight
        weight_maps[cell, y_pixels, x_pixels] = weights
        binary_masks[cell, y_pixels, x_pixels] = True
        mask_overlaps[cell, cell] = np.sum(weights ** 2)

    # Computes overlap scores between different cells.
    # For each cell that has overlaps:
    # 1. Extract coordinates and weights for overlapping pixels
    # 2. Find all other cells that share these pixels using binary masks
    # 3. For each overlapping neighbor:
    #    - Find shared pixels between the two cells
    #    - Calculate overlap score as sum of (weight_cell1 * weight_cell2) for shared pixels
    #    - Store score in mask_overlaps matrix
    for cell, mask in enumerate(cell_masks):
        if np.sum(mask["overlap"]) > 0:
            overlap_indices = mask["overlap"]
            y_pixels = mask["ypix"][overlap_indices]
            x_pixels = mask["xpix"][overlap_indices]
            weights = mask["lam"][overlap_indices]

            # Finds all cells that overlap at these pixels
            neighbor_cells, pixel_indices = np.nonzero(binary_masks[:, y_pixels, x_pixels])

            # Calculates overlap score with each neighboring cell
            unique_neighbors = np.unique(neighbor_cells)
            for neighbor in unique_neighbors[unique_neighbors != cell]:
                overlap_pixels = pixel_indices[neighbor_cells == neighbor]
                neighbor_weights = weight_maps[neighbor, y_pixels[overlap_pixels], x_pixels[overlap_pixels]]
                # Each entry [cell_number, neighbor] is the sum of (weights in mask_cell * weights in mask_neighbor
                # that overlap). This is an overlap score matrix in a sense.
                mask_overlaps[cell, neighbor] = np.sum(neighbor_weights * weights[overlap_pixels])

    # Solves system of linear equations to separate (demix) overlapping cell signals.
    # Process:
    # 1. Movie data M can be represented as: M = U @ V.T
    #    where U = cell masks, V = true cellular activity
    # 2. Our known quantities are:
    #    - mask_overlaps = U @ U.T (how much cells overlap)
    #    - baseline_corrected = U @ M.T (measured signals)
    # 3. Add L2 regularization to stabilize solution for overlapping cells
    regularization = np.mean(np.diag(mask_overlaps)) * ops["l2_reg"]
    demixed_traces = np.linalg.solve(
        mask_overlaps + regularization * np.eye(num_cells),  # Left side: U @ U.T + regularization
        baseline_corrected  # Right side: U @ M.T
    )  # Solves the linear system to obtain demixed traces (V)

    return demixed_traces, baseline_corrected, mask_overlaps, weight_maps


def bin_fluorescence_data(F, data, edges, method="mean", threshold=0):
    """Bin fluorescene data according to some value by averaging.
    
    Arguments:
        F {numpy array}             -- Fluorescence data (size: num_cells x num_frames)
        data {dataseries}           -- Values by which to bin on (number rows must equal number of frame in F)
        edges {numpy array}         -- Bin edges according to pandas.cut
    
    Returns:
        numpy array                 -- binned fluorescent data (size: num_cells x num_bins)
        numpy array                 -- number of samples in each bin.
    """
    # return bin label (from 0 to edges.size-1) for each entry (nan if outside of edges).
    bins = pd.cut(data, edges, include_lowest=True, labels=False).to_numpy()
    # I do simple list comprehension here because I also want the values for missing bins.
    uni_bin_ids = np.arange(0, edges.size - 1)
    count = np.array([sum(bins == cbin) for cbin in uni_bin_ids])
    if method == "mean":
        F = np.array(
            [F[:, bins == cbin].mean(axis=1) if sum(bins == cbin) != 0 else np.full((F.shape[0]), np.nan) for cbin in
             uni_bin_ids]).T
    if method == "sum":
        F = np.array(
            [F[:, bins == cbin].sum(axis=1) if sum(bins == cbin) != 0 else np.full((F.shape[0]), np.nan) for cbin in
             uni_bin_ids]).T
    if method == "threshold sum":
        F = np.array(
            [np.sum(F[:, bins == cbin] > threshold, axis=1) if sum(bins == cbin) != 0 else np.full((F.shape[0]), np.nan)
             for cbin in uni_bin_ids]).T
    return F, count


def df_over_f0(F, method_baseline="maximin", subtract_min=False, **kwargs):
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
        F = F - np.min(F, axis=1)[..., np.newaxis]
    f0 = baseline(F, method=method_baseline, **kwargs)
    dF = F - f0
    dF = np.divide(dF, f0)
    return dF, f0


def fold_change(F, method_baseline="maximin", **kwargs):
    f0 = baseline(F, method=method_baseline, **kwargs)
    dF = np.divide(dF, f0)
    return dF, f0


def baseline(F, method="maximin", sigma_baseline=20, window_size=600):
    """Calculate the baseline of fluorescence data.

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
    if method == "maximin":
        if (F.ndim) == 2:
            Flow = filters.gaussian_filter(F, [0., sigma_baseline])
        if (F.ndim) == 1:
            Flow = filters.gaussian_filter(F, [sigma_baseline])
        Flow = filters.minimum_filter1d(Flow, window_size)
        Flow = filters.maximum_filter1d(Flow, window_size)
    if method == "average":
        Flow = np.transpose(np.tile(np.mean(F, axis=1), (F.shape[1], 1)))
    return Flow


def quantile_max_treshold(F, base_quantile=0.25, threshold_factor=0.25):
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
    F = F.copy()  # F gets adjusted later on (nan to -inf).
    # get max and baseline value (max peak and quantile value)
    max_val = np.nanmax(F, axis=1)
    quantile_val = np.nanquantile(F, base_quantile, axis=1)
    # get mean of lowest quantile bins.
    base_val = []
    for i in range(F.shape[0]):
        temp = F[i, ~np.isnan(F[i, :])]
        base_val.append(np.nanmean(temp[temp <= quantile_val[i]]))
    # set threshold as percentage difference between baseline and max value.
    threshold = (base_val + ((max_val - base_val) * threshold_factor))[:, np.newaxis]
    threshold = np.tile(threshold, [1, F.shape[1]])
    F[np.isnan(F)] = -np.inf
    return threshold < F  # nan_to_num prevents warning.


def find_calcium_events(dF, bin_size=50, base_quantile=0.5, onset_factor=3, end_factor=0.5):
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
    num_bins = int(np.floor(dF.shape[1] / bin_size))
    bin_dF = dF[:, 0:(num_bins * bin_size)]
    bin_dF = np.reshape(bin_dF, [dF.shape[0], -1, bin_size])
    # calculate std. per bin.
    std_bin = np.std(bin_dF, axis=2)
    # get quantile.
    std_quant = np.nanquantile(std_bin, base_quantile, axis=1)
    event_mask = np.zeros(dF.shape, np.bool)
    # get start and stop of events.
    for icell in range(dF.shape[0]):
        onset_mask = scipy.signal.convolve(dF[icell, :] >= (std_quant[icell] * onset_factor), [1, -1], "same") == 1
        end_mask = scipy.signal.convolve(dF[icell, :] <= (std_quant[icell] * end_factor), [1, -1], "same") == 1
        event_mask = match_start_end(onset_mask, end_mask, event_mask, icell, dF.shape[1])
    return event_mask, std_quant


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


@jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def match_start_end(onset_mask, end_mask, event_mask, icell, num_frames):
    # get indices.
    onset_ind = np.argwhere(onset_mask)
    end_ind = np.argwhere(end_mask)
    # find minimum positive offset
    offset = np.transpose(end_ind) - onset_ind
    # remove smaller then 0
    for i in range(offset.shape[0]):
        for j in range(offset.shape[1]):
            if offset[i, j] < 0:
                offset[i, j] = 9999999
    matches = np_min(offset, axis=1)
    onset_ind = onset_ind.flatten()
    end = onset_ind + matches
    # if none then event lasts till end.
    end[end == 9999999] = num_frames
    for i, start_ind in enumerate(onset_ind):
        event_mask[icell, start_ind:end[i]] = True
    return event_mask


class ChunkShuffler:
    def __init__(self, F, num_bins):
        step_size = np.ceil(F.shape[1] / num_bins)
        self._num_bins = num_bins
        self._F_split = np.split(F,
                                 np.arange(step_size, F.shape[1], step_size, np.uint16), axis=1)

    def shuffle(self):
        shuffle_ind = np.random.choice(np.arange(self._num_bins), self._num_bins, replace=False)
        return np.concatenate([self._F_split[i] for i in shuffle_ind], axis=1)
