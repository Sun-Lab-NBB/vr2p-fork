from copy import deepcopy

import dask
import numpy as np
import colorcet as cc
import dask.array as da
from scipy.signal import convolve2d
from scipy.ndimage import label
import dask.dataframe
from skimage.measure import regionprops
import matplotlib.pyplot as plt

import vr2p
import vr2p.signal


def circular_connected_placefields(thres_im, binF, min_bins=3):
    """Takes thresholded binary image and creates labeled image of
     circularly connected region within a cell/row
    
    Arguments:
        thres_im {numpy array}      -- Thresholded binary image of binned place field activity (size: num_cells x num_bins)
        binF {numpy array}          -- Binned fluorescene data (size: num_cells x num_bins)
    
    Keyword Arguments:
        min_bins {int}              -- minimal required size of connected region (default: {3})
    
    Returns:
        PlaceFields1d                -- Detected 1D place fields.
    """
    num_bins = thres_im.shape[1]
    # make circular.
    pad_thres = np.pad(thres_im, ((0,0),(num_bins, num_bins)),mode="wrap")
    F_padded = np.pad(binF, ((0,0),(num_bins, num_bins)),mode="wrap")
    # connected components.
    label_im,_ = label(pad_thres,[[0,0,0],[1,1,1],[0,0,0]])
    props = np.array(regionprops(label_im, F_padded, cache=False))
    # Get region centers and area.
    centers = np.array([prop["weighted_centroid"] for prop in props])
    area = np.array([prop["area"] for prop in props],np.uint32)
    # select components with center within original area (for circularity) and minimum area size.
    ind = (centers[:,1]>=num_bins) & (centers[:,1]<num_bins*2) & (area>=min_bins)
    # store result.
    result_label_im = np.zeros(thres_im.shape,np.uint32)
    adj_centers = []
    for counter, prop in enumerate(props[ind]):
        coords = prop["coords"]
        new_ind = np.take(np.arange(0,num_bins), coords[:,1],mode="wrap")
        result_label_im[coords[:,0],new_ind] = counter+1
        # get center.
        center=np.array(prop["weighted_centroid"])
        center[1]-=num_bins
        adj_centers.append(center)
    return PlaceFields1d(result_label_im,binF,np.vstack(adj_centers))

def  outside_field_threshold(pf, threshold_factor=3):
    """Filter out detected placefield region if the inside field values
     are below (threshold_factor) times the outside field values. If there are multiple fields
     for one cell both are excluded to from the outside field (one outside field per cell).

    Arguments:
        pf {PlaceFields1d}          -- Placefields1d object with previously detected placedfields.
    
    Keyword Arguments:
        threshold_factor {int}      --  determines the threshold as scalar factor of the outside field signal   (default: {3})
    
    Returns:
        PlaceFields1d               -- Fitlered Placefields1d object.
    """
    # get outside limit value.
    outside_im = pf.binF.copy()
    outside_im[pf.label_im!=0] = np.nan
    outside_values = np.nanmean(outside_im,axis=1)
    # get threshold each regions
    threshold_values = outside_values[pf.cell_id]*threshold_factor
    invalid_regions = np.concatenate(np.argwhere(pf.mean_intensity<threshold_values))
    # remove invalid
    return pf.remove_fields(invalid_regions)

"""Holds data on detected placefields.
"""
class PlaceFields1d:
    def __init__(self,label_im, binF, centers=[],bin_size=1):
        self.bin_size = bin_size
        self.label_im = label_im.astype(int)
        self.binF        = binF.astype(float)
        if not centers:
            props = regionprops(self.label_im, self.binF, cache=False)
            self.centers = np.array([prop["weighted_centroid"] * np.array([1,bin_size]) for prop in props])
        else:
            self.centers  = centers
    @property
    def mean_intensity(self):
        props = regionprops(self.label_im,self.binF, cache=False)
        return np.array([prop["mean_intensity"]for prop in props])
    @property
    def max_intensity(self):
        props = regionprops(self.label_im,self.binF, cache=False)
        return np.array([prop["max_intensity"]for prop in props])
    @property
    def cell_id(self):
        props = regionprops(self.label_im,self.binF, cache=False)
        return np.array([ region["coords"][0,0]for region in props]).astype(int)
    @property
    def has_place_field(self):
        return np.any(self.label_im>0,axis=1)
    @property
    def order(self):
        num_cells = self.binF.shape[0]
        order = np.full(num_cells,np.inf)
        intensity = self.mean_intensity
        centers = self.centers
        if not centers.any():
            return order
        cell_id = centers[:,0].astype(int)
        # in case a cell has two place fields, order on one with highest mean intensity.
        for icell in range(num_cells):
            cell_ind = np.argwhere(cell_id==icell)
            if cell_ind.size>0:
                ind = np.argmax(intensity[cell_id==icell])
                order[icell] = centers[cell_ind[ind],1]
        return np.argsort(order)
    def remove_fields(self,ind):
        pf = deepcopy(self)
        # remove from label image and renumber.
        pf.label_im[np.isin(pf.label_im,ind+1)] = 0
        for counter, value in enumerate(np.unique(pf.label_im)):
            if value!=0:
                pf.label_im[pf.label_im==value] = counter
        # remove from centers
        pf.centers = np.delete(pf.centers,ind,axis=0)
        return pf
    def filter_cells(self,ind):
        pf = deepcopy(self)
        # remove from label image and renumber.
        list_ind = np.arange(0,pf.label_im.shape[0])
        cell_id = pf.cell_id
        pf.label_im[~np.isin(list_ind,ind),:] = 0
        for counter, value in enumerate(np.unique(pf.label_im)):
            if value!=0:
                pf.label_im[pf.label_im==value] = counter
        pf.centers = pf.centers[np.isin(cell_id,ind),:]
        return pf
    def plot(self, color_bar=True,title=None,sort=True,cells = None,dpi=150, vmin=None,vmax=None, **kwargs):
        plt.style.use(vr2p.styles("heatmap"))
        # format data.
        data = self.binF
        # sort.
        if sort:
            order = self.order
        else:
            order = np.arange(0,data.shape[0])
        # select specific cells
        if cells is not None:
            order = order[np.isin(order,np.argwhere(cells))]
        data = data[order,:]
        # calculate image range
        if vmin==None:
            print(vmin)
            vmin = np.nanquantile(data,0.5)
        if vmax==None:
            vmax = np.nanquantile(data,0.9)
        # setup figure.
        fig, axes = plt.subplots(1, 1,figsize=(2, 3),facecolor="white",dpi=dpi)
        if title:
            plt.title(title,fontsize=8)
        # plot heatmap
        extent = [0,self.bin_size*data.shape[1],
            1,data.shape[0]+1]
        plt.imshow(data, cmap=cc.cm.CET_CBL2,
                extent=extent,interpolation="none",
                vmin = vmin, vmax=vmax, **kwargs)
        axes.set_aspect("auto")
        plt.xlabel("Position (cm)")
        plt.ylabel("Cell #")
        # colorbar.
        if color_bar:
            cbar = plt.colorbar()
            cbar.ax.set_ylabel("dF/F0")

"""Base abstract class for 1d Placefield detection protocol.
"""
class PlaceFields1dProtocol:
    class Params:
        value = 0
    params = Params()
    def detect(self,F,pos,speed,track_length,bin_size):
        raise NotImplementedError
    def validate(self,client,num_repeats,F,pos,speed,track_length,bin_size):
        raise NotImplementedError

""" Tank Implementation of 1d Placefield detection protocol.
"""
class Tank1dProtocol(PlaceFields1dProtocol):
    class Params:
        min_speed = 5
        smooth_size = 3
        base_quantile = 0.25
        signal_threshold = 0.25
        min_bins = 3
        outside_threshold = 3
        max_int_threshold = 0.1
        num_chunks = 100
        sig_threshold = 0.05
    params = Params()
    def detect(self,F,pos,speed,track_length,bin_size,calc_df=True):
        # Delta F over F zero.
        if (calc_df):
            F = vr2p.signal.df_over_f0(F)
        # filter for speed.
        ind = speed>self.params.min_speed
        pos = pos.loc[ind]
        F = F[:,ind]
        # average bin fluorescent data.
        edges = np.arange(0,track_length + bin_size, bin_size)
        binF,_ = vr2p.signal.bin_fluorescence_data(F,pos,edges)
        # smooth.
        binF = convolve2d(binF,np.ones((1, self.params.smooth_size))/self.params.smooth_size,
                mode="same",boundary="wrap")
        # threshold.
        thres_binF = vr2p.signal.quantile_max_treshold(binF, self.params.base_quantile,
                           self.params.signal_threshold)
        # get circular placefields.
        pf = circular_connected_placefields(thres_binF, binF)
        pf.bin_size = bin_size
        # filter based on outside of field threshold.
        pf = outside_field_threshold(pf,self.params.outside_threshold)
        # one bin atleast X %
        pf = pf.remove_fields(np.argwhere(pf.max_intensity<self.params.max_int_threshold))
        return pf
    def validate(self,client,num_repeats,F,pos,speed,track_length,bin_size):
        #Delta F over F zero.
        F = vr2p.signal.df_over_f0(F)
        # prep data.
        F = da.from_array(F)
        # change speed to array
        speed = speed.to_numpy()
        speed[np.isnan(speed)] = 0
        #run task.
        # calculate original data.
        results = [dask.delayed(self.detect)(F,pos,speed,track_length,bin_size,calc_df=False).has_place_field]
        for i in range(num_repeats):
            shuffled = self._shuffle(F,i,pos,speed,track_length,bin_size)
            res = dask.delayed(self.detect)(shuffled,pos,speed,track_length,bin_size,calc_df=False).has_place_field
            results.append(res)
        results = dask.compute(results)
        results = np.vstack(results).T
        observed = results[:,0] # observed original data.
        results = results[:,1:] # shuffle data.
        p = np.sum(results,axis=1)/results.shape[1]
        # signficant cells.
        sig_cells = np.argwhere((observed) & (p<self.params.sig_threshold) ).flatten()
        return sig_cells,p

    @dask.delayed
    def _shuffle(self,data,i,pos,speed,track_length,bin_size):
        data_shuffle = np.array_split(data,self.params.num_chunks,axis=1)
        np.random.seed(i)
        shuffle_ind = np.random.choice(np.arange(self.params.num_chunks),self.params.num_chunks,replace=False)
        data_shuffle = np.concatenate([data_shuffle[i] for i in shuffle_ind],axis=1)
        return data_shuffle


""" Main interaction class for the user
"""
class DetectPlaceFields1d:
    def __init__(self, F, pos, speed, track_length, bin_size,protocol=Tank1dProtocol):
        self.F = F
        self.pos = pos
        self.speed = speed
        self.track_length = track_length
        self.bin_size = bin_size
        self.protocol = protocol()
    def run(self):
        # returns PlaceFields1d access.
        return self.protocol.detect(self.F, self.pos, self.speed, self.track_length, self.bin_size)
    def validate(self,client,num_repeats,**kwargs):
        return self.protocol.validate(client,num_repeats,
            self.F, self.pos, self.speed, self.track_length, self.bin_size,**kwargs)
