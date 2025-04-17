import os
import copy
import pandas as pd
import glob
import numpy as np
import scipy.stats as stats
from scipy.stats import gaussian_kde
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

#import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# %load_ext autoreload
# %autoreload 2
from taylor import TaylorDiagram
from metrics import *
from psd import psd

VARIABLE_NAMES= {
    # "prcp" : 'prcp'
    "2m_temperature_min": "2m_temperature_min"
}

def get_lat_weight(latitudes):
    # Convert latitudes to radians and compute weights
    lat_radians = np.deg2rad(latitudes)
    weights = np.cos(lat_radians).clip(0., 1.)
    print("Mean", np.mean(weights))
    return weights

lats = np.load("/lustre/orion/lrn036/world-shared/kurihana/regridding/dataset/datasets/ERA5-Daymet-1dy-superres/3.75_arcmin/lat.npy")
lat_weights = get_lat_weight(lats)
lat_weights = lat_weights[..., np.newaxis]
lat_weights.shape

lons = np.load("/lustre/orion/lrn036/world-shared/kurihana/regridding/dataset/datasets/ERA5-Daymet-1dy-superres/3.75_arcmin/lon.npy")

#basedatadir = "/lustre/orion/lrn036/world-shared/patrickfan/super-res-torchlight-main-118M-Model/tutorial/checkpoints/test"
basedatadir = "/lustre/orion/lrn036/world-shared/patrickfan/super-res-Daymet-V28km-4km-117M-Model/tutorial/checkpoints/test"


variables = ['2m_temperature_min']
loss_metrics =  ['mse', 'imagegradient'] 

import time
import glob 
import os

def get_data(truth_or_pred):
    Var = {}
    s1 = time.time()

    filelist = sorted(glob.glob(os.path.join(basedatadir, f"{truth_or_pred}*.npy")))

    if len(filelist) == 0:
        print(f"No files found for {truth_or_pred}")
        return Var  # Return an empty dictionary

    data_array = None
    for idx, ifile in enumerate(filelist):
        data = np.load(ifile).astype(np.float32)

        if idx == 0:
            data_array = data
        else:
            data_array = np.concatenate([data_array, data], axis=0)

    Var[variables[0]] = data_array  # Use the first variable in the list
    return Var

# Predict data
Preds = {}
Preds = get_data('prediction')


Truths = {}
Truths = get_data('groundtruth')

from scipy.ndimage import zoom
land_sea_mask = np.load("land_sea_mask.npy")
pred_data = Preds[variables[0]]
truth_data = Truths[variables[0]]
for t in range(pred_data.shape[0]):
    pred_data[t][land_sea_mask == 0] = truth_data[t][land_sea_mask == 0]

def quantile_rmse(x, y, q):
    """
        x: pred
        y: truth 
        q: 0 - 1. 1,2,3 sigma = 0.6827, 0.9545, 0.9973  
    """
    #0.6827, 0.9545, 0.9973
    index = np.where(y>=np.quantile(y, q))
    rmse =  np.sqrt(np.mean(np.square(x[index] -  y[index] )))
    return rmse

def normalize(float_array,vmax, vmin ):
    # Normalize the array to range [0, 1]
    norm_array = (float_array - vmin) / (vmax - vmin)

    # Scale and convert to integers in range [0, 255]
    int_array = (norm_array * 255).astype(np.uint8)
    return int_array

Metrics = {}
for (k, preds), (_, truths) in zip(Preds.items(),  Truths.items()):
     
    corrs = np.array([])
    wrmses = np.array([])
    s1rmses = np.array([])
    s2rmses = np.array([])
    s3rmses = np.array([])
    ssim_scores = np.array([])
    psnr_scores = np.array([])
    
    for pred, truth in zip(preds, truths ):
        
        corr = clim_pearsoner(x_sim=pred, x_obs=truth )
        #wrmse = lat_weight_rmse(x_sim=pred, x_obs=truth, lat_weights=lat_weights)
        wrmse = np.sqrt(np.mean((pred - truth) ** 2))
        corrs = np.append(corrs, corr) 
        wrmses = np.append(wrmses, wrmse)
        
        # >1, 2, 3 sigma rmses
        s1, s2, s3 = 0.6827, 0.9545, 0.9973
        s1rmse = quantile_rmse(pred, truth, s1)
        s2rmse = quantile_rmse(pred, truth, s2)
        s3rmse = quantile_rmse(pred, truth, s3)
        s1rmses = np.append(s1rmses, s1rmse)
        s2rmses = np.append(s2rmses, s2rmse)
        s3rmses = np.append(s3rmses, s3rmse)
        
        # transformation
        vmin = min(np.nanmin(pred), np.nanmin(truth))
        vmax = max(np.nanmax(pred), np.nanmax(truth))
        pred= normalize(pred, vmax=vmax, vmin=vmin)
        truth= normalize(truth, vmax=vmax, vmin=vmin)  
    
        # calc
        _ssim = ssim(pred, truth)
        _psnr = psnr(pred,truth)
        ssim_scores = np.append(ssim_scores, _ssim)
        psnr_scores = np.append(psnr_scores, _psnr)
        
    
    # Add mean
    corr_mean =  np.mean(corrs)
    wrmse_mean= np.mean(wrmses)
    ssim_mean = np.mean(ssim_scores)
    psnr_mean = np.mean(psnr_scores)
    s1rmse_mean = np.mean(s1rmses)
    s2rmse_mean = np.mean(s2rmses)
    s3rmse_mean = np.mean(s3rmses)
    
    # store at dict
    Metrics[k] = {
        'corr': corr_mean,  'rmse': wrmse_mean, 
        "rmse_sigma1": s1rmse_mean,
        "rmse_sigma2": s2rmse_mean,
        "rmse_sigma3": s3rmse_mean,
        'ssim': ssim_mean, 'psnr': psnr_mean
    }


df = pd.DataFrame(Metrics)

print (df)

for column in df.columns:
    print(f"{column}:")
    print(df[column])
    print("-" * 40)  # Separator for better readability
    
df.to_csv("metrics.csv", index=True) 


nsamples = 365 #NOTE: max 365 i.e. the length of 2021


def calc_RALSD(truths, preds, nsamples=-1):
    return np.sqrt(np.mean(np.square(10*np.log(psd(truths[:nsamples])[0].mean() / 
                                    psd(preds[:nsamples])[0].mean()))) )


RALSDs = {}
for (k, truths), (_, preds) in zip(Truths.items(), Preds.items()):
    s1 = time.time()
    RALSDs[k] = calc_RALSD(truths, preds, nsamples=nsamples)
    print(f"DONE {k} {(time.time() - s1)/60.0 } [min]")


print(RALSDs)



















