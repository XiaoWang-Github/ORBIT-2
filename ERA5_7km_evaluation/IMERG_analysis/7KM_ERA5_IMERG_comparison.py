# Evaluate the global 7km ERA5 inference against IMERGE data

import numpy as np
import os
import pandas as pd
import glob
import xarray as xr

import scipy.stats as stats
from scipy.ndimage import zoom
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


# === Load predicted data ===
pred_data = np.load("clip_0.5/preds_day_1.npy")
pred_data = np.expm1(pred_data)


# === Load truth data ===
truth_file = "../Imerg_daily_precip/interpolated_imerg_20200701.npy"
truth_data = np.load(truth_file).astype(np.float32)


# === Replace NaNs and Infs ===
fill_value = 0.00001
pred_data = np.nan_to_num(pred_data, nan=fill_value, posinf=fill_value, neginf=fill_value)
truth_data = np.nan_to_num(truth_data, nan=fill_value, posinf=fill_value, neginf=fill_value)

# === Apply log1p to truth ===
# truth_data = np.log1p(truth_data)

land_sea_mask = np.load("../land_sea_mask.npy")  # shape (720, 1440)
upsampled_mask = zoom(land_sea_mask, (4, 4), order=0)  # shape (2880, 5760)
pred_data[upsampled_mask == 0] = truth_data[upsampled_mask == 0]


def lat_weight_rmse(x_sim, x_obs, lat_weights):
    # Assumes x_sim and x_obs have shape (lat, lon)
    error_squared = (x_sim - x_obs) ** 2
    weighted_error = error_squared * lat_weights  # lat_weights shape must be (lat, 1)
    rmse = np.sqrt(np.mean(weighted_error))
    return rmse


def get_lat_weight(latitudes):
    # Convert latitudes to radians and compute weights
    lat_radians = np.deg2rad(latitudes)
    weights = np.cos(lat_radians).clip(0., 1.)
    print("Mean", np.mean(weights))
    return weights


lats =  np.linspace(-89.95, 89.95, 2880)  
lat_weights = get_lat_weight(lats)
lat_weights = lat_weights[..., np.newaxis]
print (lat_weights.shape)

# Define variable name
variables = ["precipitation"]

Preds = {variables[0]: pred_data}
Truths = {variables[0]: truth_data }


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


Metrics = {}

# Extract arrays
pred = Preds["precipitation"]  # shape: (H, W)
truth = Truths["precipitation"]  # shape: (H, W)

# Flatten for correlation
truths_flat = truth.flatten()
preds_flat = pred.flatten()

# Pearson correlation
corr_matrix = np.corrcoef(truths_flat, preds_flat)
corr = corr_matrix[0, 1]

# Weighted RMSE
wrmse = lat_weight_rmse(x_sim=pred, x_obs=truth, lat_weights=lat_weights)

# Quantile RMSEs
s1, s2, s3 = 0.6827, 0.9545, 0.9973
s1rmse = quantile_rmse(pred, truth, s1)
s2rmse = quantile_rmse(pred, truth, s2)
s3rmse = quantile_rmse(pred, truth, s3)

# Normalize for similarity metrics
vmin = min(np.nanmin(pred), np.nanmin(truth))
vmax = max(np.nanmax(pred), np.nanmax(truth))


pred_norm = (pred - vmin) / (vmax - vmin)
truth_norm = (truth - vmin) / (vmax - vmin)

print("Pred norm min/max:", pred_norm.min(), pred_norm.max())
print("Truth norm min/max:", truth_norm.min(), truth_norm.max())


# SSIM and PSNR
ssim_score = ssim(pred_norm, truth_norm, data_range=1)
psnr_score = psnr(pred_norm, truth_norm, data_range=1.0)

# Store metrics
Metrics["precipitation"] = {
    'corr': corr,
    'rmse': wrmse,
    "rmse_sigma1": s1rmse,
    "rmse_sigma2": s2rmse,
    "rmse_sigma3": s3rmse,
    'ssim': ssim_score,
    'psnr': psnr_score
}

# Convert to DataFrame
df = pd.DataFrame(Metrics)
print(df)


