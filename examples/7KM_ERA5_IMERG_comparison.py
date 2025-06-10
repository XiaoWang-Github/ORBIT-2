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
from sklearn.metrics import r2_score

import argparse
import sys

import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def nanflatten(x):
    ## flatten and remove nan
    y = x.flatten()
    mask = np.isfinite(y)
    return y[mask]


def get_lat_weight(latitudes):
    # Convert latitudes to radians and compute weights
    lat_radians = np.deg2rad(latitudes)
    weights = np.cos(lat_radians).clip(0.0, 1.0)
    print("Mean", np.mean(weights))
    return weights


def lat_weight_rmse(x_sim, x_obs, lat_weights):
    # Assumes x_sim and x_obs have shape (lat, lon)
    error_squared = (x_sim - x_obs) ** 2
    weighted_error = error_squared * lat_weights

    ## Flatten and remove nan
    weighted_error_filtered = nanflatten(weighted_error)

    rmse = np.sqrt(np.mean(weighted_error_filtered))
    return rmse


def quantile_rmse(x, y, q):
    """
    x: pred
    y: truth
    q: 0 - 1. 1,2,3 sigma = 0.6827, 0.9545, 0.9973
    """
    # 0.6827, 0.9545, 0.9973
    index = np.where(y >= np.nanquantile(y, q))
    rmse = np.sqrt(np.mean(np.square(x[index] - y[index])))
    return rmse


def pearson_corr(x, y):
    ## Flatten and remove nan
    x_flat = nanflatten(x)
    y_flat = nanflatten(y)
    assert np.isfinite(x_flat).all()
    assert np.isfinite(y_flat).all()

    # Pearson correlation
    corr_matrix = np.corrcoef(x_flat, y_flat)
    corr = corr_matrix[0, 1]
    return corr


def plot2(x, y, log=False):
    data1, data2 = x, y
    if log:
        data1 = np.log1p(x)
        data2 = np.log1p(y)

    vmin0, vmax0 = np.nanmin(data1), np.nanmin(data1)
    vmin1, vmax1 = np.nanmax(data2), np.nanmax(data2)
    print((vmin0, vmax0), (vmin1, vmax1))
    vmin, vmax = min(vmin0, vmin1), max(vmax0, vmax1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 3), constrained_layout=True)
    for ax, data in zip(axes, [data1, data2]):
        im = ax.imshow(data, vmin=vmin, vmax=vmax, origin="lower", cmap="viridis")
        ax.set_aspect("equal")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax)
    # plt.tight_layout()
    # ax.set_axis("scaled")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="pred_out_july_2020")
    parser.add_argument("--taskname", default="DAYMET")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--day", type=int, default=1)
    args = parser.parse_args()
    datestr = f"{args.year}-{args.month:02d}-{args.day:02d}"

    # ## Using JYC data
    # ds1 = xr.open_zarr(f"{args.outputdir}/imerg.zarr/")
    # ds2 = xr.open_zarr(f"{args.outputdir}/preds.zarr/")

    # truth = np.log1p(ds1["precipitation"].sel(time=datestr).values)
    # pred = np.log1p(ds2["precipitation"].sel(time=datestr).values)
    # land_sea_mask = ds1["land_sea_mask"].values
    # print(truth.shape, pred.shape, land_sea_mask.shape)

    # lat_weights = get_lat_weight(ds1["latitude"].values)
    # print(lat_weights.shape)

    # ## FIXME: use only land??
    # pred[land_sea_mask == 0] = truth[land_sea_mask == 0]
    # # pred[land_sea_mask == 0] = np.nan
    # # truth[land_sea_mask == 0] = np.nan

    ## Using Ming data
    # === Load predicted data ===
    # pred_data = np.load(f"pred_out_july_2020/preds_day_{args.day}.npy")
    # pred_data = np.expm1(pred_data)

    ## JYC qm eval
    ds = xr.open_zarr("/lustre/orion/lrn036/world-shared/jyc/frontier/weatherbench2/eval.zarr/")
    pred_data = np.expm1(ds["prediction"].sel(time=f"2021-07-{args.day:02d}").values.T)

    # === Load truth data ===
    if args.taskname == "DAYMET":
        # === Load truth data (DAYMET) ===
        ds = xr.open_zarr("/lustre/orion/world-shared/lrn036/jyc/frontier/weatherbench2/datasets/daymet/era5-qm-daymet_1980-2022-1d-240x120-bilinear.zarr")
        truth_data = ds["precipitation"].sel(time=f"2020-07-{args.day:02d}").values.T
    elif args.taskname == "IMERG":
        # === Load truth data (IMERG) ===
        # truth_file = f"pred_out_july_2020/Imerg_daily_precip/interpolated_imerg_202007{args.day:02d}.npy"
        # truth_data = np.load(truth_file).astype(np.float32)
        ds = xr.open_zarr("/lustre/orion/lrn036/world-shared/jyc/frontier/weatherbench2/datasets/IMERG/IMERG-1998-2025-1d-5760x2881-bilinear.zarr")
        truth_data = ds["precipitation"].sel(time=f"2021-07-{args.day:02d}").values[:,:-1].T

    elif args.taskname == "CHIRPS":
        # === Load truth data (CHIRPS) ===
        ds = xr.open_zarr("/lustre/orion/lrn036/world-shared/jyc/frontier/weatherbench2/datasets/chirps/chirps_20200101-20210101-5760x2881-bilinear-v2.zarr/")
        truth_data = ds["precip"].sel(time=f"2020-07-{args.day:02d}").values[:,:-1].T
    elif args.taskname == "PERSIANN":
        # === Load truth data (PERSIANN) ===
        ds = xr.open_zarr("/lustre/orion/lrn036/world-shared/jyc/frontier/weatherbench2/datasets/persiann/persiann_20200701-20200801-5760x2881-bilinear.zarr/")
        truth_data = ds["precipitation"].sel(time=f"2020-07-{args.day:02d}").values[:,:-1].T
    else:
        raise ValueError("Invalid taskname: {args.taskname}")

    print(pred_data.shape, truth_data.shape)
    # # === Replace NaNs and Infs ===
    # fill_value = 0.00001
    # pred_data = np.nan_to_num(pred_data, nan=fill_value, posinf=fill_value, neginf=fill_value)
    # truth_data = np.nan_to_num(truth_data, nan=fill_value, posinf=fill_value, neginf=fill_value)
    pred_data[np.isnan(truth_data)] = np.nan

    # === Apply log1p to truth ===
    # truth_data = np.log1p(truth_data)

    # land_sea_mask = np.load("pred_out_july_2020/land_sea_mask.npy")  # shape (720, 1440)
    # upsampled_mask = zoom(land_sea_mask, (4, 4), order=0)  # shape (2880, 5760)
    # pred_data[upsampled_mask == 0] = truth_data[upsampled_mask == 0]

    ## FIXME: use jong's mask and mask out all sea areas
    ds = xr.open_zarr("/lustre/orion/lrn036/world-shared/jyc/frontier/weatherbench2/datasets/IMERG/IMERG-1998-2025-1d-5760x2881-bilinear.zarr")
    upsampled_mask = ds["land_sea_mask"].values[:,:].T
    ## DAYMET
    ds = xr.open_zarr("/lustre/orion/world-shared/lrn036/jyc/frontier/weatherbench2/datasets/daymet/era5-qm-daymet_1980-2022-1d-240x120-bilinear.zarr")
    upsampled_mask = ds["land_sea_mask"].values[:,:].T
    pred_data[upsampled_mask == 0] = np.nan
    truth_data[upsampled_mask == 0] = np.nan

    ## IMERG global
    lats = np.linspace(-89.95, 89.95, 2880)
    ## DYAMET USA
    lats = np.linspace(24, 53.75, 120)
    
    lat_weights = get_lat_weight(lats)
    lat_weights = lat_weights[..., np.newaxis]
    print ("shape:", pred_data.shape, truth_data.shape, upsampled_mask.shape, lat_weights.shape)

    pred = pred_data 
    truth = truth_data

    # Pearson correlation
    corr = pearson_corr(truth, pred)
    # corr = pearson_corr(truth, pred)

    # Weighted RMSE
    wrmse = lat_weight_rmse(x_sim=pred, x_obs=truth, lat_weights=lat_weights)
    rmse = lat_weight_rmse(x_sim=pred, x_obs=truth, lat_weights=np.ones_like(lat_weights))

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

    print("Pred norm min/max:", np.nanmin(pred_norm), np.nanmax(pred_norm))
    print("Truth norm min/max:", np.nanmin(truth_norm), np.nanmax(truth_norm))

    # SSIM and PSNR
    ssim_score = ssim(nanflatten(pred_norm), nanflatten(truth_norm), data_range=1)
    psnr_score = psnr(nanflatten(pred_norm), nanflatten(truth_norm), data_range=1.0)

    # R2 score
    r2 = r2_score(nanflatten(truth), nanflatten(pred))

    # Store metrics
    Metrics = {}
    Metrics["Global"] = {
        "corr": corr,
        "rmse": rmse,
        "wrmse": wrmse,
        "rmse_sigma1": s1rmse,
        "rmse_sigma2": s2rmse,
        "rmse_sigma3": s3rmse,
        "ssim": ssim_score,
        "psnr": psnr_score,
        "r2": r2,
    }

    # Convert to DataFrame
    df = pd.DataFrame(Metrics)
    print(df)

    ## Subregion
    df = pd.read_csv(
        f"{args.prefix}/ipcc_regions.txt", delim_whitespace=True
    )  # index_col="FID")
    df["FID"] = df["FID"] + 1

    dm = xr.open_dataset(f"{args.prefix}/ipcc_reg_mask_7km_0_to_360.nc")
    dm = dm.isel(latitude=slice(0, -1))

    ## FID starting from 1
    for i in range(1, len(df) + 1):
        acronym = df[df["FID"] == i]["Acronym"].item()
        print("Subregion:", acronym)

        mask = dm["mask"].values == i
        truth_masked = np.ma.array(truth, mask=~mask).filled(np.nan)
        pred_masked = np.ma.array(pred, mask=~mask).filled(np.nan)
        assert truth_masked.shape == mask.shape
        assert pred_masked.shape == mask.shape
        print ("shape:", acronym, pred_masked.shape, truth_masked.shape, mask.shape)
        print ("mask, real:", np.sum(mask), len(nanflatten(truth_masked)))

        # Pearson correlation
        corr = pearson_corr(truth_masked, pred_masked)

        # Weighted RMSE
        wrmse = lat_weight_rmse(
            x_sim=pred_masked, x_obs=truth_masked, lat_weights=lat_weights
        )
        rmse = lat_weight_rmse(
            x_sim=pred_masked, x_obs=truth_masked, lat_weights=np.ones_like(lat_weights)
        )

        if acronym in ["SAS", "WAF", "NWS", "WSAF", "ESAF", "WNA", "CAU"]:
            mask = ~np.isnan(truth_masked)

            # Get index positions of valid values
            rows, cols = np.where(mask)

            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            truth_subset = truth_masked[min_row:max_row+1, min_col:max_col+1]
            pred_subset = pred_masked[min_row:max_row+1, min_col:max_col+1]

            plot2(truth_subset, pred_subset, log=True)
            plt.savefig(f"plot-day{args.day:02d}-{acronym}.png")
            # import pdb; pdb.set_trace()


        # Quantile RMSEs
        s1, s2, s3 = 0.6827, 0.9545, 0.9973
        s1rmse = quantile_rmse(pred_masked, truth_masked, s1)
        s2rmse = quantile_rmse(pred_masked, truth_masked, s2)
        s3rmse = quantile_rmse(pred_masked, truth_masked, s3)

        # Normalize for similarity metrics
        vmin = min(np.nanmin(pred_masked), np.nanmin(truth_masked))
        vmax = max(np.nanmax(pred_masked), np.nanmax(truth_masked))

        pred_norm = (pred_masked - vmin) / (vmax - vmin)
        truth_norm = (truth_masked - vmin) / (vmax - vmin)

        print("Pred norm min/max:", np.nanmin(pred_norm), np.nanmax(pred_norm))
        print("Truth norm min/max:", np.nanmin(truth_norm), np.nanmax(truth_norm))

        # SSIM and PSNR
        try:
            ssim_score = ssim(nanflatten(pred_norm), nanflatten(truth_norm), data_range=1)
        except:
            ssim_score = np.nan

        try:
            psnr_score = psnr(nanflatten(pred_norm), nanflatten(truth_norm), data_range=1.0)
        except:
            psnr_score = np.nan

        try:
            r2 = r2_score(nanflatten(truth_masked), nanflatten(pred_masked))
        except:
            r2 = np.nan

        Metrics[acronym] = {
            "corr": corr,
            "rmse": rmse,
            "wrmse": wrmse,
            "rmse_sigma1": s1rmse,
            "rmse_sigma2": s2rmse,
            "rmse_sigma3": s3rmse,
            "ssim": ssim_score,
            "psnr": psnr_score,
            "r2": r2,
        }

    df = pd.DataFrame(Metrics)
    print(df)
    dirname = f"{args.prefix}/{args.taskname}"
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    fname = f"{dirname}/metrics-{datestr}.csv"
    print("Saved:", fname)
    df.to_csv(fname)
