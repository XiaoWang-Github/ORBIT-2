"""Example quantile mapping workflow for daily precipitation.

Quantile mapping is a statistical bias-correction method that transforms a
model variable so its cumulative distribution matches a reference
distribution, typically observations.

This script applies quantile mapping to ERA5 `total_precipitation_24hr` using
IMERG precipitation as the reference. The correction is estimated over a
calibration period and then applied to ERA5 total_precipitation_24hr values.
"""
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

# Calibration period and quantiles
selected = slice("1998-01-01", "2021-12-31")
quantiles = np.linspace(0.001, 0.999, 999)

# Reference (IMERG) and source (ERA5)
ds_ref = xr.open_zarr("datasets/IMERG/IMERG-1998-2025-1d-3600x1800.zarr/")
ds_ref["precipitation"] = ds_ref["precipitation"] / 1000.0

ds_src = xr.open_zarr("datasets/IMERG/era5-imerg_1998-2022-1d-1440x721-bilinear.zarr/")
p_src = ds_src["total_precipitation_24hr"].where(ds_src["total_precipitation_24hr"] >= 0, 0.0)

# Flatten space and time for empirical CDF estimation
ref_flat = ds_ref["precipitation"].sel(time=selected).stack(points=("time", "latitude", "longitude"))
src_flat = p_src.sel(time=selected).stack(points=("time", "latitude", "longitude"))

# Quantiles in log-space to better represent low precipitation values
ref_q = np.log(ref_flat + 1e-3).quantile(quantiles, dim="points").compute().values
src_q = np.log(src_flat + 1e-3).quantile(quantiles, dim="points").compute().values

# Mapping from ERA5 quantiles to IMERG quantiles
qm_func = interp1d(src_q, ref_q, bounds_error=False, fill_value="extrapolate")

def apply_qm_to_precip(precip_da: xr.DataArray) -> xr.DataArray:
    precip_da = precip_da.where(precip_da >= 0, 0.0)
    corrected_log = xr.apply_ufunc(qm_func, np.log(precip_da + 1e-3), vectorize=True)
    corrected = np.exp(corrected_log) - 1e-3
    return corrected.clip(min=0.0)

# Corrected ERA5 daily precipitation field
corrected_tp24 = apply_qm_to_precip(p_src)
