import numpy as np
import scipy.stats as stats

def clim_pearsoner(x_sim, x_obs, mean=None, std=None, p_val=None):
    if isinstance(mean, type(None)) and isinstance(std, type(None)):
        x_sim = x_sim.ravel() 
        x_obs = x_obs.ravel()
    else:
        x_sim = x_sim.ravel() * std + mean
        x_obs = x_obs.ravel() * std + mean
    corr_coef, p_value = stats.pearsonr(x_sim, x_obs)
    if p_val:
        return corr_coef, p_value
    return corr_coef

def lat_weight_rmse(x_sim, x_obs, lat_weights=None):
    error = np.square(x_sim - x_obs)
    if lat_weights is not None:
        error = error * lat_weights
    error = np.sqrt(np.mean(error))
    return error


def get_lat_weight(latitudes):
    # Convert latitudes to radians and compute weights
    lat_radians = np.deg2rad(latitudes)
    weights = np.cos(lat_radians).clip(0., 1.)
    print("Mean", np.mean(weights))

    #  Normalize weights
    weights_normalized = weights / np.mean(weights)
    weights_normalized = weights_normalized.reshape(-1,1)
    return weights_normalized

def lat_weight_mae(x_sim, x_obs, lat_weights=None):
    error = np.abs(x_sim - x_obs)
    if lat_weights is not None:
        error = error * lat_weights
    error = np.mean(error)
    return error

def calc_bias(x_sim, x_obs):
    return x_obs - x_sim


#def quantile_mse(x, q):
#    """
#        x: truth 
#        q: 0 - 1. 1,2,3 sigma = 0.6827, 0.9545, 0.9973  
#    """
    