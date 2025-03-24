import os
import numpy as np
from scipy.stats import rankdata
from tqdm import tqdm
from ..data.processing.era5_constants import VAR_TO_UNIT as ERA5_VAR_TO_UNIT
from ..data.processing.cmip6_constants import VAR_TO_UNIT as CMIP6_VAR_TO_UNIT

import torch 
from climate_learn.data.processing.era5_constants import (
    CONSTANTS
)

def test_on_many_images_lighting(mm, dm, in_transform, out_transform, variable, src, outputdir, index=0):
    print("Start Inference",flush=True)

    lat, lon = dm.get_lat_lon()
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    out_channel = dm.hparams.out_vars.index(variable)
    in_channel = dm.hparams.in_vars.index(variable)

    history = dm.hparams.history

    print("dm.hparams",dm.hparams,flush=True)
    print("out_channel",out_channel,"history",history,flush=True)

    if src == "era5":
        variable_with_units = f"{variable} ({ERA5_VAR_TO_UNIT[variable]})"
    elif src == "cmip6":
        variable_with_units = f"{variable} ({CMIP6_VAR_TO_UNIT[variable]})"
    elif src == "prism":
        variable_with_units = f"Daily Max Temperature (C)"
    else:
        raise NotImplementedError(f"{src} is not a supported source")

    counter = 0
    adj_index = 0
    for batch in dm.test_dataloader():
        #FIXME select "second" index and then flip
        xx, y = batch[:2]
        batch_size = xx.shape[0]
        xx = xx.to(mm.device)
        pred = mm.forward(xx)

        if counter == 0: print(f"xx {xx.shape} Batch size: {batch_size}")
        if dm.hparams.task == "downscaling":
            img = in_transform(xx)[:, in_channel].detach().cpu().numpy()
        else:
            img = in_transform(xx[0])[in_channel].detach().cpu().numpy()
        if src == "era5":
            if len(img.shape) == 2:
                img = np.flip(img, 0)
            elif len(img.shape) == 3:
                img = np.flip(img, 1)

        # Plot the ground truth
        yy = out_transform(y)
        yy = yy[:, out_channel].detach().cpu().numpy()
        
        if src == "era5":
            if len(yy.shape) == 2:
                yy = np.flip(yy, 0)
            elif len(yy.shape) == 3:
                yy = np.flip(yy, 1)
                

        # Plot the prediction
        ppred = out_transform(pred)
        ppred = ppred[:, out_channel].detach().cpu().numpy()
        if src == "era5":
            if len(ppred.shape) == 2:
                ppred = np.flip(ppred, 0)
            elif len(ppred.shape) == 3:
                ppred = np.flip(ppred, 1)

        # Save image datasets
        os.makedirs(outputdir, exist_ok=True)
        if counter == 0: np.save(os.path.join(outputdir, f'input_{str(counter).zfill(4)}.npy'), img)
        np.save(os.path.join(outputdir, f'groundtruth_{str(counter).zfill(4)}.npy'), yy)
        np.save(os.path.join(outputdir, f'prediction_{str(counter).zfill(4)}.npy'), ppred)

        # Counter
        print(f"Save image data {counter}...")
        counter += 1
        
def clip_replace_constant(y, yhat, out_variables):

    prcp_index = out_variables.index("total_precipitation_24hr")
    for i in range(yhat.shape[1]):
        if i==prcp_index:
            torch.clamp_(yhat[:,prcp_index,:,:], min=0.0)

    for i in range(yhat.shape[1]):
        # if constant replace with ground-truth value
        if out_variables[i] in CONSTANTS:
            yhat[:, i] = y[:, i]
    return yhat

def test_on_many_images_org(mm, dm, in_transform, out_transform, variable, src, outputdir, device, index=0):
    """native_pytorch version """
    print("Start Inference",flush=True)

    lat, lon = dm.get_lat_lon()
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    out_channel = dm.out_vars.index(variable)
    in_channel = dm.in_vars.index(variable)

    history = mm.history

    print("out_channel",out_channel,"history",history,flush=True)

    if src == "era5":
        variable_with_units = f"{variable} ({ERA5_VAR_TO_UNIT[variable]})"
    elif src == "cmip6":
        variable_with_units = f"{variable} ({CMIP6_VAR_TO_UNIT[variable]})"
    elif src == "prism":
        variable_with_units = f"Daily Max Temperature (C)"
    else:
        raise NotImplementedError(f"{src} is not a supported source")

    counter = 0
    adj_index = 0
    for batch in dm.test_dataloader():
        #FIXME select "second" index and then flip
        xx, y = batch[:2]
        batch_size = xx.shape[0]
        xx = xx.to(device)
        pred = mm.forward(xx)

        if counter == 0: print(f"xx {xx.shape} Batch size: {batch_size}")
        if dm.task == "downscaling":
            img = in_transform(xx)[:, in_channel].detach().cpu().numpy()
        else:
            img = in_transform(xx[0])[in_channel].detach().cpu().numpy()
        if src == "era5":
            if len(img.shape) == 2:
                img = np.flip(img, 0)
            elif len(img.shape) == 3:
                img = np.flip(img, 1)

        # Plot the ground truth
        yy = out_transform(y)
        yy = yy[:, out_channel].detach().cpu().numpy()
        
        if src == "era5":
            if len(yy.shape) == 2:
                yy = np.flip(yy, 0)
            elif len(yy.shape) == 3:
                yy = np.flip(yy, 1)
                

        # Plot the prediction``
        ppred = out_transform(pred)
        ppred = ppred[:, out_channel].detach().cpu().numpy()
        if src == "era5":
            if len(ppred.shape) == 2:
                ppred = np.flip(ppred, 0)
            elif len(ppred.shape) == 3:
                ppred = np.flip(ppred, 1)

        # Save image datasets
        os.makedirs(outputdir, exist_ok=True)
        if counter == 0: np.save(os.path.join(outputdir, f'input_{str(counter).zfill(4)}.npy'), img)
        np.save(os.path.join(outputdir, f'groundtruth_{str(counter).zfill(4)}.npy'), yy)
        np.save(os.path.join(outputdir, f'prediction_{str(counter).zfill(4)}.npy'), ppred)

        # Counter
        print(f"Save image data {counter}...")
        counter += 1

def test_on_many_images(mm, dm, in_variables, out_variables, in_transform, out_transform, variable, src, outputdir, device, index=-1,
                        ground_truth=True):
    """native_pytorch version 
    
        ground_truth (bool): set False for infrence on global era5
    """
    print("Start Inference",flush=True)

    # set dtype 
    torch.set_default_dtype(torch.float32)
    print(f"Dtype {torch.get_default_dtype()}")
    with torch.no_grad():

        lat, lon = dm.get_lat_lon()
        extent = [lon.min(), lon.max(), lat.min(), lat.max()]
        out_channel = dm.out_vars.index(variable)
        try:
            in_channel = dm.in_vars.index(variable)
        except KeyError:
            print(f'in channel does not include {variable}. Set in_channel = -1')
            in_channel = -1 
            pass

        history = mm.history

        print("out_channel",out_channel,"history",history,flush=True)

        counter = 0
        for idx, batch in enumerate(dm.test_dataloader()):
            xx, y = batch[:2]
            batch_size = xx.shape[0]
            xx = xx.to(device)
            if torch.isnan(xx).any(): 
                print(f"Any NaN? in test data batch indx = {idx}", torch.isnan(xx).any())
            pred = mm.forward(xx, in_variables,out_variables)

            if counter == 0: print(f"xx {xx.shape} Batch size: {batch_size}")
            if dm.task == "downscaling":
                if in_channel >= 0:
                    img = in_transform(xx)[:, in_channel].detach().cpu().numpy()
                else:
                    img = None
            else:
                img = in_transform(xx[0])[in_channel].detach().cpu().numpy()
            if src == "era5":
                if len(img.shape) == 2:
                    img = np.flip(img, 0)
                elif len(img.shape) == 3:
                    img = np.flip(img, 1)

            # Plot the ground truth
            yy = out_transform(y)
            yy = yy[:, out_channel].detach().cpu().numpy()
            
            if src == "era5":
                if len(yy.shape) == 2:
                    yy = np.flip(yy, 0)
                elif len(yy.shape) == 3:
                    yy = np.flip(yy, 1)
                    

            # Plot the prediction``
            ppred = out_transform(pred)
            ppred = clip_replace_constant(yy, ppred, out_variables)
            ppred = ppred[:, out_channel].detach().cpu().numpy()
            if torch.isnan(ppred).any():
                print("Prediction includes NaN\n", np.isnan(ppred))
            if src == "era5":
                if len(ppred.shape) == 2:
                    ppred = np.flip(ppred, 0)
                elif len(ppred.shape) == 3:
                    ppred = np.flip(ppred, 1)

            # Save image datasets
            os.makedirs(outputdir, exist_ok=True)
            if not isinstance(img, type(None)) and counter == 0: np.save(os.path.join(outputdir, f'input_{str(counter).zfill(4)}.npy'), img)
            if ground_truth: np.save(os.path.join(outputdir, f'groundtruth_{str(counter).zfill(4)}.npy'), yy)
            np.save(os.path.join(outputdir, f'prediction_{str(counter).zfill(4)}.npy'), ppred)

            # Counter
            print(f"Save image data {counter}...")
            
            if counter == index:
                break
            
            counter += 1
