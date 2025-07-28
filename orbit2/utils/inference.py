import os
import numpy as np
from scipy.stats import rankdata
from tqdm import tqdm
from ..data.processing.era5_constants import VAR_TO_UNIT as ERA5_VAR_TO_UNIT
from ..data.processing.cmip6_constants import VAR_TO_UNIT as CMIP6_VAR_TO_UNIT


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
        
def test_on_many_images(mm, dm, in_transform, out_transform, variable, src, outputdir, device, index=0):
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