import torch
import os
import numpy as np
from ..data.processing.era5_constants import VAR_TO_UNIT as ERA5_VAR_TO_UNIT
from ..data.processing.cmip6_constants import VAR_TO_UNIT as CMIP6_VAR_TO_UNIT


def enable_dropout(model_module):
    for m in model_module.modules():
        if m._get_name() == "Dropout":
            m.train()


def get_monte_carlo_predictions(batch, model_module, n_ensemble_members):
    model_module.eval()
    enable_dropout(model_module)
    ensemble_predictions = []
    for _ in range(n_ensemble_members):
        with torch.no_grad():
            prediction = model_module.forward(batch)
        ensemble_predictions.append(prediction)
    ensemble_predictions = torch.stack(ensemble_predictions)
    return ensemble_predictions


def get_mcpreds_at_index(mm, dm, in_transform, out_transform, variable, src, index=0):
    print("reach here",flush=True)

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
    adj_index = None
    for batch in dm.test_dataloader():
        x, y = batch[:2]
        batch_size = x.shape[0]
        if index in range(counter, counter + batch_size):
            adj_index = index - counter
            x = x.to(mm.device)
            ensemble_predictions = get_monte_carlo_predictions(
                x, mm, 1000)
            break
        counter += batch_size
        print("counter",counter,"adj_index",adj_index,flush=True)

    if adj_index is None:
        raise RuntimeError("Given index could not be found")
    xx = x[adj_index]
    if dm.hparams.task == "continuous-forecasting":
        xx = xx[:, :-1]

    # Create animation/plot of the input sequence
    if history > 1:
        print("entering animation code block")
        # in_fig, in_ax = plt.subplots()
        # in_ax.set_title(f"Input Sequence: {variable_with_units}")
        # in_ax.set_xlabel("Longitude")
        # in_ax.set_ylabel("Latitude")
        # imgs = []
        # for time_step in range(history):
        #     img = in_transform(xx[time_step])[in_channel].detach().cpu().numpy()
        #     if src == "era5":
        #         img = np.flip(img, 0)
        #     img = in_ax.imshow(img, cmap=plt.cm.coolwarm, animated=True, extent=extent)
        #     imgs.append([img])
        # cax = in_fig.add_axes(
        #     [
        #         in_ax.get_position().x1 + 0.02,
        #         in_ax.get_position().y0,
        #         0.02,
        #         in_ax.get_position().y1 - in_ax.get_position().y0,
        #     ]
        # )
        # in_fig.colorbar(in_ax.get_images()[0], cax=cax)
        # anim = animation.ArtistAnimation(in_fig, imgs, interval=1000, repeat_delay=2000)
        # plt.close()
    else:
        print("xx.shape",xx.shape,"in_channel",in_channel,flush=True)

        if dm.hparams.task == "downscaling":
            img = in_transform(xx)[in_channel].detach().cpu().numpy()
        else:
            img = in_transform(xx[0])[in_channel].detach().cpu().numpy()
        if src == "era5":
            img = np.flip(img, 0)
        # visualize_sample(img, extent, f"Input: {variable_with_units}")
        # anim = None
        # plt.show()
        # plt.savefig('input.png')
        # print("input shape",img.shape,flush=True)

        np.save('x_in.npy',img)

    # Plot the ground truth
    yy = out_transform(y[adj_index])
    yy = yy[out_channel].detach().cpu().numpy()
    if src == "era5":
        yy = np.flip(yy, 0)

    print("ground truth yy.shape",yy.shape,"extent",extent,flush=True)

    yy_min = np.min(yy)
    yy_max = np.max(yy)

    # visualize_sample(yy, extent, f"Ground truth: {variable_with_units}",vmin=yy_min,vmax=yy_max)
    # plt.show()
    # plt.savefig('groundtruth.png')
    np.save("y_gt.npy", yy)

    # Plot the prediction
    ppred = out_transform(ensemble_predictions)
    ppred = ppred[:].detach().cpu().numpy()
    if src == "era5":
        ppred = np.flip(ppred, 3)
    np.save("y_preds.npy", ppred)

    return None


def mctest_on_image(
        mm,
        dm,
        in_variables,
        out_variables,
        in_transform,
        out_transform,
        variable,
        src,
        outputdir,
        device,
        index=-1,
        num_samples=100):
    """native_pytorch version """
    print("Start Inference",flush=True)

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

    mm.eval()
    enable_dropout(mm)
    ensemble_predictions = []

    counter = 0
    for batch in dm.test_dataloader():
        #FIXME select "second" index and then flip
        xx, y = batch[:2]
        batch_size = xx.shape[0]
        xx = xx.to(device)
        for ind in range(num_samples):
            pred = mm.forward(xx, in_variables,out_variables)

            # Plot the prediction``
            ppred = out_transform(pred)
            ppred = ppred[:, out_channel].detach().cpu().numpy()
            if src == "era5":
                if len(ppred.shape) == 2:
                    ppred = np.flip(ppred, 0)
                elif len(ppred.shape) == 3:
                    ppred = np.flip(ppred, 1)
            ensemble_predictions.append(ppred)

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
        #ppred = out_transform(pred)
        #ppred = ppred[:, out_channel].detach().cpu().numpy()
        #if src == "era5":
        #    if len(ppred.shape) == 2:
        #        ppred = np.flip(ppred, 0)
        #    elif len(ppred.shape) == 3:
        #        ppred = np.flip(ppred, 1)

        # Save image datasets
        os.makedirs(outputdir, exist_ok=True)
        if not isinstance(img, type(None)) and counter == 0: np.save(os.path.join(outputdir, f'input_{str(counter).zfill(4)}.npy'), img)
        np.save(os.path.join(outputdir, f'groundtruth_{str(counter).zfill(4)}.npy'), yy)
        np.save(os.path.join(outputdir, f'prediction_{str(counter).zfill(4)}.npy'), ppred)
        for ind in range(len(ensemble_predictions)):
            np.save(os.path.join(outputdir, f'predictions_{str(counter).zfill(4)}_pred{str(ind).zfill(4)}.npy'), ensemble_predictions[ind])
        # np.save(os.path.join(outputdir, f'predictions_{str(counter).zfill(4)}.npy'), np.array(ensemble_predictions))

        # Counter
        print(f"Save image data {counter}...")

        if counter == index:
            break

        counter += 1




