import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torchvision
import torch
from scipy.stats import rankdata
from tqdm import tqdm
from ..data.processing.era5_constants import VAR_TO_UNIT as ERA5_VAR_TO_UNIT
from ..data.processing.cmip6_constants import VAR_TO_UNIT as CMIP6_VAR_TO_UNIT

def min_max_normalize(data):
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)


def visualize_at_index(mm, dm, out_list, in_transform, out_transform,variable, src, device, index=0):

    lat, lon = dm.get_lat_lon()
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    out_channel = dm.out_vars.index(variable)
    in_channel = dm.in_vars.index(variable)

    history = mm.history

    print("out_channel",out_channel,"in_channel",in_channel,"history",history,flush=True)

    if src == "era5":
        variable_with_units = f"{variable} ({ERA5_VAR_TO_UNIT[variable]})"
    elif src == "cmip6":
        variable_with_units = f"{variable} ({CMIP6_VAR_TO_UNIT[variable]})"
    elif src == "prism":
        variable_with_units = f"{variable}"
    else:
        raise NotImplementedError(f"{src} is not a supported source")

    counter = 0
    adj_index = None
    for batch in dm.test_dataloader():
        x, y = batch[:2]
        batch_size = x.shape[0]
        if index in range(counter, counter + batch_size):
            adj_index = index - counter
            x = x.to(device)
            pred = mm.forward(x)
            break
        counter += batch_size



    if adj_index is None:
        raise RuntimeError("Given index could not be found")
    xx = x[adj_index]
    if dm.task == "continuous-forecasting":
        xx = xx[:, :-1]

    # Create animation/plot of the input sequence
    if history > 1:
        in_fig, in_ax = plt.subplots()
        in_ax.set_title(f"Input Sequence: {variable_with_units}")
        in_ax.set_xlabel("Longitude")
        in_ax.set_ylabel("Latitude")
        imgs = []
        for time_step in range(history):
            img = in_transform(xx[time_step])[in_channel].detach().cpu().numpy()
            if src == "era5":
                img = np.flip(img, 0)
            img = in_ax.imshow(img, cmap=plt.cm.coolwarm, animated=True, extent=extent)
            imgs.append([img])
        cax = in_fig.add_axes(
            [
                in_ax.get_position().x1 + 0.02,
                in_ax.get_position().y0,
                0.02,
                in_ax.get_position().y1 - in_ax.get_position().y0,
            ]
        )
        in_fig.colorbar(in_ax.get_images()[0], cax=cax)
        anim = animation.ArtistAnimation(in_fig, imgs, interval=1000, repeat_delay=2000)
        plt.close()
    else:

        print("xx.shape",xx.shape,"in_channel",in_channel,flush=True)

        if dm.task == "downscaling":
            temp = xx[in_channel]
            
            temp = temp.repeat(len(out_list),1,1)
 
            img = in_transform(temp)[0].detach().cpu().numpy()
        else:
            img = in_transform(xx[0])[in_channel].detach().cpu().numpy()
        if src == "era5":
            img = np.flip(img, 0)


        visualize_sample(img, extent, f"Input: {variable_with_units}")
        anim = None
        plt.show()
        plt.savefig('input.png')
        print("input shape",img.shape,flush=True)

    # Plot the prediction
    ppred = out_transform(pred[adj_index])

    ppred = ppred[out_channel].detach().cpu().numpy()
    if src == "era5":
        ppred = np.flip(ppred, 0)


    ppred_min = np.min(ppred)
    ppred_max = np.max(ppred)



    visualize_sample(ppred, extent, f"Prediction: {variable_with_units}",vmin=ppred_min,vmax=ppred_max)
    plt.show()
    plt.savefig('prediction.png')

    print("prediction ppred.shape",ppred.shape,"extent",flush=True)



    # Plot the ground truth
    yy = out_transform(y[adj_index])
    yy = yy[out_channel].detach().cpu().numpy()
    if src == "era5":
        yy = np.flip(yy, 0)

    print("ground truth yy.shape",yy.shape,"extent",extent,flush=True)



    if yy.shape[0]!=ppred.shape[0] or yy.shape[1]!=ppred.shape[1]:
        yy= yy[0:ppred.shape[0],0:ppred.shape[1]]


    visualize_sample(yy, extent, f"Ground truth: {variable_with_units}",vmin=ppred_min,vmax=ppred_max)
    plt.show()
    plt.savefig('groundtruth.png')



    # Plot the bias
    bias = ppred - yy
    visualize_sample(bias, extent, f"Bias: {variable_with_units}")
    plt.show()

    # None, if no history
    return anim


def visualize_sample(img, extent, title,vmin=-1,vmax=-1):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cmap = plt.cm.coolwarm
    cmap.set_bad("black", 1)
    if vmin!=-1 and vmax!=-1:
        ax.imshow(img, cmap=cmap, extent=extent,vmin=vmin,vmax=vmax)
    else:
        ax.imshow(img, cmap=cmap, extent=extent)

    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.02,
            ax.get_position().y0,
            0.02,
            ax.get_position().y1 - ax.get_position().y0,
        ]
    )
    fig.colorbar(ax.get_images()[0], cax=cax)
    return (fig, ax)


def visualize_mean_bias(dm, mm, out_transform, variable, src):
    lat, lon = dm.get_lat_lon()
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    channel = dm.out_vars.index(variable)
    if src == "era5":
        variable_with_units = f"{variable} ({ERA5_VAR_TO_UNIT[variable]})"
    elif src == "cmip6":
        variable_with_units = f"{variable} ({CMIP6_VAR_TO_UNIT[variable]})"
    elif src == "prism":
        variable_with_units = f"Daily Max Temperature (C)"
    else:
        raise NotImplementedError(f"{src} is not a supported source")

    all_biases = []
    for batch in tqdm(dm.test_dataloader()):
        x, y = batch[:2]
        x = x.to(mm.device)
        y = y.to(mm.device)
        pred = mm.forward(x)
        pred = out_transform(pred)[:, channel].detach().cpu().numpy()
        obs = out_transform(y)[:, channel].detach().cpu().numpy()
        bias = pred - obs
        all_biases.append(bias)

    fig, ax = plt.subplots()
    all_biases = np.concatenate(all_biases)
    mean_bias = np.mean(all_biases, axis=0)
    if src == "era5":
        mean_bias = np.flip(mean_bias, 0)
    ax.imshow(mean_bias, cmap=plt.cm.coolwarm, extent=extent)
    ax.set_title(f"Mean Bias: {variable_with_units}")

    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.02,
            ax.get_position().y0,
            0.02,
            ax.get_position().y1 - ax.get_position().y0,
        ]
    )
    fig.colorbar(ax.get_images()[0], cax=cax)
    plt.show()


# based on https://github.com/oliverangelil/rankhistogram/tree/master
def rank_histogram(obs, ensemble, channel):
    obs = obs.numpy()[:, channel]
    ensemble = ensemble.numpy()[:, :, channel]
    combined = np.vstack((obs[np.newaxis], ensemble))
    ranks = np.apply_along_axis(lambda x: rankdata(x, method="min"), 0, combined)
    ties = np.sum(ranks[0] == ranks[1:], axis=0)
    ranks = ranks[0]
    tie = np.unique(ties)
    for i in range(1, len(tie)):
        idx = ranks[ties == tie[i]]
        ranks[ties == tie[i]] = [
            np.random.randint(idx[j], idx[j] + tie[i] + 1, tie[i])[0]
            for j in range(len(idx))
        ]
    hist = np.histogram(
        ranks, bins=np.linspace(0.5, combined.shape[0] + 0.5, combined.shape[0] + 1)
    )
    plt.bar(range(1, ensemble.shape[0] + 2), hist[0])
    plt.show()
