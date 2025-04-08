import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torchvision
import torch
import torch.distributed as dist
from scipy.stats import rankdata
from tqdm import tqdm
from ..data.processing.era5_constants import VAR_TO_UNIT as ERA5_VAR_TO_UNIT
from ..data.processing.cmip6_constants import VAR_TO_UNIT as CMIP6_VAR_TO_UNIT
from climate_learn.data.processing.era5_constants import (
    CONSTANTS
)
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def min_max_normalize(data):
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)


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



def visualize_at_index(mm, dm, dm_vis, out_list, in_transform, out_transform,variable, src, device, div, overlap,index=0, tensor_par_size=1,tensor_par_group=None):

    lat, lon = dm.get_lat_lon()
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    out_channel = dm.out_vars.index(variable)
    in_channel = dm.in_vars.index(variable)

    history = mm.history

    yout = len( lat )
    xout = len( lon )
    yinp = yout // 4
    xinp = xout // 4

    asets = div ** 2

    #hoverlap = overlap * 2
    #voverlap = overlap
    if overlap % 2 == 0:
        top = bottom = overlap // 2
        left = right = overlap // 2 * 2
    else:
        left = overlap // 2 * 2
        right = ( overlap // 2 + 1 ) * 2
        top = overlap // 2
        bottom = overlap // 2 + 1

    print('yout', yout, 'xout', xout, 'yinp', yinp, 'xinp', xinp, 'asets', asets, 'left', left, 'right', right, 'top', top, 'bottom', bottom )

    print("out_channel",out_channel,"in_channel",in_channel,"history",history,"src",src,"index",index,flush=True)

    if "ERA5" in src:
        variable_with_units = f"{variable} ({ERA5_VAR_TO_UNIT[variable]})"
    elif src == "CMIP6":
        variable_with_units = f"{variable} ({CMIP6_VAR_TO_UNIT[variable]})"
    elif src == "PRISM":
        variable_with_units = f"{variable}"
    elif "DAYMET" in src:
        variable_with_units = f"{variable}"

    else:
        raise NotImplementedError(f"{src} is not a supported source")

    hmul = xout // xinp
    vmul = yout // yinp

    groundtruths_arr = []
    predictions_arr = []
    psnr_arr = []
    ssim_arr = []

    counter = 0
    try:
        #for index in range(num_samples):
        for batch in dm_vis.test_dataloader():
            adj_index = None

            groundtruths = np.zeros((yout, xout),dtype=np.float32)
            preds = np.zeros((yout, xout),dtype=np.float32)
            inputs = np.zeros((yinp, xinp),dtype=np.float32)

            num_samples = div ** 2

            # Arrays to store inputs, predictions, and ground truths
            inputs_arr = []
            predictions_arr = []
            truths_arr = []

            # retrieve the image to test
            hit = False
            adj_index = 0
            #for batch in dm_vis.test_dataloader():
            x, y = batch[:2]
            in_variables = batch[2]
            out_variables = batch[3]

            batch_size = x.shape[0]
            # if index in range(counter, counter + batch_size):
            if index == counter:
                hit = True            
            counter += batch_size

            #yadj = y[ adj_index ]
            #y_adj = y

            for vindex in range(div):
                for hindex in range(div):
                    div_index = vindex * div + hindex
                    if div == 1:
                        xi1 = 0
                        xi2 = xinp
                        xo1 = 0
                        xo2 = xout
                        xi1t = 0
                        xi2t = xinp
                        xo1t = 0
                        xo2t = xout
                        xi1r = 0
                        xi2r = xinp
                        xo1r = 0
                        xo2r = xout
                    else:
                        xi1 = xinp // div * hindex
                        xi2 = xinp // div * (hindex+1)
                        xo1 = xout // div * hindex
                        xo2 = xout // div * (hindex+1)

                        xi1r = xinp // div * hindex
                        xi2r = xinp // div * (hindex+1)
                        xo1r = xout // div * hindex
                        xo2r = xout // div * (hindex+1)

                        if hindex == 0:
                            xi2 += left
                            xo2 += left * hmul
                        else:
                            xi1 -= left
                            xo1 -= left * hmul
                        if hindex == div - 1:
                            xi1 -= right
                            xo1 -= right * hmul
                        else:
                            xi2 += right
                            xo2 += right * hmul

                        if hindex == 0:
                            xi1t = 0
                            xi2t = xinp // div
                            xo1t = 0
                            xo2t = xout // div
                        elif hindex == div - 1:
                            xi1t = left + right #hoverlap * 2
                            xi2t = xi1t + xinp // div
                            xo1t = ( left + right ) * hmul #hoverlap * 2 * hmul
                            xo2t = xo1t + xout // div
                        else:
                            xi1t = left #hoverlap
                            xi2t = xi1t + xinp // div
                            xo1t = left * hmul #hoverlap * hmul
                            xo2t = xo1t + xout // div

                    if div == 1:
                        yi1 = 0
                        yi2 = yinp
                        yo1 = 0
                        yo2 = yout
                        yi1t = 0
                        yi2t = yinp
                        yo1t = 0
                        yo2t = yout
                        yi1r = 0
                        yi2r = yinp
                        yo1r = 0
                        yo2r = yout
                    else:
                        yi1 = yinp // div * vindex
                        yi2 = yinp // div * (vindex+1)
                        yo1 = yout // div * vindex
                        yo2 = yout // div * (vindex+1)

                        yi1r = yinp // div * vindex
                        yi2r = yinp // div * (vindex+1)
                        yo1r = yout // div * vindex
                        yo2r = yout // div * (vindex+1)

                        if vindex == 0:
                            yi2 += top
                            yo2 += top * vmul
                        else:
                            yi1 -= top
                            yo1 -= top ** vmul
                        if vindex == div - 1:
                            yi1 -= bottom
                            yo1 -= bottom * vmul
                        else:
                            yi2 += bottom
                            yo2 += bottom * vmul

                        if vindex == 0:
                            yi1t = 0
                            yi2t = yinp // div
                            yo1t = 0
                            yo2t = yout // div
                        elif vindex == div - 1:
                            yi1t = top + bottom #voverlap * 2
                            yi2t = yi1t + yinp // div
                            yo1t = ( top + bottom ) * vmul #voverlap * 2 * vmul
                            yo2t = yo1t + yout // div
                        else:
                            yi1t = top #voverlap
                            yi2t = yi1t + yinp // div
                            yo1t = top * vmul #voverlap * vmul
                            yo2t = yo1t + yout // div

                    xdiv = x[:,:,yi1:yi2,xi1:xi2]
                    ydiv = y[:,:,yo1:yo2,xo1:xo2]
                    ydiv = ydiv[adj_index]

                    #xdiv = x
                    #ydiv = yadj

                    print( 'xi1', xi1, 'xi2', xi2, 'yi1', yi1, 'yi2', yi2, 'xo1', xo1, 'xo2', xo2, 'yo1', yo1, 'yo2', yo2 )
                    print( 'xi1t', xi1t, 'xi2t', xi2t, 'yi1t', yi1t, 'yi2t', yi2t, 'xo1t', xo1t, 'xo2t', xo2t, 'yo1t', yo1t, 'yo2t', yo2t )
                    print( 'xi1r', xi1r, 'xi2r', xi2r, 'yi1r', yi1r, 'yi2r', yi2r, 'xo1r', xo1r, 'xo2r', xo2r, 'yo1r', yo1r, 'yo2r', yo2r )
                    print('hjy',xdiv.shape,ydiv.shape, x.shape, y.shape)

                    xdiv = xdiv.to(device)
                    pred = mm.forward(xdiv, in_variables, out_variables)
                    pred = clip_replace_constant(ydiv, pred, out_variables)
                    print("x.shape",x.shape,"y.shape",y.shape,"pred.shape",pred.shape,flush=True)

                    xx = xdiv[adj_index]
                    if dm.task == "continuous-forecasting":
                        xx = xx[:, :-1]

                    print(f"xx.shape {xx.shape}, in_channel {in_channel}", flush=True)

                    temp = xx[in_channel]
                    temp = temp.repeat(len(out_list), 1, 1)
                    img = in_transform(temp)[out_channel].detach().cpu().numpy()

                    if "ERA5" in src or src == "PRISM" or "DAYMET" in src:
                        img = np.flip(img, 0)
                        print('before',yi1t,yi2t,yinp//div+top+bottom)#voverlap*2)
                        yi2tp = yinp//div + ( top + bottom ) - yi1t # voverlap * 2 - yi1t
                        yi1tp = yinp//div + ( top + bottom ) - yi2t # voverlap * 2 - yi2t
                        yi1t = yi1tp
                        yi2t = yi2tp
                        print('after',yi1t,yi2t)
                        yi2rp = yinp - yi1r
                        yi1rp = yinp - yi2r
                        yi1r = yi1rp
                        yi2r = yi2rp


                    img_min = np.min(img)
                    img_max = np.max(img)

                    inputs[yi1r:yi2r, xi1r:xi2r] = img[yi1t:yi2t, xi1t:xi2t]
                    print(f"img.shape {img.shape}, min {img_min}, max {img_max}", flush=True)

                    ppred = out_transform(pred.squeeze(0))
                    ppred = ppred[out_channel].detach().cpu().numpy()
                    if "ERA5" in src or src == "PRISM" or "DAYMET" in src:
                        ppred = np.flip(ppred, 0)
                        print('before',yo1t,yo2t)
                        yo2tp = yout//div + ( top + bottom ) * vmul - yo1t # voverlap * vmul * 2 - yo1t
                        yo1tp = yout//div + ( top + bottom ) * vmul - yo2t # voverlap * vmul * 2 - yo2t
                        yo1t = yo1tp
                        yo2t = yo2tp
                        print('after',yo1t,yo2t)
                        yo2rp = yout - yo1r
                        yo1rp = yout - yo2r
                        yo1r = yo1rp
                        yo2r = yo2rp

                    preds[yo1r:yo2r, xo1r:xo2r] = ppred[yo1t:yo2t, xo1t:xo2t]


                    yy = out_transform(ydiv)
                    yy = yy[out_channel].detach().cpu().numpy()
                    if "ERA5" in src or src == "PRISM" or "DAYMET" in src:
                        yy = np.flip(yy, 0)

                    print(f"ground truth yy.shape {yy.shape}, extent {extent}", flush=True)

                    if yy.shape[0] != ppred.shape[0] or yy.shape[1] != ppred.shape[1]:
                        yy = yy[0 : ppred.shape[0], 0 : ppred.shape[1]]

                    groundtruths[yo1r:yo2r, xo1r:xo2r] = yy[yo1t:yo2t, xo1t:xo2t]
        
            ###
            if hit:
                img_min = np.min(inputs)
                img_max = np.max(inputs)

                plt.figure(figsize=(inputs.shape[1]/10,inputs.shape[0]/10))
                plt.imshow(inputs,cmap='coolwarm',vmin=img_min,vmax=img_max)
                plt.show()
                plt.savefig('input.png')

                print("img.shape",inputs.shape,"min",img_min,"max",img_max,flush=True)

            if hit:
                ppred_min = np.min(preds)
                ppred_max = np.max(preds)


                plt.figure(figsize=(preds.shape[1]/10,preds.shape[0]/10))
                plt.imshow(preds,cmap='coolwarm',vmin=img_min,vmax=img_max)
                plt.show()
                plt.savefig('prediction.png')

                print("ppred.shape",preds.shape,"min",ppred_min,"max",ppred_max,flush=True)


            if groundtruths.shape[0]!=preds.shape[0] or groundtruths.shape[1]!=preds.shape[1]:
                groundtruths= groundtruths[0:preds.shape[0],0:preds.shape[1]]

            if hit:
                print("ground truth yy.shape",yy.shape,"extent",extent,flush=True)

                plt.figure(figsize=(groundtruths.shape[1]/10,groundtruths.shape[0]/10))
                plt.imshow(groundtruths,cmap='coolwarm',vmin=img_min,vmax=img_max)
                plt.show()
                plt.savefig('groundtruth.png')

            groundtruths_arr.append(groundtruths)
            predictions_arr.append(preds)

            sr_array = preds
            hr_array = groundtruths

            psnr = peak_signal_noise_ratio(hr_array, sr_array, data_range=hr_array.max() - hr_array.min())
            ssim = structural_similarity(hr_array, sr_array, data_range=hr_array.max() - hr_array.min())

            psnr_arr.append(psnr)
            ssim_arr.append(ssim)

            print( f"Image: {counter}, Goodness of fit: PSNR {psnr} , SSIM {ssim}", flush= True )
    
    except Exception as e:
        import logging
        print("Error in visualization", flush=True)
        logging.error("An error occurred: %s", e)
        quit()


    print("Average PSNR",np.mean(psnr_arr), flush=True )
    print("Average SSIM",np.mean(ssim_arr), flush=True )

    groundtruths_arr = np.array(groundtruths_arr)
    predictions_arr = np.array(predictions_arr)

    np.save('groundtruths.npy', groundtruths_arr)
    np.save('predictions.npy', predictions_arr)

    anim = None
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

