import os
import numpy as np
import matplotlib.pyplot as plt

def post_processing(x, thres_mm_per_day=0.25):
    thres_log = np.log1p(thres_mm_per_day)
    x[ np.where(x <= thres_log) ] = 0
    return x

def prcp_special(truths, preds, vmin, vmax, varname='prcp', index=0):
    fig = plt.figure(figsize=(18,6))
    ax = plt.subplot(1,3,1)
    im1 = ax.imshow(truths[index], vmin=vmin, vmax=vmax, cmap=plt.cm.coolwarm)
    #ax.set_title('Truth [ERA5]')
    ax.set_title('Truth [Daymet]')
    ax = plt.subplot(1,3,2)
    im2 = ax.imshow(preds[index], vmin=vmin, vmax=vmax, cmap=plt.cm.coolwarm)
    ax.set_title('Pre-process only Pred \n [Era5 0.25deg ->Daymet 0.0625]')

    ax = plt.subplot(1,3,3)
    post_preds = post_processing(preds)
    im3 = ax.imshow(post_preds[index], vmin=vmin, vmax=vmax, cmap=plt.cm.coolwarm)
    ax.set_title('Pos-process Pred \n [Era5 0.25deg ->Daymet 0.0625]')

    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.04])
    # Add colorbar
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')

    plt.savefig(os.path.join(basedir, f'era5_daymet_{varname}_2021_finetune.png'))
    plt.show()
    plt.close()


epoch=35
loss='mse'
varname='prcp'
expname="3190616"
basedir=f"/lustre/orion/cli138/proj-shared/kurihana/super-res-torchlight/tutorial/checkpoints/climate/{loss}/{expname}/test"

ginput = os.path.join(basedir, 'groundtruth_0000.npy') # (32, 720, 1440)
pinput = os.path.join(basedir, 'prediction_0000.npy')

truths = np.load(ginput)
preds = np.load(pinput)


index = 0
vmax = np.nanmax(truths[index])
vmin = np.nanmin(truths[index])

#if varname == 'prcp':
#    print('Prcp special plotting')
#    prcp_special(truths, preds, vmin, vmax, varname, index=index)
#    exit(0)

# plt
fig = plt.figure(figsize=(12,6))
ax = plt.subplot(1,2,1)
im1 = ax.imshow(truths[index], vmin=vmin, vmax=vmax, cmap=plt.cm.coolwarm)
#ax.set_title('Truth [ERA5]')
ax.set_title('Truth [Daymet]')
ax = plt.subplot(1,2,2)
im2 = ax.imshow(preds[index], vmin=vmin, vmax=vmax, cmap=plt.cm.coolwarm)
ax.set_title('Pred [Era5 0.25deg ->Daymet 0.0625]')
#ax.set_title('Pred [Era5->Daymet]')

cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.04])
# Add colorbar
cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')

plt.savefig(os.path.join(basedir, f'era5_daymet_{varname}_2021_finetune_epoch{epoch}.png'))
plt.show()
plt.close()