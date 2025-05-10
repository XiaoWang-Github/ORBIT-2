import os
import glob
import numpy as np
from datetime import datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader


def logtransform(x, m2mm_per_day=False, clip_threshold=0.25):
    if m2mm_per_day:
        x = x * 1000  # Convert from m/day to mm/day

    x = np.nan_to_num(x, nan=0.0)  # Replace NaNs with 0
    x = np.where(x < clip_threshold, 0, x)  # Clip small values to 0
    
    return np.log1p(x)  # Apply log(x + 1)


def logtransform1(x, m2mm_per_day=False, clip_threshold=0.25):
    if m2mm_per_day:
        x = x * 1000  # Convert from m/day to mm/day

    x = np.nan_to_num(x, nan=0.0)  # Replace NaNs with 0
    x = np.where(x < clip_threshold, 0, x)  # Clip small values to 0
    
    return x


# Load the data and apply log transform
def load_data(directory, prefix, m2mm=False, ext=".npy", clip_threshold=0.25):
    files = sorted(glob.glob(os.path.join(directory, f"{prefix}*{ext}")))
    data = [logtransform(np.load(file), m2mm_per_day=m2mm, clip_threshold=clip_threshold) for file in files]
    return np.array(data)

def load_data1(directory, prefix="preds_day_", m2mm=False, ext=".npy", clip_threshold=0.25):
    pattern = os.path.join(directory, f"{prefix}*{ext}")
    files = sorted(glob.glob(pattern), key=lambda x: int(x.split("_day_")[1].split(".")[0]))

    data = [
        logtransform1(np.load(file), m2mm_per_day=m2mm, clip_threshold=clip_threshold)
        for file in files
    ]
    return np.array(data)


era5_data = load_data1("ERA5_117M")  #ERA5 data does not require log transform
imerg_data = load_data("Imerg_daily_precip", "interpolated_imerg_", m2mm=False)  #IMERG data require log transform


from scipy.ndimage import zoom
land_sea_mask = np.load("land_sea_mask.npy")  # shape: (720, 1440)
upsampled_mask = zoom(land_sea_mask, (4, 4), order=0)  # -> (2880, 5760)

# Broadcast the mask to match (30, 2880, 5760)
mask_broadcasted = np.repeat(upsampled_mask[np.newaxis, :, :], era5_data.shape[0], axis=0)
era5_data[mask_broadcasted == 0] = 0
imerg_data[mask_broadcasted == 0] = 0


def load_data_ERA5_28km(directory, prefix="ERA5_28km_", m2mm=True, ext=".npy", clip_threshold=0.5):
    pattern = os.path.join(directory, f"{prefix}*{ext}")
    files = sorted(glob.glob(pattern), key=lambda x: int(os.path.basename(x).replace(prefix, "").replace(ext, "")))

    data = [
        logtransform(np.load(file), m2mm_per_day=m2mm, clip_threshold=clip_threshold)
        for file in files
    ]
    return np.array(data)


era5_28km = load_data_ERA5_28km("ERA5_28km")  #ERA5 data does not require log transform

land_sea_mask = np.load("land_sea_mask.npy")  # shape: (720, 1440)
mask_broadcasted_1 = np.repeat(land_sea_mask[np.newaxis, :, :], era5_28km.shape[0], axis=0)
era5_28km[mask_broadcasted_1 == 0] = 0


if len(era5_28km.shape) == 3:
    era5_28km = np.flip(era5_28km, 1)
else:  # 2D case
    era5_28km = np.flip(era5_28km, 0)


if len(era5_data.shape) == 3:
    era5_data = np.flip(era5_data, 1)
else:  # 2D case
    era5_data = np.flip(era5_data, 0)


if len(imerg_data.shape) == 3:
    imerg_data = np.flip(imerg_data, 1)
else:  # 2D case
    imerg_data = np.flip(imerg_data, 0)


timestamps = [
    (datetime(2020, 7, 1) + timedelta(days=i)).strftime("%Y-%m-%d") 
    for i in range(31)
]

# Plot settings
ncols = 3  # Three subplots: ERA5, IMERG, Difference
nrows = 1
vmin, vmax = 0, 4  # Precipitation range for visualization
lonmin, lonmax = 0, 360
latmin, latmax = -90, 90

# Set up the figure and axes
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5),
                         subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
axes = axes.flatten()

# --- Custom colormap for ERA5 and IMERG ---
colors = [
    (1, 1, 1),          # White
    (0.85, 0.85, 0.85), # Light Grey
    (0.7, 0.7, 0.7),    # Grey
    (0.6, 0.75, 0.6),   # Pale Green
    (0.5, 0.8, 0.5),    # Light Green
    (0.25, 0.9, 0.75),  # Greenish Cyan
    (0.0, 1.0, 1.0),    # Cyan
    (0.6, 0.8, 1.0),    # Light Blueish Purple
    (0.8, 0.6, 1.0),    # Light Purple
    (0.6, 0.3, 0.8),    # Medium Purple
    (0.4, 0.0, 0.4)     # Dark Purple
]
custom_cmap = LinearSegmentedColormap.from_list("custom_precip", colors, N=256)

# Diverging colormap for difference
diff_cmap = matplotlib.colormaps.get_cmap('RdBu_r')

def area(ax, iso, clr):
    shp = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
    reader = shpreader.Reader(shp)
    for n in reader.records():
        if n.attributes['ADM0_A3'] == iso:
            ax.add_geometries(n.geometry, ccrs.PlateCarree(), facecolor=clr, 
                              alpha=1.00, linewidth=0.15, edgecolor="black",
                              label=n.attributes['ADM0_A3'])
    return ax

colorbars = []


def update(frame):
    global colorbars  # access the list

    # Clear previous colorbars
    for cb in colorbars:
        cb.remove()
    colorbars = []

    era5_frame = era5_data[frame]
    imerg_frame = imerg_data[frame]
    era5_28km_frame = era5_28km[frame]
    dt = timestamps[frame]

    # 28km ERA5
    ax = axes[0]
    im0 = ax.imshow(era5_28km_frame, vmin=0, vmax=4.5, cmap=custom_cmap, transform=ccrs.PlateCarree(),
                    extent=[lonmin, lonmax, latmin, latmax])
    ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.LAND, color='lightgrey')
    ax.set_title(f"28km ERA5 {dt}")
    cb0 = fig.colorbar(im0, ax=axes[0], orientation='horizontal', fraction=0.046, pad=0.04)
    colorbars.append(cb0)
    cb0.set_label("$log(x+1) [mm/day]$")


    # === IMERG ===
    ax = axes[1]
    im1 = ax.imshow(imerg_frame, vmin=0, vmax=4.5, cmap=custom_cmap, transform=ccrs.PlateCarree(),
                    extent=[lonmin, lonmax, latmin, latmax])
    ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.LAND, color='lightgrey')
    ax.set_title(f"7km IMERG {dt}")
    cb1 = fig.colorbar(im1, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
    colorbars.append(cb1)
    cb1.set_label("$log(x+1)$ [mm/day]")

    # === ERA5 ===
    ax = axes[2]
    im2 = ax.imshow(era5_frame, vmin=0, vmax=4.5, cmap=custom_cmap, transform=ccrs.PlateCarree(),
                    extent=[lonmin, lonmax, latmin, latmax])
    ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.LAND, color='lightgrey')
    ax.set_title(f"7km ORBIT-2 {dt}")
    cb2 = fig.colorbar(im2, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
    colorbars.append(cb2)
    cb2.set_label("$log(x+1)$ [mm/day]")

    plt.tight_layout()


# Create the animation
anim = animation.FuncAnimation(fig, update, frames=range(len(era5_data)), interval=1000)

# Save the animation as a GIF (or any other format)
anim.save('./precipitation_comparison_animation.gif', writer='imagemagick', fps=1)

