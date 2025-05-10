#powerspectrum.py
# Reference:  `https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/`
import numpy as np
import scipy.stats as stats
import cv2
import matplotlib.pyplot as plt
import os
import glob
from scipy.stats import gaussian_kde

def ps_con_method(image):
    # Helper function to compute radial average
    def radial_average(image, center=None):
        y, x = np.indices(image.shape)
        if center is None:
            center = np.array(image.shape) // 2
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(np.int64)
        # Accumulate values and counts for each radius
        radial_sum = np.bincount(r.ravel(), image.ravel())
        radial_count = np.bincount(r.ravel())
        
        # Avoid division by zero
        radial_count[radial_count == 0] = 1
        radial_avg = radial_sum / radial_count
        return radial_avg
    
    if image is None:
        raise ValueError("Image is None!")
    
    # Compute the 2D FFT
    fft_image = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_image)  # Shift the zero frequency to the center
    
    # Compute the power spectrum (magnitude squared of the FFT)
    power_spectrum = np.abs(fft_shifted) ** 2
    
    # Take the logarithm for better visualization
    log_power_spectrum = np.log1p(power_spectrum)
    
    # Compute the radial average of the power spectrum
    center = tuple(np.array(power_spectrum.shape) // 2)
    radial_avg = radial_average(power_spectrum, center=center)
    
    # Create wave number array
    wave_numbers = np.arange(len(radial_avg))
    return wave_numbers, radial_avg


def exec_powspec(images):
    results = []
    for img in images:
        # print(type(img), img.shape)
        r = ps_con_method(img)
        results.append(r)
        # print ("Error 3", np.asarray(results).shape)
    return np.median(np.asarray(results),axis=0)


# Directories
ground_truth_dir = "/lustre/orion/lrn036/world-shared/patrickfan/super-res-Daymet-V28km-4km/tutorial/checkpoints/test"
prediction_8M_dir = "/lustre/orion/lrn036/world-shared/patrickfan/super-res-Daymet-V28km-4km/tutorial/checkpoints/test"
prediction_117M_dir = "/lustre/orion/lrn036/world-shared/patrickfan/super-res-Daymet-V28km-4km-117M-Model/tutorial/checkpoints/test"


def load_all_npy_files(directory, file_prefix, max_files=None):
    file_list = sorted(glob.glob(os.path.join(directory, f"{file_prefix}_*.npy")))
    print (len(file_list))
    if max_files:  # Limit number of files for faster testing
        file_list = file_list[:max_files]
    data_list = []
    for file in file_list:
        data = np.load(file, mmap_mode='r') -273.15  # Memory-map mode for efficiency
        data_list.append(data)  # Reshape to 1D immediately
    return np.concatenate(data_list, dtype=np.float32) if data_list else None  # Use float32 to save memory

# Load datasets (limit number of files for speed if needed)
ground_truth_data = load_all_npy_files(ground_truth_dir, "groundtruth", max_files=200)[:200]
prediction_8M_data = load_all_npy_files(prediction_8M_dir, "prediction", max_files=200)[:200]
prediction_117M_data = load_all_npy_files(prediction_117M_dir, "prediction", max_files=200)[:200]

print ("Shape:", ground_truth_data.shape, prediction_8M_data.shape, prediction_117M_data.shape )

# Shape: (200, 480, 960) (200, 480, 960) (200, 480, 960)

# turth
PSs = {}
PSs['truth'] = exec_powspec(ground_truth_data)
PSs['tmin_2m_8M'] = exec_powspec(prediction_8M_data)
PSs['tmin_2m_117M'] = exec_powspec(prediction_117M_data)


# MEDIAN
fs=15
fss=14
plt.figure(figsize=(6,4))
lnames = {
    'truth': 'Daymet -7km [Truth]',
    'tmin_2m_8M': '9.5M - 7km', 
    'tmin_2m_117M': '126M - 7km'
}

plot_styles = {
    'truth': {
        'color': 'red', 
        'linestyle': '-', 
        'marker': 'o',
        'markersize': 1
    },
    'tmin_2m_8M': {
        'color': 'blue', 
        'linestyle': '--', 
        'marker': 's',
        'markersize': 1
    },
    'tmin_2m_117M': {
        'color': 'black', 
        'linestyle': ':', 
        'marker': '^',
        'markersize': 1
    }
}

for k, v in PSs.items():
    kvals, Abins = v
    # Use the plot styles from the dictionary
    plt.loglog(kvals[1:-10], Abins[1:-10], 
               label=lnames[k], 
               color=plot_styles[k]['color'],
               linestyle=plot_styles[k]['linestyle'],
               marker=plot_styles[k]['marker'],
               markersize=plot_styles[k]['markersize'],
               markevery=5)  # Plot a marker every 5 points to avoid overcrowding

plt.legend(fontsize=fs, loc='upper right')
plt.xlabel("Wavenumber", fontsize=fs)
plt.ylabel("Power spectrum", fontsize=fs)
plt.title("Tmin Power Spectrum Comparison", fontsize=fs)
plt.tick_params(axis='both', labelsize=fss)
plt.xlim(1, None)  # Start from 1 to avoid log(0)
plt.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.savefig("Power_spectrum.png", dpi=300)
plt.show()

