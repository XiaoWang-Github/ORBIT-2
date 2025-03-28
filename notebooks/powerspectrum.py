#powerspectrum.py
# Reference:  `https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/`
import numpy as np
import scipy.stats as stats

def ps_bin_method(image):
    """ Compute power spectrum between bins of frequencies. Outputs are discreate values
    Reference:  `https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/`
    """

    assert len(image.shape) == 2 #'Image should be (H,W)'
    npixY, npixX = image.shape
    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)**2
    
    # freq for each axis
    kfreqY = np.fft.fftfreq(npixY) * npixY
    kfreqX = np.fft.fftfreq(npixX) * npixX
    kfreq2D = np.meshgrid(kfreqX, kfreqY)
    
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    
    kbins = np.arange(0.5, max(npixX, npixY)//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    
    # comute between bins
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                     statistic = "mean",
                                     bins = kbins)
    
    # get only real numbers and the difference
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    
    # return (freq, power spectrum)
    return kvals, Abins

#test
#ps_bin_method(np.ones( (128,128)))

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
        radial_avg = radial_sum / radial_count
        return radial_avg

    # Load a rectangular image (grayscale)
    #image_path = "path_to_your_image.jpg"  # Replace with your image path
    #image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Image not found or invalid format!")

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