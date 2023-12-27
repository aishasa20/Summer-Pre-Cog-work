# Author : Aditya
# Date : 2023-07-07

import sklearn
import numpy as np
import pandas as pd

from scipy.signal import welch, spectrogram

def compute_power_bands(signal, freq_bands: dict, Fs: int=1000):
    f, welch_signal = welch(signal, fs = Fs, axis=1, nperseg = 512)
    signal_power = []
    
    for band in freq_bands:
        low_freq, high_freq = freq_bands[band]
        band_indices = np.where((f >= low_freq) & (f <= high_freq))
        band_power = np.mean(welch_signal[:, band_indices], axis=2).reshape(-1)
        signal_power.append(band_power)
    
    signal_power = np.array(signal_power).T
    
    return signal_power

def feature_extraction(signal, feature_type: str = "psd", freq_bands: dict = None):
    """ Function to extract features from the signal """
    if feature_type == "psd":
        # Compute the PSD
        f, psd = welch(signal, fs=512, axis=1)
        return psd
    elif feature_type == "spectrogram":
        f, t, Sxx = spectrogram(signal, fs=512, axis=1, nperseg = 64, noverlap = 32)
        return Sxx
    elif feature_type == "power_bands":
        return compute_power_bands(signal, freq_bands)

if __name__ == "__main__":
    # Sanity Check for the power bands
    freq_bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 45)}
    signal = np.random.randn(128, 512)
    signal_power = feature_extraction(signal, "power_bands", freq_bands)
    print('Signal Power Shape: ', signal_power.shape)