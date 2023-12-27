# Code to compute the power from a given set of bands
from scipy.signal import welch
import numpy as np

def compute_power_bands(signal, freq_bands: dict, Fs: int=1000):
    n_channels = signal.shape[0]
    signal_power = []

    for channel in range(n_channels):
        channel_power = []
        f, welch_signal = welch(signal[channel, :], fs = Fs, nperseg = 256)
        for band in freq_bands:
            low_freq, high_freq = freq_bands[band]
            band_indices = np.where((f >= low_freq) & (f <= high_freq))
            band_power = np.mean(welch_signal[band_indices])
            channel_power.append(band_power)
        signal_power.append(channel_power)
    
    signal_power = np.array(signal_power)
    return signal_power

if __name__ == '__main__':
    # Sanity Check for the power bands
    freq_bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 45)}
    signal = np.random.randn(64, 1000)
    signal_power = compute_power_bands(signal, freq_bands)
    print('Signal Power Shape: ', signal_power.shape)
            