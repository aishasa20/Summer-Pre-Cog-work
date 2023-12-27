# Author: Aditya
# Date: 2023-07-04

import torch.utils.data as data
import pandas as pd
import numpy as np
import torch
from scipy.signal import welch
from scipy.signal import welch, spectrogram


FREQ_BANDS={'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 45)}
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

class EEGDataset(data.Dataset):
    """ Class to create a PyTorch dataset of EEG data """

    def __init__(self, epoched_data_path: str, subjects: list, feature_type: str=None):
        """Dataset to load the proxy data"""
        self.epoched_data_path = epoched_data_path

        # Create the dataset
        self.X = []
        self.y = []

        epoched_data = {}

        for subject in subjects:
            epoched_data_file = "{}/{}_epoch_data.pkl".format(epoched_data_path, subject)
            epoched_data[subject] = pd.read_pickle(epoched_data_file)
            
            congruent_epochs = epoched_data[subject]["congruent_epochs"]
            incongruent_epochs = epoched_data[subject]["incongruent_epochs"]

            self.X.extend(congruent_epochs)
            self.X.extend(incongruent_epochs)   

            self.y.extend([1] * len(congruent_epochs))
            self.y.extend([0] * len(incongruent_epochs))
        
        self.X = np.array(self.X)
        # Exlude the last 9 channels
        self.X = self.X[:, :128, :]

        self.y = np.array(self.y)

        # Compute the features if feature type is not None
        if feature_type is not None:
            X_features = []
            for i in range(len(self.X)):
                X_features.append(feature_extraction(self.X[i], feature_type=feature_type, freq_bands=FREQ_BANDS))
            self.X = np.array(X_features)
        
        print("Dataset shape: {}".format(self.X.shape))

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        # Unsqueeze the data
        X = self.X[index]
        X = np.expand_dims(X, axis=0)
        
        y = self.y[index]
        # Convert to one hot
        y = np.eye(2)[y]
        return X, y

if __name__ == "__main__":
    epoched_data_path = "/media/data/PRECOG_Data/2022N400_Epoched/"
    subjects = ["sub-{:02d}".format(i) for i in range(1, 10) if i not in [5, 10, 15, 18]]

    dataset = EEGDataset(epoched_data_path, subjects=subjects)
    print(len(dataset))

    # Get one sample
    X, y = dataset[0]
    print(X.shape)
    print(y)