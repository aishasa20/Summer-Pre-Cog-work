# Code to load the pilot data from mat file
import scipy.io
import glob
import numpy as np

# Brainvision Parameters
Fs = 1000
n_channels = 64

def load_pilot_data(folder):
    eeg_files = glob.glob(folder + '/*.mat')
    eeg_files.sort()

    eeg_data = []
    for file in eeg_files:
        dict_file = scipy.io.loadmat(file)
        eeg_data.append(dict_file['F'])

    return eeg_data

if __name__ == '__main__':
    folder = '../data/pilot_data'
    eeg_data = load_pilot_data(folder)
    print('EEG Data Shape: ', eeg_data.shape)
    