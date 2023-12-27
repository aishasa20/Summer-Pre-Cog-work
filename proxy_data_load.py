# Code to epoch the proxy data and extract the last word from the stimuli
import mne.io
import mne
import pandas as pd
import numpy as np

def preprocessing_proxy(eeg_signal):
    """
    Function to preprocess the proxy data
    """
    # Notch Filter at 60Hz
    eeg_signal = eeg_signal.copy().notch_filter(freqs=60, method='iir', picks='eeg')

    # Low Pass Filter at 40 Hz and high pass at 1Hz
    eeg_signal = eeg_signal.copy().filter(l_freq=0.5, h_freq=50, picks="eeg")

    # Eye Blink Correction
    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
    ica.fit(eeg_signal)

    eog_indices, eog_scores = ica.find_bads_eog(eeg_signal, ch_name="C17")
    ica.exclude = eog_indices

    eeg_signal = ica.apply(eeg_signal.copy())

    # DC offset correction
    eeg_signal = eeg_signal.apply_function(lambda x: x - np.mean(x, axis=1).reshape(-1, 1), channel_wise=False, picks="eeg")

    return eeg_signal

def epoch_data_subject(subject_number: str="sub-01", base_path: str=""):
    # Load the data
    eeg_path = "{}/{}/eeg/{}_task-N400Stimset_eeg.bdf".format(base_path, subject_number, subject_number)
    eeg_data = mne.io.read_raw_bdf(eeg_path, preload=True)
    ch_names = eeg_data.ch_names

    # preprocessing the data
    eeg_data = preprocessing_proxy(eeg_data)

    # Load the events file
    events_path = "{}/{}/eeg/{}_task-N400Stimset_events.tsv".format(base_path, subject_number, subject_number)
    events_data = pd.read_csv(events_path, sep="\t")

    # Sampling frequency of the EEG data
    freq = eeg_data.info["sfreq"]

    # Selecting the rows wherein onset and duration are not NaN
    events_data = events_data[events_data["onset"].notna() & events_data["duration"].notna()]

    # Congruent Events
    congruent_events = events_data[events_data["trial_type"] == "NPC"]

    # Incongruent Events
    incongruent_events = events_data[events_data["trial_type"] == "NPI"]

    # Epoching the EEG data using the congruent events onset time

    eeg_data_np = eeg_data.get_data()
    congruent_epochs = []

    # Looping through the congruent events
    for index, row in congruent_events.iterrows():
        # Extracting the onset time
        onset_time = row["onset"]
        # Get the EEG from -0.2 to 0.8 seconds
        start_time = onset_time - 0.2
        end_time = onset_time + 0.8
        # Get the start and end sample
        start_sample = int(start_time * freq)
        end_sample = int(end_time * freq)
        # Get the EEG data
        eeg_data = eeg_data_np[:, start_sample:end_sample]
        # Append the EEG data to the list
        congruent_epochs.append(eeg_data)
    
    # Epoching Incongruent epochs
    incongruent_epochs = []

    # Looping through the incongruent events
    for index, row in incongruent_events.iterrows():
        # Extracting the onset time
        onset_time = row["onset"]
        # Get the EEG from -0.2 to 0.8 seconds
        start_time = onset_time - 0.2
        end_time = onset_time + 0.8
        # Get the start and end sample
        start_sample = int(start_time * freq)
        end_sample = int(end_time * freq)
        # Get the EEG data
        eeg_data = eeg_data_np[:, start_sample:end_sample]
        # Append the EEG data to the list
        incongruent_epochs.append(eeg_data)

    # Dictionary to store the congruent, incongruent epochs and the channel order
    epoch_data = {
        "congruent_epochs": congruent_epochs,
        "incongruent_epochs": incongruent_epochs,
        "channel_order": ch_names
    }
    
    # Return the epoch data
    return epoch_data

def epoch_proxy_data(subjects: list=[], base_path: str="", destination_path: str=""):
    
    # Loop through the subjects
    for subject in subjects:
        # Get the epoch data for the subject
        epoch_data = epoch_data_subject(subject_number=subject, base_path=base_path)
        
        # Save the epoch data to the destination path
        destination_file = "{}/{}_epoch_data.pkl".format(destination_path, subject)
        pd.to_pickle(epoch_data, destination_file)

if __name__ == "__main__":
    subjects = []
    for i in range(1, 25):
        subjects.append("sub-{:02d}".format(i))
    
    epoch_proxy_data(subjects=subjects, base_path="/media/data/PRECOG_Data/2022N400_Jan/", destination_path="/media/data/PRECOG_Data/2022N400_Epoched/")
    # eeg_path = "/data/akommine/PRECOG_DATA/N400-EEG/sub-01/eeg/sub-01_task-N400Stimset_eeg.bdf"
    # eeg_data = mne.io.read_raw_bdf(eeg_path, preload=True)

    # preprocessed_data = preprocessing_proxy(eeg_data)
