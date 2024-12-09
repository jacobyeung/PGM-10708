import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq


def calculate_power_bands(signal, sampling_rate=1024, num_timepoints=4000):
    frequency_bands = {
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 100),
        "high_gamma": (100, 200)
    }
    
    fft_values = fft(signal, axis=1)
    freqs = fftfreq(num_timepoints, d=1/sampling_rate)
    
    positive_freqs = freqs[:num_timepoints // 2]
    positive_fft_values = np.abs(fft_values[:, :num_timepoints // 2]) ** 2

    power_bands = np.zeros((signal.shape[0], len(frequency_bands)))

    for i, (band_name, (low_freq, high_freq)) in enumerate(frequency_bands.items()):
        band_indices = np.where((positive_freqs >= low_freq) & (positive_freqs < high_freq))[0]
        power_bands[:, i] = np.mean(positive_fft_values[:, band_indices], axis=1)
    
    return power_bands



key_file_path = '/Users/yuxinguo/Desktop/PGM project/pgm_data/p8/p8_trials_key.npy'
stim_file_path = '/Users/yuxinguo/Desktop/PGM project/pgm_data/p8/p8_trials_stim.npy'
behave_file_path = '/Users/yuxinguo/Desktop/PGM project/pgm_data/p8/p8_behavior.csv'

# Load the data from key.npy and stim.npy into variables
key_data = np.load(key_file_path)
stim_data = np.load(stim_file_path)
behave_data = pd.read_csv(behave_file_path)
labels = behave_data.iloc[:,7].to_numpy()

#sEEG time series for brain regions 
start = 2500
end = 3000
left_amygdala_signal = stim_data[:,1:3,start:end]
left_amygdala_signal = np.mean(left_amygdala_signal, axis=1)
right_amygdala_signal = stim_data[:,92:95,start:end]
right_amygdala_signal = np.mean(right_amygdala_signal, axis=1)
left_hippocampus_signal = stim_data[:,10:12,start:end]
left_hippocampus_signal = np.mean(left_hippocampus_signal, axis=1)
right_hippocampus_signal = stim_data[:,102:105,start:end]
right_hippocampus_signal = np.mean(right_hippocampus_signal, axis=1)
#left posterior hippocampus

left_hippocampus_pos_signal = stim_data[:,20:23,start:end]
left_hippocampus_pos_signal = np.mean(left_hippocampus_pos_signal, axis=1)

#power in 5 freq bands
left_amygdala_power_bands = calculate_power_bands(left_amygdala_signal, num_timepoints=end-start)
right_amygdala_power_bands = calculate_power_bands(right_amygdala_signal, num_timepoints=end-start)
left_hippocampus_power_bands = calculate_power_bands(left_hippocampus_signal, num_timepoints=end-start)
right_hippocampus_power_bands = calculate_power_bands(right_hippocampus_signal, num_timepoints=end-start)
left_hippocampus_pos_power_bands = calculate_power_bands(left_hippocampus_pos_signal, num_timepoints=end-start)
print(left_hippocampus_pos_power_bands.shape)

np.savez('power_bands_data.npz',
         left_amygdala_power_bands=left_amygdala_power_bands,
         right_amygdala_power_bands=right_amygdala_power_bands,
         left_hippocampus_power_bands=left_hippocampus_power_bands,
         right_hippocampus_power_bands=right_hippocampus_power_bands,
         left_hippocampus_pos_power_bands=left_hippocampus_pos_power_bands,
         labels=labels)
