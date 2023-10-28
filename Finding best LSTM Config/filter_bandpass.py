from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt
import numpy as np

from data_metrics import signal_to_noise
from data_preprocess import add_noise


def create_bandpass_filter(low, high, fs, order=6):
    nyq = 0.5 * fs
    normal_low  = low / nyq
    normal_high = high / nyq
    sos = butter(order,normal_high, btype='low', analog=False, output='sos')
    return sos


def filter_bandpass(data, low, high, fs, order=5):
    output = []
    data = data.squeeze()
    for sample in data:
        sos = create_bandpass_filter(low, high, fs, order=order)
        y = sosfiltfilt(sos, sample)
        output.append(y)
    y = sosfilt(sos, data)
    return np.array(output)

