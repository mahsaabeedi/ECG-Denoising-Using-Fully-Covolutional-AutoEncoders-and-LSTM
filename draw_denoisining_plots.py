import matplotlib.pyplot as plt
import numpy as np
from keras.engine.saving import load_model
from sklearn.metrics import mean_squared_error

from data_metrics import signal_to_noise
from data_preprocess import reshape_into_samples, preprocess_data
from filter_bandpass import filter_bandpass
from filter_wavelet import waveletSmooth

np.random.seed(1)

real_dataset = np.load("real_dataset/dataset.npy")[0:1]

plt.style.use("seaborn")

plot_number = 1

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.1)

plt.rcParams["axes.grid"] = True
plt.rcParams["axes.edgecolor"] = ".0"
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams['axes.facecolor']='white'

def draw_plot(dataset, label, color='blue'):
    global plot_number
    plt.subplot(3, 4, plot_number)
    plot_number += 1
    plt.xlim(0, 1200)
    plt.ylim(-1.5, 1.5)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    frame.axes.spines["top"].set_visible(True)

    plt.plot(dataset, color=color,  label=None, linewidth=0.5)
    plt.legend(loc='upper right')


def draw_plot_with_stats(dataset, label):
    print(label, dataset.shape)
    print("MSE: %f" % mean_squared_error(dataset, original))
    print("STN: %f [dB]" % signal_to_noise(dataset, original))
    draw_plot(dataset, label)


model = load_model("networks tuned/network 0 pretraining 50 tuning")


for noise_level in [0.2515, 0.45245, 0.79715]:
    noised, original = preprocess_data(real_dataset, noise_level=noise_level)
    noised = noised[0:2400]
    original = original[0:2400]
    noised_samples = reshape_into_samples(noised, points_per_sample=600)
    print("STN: %f [dB]" % signal_to_noise(noised, original))
    draw_plot(noised, 'noised signal')

    output_dnn = model.predict(noised_samples).flatten()
    draw_plot(output_dnn, 'denoised signal (DNN)')

    output_bandpass = filter_bandpass(noised_samples, fs=512, low=0.05, high=50).flatten()
    draw_plot(output_bandpass, 'denoised signal (bandpass)')

    output_wavelet = waveletSmooth(noised_samples, level=4).flatten()
    draw_plot(output_wavelet, 'denoised signal (wavelet)')


plt.show()
