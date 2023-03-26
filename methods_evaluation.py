import numpy as np
from keras.engine.saving import load_model
from sklearn.metrics import mean_squared_error

from data_metrics import signal_to_noise
from data_preprocess import preprocess_data, reshape_into_samples
from filter_bandpass import filter_bandpass
from filter_wavelet import waveletSmooth

real_dataset = np.load("real_dataset/dataset.npy")[48:58]

points_per_sample = 600

directory = "networks tuned"
file = "network 0 pretraining 50 tuning"
model = load_model("%s/%s" % (directory, file))

for noise_level in [0.7]:
    noised, original = preprocess_data(real_dataset, noise_level)
    noised = reshape_into_samples(noised, points_per_sample)
    original = reshape_into_samples(original, points_per_sample)
    np.savetxt("original.csv", original.squeeze())
    np.savetxt("noised.csv", noised.squeeze())
    print(original.shape)
    output_dnn = model.predict(noised)
    output_wavelet = waveletSmooth(noised, level=4)
    output_bandpass = filter_bandpass(noised, fs=512, low=0.05, high=50).flatten()

    print("DNN")
    print("ORI: %f" % signal_to_noise(noised, original))
    print("STN: %f" % signal_to_noise(output_dnn, original))