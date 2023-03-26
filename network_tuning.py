from os import listdir

from keras.engine.saving import load_model
from keras.layers import *

from data_preprocess import preprocess_data, reshape_into_samples
import numpy as np

# Tunes up the network with real data

np.random.seed(1)

real_dataset = np.load("real_dataset/dataset.npy")

noised_train, original_train = preprocess_data(real_dataset[10:50])
noised_test, original_test = preprocess_data(real_dataset[0:10])

points_per_sample = 600

print(noised_train.shape)
noised_train = reshape_into_samples(noised_train, points_per_sample=points_per_sample)
print(noised_train.shape)
original_train = reshape_into_samples(original_train, points_per_sample=points_per_sample)

noised_test = reshape_into_samples(noised_test, points_per_sample=points_per_sample)
original_test = reshape_into_samples(original_test, points_per_sample=points_per_sample)

print("noised_train set: %s, original_train: %s" % (noised_train.shape, original_train.shape))

tuning_epochs = 20
network_dir = "network history"
f = "network.10.hdf5"

model = load_model("%s/%s" %(network_dir, f))

model.summary()
model.fit(noised_train, original_train, validation_data=(noised_test, original_test),
          epochs=tuning_epochs,
          batch_size=128, verbose=2)
model.save("networks tuned/network %s pretraining %s tuning" % (0, tuning_epochs))