import numpy as np


def add_noise(dataset, noise_level=0.5):
    noised = []
    for sequence in dataset:
        noised.append(sequence + np.random.normal(0, noise_level, sequence.shape))
    noised = np.array(noised)
    return noised




def reshape_into_samples(dataset, points_per_sample=300):
    return dataset.reshape((-1, points_per_sample, 1))


def normalize(data):
    mean_val = np.mean(data)
    mean_array = np.full_like(data, mean_val)
    centered = (data - mean_array)
    max_val = max(abs(centered))
    return centered/ max_val


def preprocess_data(dataset, noise_level=0.5):
    noised = []
    original = []
    for sample in dataset:
        normalized_sample = normalize(sample)
        noised_sample = add_noise(normalized_sample, noise_level)
        original.append(normalized_sample)
        noised.append(noised_sample)


    noised = np.array(noised).flatten()
    original = np.array(original).flatten()
    return noised, original