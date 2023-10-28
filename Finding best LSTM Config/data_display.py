import numpy as np
from numpy import shape
import matplotlib.pyplot as plt

from data_preprocess import normalize

signal = np.loadtxt("artificialecg_dataset/myfile.csv", delimiter=",")
print(signal.shape[1])
 
for i in signal.shape[0]:
    signal[i] = normalize(signal[i]) * 0.5

real_data = np.load("real_dataset/dataset.npy")
print(shape(real_data))

#real_data = normalize(real_data)

plt.style.use("seaborn")

for i in range(100,101):
    plt.plot(signal[i])
    print(signal[i])
    print()

plt.ylabel("Voltage [mV]")
plt.xlabel("Time [steps]")

plt.ylim(-1,1)
plt.xlim(222,822)
plt.show()
