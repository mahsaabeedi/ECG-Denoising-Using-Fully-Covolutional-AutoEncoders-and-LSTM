import matplotlib.pyplot as plt
import numpy as np

from data_preprocess import preprocess_data

np.random.seed(1)

artificial_dataset = np.loadtxt("artificialecg_dataset/myfile.csv", delimiter=",")[17:18]
noised_artificial, original_artificial = preprocess_data(artificial_dataset, noise_level=0)


plt.style.use("seaborn")

plot_number = 1

plt.rcParams["axes.grid"] = True
plt.rcParams["axes.edgecolor"] = ".0"
plt.rcParams["axes.linewidth"] = 0.5

points = 600


def draw_plot(dataset, label, color='blue'):
    global plot_number
    plt.subplot(1, 1, plot_number)
    plot_number += 1
    plt.xlim(0, points)
    plt.ylim(-1.5, 1.5)
    plt.xticks(np.arange(0,points, 100))
    plt.xlabel("Time [steps]")
    plt.ylabel("Voltage [mV]")
    plt.title(label)
    plt.plot(dataset, color=color, linewidth=0.5)
    plt.legend(loc='upper right')


draw_plot(original_artificial[200:200+points], "ECG cycle")


plt.show()
