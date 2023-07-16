import matplotlib.pyplot as plt
import numpy as np

from data_preprocess import preprocess_data

np.random.seed(1)





plt.style.use("seaborn")

plot_number = 1

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.5)

plt.rcParams["axes.grid"] = True
plt.rcParams["axes.edgecolor"] = ".0"
plt.rcParams["axes.linewidth"] = 0.5

points = 3000


def draw_plot(dataset, label, color='blue'):
    global plot_number
    plt.subplot(4, 4, plot_number)
    plot_number += 1
    plt.xlim(0, points)
    plt.ylim(-1.5, 1.5)
    plt.xticks(np.arange(0,points, 1000))
    plt.plot(dataset, color=color, linewidth=0.5)

artificial_dataset = np.loadtxt("artificial_dataset/myfile.csv", delimiter=",")

for i in range(0,16):
    noised_artificial, original_artificial = preprocess_data(artificial_dataset[i:i+1], noise_level=0)
    draw_plot(original_artificial[0:points], "Artificial signal")


plt.show()
