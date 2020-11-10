import numpy as np
import matplotlib.pyplot as plt

ntracks = 1
filename = f"all_bins_{ntracks}_tracks.txt"
a = np.loadtxt(filename)

rate = []
for line in a.T:
    rate.append(line[ntracks] / np.sum(line))

rate = np.asarray(rate)
x_rate = np.arange(len(rate))

plt.figure(filename)
plt.plot(x_rate / 100, rate * 100)
plt.ylabel("Reconstructed SAETAs %")
plt.grid(axis="both")
plt.xlabel("Cut Value")
plt.savefig(f"{filename[:-4]}.png")
