import os
import pandas as pd
import matplotlib.pyplot as plt

generate_csv = True
if generate_csv:
    for i in range(900):
        os.system("python3 kalmanTRAG_lineal.py")

headers = ["dX0", "dXP", "dY0", "dYP", "dT0", "dS0", "prob"]
df = pd.read_csv("saetas_file.csv", names=headers)

plt.close('all')

for hd in headers:
    plt.figure(hd)
    plt.title(hd)
    plt.ylabel(hd)
    hist_val = plt.hist(df[hd])
    plt.grid(which="major")
    plt.xlabel(f"{hd}")
