import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter, defaultdict
from math import ceil

file_path = "/home/melassal/Workspace/Server files/mnist_2/3/mnist_2.csv"

information = pd.read_csv(file_path, header=None)

size = ceil(np.sqrt(pd.unique(information[0]).size))
fig, axs = plt.subplots(size, size)
l = 0
for neuron in information.groupby([0]):
    print("="*30)
    print("Neuron: " , neuron[0])
    labels = defaultdict()

    for label in neuron[1].groupby([1]):
        labels[label[0]] = label[1].groupby([1]).count().values[0][0]
        print("label ", label[0], ":", labels[label[0]])

    axs[l%size, l//size].bar(labels.keys(), labels.values())
    axs[l%size, l//size].set(xlim=(list(labels.keys())[0]-0.5, list(labels.keys())[-1]+0.5), ylim=(0, 300))
    axs[l%size, l//size].set_title(f"Features: {neuron[0]}")
    l += 1

plt.show()