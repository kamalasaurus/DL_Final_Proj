from anotherJEPA3 import TrajectoryDataset, flip_and_shift_augmentation
import random
import torch
import PIL
import matplotlib.pyplot as plt

import numpy as np


# Load the dataset
dataset = TrajectoryDataset('/scratch/DL24FA/train/states.npy', '/scratch/DL24FA/train/actions.npy')

# Collect means for each datapoint
means = []

# Iterate through the dataset
for i in range(len(dataset)):
    states, _ = dataset[i]
    # Calculate the mean of the states tensor
    mean = states.mean().item()
    means.append(mean)

# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(means, bins=30, edgecolor='black')
plt.title('Histogram of State Tensor Means')
plt.xlabel('Mean Value')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('state_means_histogram.png')
plt.show()

# Print some additional statistics
print("Mean of means:", np.mean(means))
print("Median of means:", np.median(means))
print("Standard deviation of means:", np.std(means))