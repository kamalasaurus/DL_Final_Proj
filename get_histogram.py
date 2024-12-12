from anotherJEPA3 import TrajectoryDataset, flip_and_shift_augmentation
import random
import torch
import PIL
import matplotlib.pyplot as plt

import numpy as np


# Load the dataset
dataset = TrajectoryDataset('/scratch/DL24FA/train/states.npy', '/scratch/DL24FA/train/actions.npy')

# Collect means for each datapoint
wall_position_means = []
door_position_means = []

# Iterate through the dataset
for i in range(len(dataset)):
    states, _ = dataset[i]
    # Calculate the mean of the states tensor
    wall_non_zeros = torch.nonzero(states[-1, 1, 0, 5:-5] != 0)
    wall_min = wall_non_zeros.min().item()
    wall_max = wall_non_zeros.max().item()
    wall_pos = int((wall_min+wall_max)/2)
    wall_position_means.append(wall_pos)

    door_zeros, _, _ = torch.nonzero(states[-1, 1, 5:-5, torch.nonzero(states[-1, 1, 0, 5:-5] != 0)+5] == 0, as_tuple=True)
    door_min = door_zeros.min().item()
    door_max = door_zeros.max().item()
    door_pos = int((door_min+door_max)/2)
    door_position_means.append(door_pos)

# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(wall_position_means, bins=30, edgecolor='black')
plt.title('Histogram of Wall positions')
plt.xlabel('Wall position')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('wall_position_histogram.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(door_position_means, bins=30, edgecolor='black')
plt.title('Histogram of door positions')
plt.xlabel('Door position')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('door_position_histogram.png')
plt.show()