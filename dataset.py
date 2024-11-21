from typing import NamedTuple, Optional
import torch
import numpy as np


class WallSample(NamedTuple):
    states: torch.Tensor
    locations: torch.Tensor
    actions: torch.Tensor


class WallDataset:
    def __init__(
        self,
        data_path,
        probing=False,
        device="cuda",
    ):
        self.device = device
        self.states = np.load(f"{data_path}/states.npy", mmap_mode="r")
        self.actions = np.load(f"{data_path}/actions.npy")

        if probing:
            self.locations = np.load(f"{data_path}/locations.npy")
        else:
            self.locations = None

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        states = torch.from_numpy(self.states[i]).float().to(self.device)
        actions = torch.from_numpy(self.actions[i]).float().to(self.device)

        if self.locations is not None:
            locations = torch.from_numpy(self.locations[i]).float().to(self.device)
        else:
            locations = torch.empty(0).to(self.device)

        return WallSample(states=states, locations=locations, actions=actions)

class WallDatasetWithSplits(WallDataset):
    "Assign indices to training dataset and generate splits for encoder training"
    def __init__(self, data_path, split, device="cuda"):
        super().__init__(data_path, device=device)
        num_samples = len(self.states)
        indices = np.arange(num_samples)

        train_idx, test_idx = train_test_split(indices, test_size=0.2) # 80% train, 20% val and test
        val_idx, test_idx = train_test_split(test_idx, test_size=0.5) # 10% val, 10% test

        if split == "train":
            self.indices = train_idx
        elif split == "val":
            self.indices = val_idx
        elif split == "test":
            self.indices = test_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return super().__getitem__(idx)

def create_wall_dataloader(
    data_path,
    probing=False,
    device="cuda",
    batch_size=64,
    train=True,
):
    ds = WallDataset(
        data_path=data_path,
        probing=probing,
        device=device,
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size,
        shuffle=train,
        drop_last=True,
        pin_memory=False,
    )

    return loader

def create_wall_encoder_dataloader(data_path, split, device="cuda", batch_size=64):
    "For Enconder Only"
    # Split: "train", "val", "test"

    encoder_ds = WallDatasetWithSplits(
        data_path, 
        split, 
        device=device,
    )
    
    encoder_loader = torch.utils.data.DataLoader(
        encoder_ds, 
        batch_size=batch_size, 
        shuffle=(split == "train"), 
        drop_last=True, 
        pin_memory=False,
    )
    
    return encoder_loader
