import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import random


#########################
# Dataset and Dataloader
#########################


class TrajectoryDataset(Dataset):
    def __init__(self, states_path, actions_path, augmentations=None):
        """
        Args:
            states_path (str): Path to the states .npy file.
            actions_path (str): Path to the actions .npy file.
            augmentations (callable, optional): A function or transform to apply to the states and actions.
        """
        self.states = np.load(states_path, mmap_mode='r')
        self.actions = np.load(actions_path, mmap_mode='r')
        # self.states = torch.tensor(self.states, dtype=torch.float32)
        # self.actions = torch.tensor(self.actions, dtype=torch.float32)
        
        self.augmentations = augmentations

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        states = torch.tensor(self.states[idx], dtype=torch.float32)
        actions = torch.tensor(self.actions[idx], dtype=torch.float32)
        # states, actions = self.states[idx], self.actions[idx]
        
        # Apply augmentations if specified
        if self.augmentations:
            states, actions = self.augmentations(states, actions)
        
        return states, actions

# Example augmentation function
def flip_and_shift_augmentation(states, actions):
    """
    Example augmentation function for the TrajectoryDataset.
    Args:
        states (Tensor): Tensor of shape (T, 2, 64, 64).
        actions (Tensor): Tensor of shape (T-1, 2).
    
    Returns:
        Tuple[Tensor, Tensor]: Augmented states and actions.
    """
    # Random horizontal flip
    if random.random() > 0.5:
        states = torch.flip(states, dims=[-1])  # Flip along the width
        actions[:, 0] = -actions[:, 0]  # Invert x-axis action

    # Random vertical flip
    if random.random() > 0.5:
        states = torch.flip(states, dims=[-2])  # Flip along the height
        actions[:, 1] = -actions[:, 1]  # Invert y-axis action


    # Check for edges of the agent
    _, _, width_non_zeros = torch.nonzero((states[:, 0] != 0), as_tuple=True)
    width_min = width_non_zeros.min().item()
    width_max = width_non_zeros.max().item()

    # Check for edges of the walls
    wall_non_zeros = torch.nonzero(states[-1, 1, 0, 5:-5] != 0)
    wall_min = wall_non_zeros.min().item()
    wall_max = wall_non_zeros.max().item()

    # Identify range of the data (lowest and highest index where it is not empty space)
    global_min_all = min(width_min, width_max, wall_min, wall_max)
    global_max_all = max(width_min, width_max, wall_min, wall_max)


    # Randomly determine shift (without breaking out of the box)
    min_shift = 5 - global_min_all
    max_shift = 59 - global_max_all
    if min_shift is not max_shift+1 or min_shift is not max_shift:
        try:
            shift = torch.randint(min_shift, max_shift + 1, size=(1,))
        except:
            shift = 0
    else:
        shift = min_shift

    # print("shifting:", shift.item())

    # Shift left or right
    slice1 = states[:, :, :, 0:-shift]  # First part (before the shift)
    slice2 = states[:, :, :, -shift:]   # Second part (after the shift)

    shifted = torch.cat((slice2, slice1), dim=3)

    left_edge = states[:, :, :, 0:5]
    core = states[:, :, :, 5:-5]  # First part (before the shift)
    right_edge = states[:, :, :, -5:]

    wall_slice1 = core[:, :, :, 0:-shift]  # First part (before the shift)
    wall_slice2 = core[:, :, :, -shift:]   # Second part (after the shift)

    shifted_walls = torch.cat((left_edge, wall_slice2, wall_slice1, right_edge), dim=3)


    states[:, 0] = shifted[:, 0]
    states[:, 1] = shifted_walls[:, 1]

    return states, actions

#########################
# Model Components
#########################

#########################
# Encoder
#########################

class Encoder(nn.Module):
    def __init__(self, in_channels=2, state_dim=128):
        super().__init__()
        # Simple CNN encoder
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )
        # After downsampling 64x64 -> approximately 8x8 feature map
        self.fc = nn.Linear(128 * 8 * 8, state_dim)

    def forward(self, x):
        # x: (B, T, C, H, W) or (B, C, H, W)
        if x.ndimension() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)  # Flatten batch and sequence dims
            h = self.conv(x)
            h = h.view(h.size(0), -1)
            s = self.fc(h)
            s = s.view(B, T, -1)  # Restore batch and sequence dims
        else:
            h = self.conv(x)
            h = h.view(h.size(0), -1)  # Flatten for FC layer
            s = self.fc(h)
        return s

#########################
# Predictor
#########################

class Predictor(nn.Module):
    def __init__(self, state_dim=128, action_dim=2, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, prev_state, prev_action):
        x = torch.cat([prev_state, prev_action], dim=-1)
        return self.fc(x)

#########################
# Regularization Utilities
#########################

def variance_regularization(latents, epsilon=1e-4):
    var = torch.var(latents, dim=0)
    return torch.mean(torch.clamp(epsilon - var, min=0))

def covariance_regularization(latents):
    latents = latents - latents.mean(dim=0)
    latents = latents.view(latents.size(0), -1)  # Flatten all dimensions except the batch dimension
    cov = torch.mm(latents.T, latents) / (latents.size(0) - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return torch.sum(off_diag ** 2)

def normalize_latents(latents):
    return latents / (torch.norm(latents, dim=-1, keepdim=True) + 1e-8)

def contrastive_loss(predicted_states, target_states, temperature=0.1):
    """
    Compute contrastive loss between predicted and target states.
    Args:
        predicted_states: Tensor of shape (B, T-1, D)
        target_states: Tensor of shape (B, T-1, D)
        temperature: Temperature scaling factor for contrastive loss
    Returns:
        loss: Contrastive loss value
    """
    B, T_minus_1, D = predicted_states.shape
    predicted_states = predicted_states.reshape(-1, D)
    target_states = target_states.reshape(-1, D)

    # Normalize the embeddings
    predicted_states = normalize_latents(predicted_states)
    target_states = normalize_latents(target_states)

    # Compute similarity scores
    logits = torch.mm(predicted_states, target_states.T) / temperature
    labels = torch.arange(B * T_minus_1, device=predicted_states.device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

#########################
# JEPA Model
#########################

class JEPA(nn.Module):
    def __init__(self, state_dim=128, action_dim=2, hidden_dim=128, ema_rate=0.99):
        super().__init__()
        self.repr_dim = state_dim

        # Online encoder (learned)
        self.online_encoder = Encoder(in_channels=2, state_dim=state_dim)

        # Target encoder (EMA copy of online encoder)
        self.target_encoder = deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor
        self.predictor = Predictor(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)

        # EMA update rate
        self.ema_rate = ema_rate

    @torch.no_grad()
    def update_target_encoder(self):
        """Update target encoder using exponential moving average (EMA)."""
        for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = self.ema_rate * target_params.data + (1 - self.ema_rate) * online_params.data

    def forward(self, states, actions):
        """
        Args:
            states: Tensor of shape (B, T, 2, 64, 64)
            actions: Tensor of shape (B, T-1, 2)

        Returns:
            predicted_states: Predicted latent states (B, T-1, D)
            target_next_states: Target latent states (B, T-1, D)
            all_states: All latent states including the first online state (B, T, D)
        """
        # Encode all states in a batch
        online_states = self.online_encoder(states)  # (B, T, D)
        with torch.no_grad():
            target_states = self.target_encoder(states)  # (B, T, D)

        # Predict future states in latent space
        predicted_states_list = []
        for t in range(1, states.shape[1]):
            prev_state = online_states[:, t - 1]  # (B, D)
            prev_action = actions[:, t - 1]       # (B, 2)
            predicted_state = self.predictor(prev_state, prev_action)
            predicted_states_list.append(predicted_state)

        predicted_states = torch.stack(predicted_states_list, dim=1)  # (B, T-1, D)
        target_next_states = target_states[:, 1:]                     # (B, T-1, D)
        all_states = torch.cat((online_states[:, :1], predicted_states), dim=1)  # (B, T, D)

        return predicted_states, target_next_states, all_states


#########################
# Training Loop Example
#########################

if __name__ == "__main__":
    device = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )

    # Hyperparams
    batch_size = 64
    lr = 3e-4
    epochs = 20
    state_dim = 128
    action_dim = 2
    hidden_dim = 32
    initial_accumulation_steps = 4  # Initial number of steps to accumulate gradients
    final_accumulation_steps = 4    # Final number of steps to accumulate gradients
    
    # Load data
    train_dataset = TrajectoryDataset("/scratch/DL24FA/train/states.npy", "/scratch/DL24FA/train/actions.npy", augmentations=flip_and_shift_augmentation)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    model = JEPA(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, ema_rate=0.99).to(device)
    if device == 'cuda':
        model = torch.compile(model)

    torch.set_float32_matmul_precision('high')

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs, eta_min=lr*0.1)
    
    loss_history = []

    model.train()
    for epoch in range(epochs):
        print(epoch)
        total_loss = 0.0
        optimizer.zero_grad()
        
        # Calculate current accumulation steps based on the epoch
        accumulation_steps = max(final_accumulation_steps, initial_accumulation_steps - (initial_accumulation_steps - final_accumulation_steps) * epoch // epochs)
        
        for step, (states, actions) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            t0 = time.time()
            states = states.to(device)
            actions = actions.to(device)
            
            predicted_states, target_states, _ = model(states, actions)
            
            # Compute loss: distance between predicted and target embeddings
            with torch.autocast(device_type=device, dtype=torch.float16):
                mse_loss = criterion(predicted_states, target_states)

                # Add regularization terms
                mse_loss += 0.01 * variance_regularization(predicted_states)
                mse_loss += 0.01 * covariance_regularization(predicted_states)

                # Add contrastive loss
                contrast_loss = contrastive_loss(predicted_states, target_states)
                loss = mse_loss + contrast_loss

                # Normalize latents
                predicted_states = normalize_latents(predicted_states)
                target_states = normalize_latents(target_states)
            loss.backward()
            
            if (step + 1) % accumulation_steps == 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                # Update target encoder
                with torch.no_grad():
                    model.update_target_encoder()

                if device == 'mps':
                    torch.mps.synchronize()
                elif device == 'cuda':
                    torch.cuda.synchronize()

                t1 = time.time()
                dt = (t1 - t0) * 1000
                
                total_loss += loss.item()
                loss_history.append(loss.item())
                print(f"loss {loss.item()}, dt {dt:.2f}ms, norm {norm:.4f}")
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    print("finished")
    # Plot the loss over time
    plt.figure()
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True)
    plt.savefig('training_loss.png')
    # plt.show()

    # Save the trained model
    torch.save(model.state_dict(), "./trained_jepa.pth")
