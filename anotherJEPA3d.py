import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

#########################
# Dataset and Dataloader
#########################

class TrajectoryDataset(Dataset):
    def __init__(self, states_path, actions_path):
        self.states = np.load(states_path)  # shape (N, T, 2, 64, 64)
        self.actions = np.load(actions_path) # shape (N, T-1, 2)
        
        self.states = torch.tensor(self.states, dtype=torch.float32)
        self.actions = torch.tensor(self.actions, dtype=torch.float32)
        
    def __len__(self):
        return self.states.shape[0]
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

#########################
# Augmentation Function
#########################

def augment_actions_and_states_fixed_length(states, actions, new_len):
    """
    Augment trajectories by resampling their length to a fixed `new_len`.
    Keeps the initial and final state fixed, and linearly interpolates intermediate states.
    Then recomputes actions from these interpolated states.
    """
    B, T, C, H, W = states.shape
    device = states.device
    
    y_coords = torch.linspace(0, H-1, H, device=device)
    x_coords = torch.linspace(0, W-1, W, device=device)
    X, Y = torch.meshgrid(x_coords, y_coords, indexing='xy')
    X, Y = X.to(device), Y.to(device)
    
    agent_map = states[:, :, 1, :, :] 
    agent_map = agent_map / (agent_map.sum(dim=(-1, -2), keepdim=True) + 1e-8)
    agent_x = (agent_map * X.unsqueeze(0).unsqueeze(0)).sum(dim=(-1, -2))
    agent_y = (agent_map * Y.unsqueeze(0).unsqueeze(0)).sum(dim=(-1, -2))

    augmented_state_list = []
    augmented_action_list = []
    
    sigma = 1.5
    
    for b in range(B):
        start_x, start_y = agent_x[b, 0], agent_y[b, 0]
        end_x, end_y = agent_x[b, -1], agent_y[b, -1]

        interp_x = torch.linspace(start_x, end_x, new_len, device=device)
        interp_y = torch.linspace(start_y, end_y, new_len, device=device)
        
        background = states[b, 0, 0, :, :].unsqueeze(0).repeat(new_len, 1, 1)
        
        y_grid = Y.unsqueeze(0).repeat(new_len, 1, 1)
        x_grid = X.unsqueeze(0).repeat(new_len, 1, 1)
        agent_stack = torch.exp(-((y_grid - interp_y.view(-1,1,1))**2 + (x_grid - interp_x.view(-1,1,1))**2)/(2*sigma*sigma))
        agent_stack = agent_stack / (agent_stack.sum(dim=(-1,-2), keepdim=True)+1e-8)
        
        new_states = torch.stack([background, agent_stack], dim=1)
        
        new_actions = torch.stack([interp_x[1:] - interp_x[:-1], interp_y[1:] - interp_y[:-1]], dim=-1)
        
        augmented_state_list.append(new_states)
        augmented_action_list.append(new_actions)
    
    augmented_states = torch.stack(augmented_state_list, dim=0)
    augmented_actions = torch.stack(augmented_action_list, dim=0)

    return augmented_states, augmented_actions

#########################
# Model Components
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

def variance_regularization(latents, epsilon=1e-4):
    var = torch.var(latents, dim=0)
    return torch.mean(torch.clamp(epsilon - var, min=0))

def covariance_regularization(latents):
    latents = latents - latents.mean(dim=0)
    latents = latents.view(latents.size(0), -1)
    cov = torch.mm(latents.T, latents) / (latents.size(0) - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return torch.sum(off_diag ** 2)

def normalize_latents(latents):
    return latents / (torch.norm(latents, dim=-1, keepdim=True) + 1e-8)

def contrastive_loss(predicted_states, target_states, temperature=0.1):
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
        online_states = self.online_encoder(states)  # (B, T, D)
        with torch.no_grad():
            target_states = self.target_encoder(states)  # (B, T, D)

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
# Collate Function for Augmentation
#########################

def augmentation_collate_fn(batch):
    states, actions = zip(*batch)
    states = torch.stack(states, dim=0)
    actions = torch.stack(actions, dim=0)

    # Choose a random new length for augmentation
    min_length, max_length = 5, 100
    new_len = np.random.randint(min_length, max_length+1)

    # Augment actions and states to this new length
    states, actions = augment_actions_and_states_fixed_length(states, actions, new_len)
    return states, actions

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
    batch_size = 32
    lr = 3e-4
    epochs = 10
    state_dim = 128
    action_dim = 2
    hidden_dim = 32
    initial_accumulation_steps = 4
    final_accumulation_steps = 4
    
    # Load data
    train_dataset = TrajectoryDataset("/Volumes/PhData2/DeepLearning/train/subset_states.npy", "/Volumes/PhData2/DeepLearning/train/subset_actions.npy")
    
    # Initially, do not use augmentation
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
        # After the first epoch, apply augmentation half of the time
        if epoch >= 1:
            if np.random.rand() > 0.5:
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=augmentation_collate_fn)
            else:
                # Use no augmentation
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

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
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
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

    # Plot the loss over time
    plt.figure()
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True)
    plt.savefig('training_loss.png')

    # Save the trained model
    torch.save(model.state_dict(), "/Volumes/PhData2/DeepLearning/trained_jepa.pth")