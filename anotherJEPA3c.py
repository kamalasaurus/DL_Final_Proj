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
        
        # No need to convert here yet; we can do it on-the-fly when loading into the device.
        self.states = torch.tensor(self.states, dtype=torch.float32)
        self.actions = torch.tensor(self.actions, dtype=torch.float32)
        
    def __len__(self):
        return self.states.shape[0]
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

#########################
# Augmentation Functions
#########################

def normalize_latents(latents):
    # latents should already be bfloat16 after autocast, but safe to do as is
    return latents / (torch.norm(latents, dim=-1, keepdim=True) + 1e-8)

def augment_actions_and_states_fixed_length(states, actions, new_len):
    B, T, C, H, W = states.shape
    device = states.device
    dtype = states.dtype
    
    y_coords = torch.linspace(0, H-1, H, device=device, dtype=dtype)
    x_coords = torch.linspace(0, W-1, W, device=device, dtype=dtype)
    X, Y = torch.meshgrid(x_coords, y_coords, indexing='xy')
    X, Y = X.to(device, dtype=dtype), Y.to(device, dtype=dtype)
    
    agent_map = states[:, :, 1, :, :] 
    agent_map = agent_map / (agent_map.sum(dim=(-1, -2), keepdim=True) + torch.tensor(1e-8, device=device, dtype=dtype))
    agent_x = (agent_map * X.unsqueeze(0).unsqueeze(0)).sum(dim=(-1, -2))
    agent_y = (agent_map * Y.unsqueeze(0).unsqueeze(0)).sum(dim=(-1, -2))

    augmented_state_list = []
    augmented_action_list = []
    
    sigma = torch.tensor(1.5, dtype=dtype, device=device)
    
    for b in range(B):
        start_x, start_y = agent_x[b, 0], agent_y[b, 0]
        end_x, end_y = agent_x[b, -1], agent_y[b, -1]

        interp_x = torch.linspace(start_x, end_x, new_len, device=device, dtype=dtype)
        interp_y = torch.linspace(start_y, end_y, new_len, device=device, dtype=dtype)
        
        background = states[b, 0, 0, :, :].unsqueeze(0).repeat(new_len, 1, 1)
        
        y_grid = Y.unsqueeze(0).repeat(new_len, 1, 1)
        x_grid = X.unsqueeze(0).repeat(new_len, 1, 1)
        dist_sq = (y_grid - interp_y.view(-1,1,1))**2 + (x_grid - interp_x.view(-1,1,1))**2
        agent_stack = torch.exp(-dist_sq/(2*(sigma**2)))
        agent_stack = agent_stack / (agent_stack.sum(dim=(-1,-2), keepdim=True)+torch.tensor(1e-8, device=device, dtype=dtype))
        
        new_states = torch.stack([background, agent_stack], dim=1)
        
        new_actions = torch.stack([interp_x[1:] - interp_x[:-1], interp_y[1:] - interp_y[:-1]], dim=-1)
        
        augmented_state_list.append(new_states)
        augmented_action_list.append(new_actions)
    
    augmented_states = torch.stack(augmented_state_list, dim=0)
    augmented_actions = torch.stack(augmented_action_list, dim=0)

    return augmented_states, augmented_actions

def augment_walls_and_agent(states, actions):
    if np.random.rand() > 0.5:
        states = torch.flip(states, dims=[4])  # Flip along width dimension
        actions[:,:,0] = -actions[:,:,0]
    return states, actions

#########################
# Model Components
#########################

class Encoder(nn.Module):
    def __init__(self, in_channels=2, state_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )
        self.fc = nn.Linear(128 * 8 * 8, state_dim)

    def forward(self, x):
        if x.ndimension() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)  
            h = self.conv(x)
            h = h.view(h.size(0), -1)
            s = self.fc(h)
            s = s.view(B, T, -1)
        else:
            h = self.conv(x)
            h = h.view(h.size(0), -1)
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

def contrastive_loss(predicted_states, target_states, temperature=0.1):
    predicted_states = normalize_latents(predicted_states.reshape(-1, predicted_states.size(-1)))
    target_states = normalize_latents(target_states.reshape(-1, target_states.size(-1)))

    temp = torch.tensor(temperature, dtype=predicted_states.dtype, device=predicted_states.device)
    logits = torch.mm(predicted_states, target_states.T) / temp
    labels = torch.arange(logits.size(0), device=predicted_states.device).long()
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

class JEPA(nn.Module):
    def __init__(self, state_dim=128, action_dim=2, hidden_dim=128, ema_rate=0.99):
        super().__init__()
        self.repr_dim = state_dim

        self.online_encoder = Encoder(in_channels=2, state_dim=state_dim)
        self.target_encoder = deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = Predictor(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.ema_rate = ema_rate

    @torch.no_grad()
    def update_target_encoder(self):
        for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = self.ema_rate * target_params.data + (1 - self.ema_rate) * online_params.data

    def forward(self, states, actions):
        online_states = self.online_encoder(states)
        with torch.no_grad():
            target_states = self.target_encoder(states)

        predicted_states_list = []
        for t in range(1, states.shape[1]):
            prev_state = online_states[:, t - 1]
            prev_action = actions[:, t - 1]
            predicted_state = self.predictor(prev_state, prev_action)
            predicted_states_list.append(predicted_state)

        predicted_states = torch.stack(predicted_states_list, dim=1)
        target_next_states = target_states[:, 1:]
        return predicted_states, target_next_states, online_states

#########################
# Collate_fn with fixed length
#########################

def augmentation_collate_fn(batch):
    states, actions = zip(*batch)
    states = torch.stack(states, dim=0)
    actions = torch.stack(actions, dim=0)

    min_length, max_length = 5, 100
    new_len = np.random.randint(min_length, max_length+1)

    states, actions = augment_actions_and_states_fixed_length(states, actions, new_len)
    states, actions = augment_walls_and_agent(states, actions)
    return states, actions

#########################
# Training Loop
#########################

if __name__ == "__main__":
    device = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )

    batch_size = 32
    lr = 1e-4
    epochs = 10
    state_dim = 128
    action_dim = 2
    hidden_dim = 32
    initial_accumulation_steps = 4
    final_accumulation_steps = 4

    train_dataset = TrajectoryDataset("/Volumes/PhData2/DeepLearning/train/subset_states.npy",
                                      "/Volumes/PhData2/DeepLearning/train/subset_actions.npy")
    # First epoch without augmentation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = JEPA(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, ema_rate=0.99).to(device=device)
    # Convert model to bfloat16
    model = model.to(dtype=torch.bfloat16)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    criterion = nn.MSELoss().to(dtype=torch.bfloat16, device=device)

    loss_history = []

    model.train()
    for epoch in range(epochs):
        if epoch >= 1:
            # After the first epoch, use augmented dataloader
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=augmentation_collate_fn)

        total_loss = 0.0
        optimizer.zero_grad()
        
        accumulation_steps = max(final_accumulation_steps, initial_accumulation_steps - (initial_accumulation_steps - final_accumulation_steps) * epoch // epochs)
        
        for step, (states, actions) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            # Move data to device with bfloat16
            states = states.to(device, dtype=torch.bfloat16)
            actions = actions.to(device, dtype=torch.bfloat16)
            
            # Autocast block for bfloat16
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                predicted_states, target_states, _ = model(states, actions)
            
                mse_loss = criterion(predicted_states, target_states)
                contrast_loss_val = contrastive_loss(predicted_states, target_states)
                loss = mse_loss + 0.1 * contrast_loss_val
            
            loss.backward()
            
            if (step + 1) % accumulation_steps == 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                with torch.no_grad():
                    model.update_target_encoder()

                total_loss += loss.item()
                loss_history.append(loss.item())
                print(f"loss {loss.item()}, norm {norm:.4f}")
        
        avg_loss = total_loss / (len(train_loader) / accumulation_steps)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    plt.figure()
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True)
    plt.savefig('training_loss.png')

    torch.save(model.state_dict(), "/Volumes/PhData2/DeepLearning/trained_jepa.pth")