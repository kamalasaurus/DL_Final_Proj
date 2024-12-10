import torch
import torch.nn as nn
from copy import deepcopy

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
        # x: (B, C, H, W)
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
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, prev_state, prev_action):
        x = torch.cat([prev_state, prev_action], dim=-1)
        return self.fc(x)

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
        """
        B, T, C, H, W = states.shape

        # Predict future states in latent space
        predicted_states_list = []
        target_states_list = []

        for t in range(1, T):
            prev_state = self.online_encoder(states[:, t - 1])  # (B, C, H, W)
            curr_state = self.target_encoder(states[:, t])      # (B, C, H, W)

            prev_action = actions[:, t - 1]                     # (B, 2)
            predicted_state = self.predictor(prev_state, prev_action)

            predicted_states_list.append(predicted_state)
            target_states_list.append(curr_state)

        predicted_states = torch.stack(predicted_states_list, dim=1)  # (B, T-1, D)
        target_next_states = torch.stack(target_states_list, dim=1)   # (B, T-1, D)

        # Store representation for downstream tasks
        self.repr = self.online_encoder(states[:, 0])  # Initial state representation (B, D)

        return predicted_states, target_next_states
