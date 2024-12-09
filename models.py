from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super().__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = 256

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        return torch.randn((self.bs, self.n_steps, self.repr_dim)).to(self.device)

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        
        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size 
        )
        
    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)

        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()

        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(batch_size, seq_len, embed_dim)

        output = self.out_proj(attention_output)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attention_output = self.attention(x)
        x = self.norm1(x + self.dropout(attention_output))

        mlp_output = self.mlp(x)
        x = self.norm2(x + self.dropout(mlp_output))

        return x

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, num_heads, mlp_dim, num_layers, dropout=0.1):
        super(VisionTransformer, self).__init__()

        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)

        num_patches = (image_size // patch_size) ** 2

        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embedding
        x = self.dropout(x)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        return x

class JEPAEncoder(torch.nn.Module):

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super().__init__()
        self.device = device
        self.bs = bs 
        self.n_steps = n_steps 
        self.repr_dim = output_dim 

        self.encoder = VisionTransformer(
            image_size=65,
            patch_size=13,
            in_channels=2,
            embed_dim=output_dim,
            num_heads=4,
            mlp_dim=512,
            num_layers=4
        )

    def forward(self, states):
        B, T, C, H, W = states.size()
        states = states.reshape(B * T, C, H, W)  
        embeddings = self.encoder(states)  
        num_patches = embeddings.size(1)
        patch_dim = int(num_patches ** 0.5)
        embeddings = embeddings.view(B, T, -1, patch_dim, patch_dim)  # [B, T, embed_dim, H', W']
        
        return embeddings

class RecurrentJEPAPredictor(nn.Module):
    def __init__(self, in_channels=2, embed_dim=256, mlp_dim=512, cnn_channels=64):
        super().__init__()
        self.action_mlp = nn.Sequential(
           nn.Linear(in_channels, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim)
        )

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim + embed_dim, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=cnn_channels, out_channels=embed_dim, kernel_size=3, padding=1)  
        )

    def forward(self, initial_embedding, actions):
        print("initial_embedding:", initial_embedding.shape)
        B, T, C, H, W = initial_embedding.size()  # [B, T, embed_dim, H, W]
        T_minus_1 = actions.size(1)

        current_embedding = initial_embedding[:, 0]  # [B, embed_dim, H, W]
        predicted_embeddings = [current_embedding] 

        actions = actions.view(-1, actions.size(-1))  # Flatten to [B * T_minus_1, action_dim]
        actions = self.action_mlp(actions).reshape(B, T_minus_1, -1, 1, 1)  # Output [B, T_minus_1, embed_dim, 1, 1]

        actions = actions.expand(-1, -1, -1, H, W)  # Shape: [B, T_minus_1, embed_dim, H, W]

        for t in range(T_minus_1):
            action = actions[:, t]  
            input_to_cnn = torch.cat((current_embedding, action), dim=1)  #Shape: [B, embed_dim + embed_dim, H, W]
            print("Before CNN:", input_to_cnn.shape)
            current_embedding = self.cnn(input_to_cnn)  #[B, embed_dim, H, W]
            print("After CNN:", current_embedding.shape)
            predicted_embeddings.append(current_embedding)

        predicted_embeddings = torch.stack(predicted_embeddings, dim=1)  #[B, T, embed_dim, H, W]
        predicted_embeddings = predicted_embeddings[:,1:]

        return predicted_embeddings.view(B, T, -1) 


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output
