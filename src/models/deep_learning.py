"""
Deep Learning Forecasting Models for FreshMind in PyTorch.
Provides PyTorch implementations of:
1. N-BEATS style block basis regression models
2. TFT-style sequence models with LSTM encoders and Self-Attention
3. PatchTST-style patch-based time-series transformer models
Includes complete training loops and self-contained data structures.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Union, List, Optional, Tuple

# Adjust path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.logging_utils import setup_logger

logger = setup_logger("deep_learning")

# Check device availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Deep learning models using device: {device}")

class SequenceDataset(Dataset):
    """
    Standard PyTorch Dataset for loading windowed time-series.
    """
    def __init__(self, y: np.ndarray, seq_len: int, horizon: int):
        self.seq_len = seq_len
        self.horizon = horizon
        self.X, self.y = self._create_windows(y)
        
    def _create_windows(self, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        X_list, y_list = [], []
        for i in range(len(y) - self.seq_len - self.horizon + 1):
            X_list.append(y[i : i + self.seq_len])
            y_list.append(y[i + self.seq_len : i + self.seq_len + self.horizon])
            
        if not X_list:
            # Fallback if series is too short
            X_list.append(np.zeros(self.seq_len))
            y_list.append(np.zeros(self.horizon))
            
        return torch.tensor(np.array(X_list), dtype=torch.float32), torch.tensor(np.array(y_list), dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ------------------ N-BEATS Style Architecture ------------------
class NBEATSBlock(nn.Module):
    """Simple N-BEATS fully-connected block mapping backcast and forecast."""
    def __init__(self, seq_len: int, horizon: int, hidden_dim: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.backcast_proj = nn.Linear(hidden_dim, seq_len)
        self.forecast_proj = nn.Linear(hidden_dim, horizon)
        
    def forward(self, x):
        h = self.fc(x)
        b = self.backcast_proj(h)
        f = self.forecast_proj(h)
        return b, f

class NBEATSNet(nn.Module):
    def __init__(self, seq_len: int, horizon: int, num_blocks: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.blocks = nn.ModuleList([NBEATSBlock(seq_len, horizon, hidden_dim) for _ in range(num_blocks)])
        
    def forward(self, x):
        residuals = x
        forecast = torch.zeros(x.shape[0], self.blocks[0].forecast_proj.out_features, device=x.device)
        
        for block in self.blocks:
            backcast, f = block(residuals)
            residuals = residuals - backcast
            forecast = forecast + f
            
        return forecast


# ------------------ TFT Style Architecture ------------------
class TFTNet(nn.Module):
    """
    Simplified Temporal Fusion Transformer.
    Uses LSTM encoders/decoders with Multi-Head Self-Attention.
    """
    def __init__(self, seq_len: int, horizon: int, hidden_dim: int = 32, num_heads: int = 2):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        
        # Encoder LSTM for historical context
        self.encoder_lstm = nn.LSTM(
            input_size=1, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        
        # Self-Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # Gated Linear Unit projection layer
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon)
        )
        
    def forward(self, x):
        # x shape: [batch, seq_len] -> convert to [batch, seq_len, 1] for LSTM
        x_lstm = x.unsqueeze(-1)
        lstm_out, _ = self.encoder_lstm(x_lstm)
        
        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Flatten and project to horizon
        attn_flat = attn_out.contiguous().view(x.shape[0], -1)
        forecast = self.proj(attn_flat)
        return forecast


# ------------------ PatchTST Style Architecture ------------------
class PatchTSTNet(nn.Module):
    """
    Simplified PatchTST model.
    Partitions the input sequence into overlapping patches, embeds them,
    and runs them through a Transformer Encoder before projecting to the forecast horizon.
    """
    def __init__(self, seq_len: int, horizon: int, patch_len: int = 8, stride: int = 4, hidden_dim: int = 32, num_layers: int = 1, num_heads: int = 2):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        
        # Calculate number of patches
        # N = (L - P)/S + 2 (with padding)
        self.num_patches = max(1, (seq_len - patch_len) // stride + 1)
        
        # Patch linear embedding
        self.patch_embed = nn.Linear(patch_len, hidden_dim)
        
        # Positional Encodings
        self.pos_encoder = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        
        # Transformer Encoder Block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            activation="relu",
            dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output prediction head
        self.head = nn.Linear(hidden_dim * self.num_patches, horizon)
        
    def forward(self, x):
        # x shape: [batch, seq_len]
        batch_size = x.shape[0]
        
        # Extract patches
        patches = []
        for i in range(self.num_patches):
            start = i * self.stride
            end = start + self.patch_len
            if end <= x.shape[1]:
                patches.append(x[:, start:end])
            else:
                # Padding if last patch overflows
                pad = torch.zeros(batch_size, end - x.shape[1], device=x.device)
                patches.append(torch.cat([x[:, start:], pad], dim=1))
                
        # Shape: [batch, num_patches, patch_len]
        patches = torch.stack(patches, dim=1)
        
        # Embed patches
        emb = self.patch_embed(patches) # [batch, num_patches, hidden_dim]
        emb = emb + self.pos_encoder
        
        # Run Transformer
        trans_out = self.transformer(emb) # [batch, num_patches, hidden_dim]
        
        # Flatten and project
        trans_flat = trans_out.contiguous().view(batch_size, -1)
        forecast = self.head(trans_flat)
        return forecast


# ------------------ PyTorch Training Wrapper ------------------
class PyTorchForecaster:
    """
    Standard Wrapper for training and predicting using PyTorch forecasting networks.
    """
    def __init__(self, model_type: str = "nbeats", seq_len: int = 14, epochs: int = 10, batch_size: int = 8, lr: float = 0.01):
        self.model_type = model_type.lower()
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.net_ = None
        self.fitted_ = False
        
    def fit(self, y: Union[np.ndarray, pd.Series], horizon: int = 14) -> 'PyTorchForecaster':
        y_arr = np.asarray(y, dtype=np.float32)
        if len(y_arr) < self.seq_len + horizon:
            logger.warning(f"Training history too short ({len(y_arr)}) for deep learning. Defaulting to Naive prediction.")
            self.fitted_ = False
            self.y_last_ = y_arr[-1]
            return self
            
        # Instantiate correct network
        if self.model_type == "nbeats":
            self.net_ = NBEATSNet(self.seq_len, horizon).to(device)
        elif self.model_type == "tft":
            self.net_ = TFTNet(self.seq_len, horizon).to(device)
        elif self.model_type == "patchtst":
            self.net_ = PatchTSTNet(self.seq_len, horizon).to(device)
        else:
            raise ValueError(f"Unknown deep learning model type: {self.model_type}")
            
        # Setup Data
        dataset = SequenceDataset(y_arr, self.seq_len, horizon)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Setup training parameters
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        self.net_.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                pred = self.net_(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_x)
                
            avg_loss = total_loss / len(dataset)
            if (epoch + 1) % max(1, self.epochs // 2) == 0:
                logger.info(f"[{self.model_type.upper()}] Epoch {epoch+1}/{self.epochs} - Training Loss: {avg_loss:.4f}")
                
        self.fitted_ = True
        self.y_last_ = y_arr[-self.seq_len:]
        return self

    def predict(self, horizon: int = 14) -> np.ndarray:
        if not self.fitted_ or self.net_ is None:
            # Fallback to Naive
            return np.full((horizon,), self.y_last_ if not isinstance(self.y_last_, np.ndarray) else self.y_last_[-1], dtype=np.float32)
            
        self.net_.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(self.y_last_, dtype=torch.float32).unsqueeze(0).to(device)
            pred = self.net_(x_tensor)
            forecast = pred.cpu().squeeze().numpy()
            
            # Floor at 0.0
            forecast = np.clip(forecast, a_min=0.0, a_max=None)
            
            if horizon == 1:
                return np.array([float(forecast)], dtype=np.float32)
            elif len(forecast.shape) == 0:
                # Handle single element outputs
                return np.full((horizon,), float(forecast), dtype=np.float32)
                
            # If forecast size doesn't match requested horizon, repeat or slice
            if len(forecast) < horizon:
                return np.pad(forecast, (0, horizon - len(forecast)), 'edge')
            else:
                return forecast[:horizon]
