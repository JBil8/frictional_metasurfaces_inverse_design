import torch
import torch.nn as nn

class SurfaceInverseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_asp = config['physics']['n_asperities'] 
        self.max_delta = config['physics']['max_delta_ratio'] * config['physics']['radius']
        self.n_steps = config['data']['n_steps']
        
        # --- Encoder (Feature Extractor) ---
        self.conv_layers = nn.Sequential(
            # Block 1: Capture broad trends (Wider kernel, more channels)
            # Input: (B, 3, 500) -> Output: (B, 32, 250)
            nn.Conv1d(3, 32, kernel_size=7, padding=3, stride=2), 
            nn.BatchNorm1d(32),
            nn.GELU(),  
            
            # Block 2: Refine features
            # Input: (B, 32, 250) -> Output: (B, 64, 125)
            nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            # Block 3: Abstract features
            # Input: (B, 64, 125) -> Output: (B, 128, 63)
            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(128),
            nn.GELU(),

            # Block 4: High-level Logic (New Layer)
            # Input: (B, 128, 63) -> Output: (B, 256, 32)
            nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(256),
            nn.GELU()
        )
        
        # --- Dynamic Size Calculation ---
        dummy_input = torch.zeros(1, 3, self.n_steps)
        
        with torch.no_grad():
            dummy_out = self.conv_layers(dummy_input)
            self.flatten_size = dummy_out.view(1, -1).size(1)
        
        # --- Decoder ---
        hidden_dim = config['model']['hidden_dim']
        hidden_dim = max(hidden_dim, self.n_asp * 10) 
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Extra Layer for complexity
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.n_asp * 2) 
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.conv_layers(x)
        features = features.view(-1, self.flatten_size)
        raw_out = self.sigmoid(self.fc_layers(features))
        
        raw_n = raw_out[:, :self.n_asp]
        raw_h = raw_out[:, self.n_asp:]
        
        pred_exponents = 1.0 + raw_n * 7.0
        scaled_h = raw_h * self.max_delta
        pred_offsets, _ = torch.sort(scaled_h, dim=1)
        pred_offsets = pred_offsets - pred_offsets[:, 0:1]
        
        return pred_exponents, pred_offsets