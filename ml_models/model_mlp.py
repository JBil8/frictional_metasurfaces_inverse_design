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
            # Block 1: Input channels = 3 (Load, Area, Stiffness)
            nn.Conv1d(3, 16, kernel_size=5, padding=2, stride=2), 
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            # Block 2
            nn.Conv1d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Block 3
            nn.Conv1d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
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