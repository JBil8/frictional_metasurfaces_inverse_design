import torch
import torch.nn as nn

class SurfaceInverseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_asp = config['physics']['n_asperities']
        self.max_delta = config['physics']['max_delta_ratio'] * config['physics']['radius']

        # --- Efficient Encoder ---
        self.conv_layers = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, padding=3, stride=2), 
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            
            # Global Pooling: Extracts the "presence" of features regardless of exact width
            nn.AdaptiveMaxPool1d(1) 
        )

        self.flatten_size = 256 
        hidden_dim = min(config['model']['hidden_dim'], 512) 

        # --- Lean Decoder ---
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, hidden_dim),
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
        raw_gaps = raw_out[:, self.n_asp:] 

        pred_exponents = 1.0 + raw_n * 2.0 
        scaled_gaps = raw_gaps * (1.2 * self.max_delta / (self.n_asp - 1))
        
        final_gaps = scaled_gaps.clone()
        final_gaps[:, 0] = 0.0 
        pred_offsets = torch.cumsum(final_gaps, dim=1)

        return pred_exponents, pred_offsets