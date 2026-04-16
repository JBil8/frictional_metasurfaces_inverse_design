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
        )

        # The Silver Bullet: Compress the remaining spatial dimension to 1
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        hidden_dim = min(config['model']['hidden_dim'], 512) 

        # --- Decoder ---
        self.fc_layers = nn.Sequential(
            # Input is now strictly 256, regardless of sequence length
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, self.n_asp * 2)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.conv_layers(x)
        features = self.global_pool(features)     # Shape: (Batch, 256, 1)
        features = features.view(features.size(0), -1) # Shape: (Batch, 256)
        
        raw_out = self.sigmoid(self.fc_layers(features))

        raw_n = raw_out[:, :self.n_asp]
        raw_gaps = raw_out[:, self.n_asp:] 

        pred_exponents = 1.0 + raw_n * 2.0 
        
        scaled_gaps = raw_gaps * (2.0 * self.max_delta / (self.n_asp - 1))
        
        final_gaps = scaled_gaps.clone()
        final_gaps[:, 0] = 0.0 
        pred_offsets = torch.cumsum(final_gaps, dim=1)

        return pred_exponents, pred_offsets