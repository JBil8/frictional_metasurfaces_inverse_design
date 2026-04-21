import torch
import torch.nn as nn

class SurfaceInverseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_asp = config['physics']['n_asperities']
        self.max_delta = config['physics']['max_delta_ratio'] * config['physics']['radius']
        self.n_steps = config['data']['n_steps']
        # The input is (Batch, 3 channels, n_steps spatial points)
        # We flatten this immediately to lock in absolute positioning
        self.input_dim = 3 * self.n_steps 
        
        # We need a wider initial capacity to handle the raw 1500-vector
        hidden_dim = min(config['model']['hidden_dim'], 1024) 

        # --- Pure Dense Encoder/Decoder ---
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.GELU(),
            
            nn.Linear(hidden_dim // 4, self.n_asp * 2)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Flatten the spatial and channel dimensions instantly: 
        # Shape goes from (Batch, 3, n_steps) -> (Batch, 1500)
        x_flat = x.view(x.size(0), -1) 
        
        raw_out = self.sigmoid(self.net(x_flat))

        # --- The Physics Scaling (Unchanged) ---
        raw_n = raw_out[:, :self.n_asp]
        raw_gaps = raw_out[:, self.n_asp:] 

        pred_exponents = 1.0 + raw_n * 2.0 
        
        scaled_gaps = raw_gaps * (2.0 * self.max_delta / (self.n_asp - 1))
        
        final_gaps = scaled_gaps.clone()
        final_gaps[:, 0] = 0.0 
        pred_offsets = torch.cumsum(final_gaps, dim=1)

        return pred_exponents, pred_offsets