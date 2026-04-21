import torch
import torch.nn as nn

class SurfaceInverseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_asp = config['physics']['n_asperities']
        self.max_delta = config['physics']['max_delta_ratio'] * config['physics']['radius']
        self.n_steps = config['data']['n_steps']
        
        # --- INPUT DIMENSION CALCULATION ---
        # array input (Batch, 2 channels, n_steps) -> alpha_hat, stiff_hat
        # scalar input (Batch, 2) -> log10(P_max), log10(Alpha_max)
        self.array_dim = 2 * self.n_steps
        self.scalar_dim = 2
        
        # Total dimension entering the first Linear layer
        self.input_dim = self.array_dim + self.scalar_dim 
        
        # We need a wider initial capacity to handle the raw vector
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

    def forward(self, x_arrays, x_scalars):
        # Flatten the spatial and channel dimensions instantly: 
        # Shape goes from (Batch, 2, n_steps) -> (Batch, 1024)
        x_flat = x_arrays.view(x_arrays.size(0), -1) 
        
        # Concatenate the normalized arrays with the physical magnitude scalars
        # Shape goes to (Batch, 1026)
        x_combined = torch.cat([x_flat, x_scalars], dim=1)
        
        # 3. Pass through the rigid dense network
        raw_out = self.sigmoid(self.net(x_combined))

        # --- The Physics Scaling ---
        raw_n = raw_out[:, :self.n_asp]
        raw_gaps = raw_out[:, self.n_asp:] 

        pred_exponents = 1.0 + raw_n * 2.0 
        
        scaled_gaps = raw_gaps * (2.0 * self.max_delta / (self.n_asp - 1))
        
        final_gaps = scaled_gaps.clone()
        final_gaps[:, 0] = 0.0 
        pred_offsets = torch.cumsum(final_gaps, dim=1)

        return pred_exponents, pred_offsets