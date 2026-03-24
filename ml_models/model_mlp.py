import torch
import torch.nn as nn

class SurfaceInverseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_asp = config['physics']['n_asperities']
        
        self.max_delta = config['physics']['max_delta_ratio'] * \
                         config['physics']['radius']
        self.n_steps = config['data']['n_steps']

        # --- "Spike-Preserving" Encoder ---
        self.conv_layers = nn.Sequential(
            # Layer 1: Look at the raw, high-res cliffs. NO STRIDE.
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # Preserves the sharpest spike

            # Layer 2: Extract local patterns. NO STRIDE.
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Layer 3 & 4: Safe to stride now that features are localized
            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(256),
            nn.GELU()
        )

        dummy_input = torch.zeros(1, 1, self.n_steps)
        with torch.no_grad():
            dummy_out = self.conv_layers(dummy_input)
            self.flatten_size = dummy_out.view(1, -1).size(1)

        hidden_dim = config['model']['hidden_dim']
        hidden_dim = max(hidden_dim, self.n_asp * 10)

        # Decoder is fine as-is
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

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
        raw_gaps = raw_out[:, self.n_asp:] 

        # CRITICAL FIX: Restore the correct physical bound for Exponents
        pred_exponents = 1.0 + raw_n * 2.0 
        
        # We give the gaps a tiny bit of breathing room (1.2x max_delta) 
        # so the sigmoid doesn't have to push exactly to 1.0 to reach the end.
        scaled_gaps = raw_gaps * (1.2 * self.max_delta / (self.n_asp - 1))
        
        final_gaps = scaled_gaps.clone()
        final_gaps[:, 0] = 0.0 
        
        pred_offsets = torch.cumsum(final_gaps, dim=1)

        return pred_exponents, pred_offsets