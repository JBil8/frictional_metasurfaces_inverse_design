import torch

def batched_interp1d(x_target, x_pred, y_pred, pad_value=-1.0):
    """
    Fast, differentiable batched 1D interpolation.
    Args:
        x_target: (N_steps,) Fixed global grid (e.g., P* grid)
        x_pred: (Batch, N_steps) The x-values predicted by the physics engine
        y_pred: (Batch, N_steps) The y-values predicted by the physics engine
        pad_value: Value to assign when x_target > max(x_pred)
    Returns:
        y_interp: (Batch, N_steps) Interpolated y-values padded with pad_value
    """
    B, N_p = x_pred.shape
    
    # Expand target to match batch
    x_target_exp = x_target.unsqueeze(0).expand(B, -1)
    
    # Find insertion indices (Requires x_pred to be monotonically increasing)
    x_pred_contig = x_pred.contiguous()
    x_target_exp_contig = x_target_exp.contiguous()
    
    # Find insertion indices 
    idx = torch.searchsorted(x_pred_contig, x_target_exp_contig)
    
    # Clamp indices to avoid out-of-bounds errors during gather
    idx_lower = torch.clamp(idx - 1, min=0)
    idx_upper = torch.clamp(idx, max=N_p - 1)
    
    # Gather the bounding x and y values
    x_lower = torch.gather(x_pred, 1, idx_lower)
    x_upper = torch.gather(x_pred, 1, idx_upper)
    y_lower = torch.gather(y_pred, 1, idx_lower)
    y_upper = torch.gather(y_pred, 1, idx_upper)
    
    # Calculate interpolation weights
    dx = x_upper - x_lower
    # Avoid division by zero where x_lower == x_upper (e.g., flat regions)
    dx = torch.where(dx == 0, torch.ones_like(dx), dx)
    weight = (x_target_exp - x_lower) / dx
    
    # Linear interpolation
    y_interp = y_lower + weight * (y_upper - y_lower)
    
    # Apply out-of-bounds padding
    max_x_pred = x_pred[:, -1].unsqueeze(1)
    mask = x_target_exp > max_x_pred
    y_interp[mask] = pad_value
    
    return y_interp