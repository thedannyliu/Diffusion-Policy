"""
Flow Matching / Rectified Flow-style Loss for Diffusion Policy.

Key differences from DDPM:
1. Time t is continuous in [0, 1] instead of discrete timesteps
2. We predict velocity (u_t = x_1 - x_0) instead of noise
3. Interpolation is linear: x_t = t*x_1 + (1-t)*x_0 (linear OT path)

References:
- Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
- Liu et al., "Rectified Flow: A Marginal Preserving Approach to Optimal Transport", 2023
"""

import torch
import torch.nn.functional as F
from typing import Optional


class FlowMatchingLoss:
    """
    Conditional Flow Matching / Rectified Flow-style loss for action sequences.

    Uses a linear optimal-transport path: straight line from Gaussian noise to data.
    This corresponds to the OT-CFM objective that enables few-step generation.
    """
    
    def __init__(self, sigma_min: float = 0.0):
        """
        Args:
            sigma_min: Minimum noise scale to avoid numerical issues at t=1.
                       Set to 0 for pure OT path (default).
        """
        self.sigma_min = sigma_min
    
    def __call__(
        self, 
        model: torch.nn.Module, 
        x_1: torch.Tensor, 
        global_cond: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute flow matching loss.
        
        Args:
            model: Network that predicts velocity v_θ(x_t, t, cond)
                   Expected signature: model(sample, timestep, global_cond=global_cond)
            x_1: Target data (normalized action sequence) [B, T, D]
            global_cond: Conditioning features [B, cond_dim]
            loss_mask: Optional mask for loss computation [B, T, D] (True = compute loss)
        
        Returns:
            loss: Scalar MSE loss
        """
        batch_size = x_1.shape[0]
        device = x_1.device
        
        # Sample t uniformly from [0, 1]
        t = torch.rand(batch_size, device=device)
        
        # Sample x_0 from standard Gaussian (same shape as x_1)
        x_0 = torch.randn_like(x_1)
        
        # Linear optimal transport interpolation: x_t = t * x_1 + (1 - t) * x_0
        # This creates a straight-line path from noise to data
        t_expand = t.view(-1, 1, 1)  # [B, 1, 1] for broadcasting
        x_t = t_expand * x_1 + (1 - t_expand) * x_0
        
        # Optional: add small noise at t≈1 for stability (usually not needed)
        if self.sigma_min > 0:
            x_t = x_t + self.sigma_min * torch.randn_like(x_t)
        
        # Target velocity: constant along the OT path
        # u_t = dx_t/dt = x_1 - x_0 (independent of t for linear interpolation)
        u_t = x_1 - x_0
        
        # Predict velocity using the same UNet architecture as DDPM
        # The model expects timestep in [0, 1] for FM or discrete steps for DDPM
        # We pass continuous t directly
        v_pred = model(x_t, t, global_cond=global_cond)
        
        # Compute MSE loss
        loss = F.mse_loss(v_pred, u_t, reduction='none')
        
        # Apply loss mask if provided (same masking logic as DDPM)
        if loss_mask is not None:
            loss = loss * loss_mask.float()
        
        # Reduce: mean over all dimensions
        loss = loss.mean()
        
        return loss


def compute_flow_matching_loss(
    model: torch.nn.Module,
    x_1: torch.Tensor,
    global_cond: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    sigma_min: float = 0.0
) -> torch.Tensor:
    """
    Functional interface for Flow Matching loss.
    
    See FlowMatchingLoss class for details.
    """
    loss_fn = FlowMatchingLoss(sigma_min=sigma_min)
    return loss_fn(model, x_1, global_cond=global_cond, loss_mask=loss_mask)
