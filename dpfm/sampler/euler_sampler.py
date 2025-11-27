"""
Euler ODE Sampler for Flow Matching.

Integrates the learned velocity field from t=0 (noise) to t=1 (data).
Unlike DDPM which requires 50-100 denoising steps, Flow Matching with
Euler integration can generate high-quality samples in just 1-8 steps.

Key insight: The OT-CFM objective learns a nearly straight velocity field,
so simple Euler integration is sufficient (no need for higher-order methods).
"""

import torch
import time
from typing import Optional, Tuple


class EulerSampler:
    """
    Euler method sampler for Flow Matching.
    
    Performs simple forward Euler integration of the learned velocity field
    to transform samples from the prior distribution (Gaussian) to the data
    distribution (actions).
    """
    
    def __init__(self, num_steps: int = 4):
        """
        Args:
            num_steps: Number of Euler integration steps (1, 2, 4, or 8).
                       More steps = higher quality but slower inference.
        """
        self.num_steps = num_steps
        self.dt = 1.0 / num_steps
    
    @torch.no_grad()
    def sample(
        self, 
        model: torch.nn.Module, 
        shape: Tuple[int, ...], 
        global_cond: Optional[torch.Tensor] = None, 
        device: str = 'cuda',
        return_latency: bool = True
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """
        Generate samples via Euler integration.
        
        The ODE we integrate is: dx/dt = v_θ(x, t, cond)
        Starting from x(0) ~ N(0, I), we integrate to x(1) ≈ data.
        
        Args:
            model: Velocity network v_θ(x, t, cond)
            shape: Output shape (B, T, D)
            global_cond: Conditioning [B, cond_dim]
            device: Compute device
            return_latency: Whether to measure and return latency
        
        Returns:
            x: Generated samples [B, T, D]
            latency_ms: Inference time in milliseconds (if return_latency=True)
        """
        if return_latency:
            # Ensure GPU synchronization for accurate timing
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
        
        # Initialize from Gaussian noise: x(0) ~ N(0, I)
        x = torch.randn(shape, device=device)
        
        # Euler integration from t=0 to t=1
        # x(t + dt) = x(t) + v_θ(x(t), t) * dt
        for step in range(self.num_steps):
            # Current time
            t_val = step * self.dt
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.float32)
            
            # Predict velocity at current state
            v = model(x, t, global_cond=global_cond)
            
            # Euler step
            x = x + v * self.dt
        
        if return_latency:
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - start_time) * 1000
            return x, latency_ms
        else:
            return x, None


class MidpointSampler:
    """
    Midpoint method (RK2) sampler for Flow Matching.
    
    Uses second-order Runge-Kutta integration for potentially better
    accuracy with the same number of function evaluations.
    """
    
    def __init__(self, num_steps: int = 4):
        """
        Args:
            num_steps: Number of integration steps.
        """
        self.num_steps = num_steps
        self.dt = 1.0 / num_steps
    
    @torch.no_grad()
    def sample(
        self, 
        model: torch.nn.Module, 
        shape: Tuple[int, ...], 
        global_cond: Optional[torch.Tensor] = None, 
        device: str = 'cuda',
        return_latency: bool = True
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """
        Generate samples via Midpoint (RK2) integration.
        """
        if return_latency:
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
        
        x = torch.randn(shape, device=device)
        
        for step in range(self.num_steps):
            t_val = step * self.dt
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.float32)
            
            # Predict velocity at current point
            v1 = model(x, t, global_cond=global_cond)
            
            # Midpoint prediction
            x_mid = x + v1 * (self.dt / 2)
            t_mid = torch.full((shape[0],), t_val + self.dt / 2, device=device, dtype=torch.float32)
            v2 = model(x_mid, t_mid, global_cond=global_cond)
            
            # Full step using midpoint velocity
            x = x + v2 * self.dt
        
        if return_latency:
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - start_time) * 1000
            return x, latency_ms
        else:
            return x, None
