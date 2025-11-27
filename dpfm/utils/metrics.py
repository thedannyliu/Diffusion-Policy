"""
Metrics for evaluating Diffusion Policy with Flow Matching.

This module provides utilities for computing and logging metrics that are
central to our hypothesis: latency and action jerk.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import time


def compute_action_jerk(actions: np.ndarray) -> float:
    """
    Compute mean action jerk from a sequence of actions.
    
    Jerk measures the "smoothness" of actions. Lower jerk = smoother control.
    
    Definition:
        Δa_t = a_t - a_{t-1}
        jerk_t = ||Δa_t||²
        jerk_mean = (1 / (T-1)) * Σ jerk_t
    
    Args:
        actions: Action sequence [T, D] or [B, T, D]
        
    Returns:
        Mean jerk (squared L2 norm of action differences)
    """
    if actions.ndim == 2:
        # [T, D] -> add batch dim
        actions = actions[np.newaxis, ...]
    
    # Compute differences: Δa_t = a_t - a_{t-1}
    # Shape: [B, T-1, D]
    action_diffs = np.diff(actions, axis=1)
    
    # Compute squared L2 norm: ||Δa_t||²
    # Shape: [B, T-1]
    jerk_per_step = np.sum(action_diffs ** 2, axis=-1)
    
    # Mean over all steps and batches
    mean_jerk = np.mean(jerk_per_step)
    
    return float(mean_jerk)


def compute_action_jerk_torch(actions: torch.Tensor) -> torch.Tensor:
    """
    Compute mean action jerk (PyTorch version).
    
    Args:
        actions: Action sequence [T, D] or [B, T, D]
        
    Returns:
        Mean jerk as a scalar tensor
    """
    if actions.ndim == 2:
        actions = actions.unsqueeze(0)
    
    # Differences
    action_diffs = actions[:, 1:, :] - actions[:, :-1, :]
    
    # Squared L2 norm
    jerk_per_step = torch.sum(action_diffs ** 2, dim=-1)
    
    # Mean
    return jerk_per_step.mean()


class LatencyTracker:
    """
    Track inference latency statistics.
    """
    
    def __init__(self):
        self.latencies = []
        
    def record(self, latency_ms: float):
        """Record a latency measurement."""
        self.latencies.append(latency_ms)
    
    def reset(self):
        """Clear all measurements."""
        self.latencies = []
    
    def get_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        if len(self.latencies) == 0:
            return {}
        
        arr = np.array(self.latencies)
        return {
            'latency_mean_ms': float(np.mean(arr)),
            'latency_std_ms': float(np.std(arr)),
            'latency_p50_ms': float(np.percentile(arr, 50)),
            'latency_p95_ms': float(np.percentile(arr, 95)),
            'latency_min_ms': float(np.min(arr)),
            'latency_max_ms': float(np.max(arr)),
            'latency_count': len(self.latencies)
        }


class JerkTracker:
    """
    Track action jerk statistics across episodes.
    """
    
    def __init__(self):
        self.jerks = []
        
    def record_episode(self, actions: np.ndarray):
        """
        Record jerk for an episode.
        
        Args:
            actions: Action sequence [T, D] for one episode
        """
        if len(actions) > 1:
            jerk = compute_action_jerk(actions)
            self.jerks.append(jerk)
    
    def reset(self):
        """Clear all measurements."""
        self.jerks = []
    
    def get_stats(self) -> Dict[str, float]:
        """Get jerk statistics."""
        if len(self.jerks) == 0:
            return {}
        
        arr = np.array(self.jerks)
        return {
            'jerk_mean': float(np.mean(arr)),
            'jerk_std': float(np.std(arr)),
            'jerk_min': float(np.min(arr)),
            'jerk_max': float(np.max(arr)),
            'jerk_count': len(self.jerks)
        }


def measure_inference_latency(
    model: torch.nn.Module,
    obs_dict: Dict[str, torch.Tensor],
    num_warmup: int = 5,
    num_trials: int = 20
) -> Dict[str, float]:
    """
    Measure inference latency of a policy.
    
    Args:
        model: Policy model with predict_action method
        obs_dict: Example observation dictionary
        num_warmup: Number of warmup iterations (not counted)
        num_trials: Number of timing trials
        
    Returns:
        Dictionary with latency statistics
    """
    device = next(model.parameters()).device
    
    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.predict_action(obs_dict)
    
    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(num_trials):
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.perf_counter()
            
            _ = model.predict_action(obs_dict)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)  # ms
    
    arr = np.array(latencies)
    return {
        'latency_mean_ms': float(np.mean(arr)),
        'latency_std_ms': float(np.std(arr)),
        'latency_p50_ms': float(np.percentile(arr, 50)),
        'latency_p95_ms': float(np.percentile(arr, 95)),
    }
