"""
Flow Matching UNet Image Policy for Diffusion Policy.

This policy is a drop-in replacement for DiffusionUnetImagePolicy that uses
Flow Matching instead of DDPM for training and inference.

Key modifications from DDPM baseline:
1. Training: Replace noise prediction with velocity prediction (FM loss)
2. Inference: Replace DDPM reverse diffusion with Euler ODE integration
3. Time encoding: Continuous t ∈ [0, 1] instead of discrete timesteps

Everything else (encoder, UNet architecture, normalizer, action horizon) is 
kept identical for fair comparison.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

# Import from diffusion_policy (same components as baseline)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

# Import our FM components
from dpfm.sampler.euler_sampler import EulerSampler


class FlowMatchingUnetImagePolicy(BaseImagePolicy):
    """
    Flow Matching-based policy for image observations.
    
    This is functionally equivalent to DiffusionUnetImagePolicy but replaces
    DDPM with Flow Matching for faster inference (1-8 steps instead of 100).
    """
    
    def __init__(self, 
            shape_meta: dict,
            obs_encoder: MultiImageObsEncoder,
            horizon: int, 
            n_action_steps: int, 
            n_obs_steps: int,
            # Flow Matching specific
            num_inference_steps: int = 4,
            sigma_min: float = 0.0,
            # UNet parameters (same as DDPM)
            obs_as_global_cond: bool = True,
            diffusion_step_embed_dim: int = 256,
            down_dims: tuple = (256, 512, 1024),
            kernel_size: int = 5,
            n_groups: int = 8,
            cond_predict_scale: bool = True,
            **kwargs):
        """
        Args:
            shape_meta: Metadata about observation and action shapes
            obs_encoder: Image encoder (MultiImageObsEncoder)
            horizon: Action prediction horizon
            n_action_steps: Number of action steps to execute
            n_obs_steps: Number of observation steps for conditioning
            num_inference_steps: Number of Euler steps for sampling (1, 2, 4, 8)
            sigma_min: Minimum noise for training stability (usually 0)
            obs_as_global_cond: Whether to use obs as global conditioning
            diffusion_step_embed_dim: Dimension of time embedding
            down_dims: UNet channel dimensions
            kernel_size: Conv kernel size
            n_groups: GroupNorm groups
            cond_predict_scale: Whether to predict scale in FiLM
        """
        super().__init__()

        # Parse shapes (same as DDPM)
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        
        # Get feature dim from encoder
        obs_feature_dim = obs_encoder.output_shape()[0]

        # Create UNet model (identical to DDPM)
        input_dim = action_dim
        global_cond_dim = None
        if obs_as_global_cond:
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        # Initialize components
        self.obs_encoder = obs_encoder
        self.model = model
        self.normalizer = LinearNormalizer()
        
        # Mask generator (same as DDPM - for inpainting-style conditioning)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        # Flow Matching sampler (replaces DDPM scheduler)
        self.sampler = EulerSampler(num_steps=num_inference_steps)
        
        # Store hyperparameters
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.num_inference_steps = num_inference_steps
        self.sigma_min = sigma_min
        self.kwargs = kwargs
        
        # Latency tracking for evaluation
        self._last_latency_ms = None

    # ========= Inference ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predict action from observations using Flow Matching sampling.
        
        This method has the same interface as DiffusionUnetImagePolicy.predict_action
        to ensure compatibility with the evaluation infrastructure.
        
        Args:
            obs_dict: Dictionary containing observation tensors
            
        Returns:
            Dictionary with 'action' and 'action_pred' keys
        """
        assert 'past_action' not in obs_dict  # Not implemented yet
        
        # Normalize input (same as DDPM)
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        # Get device/dtype
        device = self.device
        dtype = self.dtype

        # Encode observations (same as DDPM)
        if self.obs_as_global_cond:
            # Condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # Reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
        else:
            raise NotImplementedError("Only obs_as_global_cond=True is supported")

        # Sample using Flow Matching (Euler integration)
        sample_shape = (B, T, Da)
        nsample, latency_ms = self.sampler.sample(
            model=self.model,
            shape=sample_shape,
            global_cond=global_cond,
            device=device,
            return_latency=True
        )
        
        # Store latency for logging
        self._last_latency_ms = latency_ms
        
        # Unnormalize prediction
        naction_pred = nsample
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # Get action for execution (same slicing as DDPM)
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred,
            'latency_ms': latency_ms  # Extra info for evaluation
        }
        return result

    # ========= Training ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set normalizer from dataset (same as DDPM)."""
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute Flow Matching loss.
        
        Instead of predicting noise (DDPM), we predict velocity along the
        optimal transport path from noise to data.
        
        FM Loss: L = E_{t,x_0,x_1}[||v_θ(x_t, t) - (x_1 - x_0)||²]
        
        where x_t = t * x_1 + (1-t) * x_0
        
        Args:
            batch: Training batch with 'obs' and 'action' keys
            
        Returns:
            Scalar loss value
        """
        # Normalize inputs (same as DDPM)
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # Encode observations for conditioning (same as DDPM)
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, 
                lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
            trajectory = nactions
        else:
            raise NotImplementedError("Only obs_as_global_cond=True is supported")

        # Generate mask for loss computation (same as DDPM)
        condition_mask = self.mask_generator(trajectory.shape)
        loss_mask = ~condition_mask

        # === Flow Matching Loss ===
        device = trajectory.device
        
        # Sample t uniformly from [0, 1]
        t = torch.rand(batch_size, device=device)
        
        # Sample x_0 from standard Gaussian
        x_0 = torch.randn_like(trajectory)
        
        # x_1 is the target (normalized actions)
        x_1 = trajectory
        
        # Linear interpolation: x_t = t * x_1 + (1 - t) * x_0
        t_expand = t.view(-1, 1, 1)
        x_t = t_expand * x_1 + (1 - t_expand) * x_0
        
        # Target velocity: u_t = x_1 - x_0 (constant along OT path)
        u_t = x_1 - x_0
        
        # Predict velocity
        v_pred = self.model(x_t, t, global_cond=global_cond)
        
        # Compute loss with masking (same as DDPM)
        loss = F.mse_loss(v_pred, u_t, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        return loss

    def get_last_latency_ms(self) -> Optional[float]:
        """Get latency from last predict_action call for logging."""
        return self._last_latency_ms
