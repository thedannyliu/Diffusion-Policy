"""
Flow Matching Hybrid Image Policy for Diffusion Policy.

This is the FM counterpart to DiffusionUnetHybridImagePolicy.
Uses both image and agent_pos as input (same as original Push-T setting).
"""

from typing import Dict
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
import copy

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules

from dpfm.sampler.euler_sampler import EulerSampler


class FlowMatchingUnetHybridImagePolicy(BaseImagePolicy):
    """
    Flow Matching policy using the same hybrid (image + low_dim) architecture as DDPM baseline.
    
    Key differences from DiffusionUnetHybridImagePolicy:
    - Uses Flow Matching loss instead of DDPM noise prediction
    - Uses Euler ODE sampler instead of DDPM reverse diffusion
    - Time embedding uses continuous t in [0, 1] instead of discrete timesteps
    """
    
    def __init__(self, 
            shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=4,  # Default 4-step Euler
            obs_as_global_cond=True,
            crop_shape=(84, 84),
            diffusion_step_embed_dim=128,
            down_dims=(512, 1024, 2048),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=True,
            eval_fixed_crop=True,
            sigma_min=0.0,
            **kwargs):
        super().__init__()

        # Parse shape_meta (same as DDPM)
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # Get raw robomimic config (same as DDPM)
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            config.observation.modalities.obs = obs_config
            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # Initialize obs utils
        ObsUtils.initialize_obs_utils_with_config(config)

        # Load robomimic obs encoder (same as DDPM)
        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device='cpu',
        )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, 
                    num_channels=x.num_features)
            )

        # Replace crop randomizer for eval (same as DDPM)
        rgb_keys = list()
        low_dim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                low_dim_keys.append(key)
        
        if crop_shape is not None:
            for key in rgb_keys:
                if eval_fixed_crop:
                    replace_submodules(
                        root_module=obs_encoder,
                        predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                        func=lambda x: dmvc.CropRandomizer(
                            input_shape=x.input_shape,
                            crop_height=x.crop_height,
                            crop_width=x.crop_width,
                            num_crops=x.num_crops,
                            pos_enc=x.pos_enc
                        )
                    )

        # Compute obs feature dim
        with torch.no_grad():
            obs_encoder.eval()
            example_obs = dict()
            for key, shape in obs_key_shapes.items():
                example_obs[key] = torch.zeros((1, n_obs_steps) + tuple(shape))
            obs_dict = dict_apply(example_obs, lambda x: x[:, :n_obs_steps].reshape(-1, *x.shape[2:]))
            result = obs_encoder(obs_dict)
            obs_feature_dim = result.shape[-1]
            obs_encoder.train()

        # Create diffusion model (same architecture as DDPM)
        input_dim = action_dim
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

        # Store components
        self.obs_encoder = obs_encoder
        self.model = model
        self.normalizer = LinearNormalizer()
        
        # Sampler for inference
        self.sampler = EulerSampler(num_steps=num_inference_steps)
        
        # Store config
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.num_inference_steps = num_inference_steps
        self.sigma_min = sigma_min
        self.kwargs = kwargs
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys

        # Latency tracking
        self._last_latency_ms = 0.0

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate action using Flow Matching with Euler ODE sampling.
        
        Latency is measured for the ODE sampling step ONLY, excluding:
        - Observation encoding
        - Normalization/unnormalization
        - CPU-GPU sync for latency measurement
        """
        # Normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        # Encode observations (same as DDPM) - NOT included in latency
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(B, -1)

        # === LATENCY MEASUREMENT: ODE Sampling Only ===
        # Sync GPU before timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        # Sample using Euler ODE integration
        # Note: We measure latency here instead of in sampler for consistency with DDPM
        sample_shape = (B, T, Da)
        nsample, _ = self.sampler.sample(
            model=self.model,
            shape=sample_shape,
            global_cond=global_cond,
            device=device,
            return_latency=False  # We measure latency here, not in sampler
        )

        # Sync GPU after sampling
        if device.type == 'cuda':
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start_time) * 1000
        self._last_latency_ms = latency_ms
        # === END LATENCY MEASUREMENT ===

        # Unnormalize
        action_pred = self.normalizer['action'].unnormalize(nsample)

        # Get action for execution
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {
            'action': action,
            'action_pred': action_pred,
            'latency_ms': latency_ms  # Return latency in result dict too
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        """Compute Flow Matching loss."""
        # Normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]

        # Encode observations
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(batch_size, -1)

        # Sample time uniformly
        t = torch.rand(batch_size, device=self.device)

        # Sample x_0 from standard Gaussian
        x_0 = torch.randn_like(nactions)

        # Linear interpolation (Rectified Flow path)
        t_expand = t.view(-1, 1, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * nactions

        # Add small noise for stability
        if self.sigma_min > 0:
            x_t = x_t + self.sigma_min * torch.randn_like(x_t)

        # Target velocity
        target_velocity = nactions - x_0

        # Predict velocity
        pred_velocity = self.model(x_t, t, global_cond=global_cond)

        # MSE loss
        loss = F.mse_loss(pred_velocity, target_velocity)

        return loss
