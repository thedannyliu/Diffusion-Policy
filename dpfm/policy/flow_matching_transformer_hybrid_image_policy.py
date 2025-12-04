"""
Flow Matching Transformer Hybrid Image Policy for Diffusion Policy.

This is the FM counterpart to DiffusionTransformerHybridImagePolicy.
Uses Transformer architecture with Flow Matching training and Euler ODE sampling.
"""

from typing import Dict, Tuple
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules

from dpfm.sampler.euler_sampler import EulerSampler


class FlowMatchingTransformerHybridImagePolicy(BaseImagePolicy):
    """
    Flow Matching policy using Transformer architecture with hybrid (image + low_dim) input.
    
    Key differences from DiffusionTransformerHybridImagePolicy:
    - Uses Flow Matching loss (velocity prediction) instead of DDPM noise prediction
    - Uses Euler ODE sampler instead of DDPM reverse diffusion
    - Time embedding uses continuous t in [0, 1] instead of discrete timesteps
    """
    
    def __init__(self, 
            shape_meta: dict,
            # task params
            horizon, 
            n_action_steps, 
            n_obs_steps,
            # FM params
            num_inference_steps=4,  # Default 4-step Euler (FM is much faster)
            sigma_min=0.0,
            # image
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # arch
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
            pred_action_steps_only=False,
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

        # Init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # Load model
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
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
        
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

        # Create Transformer model (same architecture as DDPM)
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers
        )

        # Create Euler sampler for FM inference
        # Note: sigma_min is stored but not used by EulerSampler currently
        self.sigma_min = sigma_min
        self.sampler = EulerSampler(
            num_steps=num_inference_steps
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.sigma_min = sigma_min
        self.num_inference_steps = num_inference_steps
        self.kwargs = kwargs
        
        # Latency tracking for fair comparison
        self._last_latency_ms = 0.0
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None):
        """
        Sample actions using Euler ODE integration.
        
        Flow Matching samples from x_0 ~ N(0, I) and integrates to x_1 (data).
        """
        model = self.model
        device = condition_data.device
        dtype = condition_data.dtype
        
        # Start from pure noise
        x = torch.randn(
            size=condition_data.shape, 
            dtype=dtype,
            device=device,
            generator=generator)
        
        # Apply conditioning at start
        x[condition_mask] = condition_data[condition_mask]
        
        # Euler integration
        num_steps = self.num_inference_steps
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = i / num_steps
            t_tensor = torch.full((x.shape[0],), t, device=device, dtype=dtype)
            
            # FM uses continuous time in [0, 1]
            # TransformerForDiffusion expects timesteps, we convert
            # t=0 corresponds to noise, t=1 corresponds to data
            # Map to scheduler timesteps range for compatibility
            timesteps = (t_tensor * 1000).long()
            
            # Predict velocity
            v = model(x, timesteps, cond)
            
            # Euler step: x_next = x + v * dt
            x = x + v * dt
            
            # Re-apply conditioning
            x[condition_mask] = condition_data[condition_mask]
        
        return x


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        start_time = time.time()
        
        assert 'past_action' not in obs_dict  # not implemented yet
        # Normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        # Handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            cond = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # Run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond)
        
        # Unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # Get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        # Track latency
        self._last_latency_ms = (time.time() - start_time) * 1000
        
        result = {
            'action': action,
            'action_pred': action_pred,
            'latency_ms': self._last_latency_ms
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            transformer_weight_decay: float, 
            obs_encoder_weight_decay: float,
            learning_rate: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def compute_loss(self, batch):
        """
        Flow Matching loss: MSE between predicted velocity and target velocity.
        
        The target velocity is: v(x_t, t) = x_1 - x_0
        where x_0 is noise and x_1 is the clean data.
        
        The interpolated point is: x_t = (1-t) * x_0 + t * x_1
        """
        # Normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps
        device = nactions.device

        # Handle different ways of passing observation
        cond = None
        trajectory = nactions  # x_1 (clean data)
        if self.obs_as_cond:
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            cond = nobs_features.reshape(batch_size, To, -1)
            if self.pred_action_steps_only:
                start = To - 1
                end = start + self.n_action_steps
                trajectory = nactions[:,start:end]
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()

        # Generate conditioning mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Compute loss mask
        loss_mask = ~condition_mask

        # Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn(trajectory.shape, device=device)
        
        # Sample time t ~ U[0, 1]
        t = torch.rand((batch_size,), device=device)
        
        # Compute interpolated point: x_t = (1-t) * x_0 + t * x_1
        t_expand = t.view(-1, 1, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * trajectory
        
        # Apply conditioning
        x_t[condition_mask] = trajectory[condition_mask]
        
        # Target velocity: v = x_1 - x_0
        target_velocity = trajectory - x_0
        
        # Convert time to timesteps for model input (scale to [0, 1000])
        timesteps = (t * 1000).long()
        
        # Predict velocity
        pred_velocity = self.model(x_t, timesteps, cond)
        
        # Compute MSE loss
        loss = F.mse_loss(pred_velocity, target_velocity, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        return loss
