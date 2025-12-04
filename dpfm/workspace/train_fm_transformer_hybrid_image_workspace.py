"""
Training Workspace for Flow Matching Transformer Hybrid Image Policy.

This workspace is aligned with TrainDiffusionTransformerHybridImageWorkspace from the original paper.
Key alignment:
1. Uses FlowMatchingTransformerHybridImagePolicy (FM version of DiffusionTransformerHybridImagePolicy)
2. Same training loop, optimizer (with transformer weight decay), scheduler, checkpointing
3. Additional metrics: latency tracking, action jerk
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from typing import Optional, Dict, Any
import time

# Import from diffusion_policy (same as baseline)
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

# Import FM policies
from dpfm.policy.flow_matching_transformer_hybrid_image_policy import FlowMatchingTransformerHybridImagePolicy

OmegaConf.register_new_resolver("eval", eval, replace=True)


def compute_action_jerk(actions: torch.Tensor, dt: float = 0.1) -> Dict[str, float]:
    """
    Compute action jerk (smoothness metric).
    
    Args:
        actions: [batch, horizon, action_dim] or [horizon, action_dim]
        dt: time step between actions
        
    Returns:
        Dictionary with jerk statistics
    """
    if actions.dim() == 2:
        actions = actions.unsqueeze(0)
    
    velocity = (actions[:, 1:, :] - actions[:, :-1, :]) / dt
    acceleration = (velocity[:, 1:, :] - velocity[:, :-1, :]) / dt
    jerk = (acceleration[:, 1:, :] - acceleration[:, :-1, :]) / dt
    
    jerk_magnitude = torch.norm(jerk, dim=-1)
    
    return {
        'jerk_mean': jerk_magnitude.mean().item(),
        'jerk_max': jerk_magnitude.max().item(),
        'jerk_std': jerk_magnitude.std().item(),
    }


class TrainFlowMatchingTransformerHybridImageWorkspace(BaseWorkspace):
    """
    Training workspace for Flow Matching Transformer Hybrid Image Policy.
    
    Aligned with TrainDiffusionTransformerHybridImageWorkspace for fair comparison.
    Uses Transformer architecture for action prediction with separate weight decay.
    """
    
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # Set seed for reproducibility
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Configure model (FlowMatchingTransformerHybridImagePolicy)
        self.model: FlowMatchingTransformerHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        # EMA model for evaluation
        self.ema_model: FlowMatchingTransformerHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # Configure optimizer (Transformer uses custom get_optimizer with weight decay)
        self.optimizer = self.model.get_optimizer(
            transformer_weight_decay=cfg.optimizer.transformer_weight_decay,
            obs_encoder_weight_decay=cfg.optimizer.obs_encoder_weight_decay,
            learning_rate=cfg.optimizer.learning_rate,
            betas=tuple(cfg.optimizer.betas)
        )

        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Metrics tracking
        self.latency_history = []
        self.jerk_history = []
        self.best_score = -float('inf')
        self.patience_counter = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # Resume training if checkpoint exists
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # Configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # Validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # Set normalizer
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # Configure LR scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )

        # Configure EMA
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # Configure environment runner
        env_runner: Optional[BaseImageRunner] = None
        if cfg.task.env_runner is not None:
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir)
            assert isinstance(env_runner, BaseImageRunner)

        # Configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update({
            "output_dir": self.output_dir,
            "method": "flow_matching_transformer",
            "num_inference_steps": cfg.policy.num_inference_steps,
            "policy_type": "FlowMatchingTransformerHybridImagePolicy"
        })

        # Checkpoint manager
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # Move to device
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # Batch for sampling visualization
        train_sampling_batch = None

        # Debug mode
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # Early stopping config
        early_stopping_enabled = getattr(cfg.training, 'early_stopping', False)
        early_stopping_patience = getattr(cfg.training, 'early_stopping_patience', 100)
        early_stopping_min_epochs = getattr(cfg.training, 'early_stopping_min_epochs', 500)

        # Training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                
                # Freeze encoder if specified
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                # Training epoch
                train_losses = list()
                epoch_start_time = time.time()
                
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # Device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # Compute FM loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # Optimizer step
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # Update EMA
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # Logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                epoch_time = time.time() - epoch_start_time
                step_log['epoch_time_sec'] = epoch_time

                # Epoch average loss
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # Evaluation
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # Rollout evaluation
                if env_runner is not None and (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    step_log.update(runner_log)
                    
                    if 'inference_latency_ms' in runner_log:
                        self.latency_history.append(runner_log['inference_latency_ms'])
                    
                    # Early stopping check
                    if early_stopping_enabled and self.epoch >= early_stopping_min_epochs:
                        current_score = runner_log.get('test/mean_score', 0)
                        if current_score > self.best_score:
                            self.best_score = current_score
                            self.patience_counter = 0
                        else:
                            self.patience_counter += 1
                            
                        if self.patience_counter >= early_stopping_patience:
                            print(f"\n=== Early Stopping at epoch {self.epoch} ===")
                            print(f"Best score: {self.best_score:.4f}")
                            break

                # Validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            step_log['val_loss'] = val_loss

                # Sample from training batch + compute jerk
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        
                        if 'latency_ms' in result:
                            step_log['sample_latency_ms'] = result['latency_ms']
                        
                        jerk_metrics = compute_action_jerk(pred_action.cpu())
                        step_log['jerk_mean'] = jerk_metrics['jerk_mean']
                        step_log['jerk_max'] = jerk_metrics['jerk_max']
                        self.jerk_history.append(jerk_metrics['jerk_mean'])
                        
                        gt_jerk = compute_action_jerk(gt_action.cpu())
                        step_log['gt_jerk_mean'] = gt_jerk['jerk_mean']
                        
                        del batch, obs_dict, gt_action, result, pred_action, mse
                
                # Checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                policy.train()

                # Log epoch
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        # Final statistics
        print(f"\n{'='*50}")
        print(f"Training Complete: {self.epoch} epochs")
        print(f"{'='*50}")
        
        if len(self.latency_history) > 0:
            print(f"\n=== Latency Statistics ===")
            print(f"p50: {np.percentile(self.latency_history, 50):.2f} ms")
            print(f"p95: {np.percentile(self.latency_history, 95):.2f} ms")
            print(f"Mean: {np.mean(self.latency_history):.2f} ms")
        
        if len(self.jerk_history) > 0:
            print(f"\n=== Jerk Statistics ===")
            print(f"Mean jerk: {np.mean(self.jerk_history):.4f}")
            print(f"Std jerk: {np.std(self.jerk_history):.4f}")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainFlowMatchingTransformerHybridImageWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
