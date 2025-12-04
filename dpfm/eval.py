"""
Evaluation script for Flow Matching and DDPM policies.

Usage:
    python -m dpfm.eval \
        --checkpoint path/to/checkpoint.ckpt \
        --output_dir path/to/output \
        --n_test 50 \
        --device cuda:0
"""

import sys
import os
import pathlib
import click
import json
import numpy as np
import torch
import dill
import hydra
import wandb
from omegaconf import OmegaConf

# Add paths
project_dir = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_dir))
sys.path.insert(0, str(project_dir / "diffusion_policy"))

# Register eval resolver for OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("now", lambda pattern: "", replace=True)  # Dummy for now resolver

from diffusion_policy.common.pytorch_util import dict_apply


def load_checkpoint(checkpoint_path: str, device: str = 'cuda:0'):
    """Load checkpoint and return policy."""
    checkpoint_path = pathlib.Path(checkpoint_path)
    
    # Load checkpoint
    payload = torch.load(checkpoint_path.open('rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    
    # Resolve config
    OmegaConf.resolve(cfg)
    
    # Get workspace class
    cls = hydra.utils.get_class(cfg._target_)
    
    # Create workspace
    workspace = cls(cfg, output_dir=str(checkpoint_path.parent.parent))
    
    # Load state
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # Get policy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    policy.to(device)
    policy.eval()
    
    return policy, cfg


def run_evaluation(
    policy,
    cfg,
    n_test: int = 50,
    output_dir: str = None,
    device: str = 'cuda:0'
):
    """Run evaluation and return metrics."""
    # Convert to mutable dict to allow modifications
    env_runner_cfg = OmegaConf.to_container(cfg.task.env_runner, resolve=True)
    
    # Override settings
    env_runner_cfg['n_test'] = n_test
    env_runner_cfg['n_test_vis'] = min(4, n_test)
    env_runner_cfg['n_train'] = 0
    env_runner_cfg['n_train_vis'] = 0
    
    # Set output dir
    if output_dir is None:
        output_dir = pathlib.Path('.').joinpath('eval_output')
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env_runner_cfg['output_dir'] = str(output_dir)
    
    # Convert back to OmegaConf for instantiation
    env_runner_cfg = OmegaConf.create(env_runner_cfg)
    
    # Create env runner
    env_runner = hydra.utils.instantiate(env_runner_cfg)
    
    # Run evaluation
    log_data = env_runner.run(policy)
    
    return log_data


def compute_action_jerk(actions: np.ndarray) -> float:
    """
    Compute action jerk metric.
    
    Args:
        actions: Array of shape [T, D] where T is time and D is action dim
        
    Returns:
        Mean squared jerk
    """
    if len(actions) < 2:
        return 0.0
    
    # Compute differences
    diffs = np.diff(actions, axis=0)  # [T-1, D]
    
    # Compute squared norm
    jerk = np.sum(diffs ** 2, axis=1)  # [T-1]
    
    return float(np.mean(jerk))


@click.command()
@click.option('--checkpoint', '-c', required=True, help='Path to checkpoint file')
@click.option('--output_dir', '-o', default=None, help='Output directory')
@click.option('--n_test', '-n', default=50, help='Number of test episodes')
@click.option('--device', '-d', default='cuda:0', help='Device to use')
@click.option('--wandb_project', default=None, help='WandB project for eval logging')
@click.option('--wandb_entity', default=None, help='WandB entity (team/user) for eval logging')
@click.option('--wandb_group', default=None, help='Optional WandB group name')
@click.option('--wandb_mode', default='online', type=click.Choice(['online', 'offline', 'disabled']), help='WandB mode')
def main(
    checkpoint: str,
    output_dir: str,
    n_test: int,
    device: str,
    wandb_project: str,
    wandb_entity: str,
    wandb_group: str,
    wandb_mode: str,
):
    """Evaluate a trained policy."""
    print(f"Loading checkpoint: {checkpoint}")
    policy, cfg = load_checkpoint(checkpoint, device)
    
    print(f"Running evaluation with {n_test} test episodes...")
    log_data = run_evaluation(policy, cfg, n_test=n_test, output_dir=output_dir, device=device)
    
    # Extract key metrics (PushT + generic)
    metrics = {
        # Original DP-style score
        'test_mean_score': log_data.get('test/mean_score', 0.0),
        # PushT-specific metrics
        'test_success_rate': log_data.get('test/success_rate', 0.0),
        'test_target_area_coverage': log_data.get('test/target_area_coverage', 0.0),
        'test_final_distance': log_data.get('test/final_distance', 0.0),
        'test_mean_step_count': log_data.get('test/mean_step_count', 0.0),
        'test_smoothness': log_data.get('test/smoothness', 0.0),
        # Latency statistics
        'inference_latency_ms': log_data.get('inference_latency_ms', 0.0),
        'inference_latency_p50_ms': log_data.get('inference_latency_p50_ms', 0.0),
        'inference_latency_p95_ms': log_data.get('inference_latency_p95_ms', 0.0),
    }
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Test Mean Score: {metrics['test_mean_score']:.4f}")
    print(f"Test Success Rate: {metrics['test_success_rate']:.4f}")
    print(f"Target-area Coverage: {metrics['test_target_area_coverage']:.4f}")
    print(f"Final Distance: {metrics['test_final_distance']:.4f}")
    print(f"Mean Step Count: {metrics['test_mean_step_count']:.2f}")
    print(f"Smoothness: {metrics['test_smoothness']:.4e}")
    print(f"Inference Latency (mean): {metrics['inference_latency_ms']:.2f} ms")
    print(f"Inference Latency (p50): {metrics['inference_latency_p50_ms']:.2f} ms")
    print(f"Inference Latency (p95): {metrics['inference_latency_p95_ms']:.2f} ms")
    print("="*50 + "\n")
    
    # Optionally log to WandB (new eval-specific run)
    if wandb_project is not None and wandb_mode != 'disabled':
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            group=wandb_group,
            mode=wandb_mode,
            name=f"eval_{pathlib.Path(checkpoint).stem}",
            config={
                "checkpoint": checkpoint,
                "n_test": n_test,
                "device": device,
            },
        )
        wandb.log(log_data)
        wandb_run.finish()

    # Save results
    if output_dir:
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / 'eval_results.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Also dump full log_data for detailed analysis
        # For the full log, convert wandb media objects to file paths
        serializable_log = {}
        for key, value in log_data.items():
            if isinstance(value, (np.floating, np.number)):
                serializable_log[key] = float(value)
            elif isinstance(value, torch.Tensor):
                serializable_log[key] = value.detach().cpu().tolist()
            elif isinstance(value, wandb.sdk.data_types.video.Video):
                serializable_log[key] = value._path
            elif isinstance(value, wandb.Image):
                serializable_log[key] = value.image.filename if hasattr(value.image, "filename") else None
            else:
                serializable_log[key] = value

        with open(output_path / 'eval_log_full.json', 'w') as f:
            json.dump(serializable_log, f, indent=2)

        print(f"Results saved to {output_path / 'eval_results.json'}")
        print(f"Full log saved to {output_path / 'eval_log_full.json'}")
    
    return metrics


if __name__ == '__main__':
    main()
