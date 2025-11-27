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
def main(checkpoint: str, output_dir: str, n_test: int, device: str):
    """Evaluate a trained policy."""
    print(f"Loading checkpoint: {checkpoint}")
    policy, cfg = load_checkpoint(checkpoint, device)
    
    print(f"Running evaluation with {n_test} test episodes...")
    log_data = run_evaluation(policy, cfg, n_test=n_test, output_dir=output_dir, device=device)
    
    # Extract key metrics
    metrics = {
        'test_mean_score': log_data.get('test/mean_score', 0.0),
        'inference_latency_ms': log_data.get('inference_latency_ms', 0.0),
        'inference_latency_p50_ms': log_data.get('inference_latency_p50_ms', 0.0),
        'inference_latency_p95_ms': log_data.get('inference_latency_p95_ms', 0.0),
    }
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Test Mean Score: {metrics['test_mean_score']:.4f}")
    print(f"Inference Latency (mean): {metrics['inference_latency_ms']:.2f} ms")
    print(f"Inference Latency (p50): {metrics['inference_latency_p50_ms']:.2f} ms")
    print(f"Inference Latency (p95): {metrics['inference_latency_p95_ms']:.2f} ms")
    print("="*50 + "\n")
    
    # Save results
    if output_dir:
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / 'eval_results.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Results saved to {output_path / 'eval_results.json'}")
    
    return metrics


if __name__ == '__main__':
    main()
