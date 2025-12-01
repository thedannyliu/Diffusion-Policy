#!/usr/bin/env python3
"""
Test FM Checkpoint Loading

This script tests that FM checkpoints can be loaded correctly
and that the policy interface works as expected for real robot deployment.
"""

import torch
import dill
import hydra
import numpy as np
from omegaconf import OmegaConf

# Register eval resolver for OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)


def test_checkpoint_loading(checkpoint_path: str, description: str):
    """Test loading a checkpoint and verify policy interface."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Checkpoint: {checkpoint_path}")
    print('='*60)
    
    try:
        # Load checkpoint
        payload = torch.load(checkpoint_path, pickle_module=dill, map_location='cuda')
        cfg = payload['cfg']
        
        print(f"[OK] Checkpoint loaded successfully")
        print(f"     Config name: {cfg.name}")
        print(f"     Config target: {cfg._target_}")
        
        # Instantiate workspace
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        print(f"[OK] Workspace instantiated: {type(workspace).__name__}")
        
        # Get policy
        policy = workspace.model
        if hasattr(workspace, 'ema_model') and workspace.ema_model is not None:
            policy = workspace.ema_model
            print(f"[OK] Using EMA model")
        else:
            print(f"[OK] Using base model (no EMA)")
        
        policy.eval().to('cuda')
        print(f"[OK] Policy moved to CUDA and set to eval mode")
        
        # Check policy attributes
        print(f"\n--- Policy Attributes ---")
        print(f"     Type: {type(policy).__name__}")
        print(f"     Horizon: {policy.horizon}")
        print(f"     n_obs_steps: {policy.n_obs_steps}")
        print(f"     n_action_steps: {policy.n_action_steps}")
        
        if hasattr(policy, 'num_inference_steps'):
            print(f"     num_inference_steps: {policy.num_inference_steps}")
        
        if hasattr(policy, 'action_dim'):
            print(f"     action_dim: {policy.action_dim}")
        
        # Test forward pass with dummy data
        print(f"\n--- Testing Forward Pass ---")
        
        # Get shape meta from config
        shape_meta = cfg.shape_meta if hasattr(cfg, 'shape_meta') else cfg.task.shape_meta
        
        # Create dummy observation
        obs_dict = {}
        for key, meta in shape_meta['obs'].items():
            shape = meta['shape']
            if meta['type'] == 'rgb':
                # Image observation: [B, T, C, H, W]
                dummy = torch.randn(1, policy.n_obs_steps, *shape, device='cuda')
            else:
                # Low-dim observation: [B, T, D]
                dummy = torch.randn(1, policy.n_obs_steps, *shape, device='cuda')
            obs_dict[key] = dummy
            print(f"     Created dummy obs '{key}': {dummy.shape}")
        
        # Run inference
        with torch.no_grad():
            import time
            start = time.time()
            result = policy.predict_action(obs_dict)
            elapsed = (time.time() - start) * 1000
        
        action = result['action']
        print(f"\n[OK] Forward pass successful!")
        print(f"     Output action shape: {action.shape}")
        print(f"     Inference latency: {elapsed:.2f} ms")
        
        # Verify action shape
        expected_action_dim = shape_meta['action']['shape'][0]
        assert action.shape[-1] == expected_action_dim, \
            f"Action dim mismatch: got {action.shape[-1]}, expected {expected_action_dim}"
        print(f"[OK] Action dimension verified: {expected_action_dim}")
        
        print(f"\n[SUCCESS] All tests passed for {description}")
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("FM Checkpoint Loading Test Suite")
    print("="*60)
    
    # Define checkpoints to test
    checkpoints = [
        # FM v1 (step=4) - Fair encoder
        {
            'path': 'data/outputs/real_robot/2025.11.27/16.22.11_fm_fair_sphere_step4_seed42/checkpoints/latest.ckpt',
            'desc': 'FM v1 Sphere (step=4, fair encoder)'
        },
        # FM v2 (step=8) - Optimized
        {
            'path': 'data/outputs/2025.11.29/00.33.00_train_fm_real_robot_fair_sphere/checkpoints/latest.ckpt',
            'desc': 'FM v2 Sphere (step=8, optimized)'
        },
    ]
    
    results = []
    for ckpt in checkpoints:
        import os
        if os.path.exists(ckpt['path']):
            success = test_checkpoint_loading(ckpt['path'], ckpt['desc'])
            results.append((ckpt['desc'], success))
        else:
            print(f"\n[SKIP] Checkpoint not found: {ckpt['path']}")
            results.append((ckpt['desc'], None))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for desc, success in results:
        if success is None:
            status = "SKIPPED"
        elif success:
            status = "PASSED"
        else:
            status = "FAILED"
        print(f"  [{status}] {desc}")
    
    # Check if any tests failed
    failed = [r for r in results if r[1] is False]
    if failed:
        print(f"\n{len(failed)} test(s) FAILED!")
        exit(1)
    else:
        print(f"\nAll tests passed!")
        exit(0)


if __name__ == '__main__':
    main()
