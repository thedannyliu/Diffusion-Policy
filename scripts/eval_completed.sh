#!/bin/bash
#SBATCH --job-name=fm_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40s:1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=inferno

# Eval script for completed FM models

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching
source ~/.bashrc
conda activate DPFM

export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/diffusion_policy"

echo "=================================================="
echo "Evaluating Completed FM Models"
echo "=================================================="
echo "Start time: $(date)"
nvidia-smi

# Create results directory
mkdir -p results/eval_fm_v2

# Evaluate each completed model
python -c "
import torch
import json
import numpy as np
from pathlib import Path
import time

# Add paths
import sys
sys.path.insert(0, 'diffusion_policy')

from diffusion_policy.env_runner.pusht_image_runner import PushTImageRunner

# Model checkpoints to evaluate
models = {
    'FM_step8_s42_lr1e-4': {
        'ckpt': 'data/outputs/2025.11.27/04.36.04_train_fm_unet_hybrid_pusht_image/checkpoints/epoch=0900-test_mean_score=0.845.ckpt',
        'steps': 8
    },
    'FM_step4_s123_lr1e-4': {
        'ckpt': 'data/outputs/2025.11.27/04.36.23_train_fm_unet_hybrid_pusht_image/checkpoints/epoch=0150-test_mean_score=0.807.ckpt',
        'steps': 4
    },
    'FM_step4_s42_lr2e-4': {
        'ckpt': 'data/outputs/2025.11.27/04.36.29_train_fm_unet_hybrid_pusht_image/checkpoints/epoch=0850-test_mean_score=0.812.ckpt',
        'steps': 4
    }
}

results = {}

for name, info in models.items():
    print(f'\n=== Evaluating {name} ===')
    ckpt_path = info['ckpt']
    
    if not Path(ckpt_path).exists():
        print(f'  Checkpoint not found: {ckpt_path}')
        continue
    
    try:
        # Load checkpoint
        payload = torch.load(ckpt_path, map_location='cuda:0')
        cfg = payload['cfg']
        
        # Get the policy class
        from hydra.utils import instantiate
        policy = instantiate(cfg.policy)
        
        # Load state dict
        state_dict = payload['state_dicts']['ema_model']
        policy.load_state_dict(state_dict)
        policy.to('cuda:0')
        policy.eval()
        
        # Create runner
        runner = PushTImageRunner(
            output_dir='results/eval_fm_v2',
            n_train=0,
            n_train_vis=0,
            train_start_seed=0,
            n_test=50,
            n_test_vis=4,
            test_start_seed=100000,
            max_steps=300,
            n_obs_steps=cfg.n_obs_steps,
            n_action_steps=cfg.n_action_steps,
            render_size=96
        )
        
        # Run evaluation
        print(f'  Running 50 test episodes...')
        start_time = time.time()
        result = runner.run(policy)
        eval_time = time.time() - start_time
        
        # Extract results
        test_mean_score = result['test/mean_score']
        
        # Get latency stats from policy if available
        latency_ms = getattr(policy, '_last_latency_ms', 0)
        
        results[name] = {
            'test_mean_score': float(test_mean_score),
            'inference_steps': info['steps'],
            'eval_time_sec': eval_time,
            'latency_ms': latency_ms
        }
        
        print(f'  Score: {test_mean_score:.4f}')
        print(f'  Eval time: {eval_time:.1f}s')
        
    except Exception as e:
        print(f'  Error: {e}')
        import traceback
        traceback.print_exc()

# Save results
with open('results/eval_fm_v2/eval_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\n=== Summary ===')
for name, res in results.items():
    print(f'{name}: score={res[\"test_mean_score\"]:.4f}, steps={res[\"inference_steps\"]}')

print('\nResults saved to results/eval_fm_v2/eval_results.json')
"

echo ""
echo "Evaluation complete!"
echo "End time: $(date)"
