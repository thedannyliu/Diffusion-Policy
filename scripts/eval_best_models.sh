#!/bin/bash
#SBATCH --job-name=eval_best
#SBATCH --output=logs/eval_best_%j.out
#SBATCH --error=logs/eval_best_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40s:1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=inferno

# Comprehensive eval script for all best models

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching
source ~/.bashrc
conda activate DPFM

export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/diffusion_policy"

echo "=================================================="
echo "Evaluating All Best Models (DDPM + FM)"
echo "=================================================="
echo "Start time: $(date)"
nvidia-smi

# Create results directory
mkdir -p results/eval_final

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
from hydra.utils import instantiate

# Model checkpoints to evaluate
models = {
    # DDPM Baseline
    'DDPM_baseline_epoch500': {
        'ckpt': '/storage/project/r-agarg35-0/eliu354/projects/Diffusion-Policy-Flow-Matching/diffusion_policy/outputs/2025-11-27/04-36-24/checkpoints/epoch=0500-test_mean_score=0.892.ckpt',
        'steps': 100,
        'type': 'ddpm'
    },
    'DDPM_baseline_epoch850': {
        'ckpt': '/storage/project/r-agarg35-0/eliu354/projects/Diffusion-Policy-Flow-Matching/diffusion_policy/outputs/2025-11-27/04-36-24/checkpoints/epoch=0850-test_mean_score=0.882.ckpt',
        'steps': 100,
        'type': 'ddpm'
    },
    # FM step8
    'FM_step8_epoch900': {
        'ckpt': 'data/outputs/2025.11.27/04.36.04_train_fm_unet_hybrid_pusht_image/checkpoints/epoch=0900-test_mean_score=0.845.ckpt',
        'steps': 8,
        'type': 'fm'
    },
    # FM step4 (different configs)
    'FM_step4_s42_lr1e-4': {
        'ckpt': 'data/outputs/2025.11.27/04.36.29_train_fm_unet_hybrid_pusht_image/checkpoints/epoch=0850-test_mean_score=0.812.ckpt',
        'steps': 4,
        'type': 'fm'
    },
    'FM_step4_s42_lr2e-4': {
        'ckpt': 'data/outputs/2025.11.27/14.28.45_train_fm_unet_hybrid_pusht_image/checkpoints/epoch=0200-test_mean_score=0.838.ckpt',
        'steps': 4,
        'type': 'fm'
    },
    'FM_step8_v2': {
        'ckpt': 'data/outputs/2025.11.27/14.28.45_train_fm_unet_hybrid_pusht_image/checkpoints/epoch=0850-test_mean_score=0.821.ckpt',
        'steps': 8,
        'type': 'fm'
    },
}

results = {}

for name, info in models.items():
    print(f'\n' + '='*50)
    print(f'Evaluating: {name}')
    print('='*50)
    
    ckpt_path = info['ckpt']
    
    if not Path(ckpt_path).exists():
        print(f'  Checkpoint not found: {ckpt_path}')
        continue
    
    try:
        # Load checkpoint
        payload = torch.load(ckpt_path, map_location='cuda:0')
        cfg = payload['cfg']
        
        # Instantiate policy
        policy = instantiate(cfg.policy)
        
        # Load state dict (use ema_model if available)
        if 'ema_model' in payload['state_dicts']:
            state_dict = payload['state_dicts']['ema_model']
        else:
            state_dict = payload['state_dicts']['model']
        policy.load_state_dict(state_dict)
        policy.to('cuda:0')
        policy.eval()
        
        # Create runner
        runner = PushTImageRunner(
            output_dir=f'results/eval_final/{name}',
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
        
        # Run evaluation with timing
        print(f'  Running 50 test episodes...')
        
        # Warmup
        for _ in range(3):
            obs = torch.randn(1, 2, 3, 96, 96).to('cuda:0')
            agent_pos = torch.randn(1, 2, 2).to('cuda:0')
            with torch.no_grad():
                _ = policy.predict_action({'image': obs, 'agent_pos': agent_pos})
        torch.cuda.synchronize()
        
        # Measure latency
        latencies = []
        obs = torch.randn(1, 2, 3, 96, 96).to('cuda:0')
        agent_pos = torch.randn(1, 2, 2).to('cuda:0')
        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = policy.predict_action({'image': obs, 'agent_pos': agent_pos})
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
        
        latency_mean = np.mean(latencies)
        latency_std = np.std(latencies)
        latency_p50 = np.percentile(latencies, 50)
        latency_p95 = np.percentile(latencies, 95)
        
        # Run full evaluation
        start_time = time.time()
        result = runner.run(policy)
        eval_time = time.time() - start_time
        
        test_mean_score = result['test/mean_score']
        
        results[name] = {
            'test_mean_score': float(test_mean_score),
            'inference_steps': info['steps'],
            'type': info['type'],
            'latency_mean_ms': float(latency_mean),
            'latency_std_ms': float(latency_std),
            'latency_p50_ms': float(latency_p50),
            'latency_p95_ms': float(latency_p95),
            'eval_time_sec': eval_time,
            'ckpt_path': ckpt_path
        }
        
        print(f'  Test Mean Score: {test_mean_score:.4f}')
        print(f'  Latency: {latency_mean:.1f} ± {latency_std:.1f} ms')
        print(f'  Eval time: {eval_time:.1f}s')
        
    except Exception as e:
        print(f'  Error: {e}')
        import traceback
        traceback.print_exc()

# Save results
output_path = 'results/eval_final/eval_results.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

# Print summary table
print('\n' + '='*80)
print('EVALUATION SUMMARY')
print('='*80)
header = '{:<30} {:>8} {:>6} {:>15} {:>10}'.format('Model', 'Score', 'Steps', 'Latency (ms)', 'Speedup')
print(header)
print('-'*80)

# Sort by type (ddpm first, then fm)
sorted_results = sorted(results.items(), key=lambda x: (0 if x[1]['type']=='ddpm' else 1, -x[1]['test_mean_score']))

baseline_latency = None
for name, res in sorted_results:
    if res['type'] == 'ddpm' and baseline_latency is None:
        baseline_latency = res['latency_mean_ms']
    
    speedup = baseline_latency / res['latency_mean_ms'] if baseline_latency else 1.0
    row = '{:<30} {:>8.4f} {:>6} {:>15.1f} {:>10.1f}x'.format(name, res['test_mean_score'], res['inference_steps'], res['latency_mean_ms'], speedup)
    print(row)

print('='*80)
print(f'\nResults saved to: {output_path}')
"

echo ""
echo "Evaluation complete!"
echo "End time: $(date)"
