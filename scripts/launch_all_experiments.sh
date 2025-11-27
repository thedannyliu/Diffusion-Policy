#!/bin/bash
# === Launch All Experiments ===
# This script submits all experiment jobs for the paper-aligned Flow Matching study
#
# Experiments:
# 1. FM step sweep: {2, 4, 8, 16} with seed=42
# 2. Multi-seed: steps=4 with seeds {42, 123}
# 3. LR sweep: steps=4, seed=42 with LR {5e-5, 1e-4, 2e-4}
# 4. Baseline (DDPM): seeds {42, 123}
#
# Note: Steps 2 duplicates steps=4,seed=42 from step 1, so we skip it there
# Total jobs: 4 (step sweep) + 1 (seed=123 only) + 2 (LR non-default) + 2 (baseline) = 9
#
# Job naming convention:
#   FM: fm_<steps>step_s<seed>_lr<lr>
#   DDPM: ddpm_s<seed>

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching

echo "=================================================="
echo "Launching All Experiments"
echo "=================================================="
echo "Total training time: ~24h per job (3050 epochs)"
echo "Account: gts-agarg35-ideas_l40s"
echo "Partition: gpu-l40s"
echo "=================================================="

# Create logs directory
mkdir -p logs

# === 1. FM Step Sweep (seed=42, lr=1e-4) ===
echo ""
echo "=== FM Step Sweep ==="
for STEPS in 2 4 8 16; do
    echo "Submitting FM steps=${STEPS}, seed=42, lr=1e-4..."
    sbatch scripts/train_fm_hybrid.sh ${STEPS} 42 1e-4
    sleep 1
done

# === 2. Multi-Seed (steps=4, lr=1e-4) ===
# Note: seed=42 with steps=4 is already submitted in step sweep
echo ""
echo "=== Multi-Seed Experiments ==="
echo "Submitting FM steps=4, seed=123..."
sbatch scripts/train_fm_hybrid.sh 4 123 1e-4
sleep 1

# === 3. LR Sweep (steps=4, seed=42) ===
# Note: lr=1e-4 is already submitted in step sweep
echo ""
echo "=== LR Sweep ==="
for LR in 5e-5 2e-4; do
    echo "Submitting FM steps=4, seed=42, lr=${LR}..."
    sbatch scripts/train_fm_hybrid.sh 4 42 ${LR}
    sleep 1
done

# === 4. Baseline DDPM (multi-seed) ===
echo ""
echo "=== Baseline DDPM ==="
for SEED in 42 123; do
    echo "Submitting DDPM baseline seed=${SEED}..."
    sbatch scripts/train_baseline_hybrid.sh ${SEED}
    sleep 1
done

echo ""
echo "=================================================="
echo "All jobs submitted! Total: 9 jobs"
echo "Use 'squeue -u \$USER' to check status"
echo "=================================================="
echo ""
echo "Jobs breakdown:"
echo "  - Step sweep (2,4,8,16), seed=42: 4 jobs"
echo "  - Multi-seed (steps=4, seed=123): 1 job"
echo "  - LR sweep (5e-5, 2e-4), steps=4, seed=42: 2 jobs"
echo "  - Baseline DDPM (seed=42,123): 2 jobs"
echo ""
echo "WandB project: dpfm_pusht_v2"
echo "WandB groups: fm_step_sweep, fm_lr_sweep, baseline"
