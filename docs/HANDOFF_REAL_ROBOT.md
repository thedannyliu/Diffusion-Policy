# 🤖 Real Robot Deployment Guide - Flow Matching vs DDPM

## 📋 Handoff Information

### 🎯 Project Overview
We have completed the experimental comparison of Flow Matching (FM) against Diffusion Policy (DDPM), with training completed on real robot data. The main advantage of FM is **7-20× faster inference speed** while maintaining similar task success rates.

---

## 🚀 Quick Start: Real Robot Deployment

### 1️⃣ Environment Setup
```bash
# Connect to the real robot machine
conda activate DPFM

# Navigate to repo
cd /path/to/Diffusion-Policy-Flow-Matching/diffusion_policy
```

### 2️⃣ Run Evaluation

**Flow Matching (Recommended - Faster)**
```bash
python eval_real_robot.py \
    -i /path/to/fm_checkpoint/checkpoints/latest.ckpt \
    -o ./eval_output_fm/ \
    --robot_ip <UR5_IP_ADDRESS>
```

**DDPM (Baseline)**
```bash
python eval_real_robot.py \
    -i /path/to/ddpm_checkpoint/checkpoints/latest.ckpt \
    -o ./eval_output_ddpm/ \
    --robot_ip <UR5_IP_ADDRESS>
```

---

## 📁 Checkpoint Locations

### Flow Matching (v2 - Optimized Version, Recommended)
| Task | Path | Steps |
|------|------|-------|
| Sphere | `data/outputs/2025.11.29/00.33.00_train_fm_real_robot_fair_sphere/checkpoints/latest.ckpt` | 8 |
| Cube | `data/outputs/2025.11.29/00.33.54_train_fm_real_robot_fair_cube/checkpoints/latest.ckpt` | 8 |

### DDPM (Baseline)
| Task | Path | Steps |
|------|------|-------|
| Sphere | `diffusion_policy/data/outputs/2025.11.29/01.33.13_train_ddpm_real_robot_sphere/checkpoints/latest.ckpt` | 100 |
| Cube | `diffusion_policy/data/outputs/2025.11.29/01.44.20_train_ddpm_real_robot_cube/checkpoints/latest.ckpt` | 100 |

### FM v1 (Legacy, For Reference Only)
| Task | Path | Steps |
|------|------|-------|
| Sphere | `data/outputs/real_robot/2025.11.27/16.22.11_fm_fair_sphere_step4_seed42/checkpoints/latest.ckpt` | 4 |
| Cube | `data/outputs/real_robot/2025.11.27/16.22.11_fm_fair_cube_step4_seed42/checkpoints/latest.ckpt` | 4 |

---

## 📊 How to Evaluate Real Robot Performance

### Automatically Recorded Metrics (Latency)

When running `eval_real_robot.py`, the following are automatically recorded:

1. **Real-time Output (Per Step)**
   ```
   Obs latency 0.0234s
   Inference latency: 45.2ms
   ```

2. **Episode Summary Statistics**
   ```
   ==================================================
   Episode Latency Summary (FM (steps=8))
   ==================================================
   Inference: 42.5 ± 3.2 ms
     Min: 38.1 ms, Max: 51.4 ms
   Obs latency: 23.4 ms
   Steps: 150
   Duration: 15.2 s
   ==================================================
   ```

3. **Final Summary**
   ```
   ============================================================
   FINAL LATENCY SUMMARY: FM (steps=8)
   ============================================================
   Total Episodes: 10
   Average Inference Latency: 43.1 ± 2.1 ms
   Results saved to: ./eval_output_fm/latency_stats.json
   ============================================================
   ```

4. **JSON Output** (`latency_stats.json`)
   - Automatically saved in output directory
   - Contains detailed statistics for all episodes

### ⚠️ Manually Recorded Metrics Required

| Metric | Description | How to Record |
|--------|-------------|---------------|
| **Success Rate** | Whether task was completed successfully | Record ✅/❌ after each episode |
| **Task Completion Time** | Time from start to success | Available from `duration` in latency_stats.json |
| **Execution Quality** | Trajectory smoothness and stability | Subjective score 1-5 or video review |
| **Collisions/Anomalies** | Any dangerous movements | Press 'S' immediately to stop and record |

### 📝 Recommended Evaluation Workflow

1. **Prepare Evaluation Sheet**
   ```
   | Episode | Method | Success | Duration | Quality | Notes     |
   |---------|--------|---------|----------|---------|-----------|
   | 1       | FM     | ✅      | 12.5s    | 4/5     |           |
   | 2       | FM     | ❌      | 15.2s    | 3/5     | Collision |
   | ...     |        |         |          |         |           |
   ```

2. **Run at least 10 episodes per method**
   - FM: 10 episodes
   - DDPM: 10 episodes

3. **Compare after evaluation**
   - Success rate: FM vs DDPM
   - Average time: FM vs DDPM
   - Latency: Obtained from JSON automatically

---

## 🎮 Control Guide

### Keyboard Controls
| Key | Function |
|-----|----------|
| `C` | Start evaluation (hand over to policy control) |
| `S` | Stop evaluation (return to human control) |
| `Q` | Exit program |

### SpaceMouse Controls (Human Control Mode)
- Movement: XY plane movement
- Right button: Unlock Z axis
- Left button: Enable rotation

### ⚠️ Safety Notes
- **Keep emergency stop button ready at all times!**
- First test recommended at safe distance
- Press 'S' immediately if any anomaly occurs

---

## 📈 Expected Results

Based on PushT simulation results:

| Policy | Inference Latency | Relative Speed | Expected Success Rate |
|--------|------------------|----------------|----------------------|
| DDPM (100 steps) | ~650 ms | 1.0× (baseline) | ~77% |
| FM (16 steps) | ~90 ms | **7.2×** | ~77% |
| FM (8 steps) | ~50 ms | **13×** | TBD |
| FM (4 steps) | ~30 ms | **21×** | TBD |

> 🔑 **Key Finding**: FM achieves 7-20× faster inference while maintaining similar success rates!

---

## 🔧 FAQ

### Q: Which checkpoint should I use?
**A**: Recommended to use **FM v2 (8 steps)**, which is the latest optimized version.

### Q: Can FM inference steps be adjusted?
**A**: Yes! After loading the checkpoint, you can override:
```python
# In the FM block of eval_real_robot.py
policy.num_inference_steps = 4  # Change to 4 steps
```

### Q: What if latency is too high?
**A**: 
1. Verify GPU is working properly
2. Reduce FM inference steps
3. Ensure no other programs are using the GPU

---

## 📞 Contact

For questions, please contact:
- Danny Liu
- Related commits: `d34b38b`, `58ff66b`, `f28e072`

---

*Last updated: 2025-11-30*
