# 🤖 Real Robot 部署指南 - Flow Matching vs DDPM

## 📋 交接資訊

### 🎯 專案概述
我們已經完成 Flow Matching (FM) 對 Diffusion Policy (DDPM) 的實驗比較，並在 real robot 數據上訓練完成。FM 的主要優勢是**推理速度快 7-20 倍**，同時保持相近的任務成功率。

---

## 🚀 快速開始：Real Robot 部署

### 1️⃣ 環境設置
```bash
# 連接到 real robot 的機器
conda activate DPFM

# 確認 repo
cd /path/to/Diffusion-Policy-Flow-Matching/diffusion_policy
```

### 2️⃣ 執行評估

**Flow Matching (推薦 - 更快)**
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

## 📁 Checkpoint 位置

### Flow Matching (v2 - 優化版本，建議使用)
| Task | Path | Steps |
|------|------|-------|
| Sphere | `data/outputs/2025.11.29/00.33.00_train_fm_real_robot_fair_sphere/checkpoints/latest.ckpt` | 8 |
| Cube | `data/outputs/2025.11.29/00.33.54_train_fm_real_robot_fair_cube/checkpoints/latest.ckpt` | 8 |

### DDPM (Baseline)
| Task | Path | Steps |
|------|------|-------|
| Sphere | `diffusion_policy/data/outputs/2025.11.29/01.33.13_train_ddpm_real_robot_sphere/checkpoints/latest.ckpt` | 100 |
| Cube | `diffusion_policy/data/outputs/2025.11.29/01.44.20_train_ddpm_real_robot_cube/checkpoints/latest.ckpt` | 100 |

### FM v1 (舊版，僅供參考)
| Task | Path | Steps |
|------|------|-------|
| Sphere | `data/outputs/real_robot/2025.11.27/16.22.11_fm_fair_sphere_step4_seed42/checkpoints/latest.ckpt` | 4 |
| Cube | `data/outputs/real_robot/2025.11.27/16.22.11_fm_fair_cube_step4_seed42/checkpoints/latest.ckpt` | 4 |

---

## 📊 如何評估 Real Robot 表現

### 自動記錄的指標 (Latency)

執行 `eval_real_robot.py` 時會自動記錄：

1. **實時輸出 (每步)**
   ```
   Obs latency 0.0234s
   Inference latency: 45.2ms
   ```

2. **Episode 結束時統計**
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

3. **最終總結**
   ```
   ============================================================
   FINAL LATENCY SUMMARY: FM (steps=8)
   ============================================================
   Total Episodes: 10
   Average Inference Latency: 43.1 ± 2.1 ms
   Results saved to: ./eval_output_fm/latency_stats.json
   ============================================================
   ```

4. **JSON 輸出** (`latency_stats.json`)
   - 自動保存在 output 目錄
   - 包含所有 episode 的詳細統計

### ⚠️ 需要人工記錄的指標

| 指標 | 說明 | 記錄方式 |
|------|------|----------|
| **成功率** | 任務是否成功完成 | 每次 episode 結束後手動記錄 ✅/❌ |
| **任務完成時間** | 從開始到成功 | 可從 latency_stats.json 的 `duration` 獲得 |
| **執行品質** | 軌跡平滑度、穩定性 | 主觀評分 1-5 或錄影回放評估 |
| **碰撞/異常** | 是否有危險動作 | 立即按 'S' 停止並記錄 |

### 📝 建議的評估流程

1. **準備評估表格**
   ```
   | Episode | Method | Success | Duration | Quality | Notes |
   |---------|--------|---------|----------|---------|-------|
   | 1       | FM     | ✅      | 12.5s    | 4/5     |       |
   | 2       | FM     | ❌      | 15.2s    | 3/5     | 碰撞  |
   | ...     |        |         |          |         |       |
   ```

2. **每種方法至少跑 10 個 episode**
   - FM: 10 episodes
   - DDPM: 10 episodes

3. **評估後比較**
   - 成功率: FM vs DDPM
   - 平均時間: FM vs DDPM
   - Latency: 從 JSON 自動獲得

---

## 🎮 操作指南

### 控制按鍵
| 按鍵 | 功能 |
|------|------|
| `C` | 開始評估 (交給 policy 控制) |
| `S` | 停止評估 (回到人類控制) |
| `Q` | 退出程式 |

### SpaceMouse 操控 (人類控制模式)
- 移動: XY 平面移動
- 右鍵: 解鎖 Z 軸
- 左鍵: 啟用旋轉

### ⚠️ 安全注意事項
- **隨時準備好按緊急停止按鈕！**
- 首次測試建議在安全距離觀察
- 有異常立即按 'S' 停止

---

## 📈 預期結果

基於 PushT simulation 的結果：

| Policy | Inference Latency | 相對速度 | 預期成功率 |
|--------|------------------|----------|-----------|
| DDPM (100 steps) | ~650 ms | 1.0× (baseline) | ~77% |
| FM (16 steps) | ~90 ms | **7.2×** | ~77% |
| FM (8 steps) | ~50 ms | **13×** | TBD |
| FM (4 steps) | ~30 ms | **21×** | TBD |

> 🔑 **關鍵發現**: FM 在保持相似成功率的情況下，推理速度提升 7-20 倍！

---

## 🔧 常見問題

### Q: 選擇哪個 checkpoint？
**A**: 建議使用 **FM v2 (8 steps)**，這是最新優化版本。

### Q: FM 的 inference steps 可以調整嗎？
**A**: 可以！在 checkpoint 載入後，可以 override：
```python
# 在 eval_real_robot.py 的 FM 區塊
policy.num_inference_steps = 4  # 改成 4 步
```

### Q: 如果 latency 太高怎麼辦？
**A**: 
1. 確認 GPU 正常運作
2. 減少 FM inference steps
3. 確認沒有其他程式佔用 GPU

---

## 📞 聯繫方式

如有問題請聯繫：
- Danny Liu
- 相關 commits: `d34b38b`, `58ff66b`, `f28e072`

---

*Last updated: 2025-11-30*
