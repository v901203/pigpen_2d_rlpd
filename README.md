# Pig Pen 2D RLPD Training & Testing

強化學習豬舍環境模擬與避障訓練項目。使用 **RLPD (Reinforcement Learning from Prior Data)** 結合專家示範資料進行多機並行訓練。

## 🎯 專案概述

本項目實現一個 2D 豬舍環境，小車需要在迴避動物與障礙物的同時，依序到達 8 間豬舍的目標點。訓練策略結合：
- **專家示範資料** (RLPD 預訓練打底)
- **線上探索** (強化學習)
- **多機並行訓練** (分佈式架構)

## 📋 系統需求

- Python 3.10+
- CUDA 12.0+（推薦，CPU 可用但較慢）
- 虛擬環境管理工具 (conda / venv)

## 🔧 依賴安裝

### 1. 建立虛擬環境
```bash
# 若尚未有虛擬環境
python3 -m venv ~/Desktop/pig_pen_2d_rl/rl_env
source ~/Desktop/pig_pen_2d_rl/rl_env/bin/activate
```

### 2. 安裝核心套件
```bash
pip install -r requirements.txt
```

### 3. 驗證環境
```bash
python -c "import jax, flax, pygame; print('✓ All packages loaded')"
```

## 📁 專案結構

```
pig_pen_2d_rlpd/
├── pig_pen_env.py              # 環境定義（2D 豬舍模擬）
├── record_expert.py            # 錄製專家示範腳本
├── train_rlpd_2d.py            # 單機訓練腳本
├── train_rlpd_2d_parallel.py   # 多機並行訓練腳本
├── test_rlpd_2d.py             # 推論測試腳本
├── expert_data.npz             # 專家示範資料
├── configs/                    # 訓練配置檔
│   ├── sac_config.py
│   ├── td_config.py
│   └── ...
├── rlpd/                       # RLPD 框架代碼
│   ├── agents/                 # Agent 實現 (SAC, DRQ, etc.)
│   ├── data/                   # 資料集與 buffer
│   ├── networks/               # 神經網路架構
│   └── ...
└── checkpoints/                # 訓練模型存檔 (已追蹤，但不推送)
```

## 🎬 快速開始

### 錄製專家示範
```bash
/home/vito/Desktop/pig_pen_2d_rl/rl_env/bin/python record_expert.py
```
操作方式：
- **SPACE** - 開始錄製
- **↑↓←→ / WASD** - 控制小車
- **ESC** - 結束並存檔

產生 `expert_data.npz`（包含 obs, actions, rewards, next_obs, dones）。

### 單機訓練
```bash
/home/vito/Desktop/pig_pen_2d_rl/rl_env/bin/python train_rlpd_2d.py
```

參數選項：
- `--seed` - 隨機種子 (default: 42)
- `--max_steps` - 訓練步數 (default: 100000)
- `--batch_size` - 批大小 (default: 256)
- `--expert_data_path` - 專家資料路徑 (default: "expert_data.npz")

### 多機並行訓練 (推薦)
```bash
/home/vito/Desktop/pig_pen_2d_rl/rl_env/bin/python train_rlpd_2d_parallel.py \
  --max_steps 2000000 \
  --num_envs 8
```

參數選項：
- `--num_envs` - 並行環境數 (default: 4)
- `--seed` - 隨機種子
- `--resume_dir` - 恢復訓練的 checkpoint 目錄

### 推論測試
```bash
/home/vito/Desktop/pig_pen_2d_rl/rl_env/bin/python test_rlpd_2d.py
```

自動尋找最新 checkpoint 並進行推論。也可手動指定：
```bash
/home/vito/Desktop/pig_pen_2d_rl/rl_env/bin/python test_rlpd_2d.py \
  --checkpoint_dir rlpd/s42_online/checkpoints
```

## 🤖 環境說明

### 觀察空間 (38D)
- **LIDAR 射線** (36D) - 前方 36 道距離傳感器
- **相對距離** (1D) - 到目標點的相對距離
- **相對角度** (1D) - 到目標點的相對方向

### 動作空間 (2D，連續)
- **action[0]** ∈ [-1, 1] - 前進/後退速度
- **action[1]** ∈ [-1, 1] - 轉向角速度

### 獎勵機制
- **接近獎勵** - 每步縮短到目標點的距離 (+0.5 × 縮短距離)
- **時間懲罰** - 每步 -0.02
- **撞擊懲罰** - 撞到豬隻或牆壁 -50
- **目標獎勵** - 到達目標點 +50～1000（依序列進度）
- **停滯處理** - 長時間無進展時，等待行為 +0.5，亂動 -0.2

## 📊 訓練監控

Tensorboard 日誌存儲在 `runs/RLPD_2D_*` 目錄。

```bash
tensorboard --logdir runs/
```

## ⚙️ 配置調整

編輯 `configs/sac_config.py` 以修改：
- 學習率 (lr)
- 熵係數 (temperature)
- 評估間隔
- 批大小

## 🐛 常見問題

### Checkpoint 維度不符
- `test_rlpd_2d.py` 已支援自動對齊 obs/action 維度。

### 無法錄製專家示範
- 確保 pygame 視窗已聚焦（點擊視窗）
- 按 SPACE 開始錄製後再操控小車
- KEYDOWN/KEYUP 事件應被正確捕獲

### 訓練速度慢
- 使用 `train_rlpd_2d_parallel.py` 並增加 `--num_envs`
- 確認 CUDA/GPU 配置正確

## 📝 License

This project is provided as-is for research purposes.

## 🙏 致謝

基於 RLPD 框架：https://github.com/ikostrikov/rl_with_implicit_models

改編自豬舍環境模擬研究專案。
