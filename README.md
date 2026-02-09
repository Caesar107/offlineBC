# BC Baseline - 纯 PyTorch 实现

使用 PyTorch 直接实现的 Behavior Cloning (BC) 训练，支持 D4RL 数据集。不依赖 imitation 库，更简单可靠。

## 安装依赖

```bash
conda activate BC  # 或你的环境
pip install -r requirements.txt

# 如果还没有安装 d4rl
pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
```

## 使用方法

### 运行所有实验

```bash
bash run_bc_offline.sh
```

### 单独运行单个实验

```bash
python train_bc_d4rl.py \
    --env_name walker2d-medium-v0 \
    --exp_name bc-walker2d-medium-run1 \
    --seed 42 \
    --n_epochs 100
```

## 实验配置

- **环境**: walker2d, hopper, halfcheetah
- **数据集**: medium, medium-expert, medium-replay
- **运行次数**: 每个配置运行 3 次（seed: 42, 43, 44）
- **训练轮数**: 100 epochs
- **批次大小**: 256
- **学习率**: 1e-3

## 输出

- 模型保存在 `models/` 目录下
- 每个实验会输出评估奖励

## 参数说明

- `--env_name`: D4RL 环境名称（如 `walker2d-medium-v0`）
- `--exp_name`: 实验名称（用于保存模型）
- `--seed`: 随机种子
- `--n_epochs`: 训练轮数

