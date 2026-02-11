#!/usr/bin/env python3
"""
使用 stable-baselines3 实现 BC (Behavior Cloning)
支持 D4RL 数据集
不依赖 imitation 库，直接使用 stable-baselines3
"""
import argparse
import numpy as np
import gym
import d4rl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import csv
from tqdm import tqdm
from datetime import datetime


class D4RLDataset(Dataset):
    """D4RL 数据集包装器"""
    def __init__(self, observations, actions):
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


class BCPolicy(nn.Module):
    """简单的 BC 策略网络"""
    def __init__(self, obs_dim, action_dim, hidden_size=256, n_layers=2):
        super(BCPolicy, self).__init__()
        
        layers = []
        layers.append(nn.Linear(obs_dim, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_size, action_dim))
        layers.append(nn.Tanh())  # 假设动作在 [-1, 1] 范围内
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs):
        return self.network(obs)


def load_d4rl_dataset(env_name):
    """加载 D4RL 数据集"""
    env = gym.make(env_name)
    dataset = env.get_dataset()
    
    observations = dataset['observations'].astype(np.float32)
    actions = dataset['actions'].astype(np.float32)
    
    # 如果动作不在 [-1, 1] 范围内，需要归一化
    action_max = np.abs(actions).max()
    if action_max > 1.0:
        print(f"Actions are in range [-{action_max:.2f}, {action_max:.2f}], normalizing to [-1, 1]")
        actions = actions / action_max
    
    return observations, actions, env


def train_bc(env_name, exp_name, seed=42, n_epochs=100, batch_size=256, lr=1e-3, data_ratio=1.0):
    """训练 BC 模型"""
    print(f"==========================================")
    print(f"Training BC on: {env_name}")
    print(f"Experiment name: {exp_name}")
    print(f"Seed: {seed}")
    print(f"Epochs: {n_epochs}")
    print(f"Data ratio: {data_ratio}")
    print(f"==========================================")
    
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 强制使用 CPU
    device = torch.device('cpu')
    
    # 加载 D4RL 数据集
    print("Loading D4RL dataset...")
    observations, actions, env = load_d4rl_dataset(env_name)
    print(f"Full dataset size: {len(observations)} transitions")
    
    # 按比例截取数据
    if data_ratio < 1.0:
        n_samples = int(len(observations) * data_ratio)
        # 随机打乱后截取，保证不同 seed 取到不同子集
        indices = np.random.permutation(len(observations))[:n_samples]
        observations = observations[indices]
        actions = actions[indices]
        print(f"Using {data_ratio*100:.0f}% of data: {len(observations)} transitions")
    
    print(f"Dataset size: {len(observations)} transitions")
    print(f"Observation shape: {observations.shape}")
    print(f"Action shape: {actions.shape}")
    
    # 创建数据集和数据加载器
    dataset = D4RLDataset(observations, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建策略网络
    obs_dim = observations.shape[1]
    action_dim = actions.shape[1]
    policy = BCPolicy(obs_dim, action_dim, hidden_size=256, n_layers=2)
    policy = policy.to(device)
    print("Using CPU (forced)")
    
    # 优化器
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # 训练
    print("Training BC model...")
    policy.train()
    
    # 创建 CSV 文件记录 loss 和 reward
    csv_dir = "logs"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{exp_name}_training.csv")
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'loss', 'reward', 'timestamp'])
    
    # 使用 tqdm 显示训练进度
    epoch_pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    
    # 保存最后一个 epoch 的 loss
    final_loss = None
    
    # 评估频率：每个 epoch 都评估一次
    eval_freq = 1
    
    for epoch in epoch_pbar:
        total_loss = 0.0
        n_batches = 0
        
        # 使用 tqdm 显示每个 epoch 内的 batch 进度
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}", 
                         leave=False, unit="batch")
        
        for obs_batch, action_batch in batch_pbar:
            # 使用 CPU
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            
            # 前向传播
            predicted_actions = policy(obs_batch)
            loss = criterion(predicted_actions, action_batch)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            # 更新 batch 进度条
            batch_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / n_batches
        final_loss = avg_loss  # 保存最后一个 epoch 的 loss
        
        # 每个 epoch 都评估一次以显示 reward
        current_reward = None
        if (epoch + 1) % eval_freq == 0:
            policy.eval()
            eval_reward = 0.0
            n_eval_episodes = 3  # 快速评估，只用 3 个 episode
            
            with torch.no_grad():
                for _ in range(n_eval_episodes):
                    obs = env.reset()
                    episode_reward = 0.0
                    done = False
                    step_count = 0
                    max_steps = 1000
                    
                    while not done and step_count < max_steps:
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                        action = policy(obs_tensor).cpu().numpy()[0]
                        
                        # 如果动作被归一化了，需要反归一化
                        action_max = np.abs(actions).max()
                        if action_max > 1.0:
                            action = action * action_max
                        
                        obs, reward, done, info = env.step(action)
                        episode_reward += reward
                        step_count += 1
                    
                    eval_reward += episode_reward
            
            current_reward = eval_reward / n_eval_episodes
            policy.train()  # 切换回训练模式
        
        # 记录到 CSV（包含 loss 和 reward）
        reward_str = f'{current_reward:.2f}' if current_reward is not None else ''
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch + 1, f'{avg_loss:.6f}', reward_str, datetime.now().isoformat()])
        
        # 更新 epoch 进度条
        if current_reward is not None:
            epoch_pbar.set_postfix({
                'loss': f'{avg_loss:.6f}',
                'reward': f'{current_reward:.2f}'
            })
        else:
            epoch_pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
    
    print(f"Training completed! Loss history saved to {csv_path}")
    
    # 评估
    print("Evaluating policy...")
    policy.eval()
    total_reward = 0.0
    episode_rewards = []
    n_episodes = 10
    
    # 使用 tqdm 显示评估进度
    eval_pbar = tqdm(range(n_episodes), desc="Evaluating", unit="episode")
    
    with torch.no_grad():
        for ep in eval_pbar:
            obs = env.reset()
            episode_reward = 0.0
            done = False
            step_count = 0
            max_steps = 1000
            
            while not done and step_count < max_steps:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action = policy(obs_tensor).cpu().numpy()[0]
                
                # 如果动作被归一化了，需要反归一化
                action_max = np.abs(actions).max()
                if action_max > 1.0:
                    action = action * action_max
                
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                step_count += 1
            
            total_reward += episode_reward
            episode_rewards.append(episode_reward)
            eval_pbar.set_postfix({'reward': f'{episode_reward:.2f}'})
    
    avg_reward = total_reward / n_episodes
    std_reward = np.std(episode_rewards)
    print(f"Average evaluation reward over {n_episodes} episodes: {avg_reward:.2f} ± {std_reward:.2f}")
    
    # 记录评估结果到 CSV
    eval_csv_path = os.path.join(csv_dir, f"{exp_name}_evaluation.csv")
    with open(eval_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['episode', 'reward', 'timestamp'])
        for i, rew in enumerate(episode_rewards):
            writer.writerow([i + 1, f'{rew:.2f}', datetime.now().isoformat()])
        writer.writerow(['average', f'{avg_reward:.2f}', datetime.now().isoformat()])
        writer.writerow(['std', f'{std_reward:.2f}', datetime.now().isoformat()])
    
    print(f"Evaluation results saved to {eval_csv_path}")
    
    # 保存模型
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{exp_name}.pth")
    
    # 计算 action_max（用于反归一化）
    action_max = np.abs(actions).max()
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'action_max': action_max,
    }, model_path)
    print(f"Model saved to {model_path}")
    
    # 保存汇总信息到 CSV
    summary_csv_path = os.path.join(csv_dir, f"{exp_name}_summary.csv")
    with open(summary_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['metric', 'value', 'timestamp'])
        writer.writerow(['env_name', env_name, datetime.now().isoformat()])
        writer.writerow(['exp_name', exp_name, datetime.now().isoformat()])
        writer.writerow(['seed', seed, datetime.now().isoformat()])
        writer.writerow(['n_epochs', n_epochs, datetime.now().isoformat()])
        writer.writerow(['data_ratio', data_ratio, datetime.now().isoformat()])
        writer.writerow(['n_transitions', len(observations), datetime.now().isoformat()])
        writer.writerow(['final_loss', f'{final_loss:.6f}', datetime.now().isoformat()])
        writer.writerow(['avg_reward', f'{avg_reward:.2f}', datetime.now().isoformat()])
        writer.writerow(['std_reward', f'{std_reward:.2f}', datetime.now().isoformat()])
        writer.writerow(['n_episodes', n_episodes, datetime.now().isoformat()])
    
    print(f"Summary saved to {summary_csv_path}")
    
    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BC on D4RL datasets")
    parser.add_argument("--env_name", type=str, required=True, help="D4RL environment name")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--data_ratio", type=float, default=1.0, help="Ratio of data to use (0.0-1.0)")
    
    args = parser.parse_args()
    
    train_bc(
        env_name=args.env_name,
        exp_name=args.exp_name,
        seed=args.seed,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_ratio=args.data_ratio,
    )
