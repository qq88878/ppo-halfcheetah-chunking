from os.path import exists
import csv
import argparse

import gymnasium as gym
import torch
import numpy as np
from buffer import RolloutBuffer
from gae import compute_gae
from net import Actor, Critic
from ppo import PPO
from reward_wrapper import StandUpWrapper

rollout_size = 2048
epochs = 10
gamma = 0.99
lam = 0.95
max_iterations = 1000

parser = argparse.ArgumentParser()
parser.add_argument("--fresh", action="store_true",
                    help="从零训练，不加载已有的 actor.pt/critic.pt（默认：续训）")
parser.add_argument("--no-render", action="store_true",
                    help="训练完成后不弹窗渲染（默认：弹窗看效果）")
parser.add_argument("--iters", type=int, default=1000,
                    help="训练迭代次数（默认 1000）")
parser.add_argument("--k", type=int, default=4,
                    help="动作分块长度 K（默认 4）")
parser.add_argument("--tag", type=str, default="",
                    help="输出文件后缀，如 --tag k8 → actor_k8.pt / training_log_k8.csv")
parser.add_argument("--rolling", action="store_true",
                    help="滚动时域：决策输出整块但只执行第 1 步，每步重新决策")
args = parser.parse_args()
max_iterations = args.iters

tag = args.tag
actor_path = f"actor_{tag}.pt" if tag else "actor.pt"
critic_path = f"critic_{tag}.pt" if tag else "critic.pt"
log_path = f"training_log_{tag}.csv" if tag else "training_log.csv"

env = gym.make("HalfCheetah-v5")
env = StandUpWrapper(env, min_height=0.3, penalty=1.0)  # 加摔倒惩罚
obs_dim = env.observation_space.shape[0]  # 17
act_dim = env.action_space.shape[0]        # 6

actor = Actor(obs_dim, act_dim, K=args.k)   # K：一次决策输出 K×6 动作块
if not args.fresh and exists(actor_path):
   actor.load_state_dict(torch.load(actor_path))
critic = Critic(obs_dim)

if not args.fresh and exists(critic_path):
 critic.load_state_dict(torch.load(critic_path))
if args.rolling:
    # 滚动版：每行就是一个决策点，buffer 存整块动作（K×6 列）
    buffer = RolloutBuffer(rollout_size, obs_dim, act_dim * args.k)
else:
    buffer = RolloutBuffer(rollout_size, obs_dim, act_dim)

K = actor.K   # 动作块长度（默认 4）

obs, _ = env.reset()
episode_reward = 0
episode_rewards = []

for iteration in range(max_iterations):
    buffer.clear()

    step = 0
    while step < rollout_size:
        # ===== 决策点：决策 1 次，输出一整块动作 =====
        obs_t = torch.from_numpy(obs).float()
        with torch.no_grad():
            action_block, log_prob = actor.get_action(obs_t)   # (K,6) 动作块 + 整块 logp（标量）
        dec_obs_np = obs_t.numpy()                             # 决策点状态

        if args.rolling:
            # ===== 滚动时域：只执行第 1 步，下一步用新状态重新决策 =====
            action_np = action_block[0].numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            with torch.no_grad():
                value = critic(torch.from_numpy(obs).float()).item()
            buffer.store(obs, action_block.reshape(-1).numpy(), reward, done,
                         log_prob.item(), value, dec_obs_np)

            episode_reward += reward
            obs = next_obs
            step += 1

            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                obs, _ = env.reset()
        else:
            # ===== 开环连续执行 K 步（块中途不重新决策） =====
            for j in range(K):
                if step >= rollout_size:
                    break
                action_np = action_block[j].numpy()                # 第 j 步的动作
                next_obs, reward, terminated, truncated, _ = env.step(action_np)
                done = terminated or truncated
                with torch.no_grad():
                    value = critic(torch.from_numpy(obs).float()).item()   # 每步真实状态的 V(s)
                buffer.store(obs, action_np, reward, done, log_prob.item(), value, dec_obs_np)

                episode_reward += reward
                obs = next_obs
                step += 1

                if done:   # 块内终止：剩余动作丢弃，新 episode 从新状态重新决策
                    episode_rewards.append(episode_reward)
                    episode_reward = 0
                    obs, _ = env.reset()
                    break

    advantages, returns = compute_gae(buffer, gamma, lam)

    # === 3. PPO update ===
    ppo = PPO(actor, critic, buffer, epochs, advantages, returns, rolling=args.rolling)
    ppo.update()

    # === 4. 打印 reward + 记录日志（供 plot_results.py 出图） ===
    if iteration % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
        print(f"Iteration {iteration}, avg reward: {avg_reward:.1f}")
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if iteration == 0 and not exists(log_path):  # 首次写入表头（续训时不重复写）
                writer.writerow(["iteration", "avg_reward", "num_episodes"])
            writer.writerow([iteration, round(avg_reward, 2), len(episode_rewards)])

print("训练完成！")
# 保存 actor
torch.save(actor.state_dict(), actor_path)

# 保存 critic
torch.save(critic.state_dict(), critic_path)

if args.no_render:
    print("--no-render：跳过渲染，直接结束。")
    env.close()
    exit()

# 关闭训练环境，重新创建带渲染的环境
env.close()
env = gym.make("HalfCheetah-v5", render_mode="human")

print("测试开始")
obs, _ = env.reset()
for _ in range(100):
    env.render()
    obs_t = torch.from_numpy(obs).float()
    with torch.no_grad():
        action_block, _ = actor.get_action(obs_t)   # (K,6) 动作块
    if args.rolling:
        next_obs, _, _, _, _ = env.step(action_block[0].numpy())   # 滚动：只执行第 1 步
        obs = next_obs
    else:
        for j in range(action_block.shape[0]):          # 块内逐个执行
            next_obs, _, _, _, _ = env.step(action_block[j].numpy())
            obs = next_obs
print("测试结束")
env.close()