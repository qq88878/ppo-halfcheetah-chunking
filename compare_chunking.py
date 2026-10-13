# -*- coding: utf-8 -*-
"""chunking 版 vs 基线 tanh 修复版 训练曲线对比"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def load(path):
    iters, rews = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iters.append(int(row["iteration"]))
            rews.append(float(row["avg_reward"]))
    return iters, rews

chunk_it, chunk_rew = load(r"D:\develop\ppo-halfcheetah-chunking\training_log.csv")
base_it, base_rew = load(r"D:\develop\ppo-halfcheetah\log_tanh_fixed.csv")

fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)
ax.plot(base_it, base_rew, color="#888888", lw=1.8, label="基线 PPO (tanh修复版)  峰值 2404@970")
ax.plot(chunk_it, chunk_rew, color="#2563EB", lw=1.8, label="动作分块版 K=4  峰值 2292@990")

peak_c = max(chunk_rew); i_c = chunk_it[chunk_rew.index(peak_c)]
peak_b = max(base_rew);  i_b = base_it[base_rew.index(peak_b)]
ax.scatter([i_c], [peak_c], color="#2563EB", zorder=5)
ax.scatter([i_b], [peak_b], color="#888888", zorder=5)
ax.annotate(f"chunking 峰值 {peak_c:.0f}@{i_c}", (i_c, peak_c), textcoords="offset points", xytext=(8, -16), fontsize=10, color="#2563EB")
ax.annotate(f"基线峰值 {peak_b:.0f}@{i_b}", (i_b, peak_b), textcoords="offset points", xytext=(-110, 10), fontsize=10, color="#555555")

ax.set_xlabel("iteration")
ax.set_ylabel("avg reward (近10个episode)")
ax.set_title("动作分块版 (K=4) vs 基线 PPO：HalfCheetah 训练曲线对比")
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("reward_curve_chunking_vs_base.png", dpi=120)
print("saved: reward_curve_chunking_vs_base.png")
