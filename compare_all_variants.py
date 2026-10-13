# -*- coding: utf-8 -*-
"""四版本训练曲线对比：基线 tanh 修复版 / 开环 K4 / 开环 K8 / 滚动 K4"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def load(path, fix_rollover=False):
    """读取 CSV；fix_rollover=True 时处理 iteration 重置（续训段自动偏移拼接成连续曲线）"""
    iters, rews = [], []
    offset = 0
    prev = None
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            it = int(row["iteration"])
            rw = float(row["avg_reward"])
            if fix_rollover and prev is not None and it < prev:
                offset += prev + 10          # 遇重置：续训段偏移到上一段末尾之后
            iters.append(it + offset)
            rews.append(rw)
            prev = it
    return iters, rews

def peak(iters, rews):
    i = rews.index(max(rews))
    return max(rews), iters[i]

def smooth(xs, ys, w=5):
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - w), min(len(ys), i + w + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return xs, out

base_it, base_rew = load(r"D:\develop\ppo-halfcheetah\log_tanh_fixed.csv")
k4_it, k4_rew = load(r"D:\develop\ppo-halfcheetah-chunking\training_log.csv")
k8_it, k8_rew = load(r"D:\develop\ppo-halfcheetah-chunking\training_log_k8.csv")
roll_it, roll_rew = load(r"D:\develop\ppo-halfcheetah-chunking\training_log_rolling.csv", fix_rollover=True)

fig, ax = plt.subplots(figsize=(11, 6), dpi=120)

ax.plot(*smooth(base_it, base_rew), color="#888888", lw=1.6,
        label="基线 PPO（tanh 修复版）峰值 2404@970")
ax.plot(*smooth(k4_it, k4_rew), color="#2563EB", lw=1.6,
        label="开环动作分块 K=4 峰值 2292@990")
ax.plot(*smooth(k8_it, k8_rew), color="#F87171", lw=1.4,
        label="开环动作分块 K=8（未学会）终点 -122@490")
ax.plot(*smooth(roll_it, roll_rew), color="#16A34A", lw=1.6,
        label="滚动时域 K=4 峰值 2423@970（500+1000 续训）")

for iters, rews, c, txt, dy in [
    (base_it, base_rew, "#888888", "峰值 2404", 10),
    (k4_it, k4_rew, "#2563EB", "峰值 2292", -18),
    (k8_it, k8_rew, "#F87171", "终点 -122", -18),
    (roll_it, roll_rew, "#16A34A", "峰值 2423", 10),
]:
    p, i = peak(iters, rews)
    ax.scatter([i], [p], color=c, zorder=5)
    ax.annotate(txt, (i, p), textcoords="offset points", xytext=(8, dy),
                fontsize=9, color=c, fontweight="bold")

ax.set_xlabel("iteration")
ax.set_ylabel("avg reward（近 10 episode）")
ax.set_title("HalfCheetah 四版本训练曲线对比：基线 vs 开环 K4/K8 vs 滚动 K4")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("reward_curve_all_variants.png", dpi=120)
print("saved: reward_curve_all_variants.png")
