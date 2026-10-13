import torch.optim
import torch.nn.functional as F
from buffer import RolloutBuffer
from net import Actor, Critic


class PPO():
    def __init__(self,actor:Actor,critic:Critic,buffer:RolloutBuffer,epoch,advantages,returns,rolling=False):
        self.actor=actor
        self.critic=critic
        self.buffer=buffer
        self.returns=returns       #还给critic网络的
        self.advantages=advantages #还给actor网络的
        self.epoch=epoch           #用一轮buffer数据调节参数轮数
        self.rolling=rolling       #滚动时域模式：每行就是一个完整决策点（不按 K 拼块）
        self.lr=3e-4               #梯度下降的学习率
        self.clip_eps=0.2          #新旧参数的比值和这个进行比较，用来处理advantage实现控制变化不要太大的作用
        self.ent_coef=0.01         #损失函数里熵值的比率，降低影响的同时增大尝试的幅度
        self.val_coef=0.5          #critic网络预测结果在损失函数中乘的比率（这个是为啥）

    def update(self):
        n = self.buffer.size
        K = self.actor.K
        act_dim = self.actor.action_dim

        # ---- critic 用：每步真实状态（不变） ----
        obs_all = torch.from_numpy(self.buffer.obs[:n])
        ret = torch.from_numpy(self.returns)
        adv = torch.from_numpy(self.advantages)

        if self.rolling:
            # ===== 滚动时域：每行都是一个决策点，动作已是整块（K×6 列） =====
            dec_obs = torch.from_numpy(self.buffer.dec_obs[:n])          # (n,17) 每行独立决策点
            actions_block = torch.from_numpy(self.buffer.actions[:n])    # (n,K*6) 每行已是整块
            old_logp_block = torch.from_numpy(self.buffer.log_probs[:n]) # (n,) 每行整块 logp
            adv_block = adv                                              # 每行一个 advantage，直接对应
            adv_block = (adv_block - adv_block.mean()) / (adv_block.std() + 1e-8)
        else:
            # ===== 开环分块：每 K 行共享一个决策点，把单步动作拼回整块 =====
            n_blocks = n // K                              # 决策点数量
            dec_obs = torch.from_numpy(self.buffer.dec_obs[:n])[::K]     # (n_blocks,17) 每块取第一行
            actions_block = torch.from_numpy(self.buffer.actions[:n]).view(n_blocks, K * act_dim)  # 把 K 行动作拼回整块
            old_logp_block = torch.from_numpy(self.buffer.log_probs[:n])[::K]   # 每块一个整块 logp
            adv_block = adv.view(n_blocks, K).mean(-1)        # 块内 K 个 advantage 取平均
            adv_block = (adv_block - adv_block.mean()) / (adv_block.std() + 1e-8)   # 块级归一化

        optimizer=torch.optim.Adam(list(self.actor.parameters())+list(self.critic.parameters()),self.lr)

        for _ in range(self.epoch):
            dist=self.actor.get_dist(dec_obs)             # 决策点状态 → 整块 24 维分布
            u=0.5*torch.log((1+actions_block.clamp(-0.9999,0.9999))/(1-actions_block.clamp(-0.9999,0.9999)))
            new_logp=dist.log_prob(u).sum(-1)-torch.log(1-actions_block**2+1e-6).sum(-1)   # 整块联合概率

            ratio=torch.exp(new_logp-old_logp_block)      # 块级新旧概率比

            sur1=ratio*adv_block
            sur2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_block

            policy_loss=-torch.min(sur1,sur2).mean()

            values=self.critic(obs_all).squeeze(-1)       # 每步状态 → 每步 V(s)（不变）
            value_loss=F.mse_loss(values,ret)

            entropy=dist.entropy().mean()                 # 决策点分布的熵

            loss=policy_loss+self.val_coef*value_loss-self.ent_coef*entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters())+list(self.critic.parameters()), 0.5)
            optimizer.step()

