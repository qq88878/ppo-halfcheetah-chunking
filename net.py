import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
  def __init__(self,state_dim=17,action_dim=6,hidden=64,K=4):
     super().__init__()
     self.K=K
     self.action_dim=action_dim
     self.fc1=nn.Linear(state_dim,hidden)
     self.fc2=nn.Linear(hidden,hidden)
     self.fc3=nn.Linear(hidden,action_dim*K)   # 一次输出 K×6=24 维
     self.log_std=nn.Parameter(torch.zeros(action_dim*K)-0.5)
  def forward(self,obs):
     x=F.relu(self.fc1(obs))
     x=F.relu(self.fc2(x))
     return self.fc3(x)
  def get_dist(self,obs):
     mean=self.forward(obs)
     std=self.log_std.exp().clamp(min=1e-3)
     return torch.distributions.Normal(mean,std)
  def get_action(self,obs):
     dist=self.get_dist(obs)
     u=dist.sample()                                      # (24,) 平铺采样
     action_flat=torch.tanh(u)                            # (24,) 平铺压缩
     action=action_flat.view(self.K,self.action_dim)      # reshape 成 (4,6) 动作块
     log_prob=dist.log_prob(u).sum(-1)-torch.log(1-action_flat**2+1e-6).sum(-1)  # 整块联合概率（标量）
     return action,log_prob
class Critic(nn.Module):
 def __init__(self,obs=17,hidden=64):
     super().__init__()
     self.fc1=nn.Linear(obs,hidden)
     self.fc2=nn.Linear(hidden,hidden)
     self.fc3=nn.Linear(hidden,1)
 def forward(self,obs):
     x=F.relu(self.fc1(obs))
     x=F.relu(self.fc2(x))
     return self.fc3(x)
