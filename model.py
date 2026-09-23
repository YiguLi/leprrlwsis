
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# from ipdb import set_trace as debug  # unused, commented out

def fanout_init(size, fanout=None):
    """**fan_out** 口径的均匀初始化（历史误名为 `fanin_init`）。

    调用处一律传 `nn.Linear(...).weight.data.size()`，而 nn.Linear 的 weight 形状是
    `(out_features, in_features)` ⇒ `size[0]` 是 **out_features**，所以这里取的是
    fan_out，`bound = 1/sqrt(fan_out)`；函数名与参数名过去写成 fan_in 是**错的**。

    对照（PyTorch `nn.Linear` 的默认初始化恰好是 fan_in 口径）：

        bound(fan_in)  = 1/sqrt(size[1])      ← nn.Linear 默认
        bound(fan_out) = 1/sqrt(size[0])      ← 本函数

    要"逐层尺度不变"应取 fan_in：对 `y = Wx`、`w ~ U(-v, v)`、`v = 1/sqrt(n)` 有
    `Var(y) = fan_in/(3n) * Var(x)`，只有 `n = fan_in` 时与 n 无关；取 fan_out 则每层
    方差被乘上 `fan_in/fan_out`。

    **不要**为了"对齐名字"就把 `size[0]` 改成 `size[1]`：实测两种口径对训练无可测差异
    （8 seed 严格配对 + 纯重抽噪声对照臂，终点 |ΔR²| ≤ 0.005，而同一探针对 bsize
    的分辨力是 0.039 ⇒ 不是探针不灵敏），见 `agentutil/probe_init_convention.py`；
    但改数值会让全部历史 run 不可比。此处只纠正命名与文档，数值一行不动。
    """
    fanout = fanout or size[0]
    v = 1. / np.sqrt(fanout)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        # self.fc3 = nn.Linear(hidden2, nb_actions)
        self.fc3 = nn.Linear(hidden2, hidden1)
        self.fc4 = nn.Linear(hidden1, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        # 权重口径: fc1/fc2/fc3 用 fan_out (fanout_init, 历史误名 fanin_init);
        # fc4(输出层) 用 ±init_w=±3e-3 —— 刻意的小输出层 (残余动作 Δ≈0 起点);
        # bias **一律不碰** ⇒ 保留 PyTorch 默认 ±1/sqrt(fan_in)。
        # 注意: 各项目在 agent 构造后另有覆写(见 hsrl_agent_init.zero_actor_output /
        # td3per_main.init_wt_action_bias), 它们才是决定"初始策略"的项。
        # self.fc1.init.orthogonal_(gain=1)
        self.fc1.weight.data = fanout_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanout_init(self.fc2.weight.data.size())
        # self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.weight.data = fanout_init(self.fc3.weight.data.size())
        self.fc4.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        # out = self.sigmoid(out)
        return out

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc4 = nn.Linear(hidden1, hidden2)
        self.fc2 = nn.Linear(hidden2+nb_actions, 80)
        # self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        self.fc5 = nn.Linear(80, 50)
        self.fc3 = nn.Linear(50, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        # 本方法只写 3 个权重: fc1/fc2 用 fan_out (fanout_init), fc3(输出层, 50→1) 用
        # ±init_w=±3e-3。**fc4/fc5 的权重完全没有被覆写**, 保留 PyTorch 默认
        # U(±1/sqrt(fan_in)) —— 即教科书口径, 是"正确"的那个。生产维度
        # (--hidden1 16 / --hidden2 100) 下实测上界:
        #   fc1 0.248 (fan_in 口径应 0.5)  fc4 0.250 (=1/sqrt(16) ✓)
        #   fc2 0.112 (fan_in 口径应 0.097) fc5 0.112 (=1/sqrt(80) ✓)  fc3 0.003
        # bias 一律不碰 ⇒ 全部保持 ±1/sqrt(fan_in); 输出层 bias 的真正修正是
        # hsrl_agent_init.BiasInitTracker (把 Q 基线搬到 r̄/(1-γ))。
        # ⚠ "补齐" fc4/fc5 会改变数值且实测无收益 (agentutil/probe_init_convention.py),
        #   数值请勿改动 —— 只纠正命名与文档。
        self.fc1.weight.data = fanout_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanout_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc4(out)
        # debug()
        out = self.fc2(torch.cat([out,a],1))
        out = self.relu(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

class Actor_L(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Actor_L, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanout_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanout_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out

class Critic_L(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Critic_L, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanout_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanout_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        # debug()
        out = self.fc2(torch.cat([out,a],1))
        out = self.relu(out)
        out = self.fc3(out)
        return out