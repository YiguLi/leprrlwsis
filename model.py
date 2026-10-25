
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
        # 注意: H 层 (Actor) **不使用** tanh/sigmoid 做输出压缩 —— 见 forward() 里的
        # "⚠ 上层 (H) 的输出故意不做有界压缩" 注释 (最优解在边界上 ⇒ 饱和区学习信号为 0)。
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
        # =====================================================================
        # ⚠ 上层 (H) 的输出**故意不做有界压缩** (下面这行 sigmoid / 上面的 tanh 都不要恢复)。
        #    tanh (以及任何饱和型 squash) 在本任务是**有害**的, 实测依据:
        #      (1) **最优解恰好在约束边界上**。一维 Q(a_j) 剖面实测 (run13 ckpt40000,
        #          agentutil/probe_q_profile_boundary.py): dim4 峰 a4=+0.05, dim5 峰 a5=-0.60,
        #          即峰值贴着/越过边界 [0,0.5] / [-0.5,0.5]。tanh 在饱和区导数→0,
        #          而最优处正是饱和区 ⇒ 学习信号在**最需要它的地方**退化 (实测导数仅
        #          0.04~0.1, 等于把边界附近的学习率砍掉 10~25 倍), 收敛停滞。
        #      (2) **边界不可达**。tanh 只能渐近趋近 0.5 / 0, 永远到不了; 而最优就在边界上
        #          ⇒ 系统性偏移 (限功率不够 / 信号不够), 白丢收益。
        #      (3) **单边区间 [0,0.5] 的重标定陷阱**。本层 fc4 初始化为 ±3e-3 (输出≈0,
        #          "残余动作≈0 起点")。若用 0.5·sigmoid(pre), 初始 pre≈0 ⇒ 动作=0.25,
        #          即**每台风机一上来就 25% 限功率**, 初始化直接坏掉; 单边 squash 必须把
        #          输出层 bias 重新初始化到很负 (如 -6) 才等价, 否则"tanh 效果差"里
        #          有很大一部分其实是初始化坏掉的锅。
        #    ⇒ 越界改由**环境侧 clip (CheckAction 之后才进 env) + 奖励侧乘性罚**
        #      (--viol_derate: r1 的风机收益 × exp(-c·Σ越界量)) 处理。这样 actor 保持
        #      线性输出, 边界外仍有**强回正梯度** (实测界外斜率 ~1.25 critic 单位, 而
        #      tanh 饱和区 ~0), 边界处的学习信号不退化。
        #    注意: 下层 Actor_L (class Actor_L) 用 tanh 是**另一回事** —— 它的残差动作
        #      对称、最优点在内部, 不存在上述"边界即最优"的问题, 不要类比着一起改。
        # =====================================================================
        # out = self.sigmoid(out)
        return out

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3,
                 action_first=False, arch=None, act='relu', out_init='3e3',
                 feat_power=False, feat_power_k=1.357e-3, feat_u_index=1,
                 feat_u_scale=10.0, feat_nt=5):
        super(Critic, self).__init__()
        # 三种结构（默认 v1 = 历史口径，逐位不变）：
        #   v1    : fc1(x)→fc4→cat(a)→fc2→fc5→fc3
        #   v1af  : cat(x,a)→fc1(4·hidden1)→fc4→fc2→fc5→fc3        （动作首层进网）
        #   v2    : s: ns→hidden1→hidden2  ∥  a: na→32→64
        #           合并 [hidden2+64] →(square, 带残差)→128→64→1   （双流 + 慢压缩）
        # 动机：物理是 P ≈ c·u³·g(a) —— u 与 a 的**乘积**；且 6 维动作是非单调
        # （单机 Cp 在 1/3 取峰）、逐台不对称的。让动作先独立成型，再与状态融合。
        if arch is None:
            arch = 'v1af' if action_first else 'v1'
        self.arch = str(arch)
        self.nb_actions = int(nb_actions)
        self.action_first = (self.arch == 'v1af')
        self.act_kind = str(act)
        self.out_init = str(out_init)
        # ---- 可选特征: P(a,u)/5（无尾流解析代理）----
        #   P_hat = k·u³·Σ_i 4a_i(1−a_i)²,  k 由数据标定（R²≈0.992, 平均相对误差 3.4%）
        #   作用: 把 c1 = |P − P_G| 的折点搬到"输入坐标"上（网络不必自己合成乘法）。
        #   注意: 该特征对 (x,a) **可导** ⇒ actor 也获得一条正确的 ∂P/∂a 通路。
        self.feat_power = bool(feat_power) and (nb_actions == feat_nt + 1)
        self.feat_power_k = float(feat_power_k)
        self.feat_u_index = int(feat_u_index)
        self.feat_u_scale = float(feat_u_scale)
        self.feat_nt = int(feat_nt)
        self.in_actions = nb_actions + (1 if self.feat_power else 0)
        self.relu = nn.LeakyReLU(0.01) if self.act_kind == 'leaky' else nn.ReLU()
        if self.arch == 'v6':
            # s4->32 ∥ a6->32, 合并时**再把原始 a 传一次** => cat(32+32+na) = 70 (H)
            self.s1 = nn.Linear(nb_states, 32)
            self.a1 = nn.Linear(self.in_actions, 32)
            self.m1 = nn.Linear(32 + 32 + self.in_actions, 256)
            self.m2 = nn.Linear(256, 256)
            self.m3 = nn.Linear(256, 128)
            self.out = nn.Linear(128, 1)
        elif self.arch == 'v5':
            self.s1 = nn.Linear(nb_states, 16)      # 4 -> 16
            self.a1 = nn.Linear(self.in_actions, 32)   # 6(+1) -> 32
            w = 16 + 32                            # 48
            self.m1 = nn.Linear(w, 100)
            self.m2 = nn.Linear(100, 80)
            self.m3 = nn.Linear(80, 50)
            self.out = nn.Linear(50, 1)
        elif self.arch == 'v4':
            self.s1 = nn.Linear(nb_states, hidden1)      # 4 -> 16
            self.a1 = nn.Linear(self.in_actions, 32)     # 6(+1) -> 32
            w = hidden1 + 32                             # 48
            self.m1 = nn.Linear(w, w)                    # 48 -> 48（残差）
            self.m2 = nn.Linear(w, 32)
            self.out = nn.Linear(32, 1)
        elif self.arch in ('v2', 'v3'):
            self.a_emb = 128 if self.arch == 'v3' else 64     # 动作支路输出宽度
            self.s1 = nn.Linear(nb_states, hidden1)
            self.s2 = nn.Linear(hidden1, hidden2)
            self.a1 = nn.Linear(self.in_actions, 32)
            self.a2 = nn.Linear(32, self.a_emb)
            w = hidden2 + self.a_emb
            self.m1 = nn.Linear(w, w)          # square 层（带残差 ⇒ 等效加深、不靠"保维"硬压）
            self.m2 = nn.Linear(w, 128)
            self.m3 = nn.Linear(128, 64)
            self.out = nn.Linear(64, 1)
        elif self.arch == 'v1af':
            self.fc1 = nn.Linear(nb_states + self.in_actions, 4 * hidden1)
            self.fc4 = nn.Linear(4 * hidden1, hidden2)
            self.fc2 = nn.Linear(hidden2, 80)
            self.fc5 = nn.Linear(80, 50)
            self.fc3 = nn.Linear(50, 1)
        else:
            self.fc1 = nn.Linear(nb_states, hidden1)
            self.fc4 = nn.Linear(hidden1, hidden2)
            self.fc2 = nn.Linear(hidden2 + nb_actions, 80)
            self.fc5 = nn.Linear(80, 50)
            self.fc3 = nn.Linear(50, 1)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        # 口径：隐藏层权重用 fan_out (fanout_init)；**输出层**用 ±init_w=±3e-3
        # （历史约定，见 Agent 文档；--critic_out_init fanin 可改成 PyTorch 的 fan_in 默认）。
        # bias 一律不碰（生产路径的 baseline 由 hsrl_agent_init.BiasInitTracker 负责）。
        if self.arch == 'v6':
            for m in (self.s1, self.a1, self.m1, self.m2, self.m3):
                m.weight.data = fanout_init(m.weight.data.size())
            if self.out_init != 'fanin':
                self.out.weight.data.uniform_(-init_w, init_w)
        elif self.arch == 'v5':
            for m in (self.s1, self.a1, self.m1, self.m2, self.m3):
                m.weight.data = fanout_init(m.weight.data.size())
            if self.out_init != 'fanin':
                self.out.weight.data.uniform_(-init_w, init_w)
        elif self.arch == 'v4':
            for m in (self.s1, self.a1, self.m1, self.m2):
                m.weight.data = fanout_init(m.weight.data.size())
            if self.out_init != 'fanin':
                self.out.weight.data.uniform_(-init_w, init_w)
        elif self.arch in ('v2', 'v3'):
            for m in (self.s1, self.s2, self.a1, self.a2, self.m1, self.m2, self.m3):
                m.weight.data = fanout_init(m.weight.data.size())
            if self.out_init != 'fanin':
                self.out.weight.data.uniform_(-init_w, init_w)
        else:
            self.fc1.weight.data = fanout_init(self.fc1.weight.data.size())
            self.fc2.weight.data = fanout_init(self.fc2.weight.data.size())
            if self.out_init != 'fanin':
                self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs):
        x, a = xs
        if getattr(self, 'feat_power', False):
            # P(a,u)/5 —— 无尾流解析代理（k 由数据标定, R²≈0.992）
            # 注意: 必须保持 2-D，否则 (N,) 与 (N,1) 广播会变成 (N,N)
            u_ = (x[:, self.feat_u_index] * self.feat_u_scale).clamp(min=0.0).unsqueeze(1)
            g_ = 4.0 * a[:, :self.feat_nt] * (1.0 - a[:, :self.feat_nt]) ** 2
            pf_ = (self.feat_power_k * u_ ** 3 * g_.sum(1, keepdim=True)) / 5.0
            a = torch.cat([a, pf_], 1)
        if self.arch == 'v6':
            hs = self.relu(self.s1(x))
            ha = self.relu(self.a1(a))
            m = torch.cat([hs, ha, a], 1)          # 原始 action 再传一次
            m = self.relu(self.m1(m))
            m = self.relu(self.m2(m))
            m = self.relu(self.m3(m))
            return self.out(m)
        if self.arch == 'v5':
            hs = self.relu(self.s1(x))
            ha = self.relu(self.a1(a))
            m = torch.cat([hs, ha], 1)
            m = self.relu(self.m1(m))
            m = self.relu(self.m2(m))
            m = self.relu(self.m3(m))
            return self.out(m)
        if self.arch == 'v4':
            hs = self.relu(self.s1(x))
            ha = self.relu(self.a1(a))
            z = torch.cat([hs, ha], 1)
            m = self.relu(self.m1(z)) + z
            m = self.relu(self.m2(m))
            return self.out(m)
        if self.arch in ('v2', 'v3'):
            hs = self.relu(self.s2(self.relu(self.s1(x))))
            ha = self.relu(self.a2(self.relu(self.a1(a))))
            z = torch.cat([hs, ha], 1)
            m = self.relu(self.m1(z)) + z            # 残差：多一层深，但不靠"保维"硬压
            m = self.relu(self.m2(m))
            m = self.relu(self.m3(m))
            return self.out(m)
        if self.action_first:
            out = self.relu(self.fc1(torch.cat([x, a], 1)))
            out = self.fc4(out)
            out = self.relu(self.fc2(out))
        else:
            out = self.relu(self.fc1(x))
            out = self.fc4(out)
            out = self.relu(self.fc2(torch.cat([out, a], 1)))
        out = self.relu(self.fc5(out))
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

def infer_arch(sd, ns=4, na=6):
    """按 state_dict 推断 critic 结构（v1 / v1af / v2 / v3）。"""
    if 's1.weight' in sd:
        if 'a2.weight' not in sd:
            if 'm1.weight' not in sd:
                return 'v4'
            _c = sd['m1.weight'].shape[1]
            return {70: 'v6', 48: 'v5'}.get(_c, 'v4')
        return 'v3' if sd['a2.weight'].shape[0] == 128 else 'v2'
    if sd['fc1.weight'].shape[1] == ns + na:
        return 'v1af'
    return 'v1'


def feat_from_sd(sd, ns=4, na=6, hidden1=16, hidden2=100):
    """按 state_dict 判断是否启用了 P(a,u)/5 特征（动作输入多 1 维）。"""
    if 'a1.weight' in sd:
        return sd['a1.weight'].shape[1] == na + 1
    if 'fc1.weight' in sd:                       # v1af: 动作与状态先拼接
        return sd['fc1.weight'].shape[1] == ns + na + 1
    if 'fc2.weight' in sd:                       # v1: 动作在 fc2 拼入
        return sd['fc2.weight'].shape[1] == hidden2 + na + 1
    return False


def build_critic_from_sd(sd, ns=4, na=6, hidden1=16, hidden2=100):
    """按 state_dict 自动还原 arch + feat_power 并加载（供 probe 使用）。"""
    arch = infer_arch(sd, ns, na)
    feat = feat_from_sd(sd, ns, na, hidden1, hidden2)
    net = Critic(ns, na, hidden1=hidden1, hidden2=hidden2, arch=arch, feat_power=feat)
    net.load_state_dict(sd)
    net.eval()
    return net


class AuxModel(nn.Module):
    """1-step 监督模型：f(s,a) ≈ r1 或 ≈ c1（无 bootstrap / 无 target 网络）。

    k=1 的 model-based 策略改进用它求 ∇_a；目标在给定 (s,a) 时是确定性的 ⇒ 噪声≈0。
    """
    def __init__(self, nb_states, nb_actions, hidden=64,
                 feat_power=False, feat_power_k=1.357e-3, feat_u_index=1,
                 feat_u_scale=10.0, feat_nt=5):
        super(AuxModel, self).__init__()
        self.feat_power = bool(feat_power) and (nb_actions == feat_nt + 1)
        self.feat_power_k = float(feat_power_k)
        self.feat_u_index = int(feat_u_index)
        self.feat_u_scale = float(feat_u_scale)
        self.feat_nt = int(feat_nt)
        _na = nb_actions + (1 if self.feat_power else 0)
        self.fc1 = nn.Linear(nb_states + _na, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()
        for m in (self.fc1, self.fc2):
            m.weight.data = fanout_init(m.weight.data.size())

    def forward(self, xs):
        x, a = xs
        if self.feat_power:
            u_ = (x[:, self.feat_u_index] * self.feat_u_scale).clamp(min=0.0).unsqueeze(1)
            g_ = 4.0 * a[:, :self.feat_nt] * (1.0 - a[:, :self.feat_nt]) ** 2
            pf_ = (self.feat_power_k * u_ ** 3 * g_.sum(1, keepdim=True)) / 5.0
            a = torch.cat([a, pf_], 1)
        z = torch.cat([x, a], 1)
        z = self.relu(self.fc1(z))
        z = self.relu(self.fc2(z))
        return self.out(z)


class StructCostModel(nn.Module):
    """实验 C：结构化解码的成本模型。

    网络只学**光滑标量** z(s,a) ≈ ΔP（带符号的出力变化），成本由解析层解码：
        c_hat = |z| + K·relu(|z| - limit)        （与 env 的 c1 定义同构：K=20, limit=3）
    这样 |·| 与 20× 折点由结构精确给出，网络不必"拟合折点"，也不必把斜率挤过读出头。

    decode='soft' 时用 softplus(beta) 代替 relu（便于与现有平滑版对照）。
    """

    def __init__(self, nb_states, nb_actions, hidden=64, k=20.0, limit=3.0,
                 feat_power=False, feat_power_k=1.357e-3, feat_u_index=1,
                 feat_u_scale=10.0, feat_nt=5, decode='hard', beta=100.0):
        super(StructCostModel, self).__init__()
        self.feat_power = bool(feat_power) and (nb_actions == feat_nt + 1)
        self.feat_power_k = float(feat_power_k)
        self.feat_u_index = int(feat_u_index)
        self.feat_u_scale = float(feat_u_scale)
        self.feat_nt = int(feat_nt)
        self.k, self.limit = float(k), float(limit)
        self.decode, self.beta = str(decode), float(beta)
        _na = nb_actions + (1 if self.feat_power else 0)
        self.fc1 = nn.Linear(nb_states + _na, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.z_out = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()
        for m in (self.fc1, self.fc2):
            m.weight.data = fanout_init(m.weight.data.size())
        # |z| 在 z=0 处次梯度为 0 ⇒ 给一个非零初值，避免"死梯度"
        with torch.no_grad():
            self.z_out.bias.data.fill_(1.0)

    def z(self, xs):
        x, a = xs
        if self.feat_power:
            u_ = (x[:, self.feat_u_index] * self.feat_u_scale).clamp(min=0.0).unsqueeze(1)
            g_ = 4.0 * a[:, :self.feat_nt] * (1.0 - a[:, :self.feat_nt]) ** 2
            pf_ = (self.feat_power_k * u_ ** 3 * g_.sum(1, keepdim=True)) / 5.0
            a = torch.cat([a, pf_], 1)
        h = self.relu(self.fc1(torch.cat([x, a], 1)))
        h = self.relu(self.fc2(h))
        return self.z_out(h)

    def decode_c(self, z):
        az = z.abs()
        if self.decode == 'soft':
            sp = torch.nn.functional.softplus(self.beta * (az - self.limit)) / self.beta
            return az + self.k * sp
        return az + self.k * torch.relu(az - self.limit)

    def forward(self, xs):
        return self.decode_c(self.z(xs))


#: Cp 最优的整场基准型线（由 refs_u.csv 实测 a* 的均值；与 u 基本无关）
FARM_CENTRE = [0.3418, 0.4247, 0.2889, 0.2562, 0.3705]


def proj_farm_level(a, centre=None):
    """把动作的前 5 维**投影到"整场同比例降载"这条射线上**： a_i = s·c_i, s≥0。

    动机（用户口径）：期望的宏观行为是"高风时整体降低/抬高出力水平"，而不是逐台微调。
    投影是线性的 ⇒ 动作只保留 1 个自由度 s，actor 的梯度变成
        ∂/∂s = Σ_i c_i ∂/∂a_i
    即"5 维同向的相干和"，而非 5 个各自带噪的偏导。第 6 维(g)保持不变。
    不改变奖励/成本定义，也不改变 critic 的 6 维输入与日志口径。
    """
    if centre is None:
        centre = FARM_CENTRE
    c = torch.tensor(centre, dtype=a.dtype, device=a.device).view(1, -1)
    a5 = a[:, :5]
    s = (a5 * c).sum(1, keepdim=True) / (c * c).sum()
    s = s.clamp(min=0.0)
    out = a.clone()
    out[:, :5] = (s * c).clamp(0.0, 0.5)
    return out
