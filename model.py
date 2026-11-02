
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


#: v6 critic 的固定宽度（**与 --hidden1/--hidden2 无关**；改这里就是改 v6）。
#:
#: 2026-10-26 按用户口径重构：
#:     state : 4 → 32 → 32
#:     action: 6 → 64 → 64
#:     merge : 96 (32+64，原始 a 不再重复拼入) → 128 → 256 → 256 → 128 → 1
#: 合并输入恒为 96 ⇒ H(nb_actions=6) 与 L(nb_actions=1) 共用同一套 merge 形状。
V6_DIMS = dict(s=(32, 32), a=(64, 64), merge=(128, 256, 256, 128))

#: v7 critic 的固定宽度（**与 --hidden1/--hidden2 无关**）。
#:
#: 2026-11-02 按用户口径新增：**action 与 state 在最开始直接拼接**，然后
#:     input (= ns + na') → 400 → 300 → 400 → 1
#: 即"动作首层进网"的 3 隐藏层版本（v1af 是 4·hidden1 → hidden2 → 80 → 50）。
#: 动机：v1af 的 4·hidden1=1600 首层对 10 维输入是 160 倍扩张，参数几乎全在首层，
#: 后面的 300→80→50 才是真正的非线性栈；v7 把宽度改成 **400→300→400（宽-窄-宽）**，
#: 参数量 245 501（ns=4, na=6），且不需要 --critic_action_first。
V7_DIMS = (400, 300, 400)

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
        # 七种结构（默认 v1 = 历史口径，逐位不变）：
        #   v1    : fc1(x)→fc4→cat(a)→fc2→fc5→fc3
        #   v1af  : cat(x,a)→fc1(4·hidden1)→fc4→fc2→fc5→fc3        （动作首层进网）
        #   v2    : s: ns→hidden1→hidden2  ∥  a: na→32→64
        #           合并 [hidden2+64] →(square, 带残差)→128→64→1   （双流 + 慢压缩）
        #   v6    : s: ns→32→32  ∥  a: na'→64→64
        #           合并 [32+64]=96 →128→256→256→128→1   （**原始 a 不再重复拼入**）
        #   v7    : cat(x,a) → 400→300→400→1   （**动作与状态一开始就拼接**，
        #           与 v1af 同一动机但宽度是显式规格 V7_DIMS）
        #   v6old : 2026-10-24 的旧 v6（s 4→32 ∥ a→32，cat(32+32+na)→256→256→128→1；
        #           合并时把原始 a 再传一次）。**只保留给历史 checkpoint 复现**
        #           （output/V6/AIFFLOR-run1、documentation/critic_arch_v6_review.md），
        #           新训练不要再用；`--critic_arch` 也不暴露它。
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
            # s: ns→32→32 ∥ a: na'→64→64，合并 cat(32+64)=96（**不再重复拼原始 a**）
            #   →128→256→256→128→1。
            # 与旧 v6 的区别（2026-10-26 重构，按用户口径）：
            #   * 状态/动作支路各加深到 2 层（旧的只有 1 层、状态侧仅 192 参数）；
            #   * 动作支路 32→64；
            #   * merge 前 96 维**只来自两条支路**，原始 a 的重复拼接被去掉
            #     （旧 v6 的 ha 与原始 a 直连在均值上互相抵消，见
            #      documentation/critic_arch_v6_review.md §4）；
            #   * merge 输入恒为 96 ⇒ **H 层与 L 层共用同一套 merge 形状**
            #     （旧的 70/65 会让 infer_arch 把 L 层认成 v4）。
            _s1, _s2 = V6_DIMS['s']
            _a1, _a2 = V6_DIMS['a']
            _m1, _m2, _m3, _m4 = V6_DIMS['merge']
            self.s1 = nn.Linear(nb_states, _s1)
            self.s2 = nn.Linear(_s1, _s2)
            self.a1 = nn.Linear(self.in_actions, _a1)
            self.a2 = nn.Linear(_a1, _a2)
            self.m1 = nn.Linear(_s2 + _a2, _m1)
            self.m2 = nn.Linear(_m1, _m2)
            self.m3 = nn.Linear(_m2, _m3)
            self.m4 = nn.Linear(_m3, _m4)
            self.out = nn.Linear(_m4, 1)
        elif self.arch == 'v6old':
            # 旧 v6（只读兼容，见 init 注释）：cat(32+32+na) = 70 (H) / 65 (L)
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
        elif self.arch == 'v7':
            # 用户口径（2026-11-02）：action 与 state **在最开始直接拼接**，然后
            #   input (ns + na') → 400 → 300 → 400 → 1
            # 与 v1af 的区别只在宽度口径：v1af = 4·hidden1(1600) → hidden2(300) → 80 → 50，
            # 参数几乎全压在首层（10→1600 是 160 倍扩张）；v7 是 400→300→400 的
            # "宽-窄-宽"栈，深度相同但容量分布均匀（见 documentation/critic_arch_v7.md）。
            # ⚠ 与 v5/v6 一样，宽度写死在 V7_DIMS ⇒ --hidden1/--hidden2 不生效。
            _h1, _h2, _h3 = V7_DIMS
            self.h1 = nn.Linear(nb_states + self.in_actions, _h1)
            self.h2 = nn.Linear(_h1, _h2)
            self.h3 = nn.Linear(_h2, _h3)
            self.out = nn.Linear(_h3, 1)
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
            for m in (self.s1, self.s2, self.a1, self.a2,
                      self.m1, self.m2, self.m3, self.m4):
                m.weight.data = fanout_init(m.weight.data.size())
            if self.out_init != 'fanin':
                self.out.weight.data.uniform_(-init_w, init_w)
        elif self.arch == 'v6old':
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
        elif self.arch == 'v7':
            for m in (self.h1, self.h2, self.h3):
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
            hs = self.relu(self.s2(self.relu(self.s1(x))))
            ha = self.relu(self.a2(self.relu(self.a1(a))))
            m = torch.cat([hs, ha], 1)             # 96 = 32(s) + 64(a)；原始 a 不再拼入
            m = self.relu(self.m1(m))
            m = self.relu(self.m2(m))
            m = self.relu(self.m3(m))
            m = self.relu(self.m4(m))
            return self.out(m)
        if self.arch == 'v6old':
            hs = self.relu(self.s1(x))
            ha = self.relu(self.a1(a))
            m = torch.cat([hs, ha, a], 1)          # 旧口径：原始 action 再传一次
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
        if self.arch == 'v7':
            z = torch.cat([x, a], 1)                  # **动作与状态一开始就拼接**
            z = self.relu(self.h1(z))
            z = self.relu(self.h2(z))
            z = self.relu(self.h3(z))
            return self.out(z)
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
    """按 state_dict 推断 critic 结构（v1 / v1af / v2 / v3 / v4 / v5 / v6 / v6old / v7）。

    ⚠ 判据必须用 m1 的**完整形状**，不能只看 `shape[1]`：v4 的 m1 是方阵 (w,w)，
      v5 的是 (100,48)，v6 的是 (128,96)，旧 v6（v6old）的是 (256, 32+32+na')
      —— 只看输入维时 v4/v5 在 hidden1=16 会撞车（都是 48）。
    v7 用 h1/h2/h3/out 命名且**没有** s1/fc1，判据唯一（`h1.weight` 只有 v7 有）。
    """
    # v7: cat(x,a) → 400→300→400→1（无 s1/a1/fc1，只有 h1/h2/h3/out）
    if 'h1.weight' in sd:
        return 'v7'
    if 's1.weight' in sd:
        if 'm4.weight' in sd:                         # v6 独有：96→128→256→256→128
            return 'v6'
        if 'a2.weight' not in sd:
            if 'm1.weight' not in sd:
                return 'v4'
            sh = tuple(sd['m1.weight'].shape)          # (out, in)
            if sh[0] == 256 and sh[1] in (70, 65):     # 旧 v6: H=70 / L=65
                return 'v6old'
            if sh[0] == sh[1]:                         # v4: 方阵
                return 'v4'
            if sh == (100, 48):                        # v5
                return 'v5'
            return 'v4'
        return 'v3' if sd['a2.weight'].shape[0] == 128 else 'v2'
    if sd['fc1.weight'].shape[1] == ns + na:
        return 'v1af'
    return 'v1'


def feat_from_sd(sd, ns=4, na=6, hidden1=16, hidden2=100):
    """按 state_dict 判断是否启用了 P(a,u)/5 特征（动作输入多 1 维）。"""
    if 'h1.weight' in sd:                        # v7: 动作与状态首层拼接
        return sd['h1.weight'].shape[1] == ns + na + 1
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


def _round_half_away_ste(x, decimals=2):
    """四舍五入到 `decimals` 位小数，**前向真取整、反向恒等**（straight-through）。

    为什么必须用 STE：`torch.round` / `floor` 的导数几乎处处为 0。若投影里直接
    取整，actor 对 a0/a4 的梯度会被这一步清零 ⇒ 整个 actor 学不动（不是"梯度小"，
    是**严格 0**）。STE 把前向值换成量化值，同时让反向按 `∂y/∂x = 1` 传回去：
    训练仍在连续插值流形上做梯度，实际执行/回放的却是两位小数动作。
    """
    scale = float(10.0 ** int(decimals))
    # +1e-6（缩放后 = 动作量 1e-8）只吸收 float32 输入在 .xx5 边界上的表示抖动
    # （actor 输出是 float32，0.12 实际是 0.1199999973…），不改变正常的四舍五入。
    y = torch.sign(x) * torch.floor(torch.abs(x) * scale + 0.5 + 1e-6) / scale
    return x + (y - x).detach()


def _round_half_away_np(x, decimals=2):
    """`_round_half_away_ste` 的 numpy 版（无梯度，四舍五入 away-from-zero）。

    与 torch 版一样在缩放后加 1e-6 容差 ⇒ 同一输入的两个实现给出同一位小数。
    """
    scale = float(10.0 ** int(decimals))
    v = np.asarray(x, dtype=np.float64)
    return np.sign(v) * np.floor(np.abs(v) * scale + 0.5 + 1e-6) / scale


def proj_lin_wt(a, nt=5, decimals=2):
    """**线性行为分支**的动作投影：只保留首尾两维，中间维沿机组线性插值。

    定义（前 `nt` 维风机动作，第 6 维 g 原样保留）：
        a_i = a_0 + (a_{nt-1} − a_0) · i/(nt−1)      i = 1 … nt−2
    并对**中间维**四舍五入到 `decimals` 位（默认 2）；首尾维保持 actor 原始输出
    （不取整、不 clip —— 越界仍交给 `check_action_H_hsrl` + pen5 处理）。

    设计动机（`agentutil/probe_turbine_wake_attribution.py` 的实测）：
    单排 5 台里两端 T1/T5 是**符号稳定、量级大**的动作通道（T1 占出力 44.4%，
    T5 的下游耦合恒 0），而中间 T2–T4 的净灵敏度被"自身项 vs 下游耦合"近抵消
    （符号一致率 43%~48%，接近抛硬币）。⇒ 把 5 个自由度压成"首尾 2 个 + 线性型线"，
    既减少近零方向的维数，又保持整排动作的连贯形状。

    ⚠ 有效自由度只有 2：第 1..nt−2 维的 actor 输出**存在但无用**，
    对它们的梯度恒为 0（与 `--scalar_wt` 的"第 1..4 维存在但无用"同构）。

    不改变第 6 维（g），也不做 clip。
    """
    out = a.clone()
    n = int(nt)
    if n <= 2:
        return out
    if a.shape[-1] < n:
        raise ValueError('proj_lin_wt: 动作维数 %d < nt=%d' % (a.shape[-1], n))
    # 插值与取整在 float64 上做：float32 的 0.65/4 之类会在 .xx5 附近抖到边界另一侧，
    # 导致 torch 与 numpy 对同一输入给出不同两位小数。结果写回原 dtype。
    work = a.to(torch.float64)
    a0 = work[..., 0]
    a_end = work[..., n - 1]
    t = torch.arange(1, n - 1, dtype=torch.float64, device=a.device) / float(n - 1)
    mid = a0.unsqueeze(-1) + (a_end - a0).unsqueeze(-1) * t
    if decimals is not None and int(decimals) >= 0:
        mid = _round_half_away_ste(mid, int(decimals))
    out[..., 1:n - 1] = mid.to(out.dtype)
    return out


def proj_lin_wt_np(a, nt=5, decimals=2):
    """`proj_lin_wt` 的 numpy 版（供主循环 / select_action 用）。"""
    out = np.array(a, dtype=np.float64, copy=True).reshape(-1)
    n = int(nt)
    if n <= 2:
        return out
    if out.size < n:
        raise ValueError('proj_lin_wt_np: 动作维数 %d < nt=%d' % (out.size, n))
    t = np.arange(1, n - 1, dtype=np.float64) / float(n - 1)
    mid = out[0] + (out[n - 1] - out[0]) * t
    if decimals is not None and int(decimals) >= 0:
        mid = _round_half_away_np(mid, int(decimals))
    out[1:n - 1] = mid
    return out


def expand_lin_wt(x, nt=5, decimals=2):
    """**3 输出头**的展开：`x = [a0, a_end, g...]` → `[a0, 中..., a_end, g...]`。

    `--lin_wt` 的结构版 actor 输出层只有 **2 + (nb_actions − nt)** 个神经元：
    首端 `a0`、末端 `a_end = a_{nt−1}`，其余维（HSRL 的 `g`）原样跟在后面。
    本函数把中间 `nt−2` 维填成线性插值（并四舍五入），得到 critic / env 需要的
    完整动作向量。

    与 `proj_lin_wt` 的分工：
      * `proj_lin_wt` 吃**完整动作**（如 warmup 调度给出的 5/6 维），只保留首尾；
      * `expand_lin_wt` 吃**压缩后的 actor 输出**（2 + 额外维），负责展开。
    """
    n = int(nt)
    if n < 2:
        raise ValueError('expand_lin_wt: nt 必须 >= 2')
    if x.shape[-1] < 2:
        raise ValueError('expand_lin_wt: 输入至少要有 [a0, a_end] 两维')
    mid = torch.zeros(x.shape[:-1] + (n - 2,), dtype=x.dtype, device=x.device)
    base = torch.cat([x[..., 0:1], mid, x[..., 1:2], x[..., 2:]], dim=-1)
    return proj_lin_wt(base, nt=n, decimals=decimals)


def expand_lin_wt_np(x, nt=5, decimals=2):
    """`expand_lin_wt` 的 numpy 版（1-D 输入；供 select_action / warmup 用）。"""
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    n = int(nt)
    if n < 2:
        raise ValueError('expand_lin_wt_np: nt 必须 >= 2')
    if arr.size < 2:
        raise ValueError('expand_lin_wt_np: 输入至少要有 [a0, a_end] 两维')
    mid = np.zeros(n - 2, dtype=np.float64)
    base = np.concatenate([arr[:1], mid, arr[1:2], arr[2:]])
    return proj_lin_wt_np(base, nt=n, decimals=decimals)


def wt_bias_values(nb_out, lo=0.05, hi=0.45, sig_value=0.0, lin_wt=False,
                   rng=None):
    """actor 输出层 bias 的初始化值（`--lin_wt` 的 3 输出映射也在这里）。

    * 默认（6 输出）：前 5 维 = 风机 `a` 的“中性宽带” `U(lo,hi)`，其余 = `sig_value`；
    * `--lin_wt`（3 输出）：第 0/1 维 = `a0/a4` 的 `U(lo,hi)`，其余 = `sig_value`。

    返回 float32 numpy 向量；调用方把它写进 `actor.fc4.bias`（及 target）。
    """
    rng = np.random if rng is None else rng
    b = np.zeros(int(nb_out), dtype=np.float32)
    if b.size == 0:
        return b
    if lin_wt:
        if b.size < 2:
            raise ValueError('wt_bias_values(lin_wt): 输出至少要有 a0/a4 两维')
        b[:2] = rng.uniform(lo, hi, 2)
        b[2:] = float(sig_value)
    else:
        n_wt = min(5, b.size)
        b[:n_wt] = rng.uniform(lo, hi, n_wt)
        b[n_wt:] = float(sig_value)
    return b


def proj_scalar_wt(a, nt=5):
    """把前 nt 维风机动作压成**同一个值** a_0：`a_i = a_0`（i<nt），其余维不变。

    刻意设计的 baseline 动作空间（--scalar_wt）：五台风机共用一个轴向因子，
    因此
      * 有效自由度只有 **1**（第 0 维），第 1..nt−1 维"存在但无用"；
      * 投影是线性映射 `a[:nt] = [1,0,0,0,0]·a` ⇒ `∂/∂a_j (j≥1) ≡ 0`（梯度恒 0，
        不是"很小"）；
      * 与 `proj_farm_level` 的区别：那条射线保留逐台**形状** c_i（a_i = s·c_i），
        这里连形状都不要（a_i 全相等）。

    不改变第 6 维（g），也不做 clip —— 越界仍由 `check_action_H_hsrl` + pen5 处理。
    """
    out = a.clone()
    out[..., :nt] = a[..., :1]
    return out


def proj_scalar_wt_np(a, nt=5):
    """`proj_scalar_wt` 的 numpy 版（供主循环 / select_action 用）。"""
    out = np.array(a, dtype=np.float64, copy=True).reshape(-1)
    out[:nt] = out[0]
    return out


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
