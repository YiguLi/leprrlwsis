# -*- coding: utf-8 -*-
"""PAVE 式 critic 几何正则（MPR / VFC / Curv）—— aifflorpave arm 的核心。

来源：Lee et al., "Stabilizing the Q-Gradient Field for Policy Smoothness in
Actor-Critic Methods"（ICML 2026, arXiv:2601.22970）。该文用隐函数定理证明
贪心策略的敏感度被 critic 的几何约束：

        L  ≤  M / μ ,   M = ‖∇²_sa Q‖（状态-动作混合偏导 = 噪声敏感度）
                        μ = −λ_max(∇²_aa Q)（动作方向曲率 = 信号可分辨度）

并给出三项**只作用于 critic** 的辅助损失，把 M 压小、同时把 μ 顶住不为 0：

    L_MPR  = E_ε ‖∇_a Q(s+ε, a) − ∇_a Q(s, a)‖²      （M 的有限差分代理）
    L_VFC  = E   ‖∇_a Q(s, a)   − ∇_a Q(s', a)‖²      （轨迹上相邻状态的场一致）
    L_Curv = E_v max(0, vᵀ ∇²_aa Q(s,a) v + δ)        （保曲率，防"过平滑塌陷"）

本仓库为什么要它（实测，见会话记录）：
  * 基线 16/100 critic 的 **相邻决策 ∇_aQ 符号翻转率 35%**（该文 Base 9–35.5%，
    我们落在最差端），**5 台机组之间的梯度同号率仅 45.7%**（≈随机），
    **全 6×6 ∇²_aa Q 负定性率仅 4.5%**（该文 Base 14–47%）⇒ μ 近 0、M 偏大。
  * 该文实测：无约束 critic 的 M 随训练增长约 4×；本仓库实测 slopeSD 增长 2.7×。

设计约定（**与论文的差异已显式声明**）：
  1. **只作用于 profit 双 critic**（critic_p / critic_p2）：actor 的梯度主通道是
     min(p,p2)；cost 通道是慢变拉格朗日项，不参与几何故事。L 层（残差动作）
     用的是**基类 SP_TD3**，完全不受影响。
  2. 本 arm 以**附加优化步**实现（`SP_TD3.update_policy()` 之后单独对 profit 双
     critic 走一次几何梯度），而不是论文的联合损失。好处：基线路径 `super()`
     调用后逐位不变、可审计、可单因子消融；代价：critic 的优化步数 +1（用
     `--pave_every` 可退化为每 k 次）。TD 更新本身**没有**任何改动。
  3. MPR / VFC 按 ‖∇_aQ‖² **归一化**（无量纲）⇒ 权重与 Q 的尺度无关；
     Curv 为原始量纲，需用 `--pave_report_only 1` 标定后再给权重。
  4. Curv 需要 C² 激活：**ReLU critic 的 ∇²_aa Q 恒为 0**（分段线性），该项退化
     为无意义 ⇒ 只有 `--critic_act silu` 时才有意义（与论文为算 Hessian 而改用
     SiLU 一致）。ReLU 下本模块自动把 Curv 置 0 并在启动时提示。
"""
import numpy as np
import torch
import torch.nn as nn

try:                       # fanout_init 在 model.py 内定义（与 Critic 同源）
    from model import fanout_init
except Exception:          # pragma: no cover - 兜底（保持与 model.py 同式）
    def fanout_init(size):
        fan_in = size[0] if len(size) > 1 else size[0]
        bound = 1.0 / np.sqrt(fan_in)
        return torch.Tensor(size).uniform_(-bound, bound)


class CriticPAVE(nn.Module):
    """与 `model.Critic` **同构同初始化序**、但激活可切换的 critic。

    * `act='relu'` ⇒ 与 `model.Critic` **数值逐位相同**（不变量测试断言）。
    * `act='silu'` ⇒ C² 平滑激活，使 ∇²_aa Q 良定义（Curv 项的前提）。
      ⚠ 只换激活、不动任何权重初始化 ⇒ 与 relu 版是**单因子**对照。

    结构/参数名与 `model.Critic` 完全一致（fc1/fc4/fc2/fc5/fc3），
    因此两边 `load_state_dict` 互通（子类用它继承基类同 seed 下的初始化）。
    """

    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300,
                 init_w=3e-3, act='relu'):
        super(CriticPAVE, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc4 = nn.Linear(hidden1, hidden2)
        self.fc2 = nn.Linear(hidden2 + nb_actions, 80)
        self.fc5 = nn.Linear(80, 50)
        self.fc3 = nn.Linear(50, 1)
        self.act_name = str(act).lower()
        if self.act_name not in ('relu', 'silu'):
            raise ValueError("critic act 只支持 'relu' / 'silu'，收到 %r" % (act,))
        self.act = nn.ReLU() if self.act_name == 'relu' else nn.SiLU()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        # 与 model.Critic.init_weights **同序同操作**（fc1, fc2, fc3）
        # ⇒ 同 seed 下参数逐位相同，与激活无关。
        self.fc1.weight.data = fanout_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanout_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs):
        x, a = xs
        out = self.act(self.fc1(x))
        out = self.fc4(out)
        out = self.act(self.fc2(torch.cat([out, a], 1)))
        out = self.act(self.fc5(out))
        return self.fc3(out)


def grad_action(net, s, a):
    """∇_a Q(s,a)，保留计算图（供正则项反传到 critic 参数）。

    ⚠ 对 `a` 做 detach+clone+requires_grad_ ⇒ 正则项**不会**反传到 actor
    （论文明确：三项只作用于 critic）。但网络参数仍在图里 ⇒ 可反传。
    """
    a_var = a.detach().clone().requires_grad_(True)
    q = net([s, a_var]).sum()
    g = torch.autograd.grad(q, a_var, create_graph=True)[0]
    return g, a_var


def pave_terms(nets, s, a, s2, sigma=0.05, delta=0.0, n_v=1, do_curv=True,
               denom_floor=1e-6, term_clip=1e3):
    """返回 (mpr, vfc, curv, gscale)。

    mpr/vfc 已按 ‖∇_aQ‖² 的批均值归一化（无量纲）；curv 为原始量纲。
    `do_curv=False`（或 ReLU critic）时 curv ≡ 0 —— 分段线性下 ∇²_aa Q 恒 0，
    该项无定义。

    ⚠ 归一化的分母必须设地板：实测 **初始化时 ‖∇_aQ‖² 可为精确 0**
    （ReLU 的 fc5 全部落在负半轴 ⇒ 动作梯度整条路径置 0），此时
    `x/(0+1e-12)` 会炸成 ~1e12。故取 `denom = max(‖∇_aQ‖², denom_floor)`
    并把归一化后的两项 clip 到 `term_clip`。
    """
    mpr = torch.zeros((), device=s.device)
    vfc = torch.zeros((), device=s.device)
    cur = torch.zeros((), device=s.device)
    gs = 0.0
    n = max(1, len(nets))
    for net in nets:
        g0, a_var = grad_action(net, s, a)
        sc = (g0 ** 2).sum(-1).mean()
        gs += float(sc.detach()) / n
        denom = torch.clamp(sc.detach(), min=float(denom_floor))
        # ---- MPR: 状态扰动下的动作梯度变化 ----
        eps = torch.randn_like(s) * float(sigma)
        g1, _ = grad_action(net, s + eps, a)
        _m = ((g1 - g0) ** 2).sum(-1).mean() / denom
        mpr = mpr + torch.clamp(_m, max=float(term_clip))
        # ---- VFC: 真实相邻状态下的动作梯度变化（本仓库"翻转率"的直接对偶量）----
        g2, _ = grad_action(net, s2, a)
        _v = ((g2 - g0) ** 2).sum(-1).mean() / denom
        vfc = vfc + torch.clamp(_v, max=float(term_clip))
        # ---- Curv: Hutchinson 估计的 vᵀ∇²_aaQ v，罚"不够凹" ----
        if do_curv:
            for _ in range(int(max(1, n_v))):
                v = torch.sign(torch.randn_like(a))
                gv = (g0 * v).sum()
                hv = torch.autograd.grad(gv, a_var, create_graph=True)[0]
                cur = cur + torch.clamp((hv * v).sum(-1) + float(delta), min=0.0).mean()
    return mpr / n, vfc / n, cur / n, gs


def describe_pave(w_mpr, w_vfc, w_curv, act, sigma, delta, n_v, report_only=False):
    """启动/日志用的一行口径描述。"""
    return ('[PAVE] critic_act={} mpr={:g} vfc={:g} curv={:g} | sigma={:g} delta={:g} n_v={} '
            '| report_only={} | MPR/VFC 已按 ‖∇_aQ‖² 归一化(无量纲); '
            'Curv 仅在 act=silu 时有效(ReLU 的 ∇²_aaQ≡0)').format(
        act, w_mpr, w_vfc, w_curv, sigma, delta, n_v, int(bool(report_only)))
