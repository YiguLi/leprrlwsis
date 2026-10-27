# -*- coding: utf-8 -*-
r"""逐步奖励分解 critic（按 `macro_critic_step_reward_decomposition_architecture.md`）。

本任务是**纯监督**的"五步功率波动和"预测，因此对原架构做最小适配：

* **step reward** = 第 t 步功率波动 `|P_t − P_{t-1}|`（t=1..5）；
* **block reward** = `F5 = Σ_t |P_t − P_{t-1}|`（无终端长期价值）；
* **QComposer** = `Σ_t γ^t r̂_t + γ^f B̂`，默认 `γ=1`；
* 原架构的 `TerminalValueHead` 保留为架构占位：纯预测任务没有下一宏状态，
  其监督目标定义为 **0**（无终端价值）；`λ_B` 很小，报告中会给出 `B` 的幅值。
  若把该 critic 接回 RL，只需把 `B` 的目标改成
  `(1-done)·Q_target(S_next, π_target(S_next))`，其余结构不变。

结构（与文档 §4–§6 对应）
--------------------------
    state encoder: 1 → 64 → 64, SiLU
    action encoder: 5 → 64 → 64, SiLU
    FiLM: (γ_s, β_s) = f(h_S);  h̃_o = (1+0.1γ_s)⊙h_o + 0.1β_s
    z = [h_S; h_o; h̃_o]  (192)
    joint: 192 → 256 → 256(残差) → 128, SiLU
    reward heads: [z; e_t] (128+16) → 64 → 1,  t=0..f-1（共享头 + 阶段嵌入）
    terminal head: z → 64 → 1, SiLU
    QComposer: F5̂ = Σ_t γ^t r̂_t + γ^f B̂（无可学习参数）
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_AGENTUTIL = os.path.dirname(_HERE)
ROOT = os.path.dirname(_AGENTUTIL)
os.chdir(ROOT)
for _p in (ROOT, _AGENTUTIL, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

STAGE_DIM = 16


class StepDecompCritic(nn.Module):
    """五步波动分解 critic：逐步 reward 头 + 终端价值头 + 固定 QComposer。"""

    def __init__(self, state_dim=1, action_dim=5, f=5, stage_dim=STAGE_DIM,
                 gamma=1.0, step_bias=0.0):
        super().__init__()
        self.f = int(f)
        self.gamma = float(gamma)
        self.stage_dim = int(stage_dim)

        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
        )
        self.action_enc = nn.Sequential(
            nn.Linear(action_dim, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
        )
        self.film = nn.Linear(64, 2 * 64)
        self.joint1 = nn.Linear(64 * 3, 256)
        self.joint2 = nn.Linear(256, 256)
        self.joint3 = nn.Linear(256, 128)

        self.stage_emb = nn.Parameter(torch.randn(self.f, self.stage_dim) * 0.02)
        self.reward_head = nn.Sequential(
            nn.Linear(128 + self.stage_dim, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.terminal_head = nn.Sequential(
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

        # FiLM 初始为恒等调制；终端头初始输出 0；reward 头 bias 用步波动均值起步
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.terminal_head[-1].weight)
        nn.init.zeros_(self.terminal_head[-1].bias)
        nn.init.uniform_(self.reward_head[-1].weight, -3e-3, 3e-3)
        nn.init.constant_(self.reward_head[-1].bias, float(step_bias))

    def encode(self, state, action):
        hs = self.state_enc(state)
        ho = self.action_enc(action)
        gb = self.film(hs)
        g, b = gb.chunk(2, dim=-1)
        ht = (1.0 + 0.1 * g) * ho + 0.1 * b
        z = torch.cat([hs, ho, ht], dim=-1)
        h = F.silu(self.joint1(z))
        h = F.silu(self.joint2(h) + h)
        z2 = F.silu(self.joint3(h))
        return z2

    def forward(self, state, action):
        """返回 `(f5_hat, step_pred, terminal_pred)`。

        step_pred: (N, f) 每步波动预测；f5_hat: (N,) 组合后的五步波动和。
        """
        z = self.encode(state, action)
        n = z.shape[0]
        ze = z[:, None, :].expand(n, self.f, 128)
        ee = self.stage_emb[None, :, :].expand(n, self.f, self.stage_dim)
        step = self.reward_head(torch.cat([ze, ee], dim=-1)).squeeze(-1)  # (N,f)
        term = self.terminal_head(z).squeeze(-1)                           # (N,)
        disc = self.gamma ** torch.arange(self.f, dtype=z.dtype, device=z.device)
        q = (step * disc[None, :]).sum(dim=1) + (self.gamma ** self.f) * term
        return q, step, term


def load_step_targets(ds):
    """从 F5 数据集还原逐步波动目标 `(f_steps (N,5), f5 (N,))`。"""
    p_steps = ds['p_steps'].astype(np.float64)
    p_prev = ds['p_prev'].astype(np.float64).reshape(-1)
    f_steps = np.empty_like(p_steps)
    f_steps[:, 0] = np.abs(p_steps[:, 0] - p_prev)
    f_steps[:, 1:] = np.abs(np.diff(p_steps, axis=1))
    f5 = f_steps.sum(axis=1)
    if not np.allclose(f5, ds['f5'].astype(np.float64), atol=1e-5):
        raise RuntimeError('p_steps/p_prev 还原的 F5 与 f5 列不一致')
    return f_steps.astype(np.float32), f5.astype(np.float32)


def state_action(ds, mask=None):
    u = ds['u'].astype(np.float32)
    a = ds['a'].astype(np.float32)
    x = (u / 10.0).reshape(-1, 1)
    if mask is not None:
        x, a = x[mask], a[mask]
    return x, a


def predict(model, x, a, batch=8192):
    model.eval()
    n = x.shape[0]
    q = np.empty(n, dtype=np.float64)
    step = np.empty((n, model.f), dtype=np.float64)
    term = np.empty(n, dtype=np.float64)
    with torch.no_grad():
        for s in range(0, n, int(batch)):
            e = min(s + int(batch), n)
            qq, ss, tt = model(torch.as_tensor(x[s:e]), torch.as_tensor(a[s:e]))
            q[s:e] = qq.cpu().numpy()
            step[s:e] = ss.cpu().numpy()
            term[s:e] = tt.cpu().numpy()
    return q, step, term


def soft_update(target, source, tau):
    for pt, ps in zip(target.parameters(), source.parameters()):
        pt.data.copy_(pt.data * (1.0 - tau) + ps.data * tau)


def hard_update(target, source):
    target.load_state_dict(source.state_dict())
