# -*- coding: utf-8 -*-
"""`yawflor` H 层的**双通道对偶**：硬通道"越限绝对不许" + 软通道"固定小价换 Q 平缓"。

设计规则（用户给定）
--------------------
> **大于 3 MW 的波动绝对不允许**；**小于 3 MW 的波动原则上不管**，
> 但为了 RL（Q 值平缓之类）可以适当妥协。

原口径（`hsrl_cost` 的 uniform c1 = `P_F + 20·(P_F−3)⁺`）把两件事**混成一个数**再乘同一个 λ：
实测（`output/TD3PER-run21`）导致 ① `<3 MW` 区被隐含定价 λ ∈ [0.967, 4.833] 元/(MW·step)
= 边际收益的 10%~50%（事实上"管了"）；② 越限信号被平滑项稀释、且重尾（块和 median 12 /
max 814）⇒ 单块极值就能把 EMA 推高 16.3 > 预算 15 ⇒ λ 以 ~4 h 周期在边界间摆动；
③ λ_max = 4.833 **低于**两种平抑资源的边际成本（电池放电退化 8.17、转向交换率 26.5）
⇒ 机制上不可能实现"绝对不许"。

本模块的做法
------------
| 通道 | 承载 | 价格 | 落点 |
|---|---|---|---|
| **硬** | `c1_hard = max(0, P_F − 3) [− W_INEV·exc_inev]` | **自适应 λ_hard**（非对称乘性 + r_t 裁剪 + 上界抬到物理成本之上） | `critic_c`（`Q_p − λ·Q_c`） |
| **软** | `P_F`（全量） | **固定 κ**（建议 0.2 元/(MW·step) = 边际收益 2%） | **`r1`** |

软通道放进 `r1` ⇒ 成为"固定价偏好"（与 SOC 追踪的 `W_GAP` 同一模式）：给 `Q_p` 在低波动区
一个**单调、平滑、微小**的梯度（"为 Q 平缓的妥协"），但不与自适应 λ 混在同一通道。

λ_hard 更新式（每块一次，紧跟 `update_ema_cost`）
--------------------------------------------------
    E_t = agent.ema_cost                      # 块内 Σ c1_hard 的 EMA（α=0.02，由 main 维护）
    r_t = clip(E_t / eps, 0, r_cap)
    r_t > 1 :  λ ← λ · exp(+η_up·(r_t − 1))   # 快罚：越限出现时快速抬价
    r_t ≤ 1 :  λ ← λ · exp(−η_dn·(1 − r_t))   # 慢放：越限消失后缓慢释放
    λ ← clip(λ, λ_min, λ_max)                 # 物理价 ↦ actor 空间：×lam_scale()

* `r_cap` 修掉"单块极值一步到顶"；`η_up ≫ η_dn` 修掉"价格 4 h 周期摆动"；
* `λ_max` 必须**高于**任何平抑资源成本与任何单步收益（建议 200 ≈ 边际收益 20×），
  否则"绝对不许"与"理性"矛盾；
* `eps` 是"绝对"的可实现形式（不给 0，避免离散/量化下不可满足而抖动）。

默认**全关**（`--dual-mode ema` + `--c1-tail-only 0`）
------------------------------------------------------
* `Cal_Rew_RB` 的 r1/c1 与历史**逐位相同**；
* `update()` 直接转发 `hsrl_cost.dual_update` ⇒ 逐位相同；
⇒ 旧 9 个方法与本仓库历史 run 完全不受影响（`agentutil/test_yawflor_dual.py` 断言）。
"""
import numpy as np

import hsrl_cost as hc
from yawflor import yawflor_myfunc as _mf

# ---------------------------------------------------------------- 默认（= 历史口径）
MODE = 'ema'            # 'ema'（历史：hc.dual_update）| 'tail'（双通道）
EPS = 0.05             # 硬通道容差 [MW·step/块]（"绝对不许"的极小容差）
R_CAP = 10.0           # r_t 上裁剪（防单块极值一步到顶）
ETA_UP = 0.10          # 越限时抬价步长（快罚）
ETA_DN = 0.01          # 满足后放价步长（慢放；10:1 非对称）
LAM_MAX_PHYS = 200.0   # 物理价上界 [元/(MW·step)]（> 电池 8.17 / 转向 26.5）
LAM_MIN_PHYS = None    # None ⇒ 用 hc.LAM_MIN（= 0.9667）
KAPPA_SOFT = 0.0       # 软通道固定价 [元/(MW·step)]（>0 时同时打开 c1 只装越限）
W_INEV = 0.0           # 越限量中"电池侧不可避"部分的排除权重（0 = 全算 ⇒ 保留 H 预留 SOC 的动机）
LAST = {}              # 最近一次更新诊断（供日志/CSV）

DEFAULTS = dict(mode=MODE, eps=EPS, r_cap=R_CAP, eta_up=ETA_UP, eta_dn=ETA_DN,
                lam_max_phys=LAM_MAX_PHYS, lam_min_phys=LAM_MIN_PHYS,
                kappa_soft=KAPPA_SOFT, w_inev=W_INEV, c1_tail_only=False)


def add_args(parser):
    """把双通道开关挂到 yawflor_main 的 parser 上（默认值 = 历史口径）。"""
    g = parser.add_argument_group('yawflor 双通道对偶（硬=越限绝对不许 / 软=固定小价换 Q 平缓）')
    g.add_argument('--dual-mode', default=DEFAULTS['mode'], choices=['ema', 'tail'],
                   help='ema=历史口径（hc.dual_update，逐位不变）; tail=双通道硬通道对偶')
    g.add_argument('--kappa-soft', default=DEFAULTS['kappa_soft'], type=float,
                   help='软通道固定价 κ [元/(MW·step)]：r1 -= κ·P_F（默认 0 = 关；'
                        '建议 0.2 ≈ 边际收益 2%%，实现"<3 MW 原则上不管"）')
    g.add_argument('--c1-tail-only', default=0, type=int, choices=[0, 1],
                   help='1 ⇒ c1 只装越限超出量 (P_F−3)⁺（硬通道）；0 ⇒ 历史 c1。'
                        '给 --kappa-soft > 0 时自动按 1 处理')
    g.add_argument('--w-inev', default=DEFAULTS['w_inev'], type=float,
                   help='越限中"电池侧不可避"（rbb_info.inevitable_violation）的排除权重：'
                        '0 = 全算（默认，保留 H 预留 SOC headroom 的动机）; 1 = 完全排除')
    g.add_argument('--dual-eps', default=DEFAULTS['eps'], type=float,
                   help='硬通道容差 ε [MW·step/块]（默认 0.05 = 极紧；"绝对"的可实现形式）')
    g.add_argument('--dual-rcap', default=DEFAULTS['r_cap'], type=float,
                   help='r_t = clip(E/ε, 0, rcap) 的上裁剪（默认 10，防单块极值顶格）')
    g.add_argument('--dual-eta-up', default=DEFAULTS['eta_up'], type=float,
                   help='越限时抬价步长 η_up（默认 0.10 = 快罚）')
    g.add_argument('--dual-eta-dn', default=DEFAULTS['eta_dn'], type=float,
                   help='满足后放价步长 η_dn（默认 0.01 = 慢放；10:1 非对称）')
    g.add_argument('--dual-lammax', default=DEFAULTS['lam_max_phys'], type=float,
                   help='λ_hard 物理价上界 [元/(MW·step)]（默认 200，> 电池 8.17 / 转向 26.5）')
    g.add_argument('--dual-lammin', default=DEFAULTS['lam_min_phys'], type=float,
                   help='λ_hard 物理价下界（默认 None ⇒ 用 hsrl_cost.LAM_MIN）')
    return parser


def configure(args, verbose=True):
    """按 args 设定模块状态 + 把开关写进 `yawflor_myfunc` 的模块常量。

    ⚠ 只在 `--dual-mode tail`（或 `--kappa-soft > 0`）时改动 myfunc 的常量；
    默认路径**不写任何常量** ⇒ 历史口径逐位不变。
    """
    global MODE, EPS, R_CAP, ETA_UP, ETA_DN, LAM_MAX_PHYS, LAM_MIN_PHYS, KAPPA_SOFT, W_INEV
    MODE = str(getattr(args, 'dual_mode', DEFAULTS['mode']))
    kappa = float(getattr(args, 'kappa_soft', DEFAULTS['kappa_soft']))
    tail_only = bool(int(getattr(args, 'c1_tail_only', 0))) or (kappa > 0.0)
    EPS = float(getattr(args, 'dual_eps', DEFAULTS['eps']))
    R_CAP = float(getattr(args, 'dual_rcap', DEFAULTS['r_cap']))
    ETA_UP = float(getattr(args, 'dual_eta_up', DEFAULTS['eta_up']))
    ETA_DN = float(getattr(args, 'dual_eta_dn', DEFAULTS['eta_dn']))
    LAM_MAX_PHYS = float(getattr(args, 'dual_lammax', DEFAULTS['lam_max_phys']))
    _lm = getattr(args, 'dual_lammin', DEFAULTS['lam_min_phys'])
    LAM_MIN_PHYS = None if _lm is None else float(_lm)
    KAPPA_SOFT = kappa
    W_INEV = float(getattr(args, 'w_inev', DEFAULTS['w_inev']))

    if MODE == 'tail' or tail_only or kappa > 0.0:
        _mf.C1_TAIL_ONLY = bool(tail_only)
        _mf.KAPPA_SOFT = KAPPA_SOFT
        _mf.W_INEV = W_INEV
    txt = summary()
    if verbose:
        try:
            from util import prCyan
        except Exception:                                        # pragma: no cover
            prCyan = print
        prCyan('[yawflor-dual] ' + txt)
    return dict(mode=MODE, eps=EPS, r_cap=R_CAP, eta_up=ETA_UP, eta_dn=ETA_DN,
                lam_max_phys=LAM_MAX_PHYS, lam_min_phys=LAM_MIN_PHYS,
                kappa_soft=KAPPA_SOFT, w_inev=W_INEV, c1_tail_only=bool(_mf.C1_TAIL_ONLY))


def summary():
    kl = hc.lam_scale()
    lo_phys = hc.LAM_MIN if LAM_MIN_PHYS is None else LAM_MIN_PHYS
    if MODE != 'tail':
        return ('mode=ema（历史口径 hc.dual_update，逐位不变）'
                + ('；c1 只装越限 (κ=%.3g)' % KAPPA_SOFT if _mf.C1_TAIL_ONLY else ''))
    return ('mode=tail | 硬通道 c1=(P_F−3)⁺%s | λ_phys ∈ [%.3f, %.3f] 元/(MW·step) '
            '(actor 空间 ×%.3g = [%.4f, %.4f]) | ε=%.3g MW·step r_cap=%.3g '
            'η_up/η_dn=%.3g/%.3g | 软通道 r1 −= %.3g·P_F'
            % (' −%.3g·exc_inev' % W_INEV if W_INEV else '', lo_phys, LAM_MAX_PHYS, kl,
               lo_phys * kl, LAM_MAX_PHYS * kl, EPS, R_CAP, ETA_UP, ETA_DN, KAPPA_SOFT))


def update(agent, d_block):
    """块末对偶更新。`mode != 'tail'` ⇒ 转发 `hsrl_cost.dual_update`（逐位不变）。"""
    if MODE != 'tail':
        return hc.dual_update(agent, d_block)

    if agent is None or getattr(agent, 'ema_cost', None) is None:
        return None
    kl = hc.lam_scale()
    lam_min = (hc.LAM_MIN if LAM_MIN_PHYS is None else LAM_MIN_PHYS) * kl
    lam_max = LAM_MAX_PHYS * kl
    E = float(agent.ema_cost)
    r = 0.0 if EPS <= 0 else float(np.clip(E / EPS, 0.0, R_CAP))
    lam = float(agent.lam)
    if r > 1.0:
        lam *= float(np.exp(ETA_UP * (r - 1.0)))          # 快罚
    else:
        lam *= float(np.exp(-ETA_DN * (1.0 - r)))         # 慢放
    lam = min(max(lam, lam_min), lam_max)
    agent.lam = lam
    # 让 agent 自身的 clip 路径（update_policy 的 legacy 分支等）与新界一致
    agent.lam_min, agent.lam_max = lam_min, lam_max
    LAST.update(E=E, r=r, lam=lam, lam_min=lam_min, lam_max=lam_max,
                at_max=bool(lam >= lam_max * (1 - 1e-12)),
                at_min=bool(lam <= lam_min * (1 + 1e-12)))
    return lam
