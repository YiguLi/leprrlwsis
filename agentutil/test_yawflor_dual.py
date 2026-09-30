#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_yawflor_dual.py —— yawflor **双通道对偶**的契约测试（只读，不需 MATLAB）。

规则（用户给定）
----------------
> **大于 3 MW 的波动绝对不允许**；**小于 3 MW 的波动原则上不管**，
> 但为了 RL（Q 值平缓之类）可以适当妥协。

本测试断言三件事：

1. **默认关 = 历史口径，逐位不变**（对照改动前生成的基准
   `agentutil/yawflor_legacy_fixture.json`）：
   * `Cal_Rew_RB` 的 `r1/c1/turbine_profit/bess_profit/degradation/P_F` 全部**按 float.hex 逐位相同**；
   * `yawflor_dual.update(agent, d)` ≡ `hsrl_cost.dual_update(agent, d)`；
   * 模块常量保持默认（`C1_TAIL_ONLY=False` / `KAPPA_SOFT=0` / `W_INEV=0` / `MODE='ema'`）。
2. **tail 模式的数学**：`r_t` 裁剪、`η_up ≫ η_dn` 非对称、物理价↔actor 空间的 `lam_scale`、
   λ 上下界与贴顶/贴底标志。
3. **双通道口径**：`c1 = (P_F−3)⁺ − W_INEV·exc_inev`；`r1` 恰好减少 `κ·P_F`；
   `details` 新增 `exc_tot/exc_inev/soft_pen`；`--kappa-soft > 0` 会自动打开 tail c1。

用法
----
    python agentutil/test_yawflor_dual.py
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import hsrl_cost as hc                                 # noqa: E402
import yawflor.yawflor_myfunc as mf                    # noqa: E402
from yawflor import yawflor_dual as ydual              # noqa: E402

FIX = os.path.join(_HERE, 'yawflor_legacy_fixture.json')
FAIL = []


def check(ok, msg):
    print('  [%s] %s' % ('OK  ' if ok else 'FAIL', msg))
    if not ok:
        FAIL.append(msg)


class _Agent:
    pass


def _mk_agent(lam, ema, lam_min, lam_max, lr):
    a = _Agent()
    a.lam, a.ema_cost, a.ema_lam_lr = float(lam), float(ema), float(lr)
    a.lam_min, a.lam_max = float(lam_min), float(lam_max)
    return a


print('=' * 96)
print('1. 默认关 ⇒ 与改动前逐位不变（基准 %s）' % os.path.basename(FIX))
print('=' * 96)
snap = json.load(open(FIX, encoding='utf-8'))
hc.set_mode(snap['c1_mode'])
hc.set_value_scale(snap['value_scale'])
hc.set_cost_scale(snap['cost_scale'])
print('  基准: c1_mode=%s s_p=%g s_c=%g lam_scale=%g λ_phys∈[%.4f, %.4f]'
      % (snap['c1_mode'], snap['value_scale'], snap['cost_scale'], snap['lam_scale'],
         snap['lam_min'], snap['lam_max']))

check(mf.PENALTY_NN_H_COEF == snap['penalty_coef'],
      'PENALTY_NN_H_COEF = %.0f（基准 %.0f）' % (mf.PENALTY_NN_H_COEF, snap['penalty_coef']))
check(mf.C1_TAIL_ONLY is False and mf.KAPPA_SOFT == 0.0 and mf.W_INEV == 0.0,
      'myfunc 双通道常量默认关（C1_TAIL_ONLY=False / KAPPA_SOFT=0 / W_INEV=0）')
check(ydual.MODE == 'ema', "ydual.MODE 默认 'ema'（历史口径）")

# --- Cal_Rew_RB 逐位 ---
CASES = [
    (10.0, 580.0, 6.0, 2.0, 12.5, 0.0, 0.3, 3.0, 3.2, 0.1),
    (6.5, 300.0, 4.2, -1.5, 0.0, 0.0, 0.0, 4.5, 4.3, -0.25),
    (12.5, 980.0, 9.5, 0.8, 33.0, 0.4, 0.05, 2.0, 2.1, 0.47),
    (8.0, 580.0, 5.5, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0),
]
bad = []
for rec, (u, pr, ptur, pb, pen, gap, clip, E, E2, sig) in zip(snap['cal_rew'], CASES):
    P_G_prev = ptur - pb
    P_F = float.fromhex(rec['P_F'])
    obs = np.array([pr, u, E, P_G_prev])
    obs2 = np.array([pr, u, E2, P_G_prev + (P_F if rec['case'] % 2 == 0 else -P_F)])
    act = np.array([0.3, -0.2, 0.1, 0.0, 0.4, sig, pb])
    info = dict(pbrule=pb, unfeasible_gap=gap, clip_amount=clip,
                is_infeasible=bool(gap > 0), inevitable_violation=0.0)
    r1, c1, det = mf.Cal_Rew_RB(obs, obs2, act, info, penalty_NN_H=pen,
                                time=1.0 / 60.0, ptur=ptur, pb_cmd=pb)
    for k, v in (('r1', r1), ('c1', c1), ('turbine_profit', det['turbine_profit']),
                 ('bess_profit', det['bess_profit']), ('degradation', det['degradation']),
                 ('P_F', det['P_F'])):
        if float(v).hex() != rec[k]:
            bad.append((rec['case'], k, float(v).hex(), rec[k]))
check(not bad, 'Cal_Rew_RB 四项分解 + r1/c1 逐位相同（%d 组案例）%s'
      % (len(CASES), '' if not bad else ' 差异: %s' % bad[:3]))
check('exc_tot' in det and 'exc_inev' in det and 'soft_pen' in det,
      'details 含新增审计列 exc_tot/exc_inev/soft_pen（纯新增键）')
check(det['soft_pen'] == 0.0 and det['exc_tot'] == max(0.0, det['P_F'] - 3.0),
      '关模式下 soft_pen ≡ 0、exc_tot = max(0, P_F−3)')

# --- dual_update 逐位 ---
bad = []
for rec in snap['dual']:
    a_new = _mk_agent(rec['lam0'], rec['ema'], snap['lam_min'] * snap['lam_scale'],
                      snap['lam_max'] * snap['lam_scale'], hc.ETA_LAM)
    ydual.update(a_new, snap['d_block'])
    if a_new.lam.hex() != rec['lam1']:
        bad.append((rec, a_new.lam.hex()))
check(not bad, 'ydual.update(mode=ema) ≡ hc.dual_update 逐位相同（%d 组状态）%s'
      % (len(snap['dual']), '' if not bad else ' 差异: %s' % bad[:2]))

print()
print('=' * 96)
print('2. tail 模式数学（r_t 裁剪 / 非对称 / lam_scale / 上下界）')
print('=' * 96)
ydual.MODE = 'tail'
ydual.EPS, ydual.R_CAP = 0.05, 10.0
ydual.ETA_UP, ydual.ETA_DN = 0.10, 0.01
ydual.LAM_MAX_PHYS, ydual.LAM_MIN_PHYS = 200.0, None
kl = hc.lam_scale()
lam_min_a, lam_max_a = hc.LAM_MIN * kl, 200.0 * kl

# r_t 裁剪：E 极大时增量不超过 exp(η_up·(r_cap−1))
a = _mk_agent(lam_min_a, 1e6, 0.0, 1e9, hc.ETA_LAM)
ydual.update(a, hc.D_BLOCK)
grow_cap = np.exp(ydual.ETA_UP * (ydual.R_CAP - 1.0))
check(abs(a.lam / lam_min_a - grow_cap) < 1e-12,
      'r_t 被裁剪到 r_cap=%.0f ⇒ 单块最大增幅 %.4f×（实测 %.4f×）'
      % (ydual.R_CAP, grow_cap, a.lam / lam_min_a))

# 非对称：同 |r−1| 偏差，抬价幅度 > 放价幅度（η_up/η_dn = 10）
a_up = _mk_agent(1.0, 0.05 * 14.0, 0.0, 1e9, hc.ETA_LAM)     # r=14 → clip 10 → r−1=9
ydual.update(a_up, hc.D_BLOCK)
a_dn = _mk_agent(1.0, 0.05 * 0.0, 0.0, 1e9, hc.ETA_LAM)      # r=0 → 1−r=1
ydual.update(a_dn, hc.D_BLOCK)
check(a_up.lam / 1.0 > 1.0 and a_dn.lam / 1.0 < 1.0
      and (a_up.lam - 1.0) > (1.0 - a_dn.lam),
      '非对称：越限时快抬（%.4f×）> 满足时慢放（%.4f×）'
      % (a_up.lam / 1.0, a_dn.lam / 1.0))

# 慢放：E=0 时每块恰乘 exp(−η_dn)
a = _mk_agent(1.0, 0.0, 0.0, 1e9, hc.ETA_LAM)
ydual.update(a, hc.D_BLOCK)
check(abs(a.lam - np.exp(-ydual.ETA_DN)) < 1e-15,
      'E=0 ⇒ λ × exp(−η_dn) = %.6f（实测 %.6f）' % (np.exp(-ydual.ETA_DN), a.lam))

# 物理价 ↔ actor 空间 + 贴顶标志
a = _mk_agent(lam_max_a * 0.99, 1e6, 0.0, 1e9, hc.ETA_LAM)
ydual.update(a, hc.D_BLOCK)
check(abs(a.lam - lam_max_a) < 1e-12 and ydual.LAST['at_max'] is True,
      'λ 被 clip 到物理上界 %.1f × lam_scale(%.3g) = %.4f（actor 空间）且 at_max=True'
      % (200.0, kl, lam_max_a))
check(abs(ydual.LAST['lam_max'] - lam_max_a) < 1e-15 and abs(a.lam_min - lam_min_a) < 1e-15,
      '上下界写回 agent：λ_actor ∈ [%.4f, %.4f]' % (lam_min_a, lam_max_a))

print()
print('=' * 96)
print('3. 双通道口径（c1 = 越限 / r1 −= κ·P_F）')
print('=' * 96)
mf.C1_TAIL_ONLY, mf.KAPPA_SOFT, mf.W_INEV = True, 0.2, 0.0
for (u, pr, ptur, pb, pen, E, E2) in ((10.0, 580.0, 6.0, 0.0, 0.0, 3.0, 3.1),
                                      (10.0, 580.0, 9.0, 0.0, 0.0, 3.0, 3.1)):
    P_G_prev = ptur - pb
    P_F = 4.5 if ptur > 8 else 2.0
    obs = np.array([pr, u, E, P_G_prev])
    obs2 = np.array([pr, u, E2, P_G_prev + P_F])
    act = np.array([0.1, -0.2, 0.0, 0.3, -0.4, 0.0, pb])
    for inev in (0.0, 1.25):
        info = dict(pbrule=pb, unfeasible_gap=0.0, clip_amount=0.0,
                    is_infeasible=False, inevitable_violation=inev)
        r1, c1, det = mf.Cal_Rew_RB(obs, obs2, act, info, penalty_NN_H=pen,
                                    time=1.0 / 60.0, ptur=ptur, pb_cmd=pb)
        want_c1 = max(0.0, max(0.0, P_F - 3.0) - mf.W_INEV * inev)
        want_soft = mf.KAPPA_SOFT * P_F
        base = (mf.get_wT(det['wind_level'], det['price_level']) * pr * ptur / 60.0
                + mf.get_wB(det['price_level'], det['wind_level'], E) * (-pr * pb / 60.0)
                - 490.0 * max(-pb, 0.0) / 60.0 - pen)
        check(abs(c1 - want_c1) < 1e-15 and abs(r1 - (base - want_soft)) < 1e-12
              and abs(det['soft_pen'] - want_soft) < 1e-15,
              'P_F=%.1f inev=%.2f ⇒ c1=%.4f（=越限）且 r1 比基准少 κ·P_F=%.4f'
              % (P_F, inev, c1, det['soft_pen']))

# **软通道必须独立于 c1 口径**（修过一次："--c1-tail-only 0 且 κ>0" 时 κ 曾静默失效）
mf.C1_TAIL_ONLY, mf.KAPPA_SOFT, mf.W_INEV = False, 0.5, 0.0
_u, _pr, _ptur, _pb, _pen, _E, _E2 = 10.0, 580.0, 6.0, 0.0, 0.0, 3.0, 3.1
_P_F = 2.0
_obs = np.array([_pr, _u, _E, _ptur - _pb])
_obs2 = np.array([_pr, _u, _E2, _ptur - _pb + _P_F])
_act = np.array([0.1, -0.2, 0.0, 0.3, -0.4, 0.0, _pb])
_info = dict(pbrule=_pb, unfeasible_gap=0.0, clip_amount=0.0,
             is_infeasible=False, inevitable_violation=0.0)
_r1, _c1, _det = mf.Cal_Rew_RB(_obs, _obs2, _act, _info, penalty_NN_H=_pen,
                               time=1.0 / 60.0, ptur=_ptur, pb_cmd=_pb)
_base = (mf.get_wT(_det['wind_level'], _det['price_level']) * _pr * _ptur / 60.0
         + mf.get_wB(_det['price_level'], _det['wind_level'], _E) * (-_pr * _pb / 60.0)
         - 490.0 * max(-_pb, 0.0) / 60.0 - _pen)
check(abs(_c1 - _P_F) < 1e-12 and abs(_det['soft_pen'] - 0.5 * _P_F) < 1e-15
      and abs(_r1 - (_base - 0.5 * _P_F)) < 1e-12,
      'C1_TAIL_ONLY=False 且 κ>0 ⇒ c1 仍走历史口径、软价照常生效（修过：曾静默失效）')
mf.C1_TAIL_ONLY, mf.KAPPA_SOFT, mf.W_INEV = True, 0.2, 0.0

# configure 语义：--c1-tail-only 三态（auto 默认 / 显式 1 / 显式 0）
class _NS:
    pass


ns = _NS()
ns.dual_mode, ns.kappa_soft, ns.c1_tail_only = 'ema', 0.0, 'auto'
ns.dual_eps, ns.dual_rcap, ns.dual_eta_up, ns.dual_eta_dn = 0.05, 10.0, 0.1, 0.01
ns.dual_lammax, ns.dual_lammin, ns.w_inev = 200.0, None, 0.0
mf.C1_TAIL_ONLY, mf.KAPPA_SOFT, mf.W_INEV = False, 0.0, 0.0
ydual.configure(ns, verbose=False)
check(ydual.MODE == 'ema' and mf.C1_TAIL_ONLY is False and mf.KAPPA_SOFT == 0.0,
      'configure(--dual-mode ema --kappa-soft 0) 不动任何常量')
ns.kappa_soft = 0.2
ydual.configure(ns, verbose=False)
check(mf.C1_TAIL_ONLY is True and mf.KAPPA_SOFT == 0.2,
      'configure(--kappa-soft 0.2, auto) 自动打开 c1 只装越限（C1_TAIL_ONLY=True）')
# E2c 用到的格：显式 0 ⇒ 保留历史 c1，但仍加软价 κ（隔离"只加软价、不加硬通道"）
ns.c1_tail_only = 0
ydual.configure(ns, verbose=False)
check(mf.C1_TAIL_ONLY is False and mf.KAPPA_SOFT == 0.2,
      "configure(--c1-tail-only 0 --kappa-soft 0.2) ⇒ 历史 c1 + 软价（E2c 格可用）")
# 显式 1 ⇒ 与 auto 同义
ns.c1_tail_only = 1
ydual.configure(ns, verbose=False)
check(mf.C1_TAIL_ONLY is True, "configure(--c1-tail-only 1) ⇒ 只装越限")
# tail 模式 + auto ⇒ 只装越限（即使 κ=0）
ns.dual_mode, ns.c1_tail_only, ns.kappa_soft = 'tail', 'auto', 0.0
ydual.configure(ns, verbose=False)
check(ydual.MODE == 'tail' and mf.C1_TAIL_ONLY is True,
      'configure(--dual-mode tail, auto, κ=0) ⇒ 只装越限（硬通道）')

# 复原（避免影响同进程后续）
mf.C1_TAIL_ONLY, mf.KAPPA_SOFT, mf.W_INEV = False, 0.0, 0.0
ydual.MODE = 'ema'

print()
print('=' * 96)
if FAIL:
    print('TEST FAILED — %d 项未通过:' % len(FAIL))
    for f in FAIL:
        print('  - ' + f)
    sys.exit(1)
print('TEST PASSED — 默认关逐位不变 + tail 模式数学 + 双通道口径全部成立')
