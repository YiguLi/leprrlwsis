# -*- coding: utf-8 -*-
"""`--nolow` 分支（td3per / td3perseg）契约测试 —— **不需要 MATLAB**。

分支语义（2026-10 口径）
-----------------------
取消 H 层第 6 维（SOC 目标信号）⇒ H 动作 5 维；BESS **只做一件事**：
**全力朝 `SOC = SOC_TARGET_CENTER`(3.0) 方向充/放**，幅度取**平抑可行窗口的对应端点**
（`E < 3` ⇒ `pbrulemax` 全力充；`E > 3` ⇒ `pbrulemin` 全力放；`E = 3` ⇒ 0）。
下层 RL 完全旁路（`r2 ≡ c2 ≡ 0`），SOC 目标相关的 gap 罚恒 0。

判据
----
  A. `RuleBasedBESS.compute_pbrule_soc_seek`: 取值 = 窗口在目标方向上的端点、恒在窗口内、
     可行窗口非空时 `|pg(t)−pg(t−1)| ≤ 3` 不被打破、`clip_amount ≡ 0`；
     原 `compute_pbrule` 未被改动（同输入 = `clip(0, 窗口)`）。
  B. 5 维 H 动作链路：`CheckAction(len==5)` / `violation_amount` / `violation_derate` 正常。
  C. `nine_case_test(nolow=True, agent_L=None)` 端到端（桩 env）：跑通、`r2 = c2 = 0`、
     SOC 朝 3.0 收敛（起点 5.0 ⇒ 放电 ⇒ `dsoc < 0`；起点 1.0 ⇒ 充电 ⇒ `dsoc > 0`）。
  D. 静态接线：两个 main 的 `--nolow` / `nb_actions_H 5|6` / RUN_TAG 后缀 / `utd_l = 0`；
     两个 myfunc 的 `nine_case_test(..., nolow=False)` 与 `compute_pbrule_soc_seek` 调用。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

from rbb.rbb_rule_bess import RuleBasedBESS                       # noqa: E402
from td3per.td3per_myfunc import (CheckAction, violation_amount,  # noqa: E402
                                  violation_derate, nine_case_test)
from td3perseg.td3perseg_myfunc import nine_case_test as nct_seg  # noqa: E402

FAIL = []


def check(cond, msg):
    print('  %s %s' % ('✔' if cond else '✘', msg))
    if not cond:
        FAIL.append(msg)


# =============================================================================
print('=' * 96)
print('A. RBB compute_pbrule_soc_seek: 朝 SOC=3.0 全力充/放（取窗口端点）')
print('=' * 96)
rbb = RuleBasedBESS(max_E=6.0, min_E_factor=0.2, rated_power=10.0,
                    eta_c=0.98, eta_d=0.98, time=1.0 / 60, fluctuation_limit=3.0)
rng = np.random.RandomState(0)
bad_dir = bad_win = bad_clip = bad_pf = bad_old = n_in = n_out = n_zero = 0
for _ in range(4000):
    ptur = rng.uniform(0, 12)
    pg_prev = rng.uniform(0, 12)
    E = rng.uniform(0.6, 5.6)
    pb, info = rbb.compute_pbrule_soc_seek(ptur=ptur, pg_prev=pg_prev, E=E)
    _, ref = rbb.compute_pbrule(pb_original=0.0, ptur=ptur, pg_prev=pg_prev, E=E)
    if info['is_infeasible']:
        continue
    n_in += 1
    err = 3.0 - E
    exp = float(info['pbrulemax']) if err > 0 else (float(info['pbrulemin']) if err < 0 else 0.0)
    if abs(pb - exp) > 1e-12:
        bad_dir += 1
    if pb < info['pbrulemin'] - 1e-12 or pb > info['pbrulemax'] + 1e-12:
        bad_win += 1
    if info['clip_amount'] != 0.0 or info['pb_original'] != info['pbrule']:
        bad_clip += 1
    if abs(ptur - pb - pg_prev) > 3.0 + 1e-9:
        bad_pf += 1
    if abs(ref['pbrule'] - float(np.clip(0.0, ref['pbrulemin'], ref['pbrulemax']))) > 1e-12:
        bad_old += 1
    n_out += int(E > 3.0)
    n_zero += int(abs(err) < 1e-12)
check(bad_dir == 0, '取值方向正确（%d 个可行状态, 放电侧 %d 个）' % (n_in, n_out))
check(bad_win == 0, '取值恒在 [pbrulemin, pbrulemax] 内（平抑+物理窗口）')
check(bad_clip == 0, 'clip_amount ≡ 0 且 pb_original = pbrule')
check(bad_pf == 0, '可行窗口非空时 |pg(t)−pg(t−1)| ≤ 3（平抑约束不被打破）')
check(bad_old == 0, '原 compute_pbrule 逐位未变（= clip(0, 窗口)）')
pb0, i0 = rbb.compute_pbrule_soc_seek(ptur=6.0, pg_prev=6.0, E=3.0)
check(pb0 == 0.0, 'E = 3.0 ⇒ 功率 0（在目标上不动）, 实测 %.6f' % pb0)
pb_lo, i_lo = rbb.compute_pbrule_soc_seek(ptur=6.0, pg_prev=6.0, E=2.0)
check(pb_lo == i_lo['pbrulemax'] and pb_lo > 0, 'E = 2.0 < 3 ⇒ 全力充（%.3f = 窗口上限）' % pb_lo)
pb_hi, i_hi = rbb.compute_pbrule_soc_seek(ptur=6.0, pg_prev=6.0, E=4.0)
check(pb_hi == i_hi['pbrulemin'] and pb_hi < 0, 'E = 4.0 > 3 ⇒ 全力放（%.3f = 窗口下限）' % pb_hi)
pb_db, _ = rbb.compute_pbrule_soc_seek(ptur=6.0, pg_prev=6.0, E=3.02, deadband=0.05)
check(pb_db == 0.0, 'deadband=0.05 且 |E−3|=0.02 ⇒ 0（不动）')

# =============================================================================
print()
print('=' * 96)
print('B. 5 维 H 动作链路（nolow 的 H 动作维数）')
print('=' * 96)
ap, pen = CheckAction(np.array([-0.1, 0.2, 0.6, 0.3, 0.4]))
check(np.allclose(ap, [0.0, 0.2, 0.5, 0.3, 0.4]) and abs(pen - 80.0) < 1e-9,
      'CheckAction(len==5): 夹到 [0,0.5] 且罚 = 400×0.2 = %.1f' % pen)
check(abs(violation_amount(np.array([0.1, 0.1, 0.1, 0.1, 0.9])) - 0.4) < 1e-12,
      'violation_amount 只看前 5 维（= 0.4）')
check(violation_derate(np.array([0.2] * 5), 5.0) == 1.0, '框内 ⇒ 乘性系数 = 1.0')


# =============================================================================
print()
print('=' * 96)
print('C. nine_case_test(nolow=True, agent_L=None) 端到端（桩 env, 不需 MATLAB）')
print('=' * 96)


class StubAgent:
    def __init__(self, a, nb_states=6):
        self.a = np.asarray(a, dtype=float).ravel().copy()
        self.is_training = True
        self.nb_states = nb_states
        self.nb_actions = self.a.size

    def select_action(self, obs, *a, **k):
        return self.a.copy()

    def reset(self, obs=None, *a, **k):
        self.s_t = obs


class StubEnv:
    """只实现 nine_case_test 需要的三个方法（PowerOutput 用确定性公式代替 MATLAB）。"""

    def __init__(self, n_turbine=5, E0=5.0):
        self.num_turbine = n_turbine
        self.max_E, self.min_E, self.eta, self.time = 6.0, 0.0, 0.98, 1.0 / 60
        self.E0, self.E = E0, E0
        self._i, self._wind, self._price = 0, [], []

    def PowerOutput(self, u, act_wind):
        a = np.clip(np.asarray(act_wind, dtype=float).ravel()[:self.num_turbine], 0.0, 0.5)
        return float(a.sum() * 10.0 * min(max(float(u), 0.0), 12.0) / 12.0)

    def ScenarioReset(self, wind_seq, price_seq):
        self._wind, self._price, self._i = list(wind_seq), list(price_seq), 0
        self.E = self.E0
        pw = self.PowerOutput(self._wind[0], np.ones(self.num_turbine) * 0.3333)
        return np.array([float(self._price[0]), float(self._wind[0]), self.E, pw])

    def ScenarioStep(self, applied_action_H, act_bat, pw_tru):
        self._i = min(self._i + 1, len(self._wind) - 1)
        u, pr = float(self._wind[self._i]), float(self._price[self._i])
        pw2 = self.PowerOutput(u, applied_action_H)
        cmd = float(np.asarray(act_bat, dtype=float).ravel()[0])
        P_C, P_D = max(0.0, cmd), abs(min(cmd, 0.0))
        E_t = self.E + (P_C * self.eta - P_D / self.eta) * self.time
        if E_t > self.max_E:
            P_C = (self.max_E - self.E) / self.eta / self.time
            self.E = self.max_E
        elif E_t < self.min_E:
            P_D = (self.E - self.min_E) * self.eta / self.time
            self.E = self.min_E
        else:
            self.E = E_t
        P_G = float(pw_tru) - P_C + P_D
        return (pw2, np.array([pr, u, self.E, P_G]), np.array([P_C - P_D]),
                0.0, False, False, {'P_F': 0.0, 'pbrule': P_C - P_D, 'P_G': P_G})


A5 = [0.30, 0.25, 0.35, 0.30, 0.33]
for tag, fn in (('td3per', nine_case_test), ('td3perseg', nct_seg)):
    for E0, want_sign, lab in ((5.0, -1, 'E0=5.0 > 3 ⇒ 净放电'), (1.0, +1, 'E0=1.0 < 3 ⇒ 净充电')):
        cm, ov = fn(StubAgent(A5), None, StubEnv(E0=E0), RuleBasedBESS(), seed=42, nolow=True)
        c2 = float(np.mean([m['c2'] for m in cm]))
        r2 = float(np.mean([m['r2'] for m in cm]))
        te = float(np.mean([m['soc_track_err'] for m in cm]))
        ok = (abs(c2) < 1e-12 and abs(r2) < 1e-12 and te == 0.0
              and np.sign(ov['dsoc']) == want_sign)
        check(ok, '%s %s: dsoc = %+.4f MWh, c2 = %.4f, r2 = %.4f, track_err = %.3f, profit = %.1f'
                 % (tag, lab, ov['dsoc'], c2, r2, te, ov['profit']))
        if tag == 'td3per' and E0 == 5.0:
            print('       （对照: 旧口径"恒取最靠充电端"实测 dsoc = +1.0 ⇒ 新口径方向相反 ✓）')

# =============================================================================
print()
print('=' * 96)
print('D. 静态接线')
print('=' * 96)
for mp, tag in (('td3per/td3per_main.py', 'td3per'), ('td3perseg/td3perseg_main.py', 'td3perseg')):
    s = open(mp, encoding='utf-8').read()
    check("add_argument('--nolow'" in s, '%s: 有 --nolow 开关' % tag)
    check('nb_actions_H = 5 if getattr(args, ' in s, '%s: nb_actions_H 6→5 分支' % tag)
    check("RUN_TAG = RUN_TAG + 'nolow'" in s and "args.env = args.env + 'NOLOW'" in s,
          '%s: RUN_TAG/--env 带 nolow 后缀' % tag)
    check('args.utd_l = 0' in s, '%s: 下层 RL 旁路（utd_l = 0）' % tag)
    check('compute_pbrule_soc_seek' in s and 'soc_target=soc_target' in s,
          '%s: 调用 compute_pbrule_soc_seek(..., soc_target=soc_target)' % tag)
    check('compute_pbrule_min_discharge' not in s, '%s: 旧"放电最小"调用已全部替换' % tag)
for mp, tag in (('td3per/td3per_myfunc.py', 'td3per_myfunc'),
                ('td3perseg/td3perseg_myfunc.py', 'td3perseg_myfunc')):
    s = open(mp, encoding='utf-8').read()
    check('nolow=False' in s and 'compute_pbrule_soc_seek' in s,
          '%s: nine_case_test 支持 nolow + 新规则调用' % tag)

print()
print('=' * 96)
if FAIL:
    print('**失败 %d 项**' % len(FAIL))
    for f in FAIL:
        print('  - %s' % f)
else:
    print('nolow 分支契约测试：全部通过 ✅')
print('=' * 96)
sys.exit(1 if FAIL else 0)
