# -*- coding: utf-8 -*-
"""`--nolow` 分支（td3per / td3perseg）契约测试 —— **不需要 MATLAB**。

分支语义（2026-10 口径，run101 修正版）
------------------------------------
取消 H 层第 6 维（SOC 目标信号）⇒ H 动作 5 维；BESS **只做一件事**：
**全力朝 `SOC = SOC_TARGET_CENTER`(3.0) 方向充/放**，由**单一真源** `hst.rule_action`
（与 `SocTracker.act` 逐位一致）给出、目标钉在 3.0：

    |δ| ≤ soc_tol  ⇒  clip(0, W)          ← 窗口不含 0 时做**最小幅度被迫吞吐**（= 只平抑）
    否则           ⇒  clip(δ/g, W)        ← g 取"一步填平" ⇒ **全力朝 3.0 方向**充/放

⚠ **run101 的教训**（旧实现 `compute_pbrule_soc_seek` 恒取窗口端点、δ=0 时硬返回 0）：
SOC 恰在 3.0（= episode 起始值）时输出 0 ⇒ SOC 不变 ⇒ **死不动点**，平抑整体失效
（run101 实测 SOC 恒定 3.0、pbrule 恒 0、`P_F>3` 占比 11.31% = 电池不动作的基线）。
本测试的 A2 段就是该场景的回归。

判据
----
  A1. `hst.rule_action` 死区/全力语义：δ≈0 且窗口不含 0 ⇒ 窗口最近端（**不是 0**）；
      δ≈0 且 0 ∈ 窗口 ⇒ 0；|δ| 大 ⇒ 窗口对应端点（全力）。
  A2. **run101 回归**：SOC 恰在 3.0 + >3 MW/min 风功率爬坡 ⇒ 动作非 0 且 `|pg(t)−pg(t−1)| ≤ 3`。
  A3. RBB `compute_pbrule` 未被改动（同输入 = `clip(0, 窗口)`）；RBB 里已无 nolow 专用方法。
  B.  5 维 H 动作链路：`CheckAction(len==5)` / `violation_amount` / `violation_derate`。
  C.  `nine_case_test(nolow=True, agent_L=None)` 端到端（桩 env）：`r2 = c2 = 0`、
      SOC 朝 3.0 收敛（5.0 ⇒ 净放；1.0 ⇒ 净充；**3.0 ⇒ 稳住且不平抑失效**）、
      test9 场景 `pf_viol = 0`（窗口保证 `P_F ≤ 3`）。
  D.  静态接线：main 的 `--nolow` / `nb_actions_H 5|6` / RUN_TAG 后缀 / `utd_l = 0` /
      `hst.rule_action(目标 3.0)`；myfunc 的 `nine_case_test(..., nolow=False)` 同口径。
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
import hsrl_soctarget as hst                                      # noqa: E402
from td3per.td3per_myfunc import (CheckAction, violation_amount,  # noqa: E402
                                  violation_derate, nine_case_test)
from td3perseg.td3perseg_myfunc import nine_case_test as nct_seg  # noqa: E402

FAIL = []


def check(cond, msg):
    print('  %s %s' % ('✔' if cond else '✘', msg))
    if not cond:
        FAIL.append(msg)


def rule(tgt, E, lo, hi, n_rem=5):
    return float(hst.rule_action(tgt, E, lo, hi, n_rem,
                                 eta_c=0.98, eta_d=0.98, time=1.0 / 60, rated_power=10.0))


rbb = RuleBasedBESS(max_E=6.0, min_E_factor=0.2, rated_power=10.0,
                    eta_c=0.98, eta_d=0.98, time=1.0 / 60, fluctuation_limit=3.0)

# =============================================================================
print('=' * 96)
print('A1. hst.rule_action 语义（目标钉在 3.0）: 死区 = 窗口强制的**最小**吞吐, 否则全力')
print('=' * 96)
v1 = rule(3.0, 3.0, 2.0, 5.0)
check(abs(v1 - 2.0) < 1e-12, 'δ=0 且窗口 [2,5] 不含 0 ⇒ 取最近端 2.0（**不是 0**!）实测 %.4f' % v1)
v2 = rule(3.0, 3.0, -5.0, 5.0)
check(v2 == 0.0, 'δ=0 且 0 ∈ 窗口 ⇒ 0（不主动吞吐）')
v3 = rule(3.0, 3.0, -5.0, -2.0)
check(abs(v3 + 2.0) < 1e-12, 'δ=0 且窗口 [-5,-2] 不含 0 ⇒ 取最近端 −2.0, 实测 %.4f' % v3)
v4 = rule(3.0, 2.0, -5.0, 5.0)
check(abs(v4 - 5.0) < 1e-12, 'δ=+1 ⇒ 一步填平需求 61 MW 被窗口夹到 +5.0（全力充）, 实测 %.4f' % v4)
v5 = rule(3.0, 4.0, -5.0, 5.0)
check(abs(v5 + 5.0) < 1e-12, 'δ=−1 ⇒ 全力放至 −5.0, 实测 %.4f' % v5)
v6 = rule(3.0, 3.05, -5.0, 5.0)
check(abs(v6 + 0.05 / (1.0 / 60 / 0.98)) < 1e-9 and abs(v6) < 5.0,
      'δ=−0.05 ⇒ 比例动作 %.3f MW（不是端点 ⇒ 无自致过冲/抖动）' % v6)

# =============================================================================
print()
print('=' * 96)
print('A2. run101 回归: SOC 恰在 3.0 + 风功率爬坡 ⇒ 必须仍有动作且 P_F ≤ 3')
print('=' * 96)
E, pg_prev = 3.0, 6.0
rows = []
for ramp in (4.0, 6.0, 8.0, -5.0, -8.0):
    ptur = pg_prev + ramp
    lo, hi = ptur - pg_prev - 3.0, ptur - pg_prev + 3.0
    pb_rule = hst.rule_action(3.0, E, lo, hi, 5, eta_c=0.98, eta_d=0.98,
                              time=1.0 / 60, rated_power=10.0)
    pb, info = rbb.compute_pbrule(pb_original=pb_rule, ptur=ptur, pg_prev=pg_prev, E=E)
    PF = abs(ptur - pb - pg_prev)
    ok = (abs(pb) > 1e-9) and (PF <= 3.0 + 1e-9)
    rows.append(ok)
    print('    爬坡 %+5.1f MW/min: 规则 %.3f → 施加 %.3f MW, P_F = %.3f  %s'
          % (ramp, pb_rule, pb, PF, 'OK' if ok else '**FAIL**'))
check(all(rows), '所有爬坡情形都有非 0 动作且 P_F ≤ 3（旧实现此处恒为 0 ⇒ P_F = |爬坡|）')

# =============================================================================
print()
print('=' * 96)
print('A3. RBB compute_pbrule 未被改动 + 已无 nolow 专用方法')
print('=' * 96)
rng = np.random.RandomState(0)
bad_old = 0
for _ in range(2000):
    ptur, pg_prev, E = rng.uniform(0, 12), rng.uniform(0, 12), rng.uniform(0.6, 5.6)
    _, ref = rbb.compute_pbrule(pb_original=rng.uniform(-12, 12), ptur=ptur, pg_prev=pg_prev, E=E)
    exp = (float(np.clip(ref['pb_original'], ref['pbrulemin'], ref['pbrulemax']))
           if not ref['is_infeasible'] else ref['pbrule'])
    if abs(ref['pbrule'] - exp) > 1e-12:
        bad_old += 1
check(bad_old == 0, 'compute_pbrule 仍是 clip(命令, 窗口)（2000 随机状态 0 违例）')
s_rbb = open('rbb/rbb_rule_bess.py', encoding='utf-8').read()
check(('compute_pbrule_soc_seek' not in s_rbb) and ('min_discharge' not in s_rbb),
      'RBB 内已无 nolow 专用方法（回归到原版 ⇒ "rulebasebess 没被改坏"）')

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
#  注: RuleBasedBESS 的 SOC 下界 = max_E × min_E_factor = 1.2 MWh。E0 < 该下界时
#  平抑窗口与物理窗口必然冲突（infeasible）⇒ 被迫吞吐 + 不可避免的 P_F 越限 —— 这是
#  RBB 的既有口径（"最小化不可避免的越限"），不是 nolow 的 bug，故单独作为一档断言。
CASES_C = [
    (5.0, 'E0=5.0 > 3（可行）⇒ 净放电',
     lambda m: m['dsoc'] < -0.5 and m['r2'] == 0.0 and m['pf_viol'] == 0),
    (1.5, 'E0=1.5 < 3（可行）⇒ 净充电',
     lambda m: m['dsoc'] > +0.5 and m['r2'] == 0.0 and m['pf_viol'] == 0),
    (3.0, 'E0=3.0 = 目标（可行）⇒ 稳住不漂',
     lambda m: abs(m['dsoc']) < 0.2 and m['r2'] == 0.0 and m['pf_viol'] == 0),
    (1.0, 'E0=1.0 < 下界 1.2（窗口不可行）⇒ 被迫吞吐 + 不可避免越限',
     lambda m: m['dsoc'] > 0 and m['r2'] < 0 and m['pf_viol'] >= 1),
]
for tag, fn in (('td3per', nine_case_test), ('td3perseg', nct_seg)):
    for E0, lab, cond in CASES_C:
        cm, ov = fn(StubAgent(A5), None, StubEnv(E0=E0), RuleBasedBESS(), seed=42, nolow=True)
        c2 = float(np.mean([m['c2'] for m in cm]))
        r2 = float(np.mean([m['r2'] for m in cm]))
        pfv = int(sum(m['pf_viol'] for m in cm))
        te = float(np.mean([m['soc_track_err'] for m in cm]))
        m = {'dsoc': ov['dsoc'], 'r2': r2, 'pf_viol': pfv}
        ok = (abs(c2) < 1e-12 and te == 0.0 and cond(m))
        check(ok, '%s %s: dsoc = %+.4f MWh, c2 = %.4f, r2 = %+.4f, pf_viol = %d, '
                 'track_err = %.3f, profit = %.1f'
                 % (tag, lab, ov['dsoc'], c2, r2, pfv, te, ov['profit']))

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
    check('hst.rule_action(' in s and '_soc_tgt_L = soc_target' in s,
          '%s: nolow 规则 = hst.rule_action(目标钉在 3.0)' % tag)
    check('compute_pbrule_soc_seek' not in s and 'compute_pbrule_min_discharge' not in s,
          '%s: 旧"取窗口端点"实现已全部移除' % tag)
    ls = s.splitlines()
    i_rbb = [i for i, l in enumerate(ls) if 'pbrule, rbb_info = rbb_system.compute_pbrule(' in l]
    check(len(i_rbb) >= 1 and 'pb_original = float(pb_cmd_L)' in ls[i_rbb[0] - 1],
          '%s: RBB 收口为两条分支共用（紧邻上一行 = 统一命令值，无 nolow 专用分支）' % tag)
for mp, tag in (('td3per/td3per_myfunc.py', 'td3per_myfunc'),
                ('td3perseg/td3perseg_myfunc.py', 'td3perseg_myfunc')):
    s = open(mp, encoding='utf-8').read()
    check('nolow=False' in s and 'hst.rule_action(' in s,
          '%s: nine_case_test 支持 nolow + 走 hst.rule_action' % tag)

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
