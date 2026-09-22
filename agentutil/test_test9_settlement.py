#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test9（45 步 9 case 固定场景探针）的**存货结算**回归 —— 不需要 MATLAB。

问题
----
`profit` 是**现金流口径**：`profit = Σ[P_G·pr·dt − 490·max(−pbrule,0)·dt]`，
其中 `P_G = ptur − P_C + P_D`。储存在电池里的电其成本已扣（充电时 `P_G` 减少、
收入减少），价值却未认回 ⇒

* `ΔSOC = SOC_end − SOC_start > 0`（净充电）⇒ profit **低估**；
* `ΔSOC < 0`（净放电）⇒ 把**期初存量**卖掉记成利润 ⇒ profit **高估**。

test9 每次固定从 SOC = 5.0 开始（`ScenarioReset` 写死 `self.E = 5.`），而策略有强烈
抽干倾向，所以这是**系统性**偏差、且随训练变大 —— 会让"profit 上升"里混进"存货变现"。

结算口径（只修指标，不动训练路径）
----------------------------------
    profit_econ = profit + ΔSOC · SOC_SETTLE_PR        SOC_SETTLE_PR = hc.PR_REF = 580

参考价**必须外生常数**：若取"当时电价"，每步充放收益逐项抵消（`ΔSOC·(pr_ref−pr) ≡ 0`）
⇒ 电池对目标贡献恒为 0 ⇒ 套利激励归零。

`nine_case_test` 是纯 Python（只有 `env_test` 侧依赖 MATLAB），故本测试用
**桩 env + 桩 agent + 真 RuleBasedBESS** 真跑一遍完整路径。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np                                                  # noqa: E402

from td3prrbb.td3prrbb_myfunc import (                              # noqa: E402
    nine_case_test, SOC_SETTLE_PR)
from td3prrbb.td3prrbb_rule_bess import RuleBasedBESS               # noqa: E402

A_H = np.array([0.35, 0.2, 0.35, 0.4, 0.35, 0.0])   # 真实日志里的一个样本
MAX_E = 6.0
ETA = 0.98


class _StubAgent:
    def __init__(self, a):
        self.a = np.asarray(a, dtype=float).ravel().copy()
        self.is_training = True

    def select_action(self, obs):
        return self.a.copy()


class _StubEnv:
    """只实现 nine_case_test 需要的三个方法；PowerOutput 用确定性公式替代 MATLAB。"""

    def __init__(self, n_turbine=5):
        self.num_turbine = n_turbine
        self.max_E = MAX_E
        self.eta = ETA
        self.time = 1.0 / 60
        self.E = 5.0
        self._i = 0
        self._wind, self._price = [], []

    def PowerOutput(self, u, act_wind):
        a = np.clip(np.asarray(act_wind, dtype=float).ravel()[:self.num_turbine], 0.0, 0.5)
        return float(a.sum() * 10.0 * min(max(float(u), 0.0), 12.0) / 12.0)

    def ScenarioReset(self, wind_seq, price_seq):
        self._wind, self._price, self._i = list(wind_seq), list(price_seq), 0
        self.E = 5.0                                    # 与真 ScenarioReset 一致
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
        elif E_t < 0.0:
            P_D = self.E * self.eta / self.time
            self.E = 0.0
        else:
            self.E = E_t
        P_G = float(pw_tru) - P_C + P_D
        return (pw2, np.array([pr, u, self.E, P_G]), np.array([P_C - P_D]),
                0.0, False, False, {'P_F': 0.0})


def run(pb_cmd):
    return nine_case_test(_StubAgent(A_H), _StubAgent([pb_cmd]),
                          _StubEnv(), RuleBasedBESS(), seed=42)


print('=' * 92)
print('A. 参考价与列：SOC_SETTLE_PR = hc.PR_REF = 580；四个新列必须齐全')
print('=' * 92)
assert SOC_SETTLE_PR == 580.0, SOC_SETTLE_PR
NEED = ('soc_start', 'soc_end', 'dsoc', 'profit_econ')
cm, ov = run(0.0)
assert len(cm) == 9, len(cm)
for m in cm:
    for k in NEED:
        assert k in m, ('case 缺列', k, sorted(m))
for k in NEED:
    assert k in ov, ('overall 缺列', k, sorted(ov))
print('  SOC_SETTLE_PR=%.1f ✓  case(9)/overall 均含 %s ✓' % (SOC_SETTLE_PR, list(NEED)))

print()
print('=' * 92)
print('B. 算术：profit_econ == profit + dsoc·580（逐 case 与 overall 精确到 1e-9）')
print('=' * 92)
bad = [abs(m['profit_econ'] - (m['profit'] + m['dsoc'] * SOC_SETTLE_PR)) for m in cm]
assert max(bad) < 1e-9, max(bad)
assert abs(ov['profit_econ'] - (ov['profit'] + ov['dsoc'] * SOC_SETTLE_PR)) < 1e-9
print('  逐 case 最大偏差 %.3e ; overall 偏差 %.3e ✓'
      % (max(bad), abs(ov['profit_econ'] - (ov['profit'] + ov['dsoc'] * SOC_SETTLE_PR))))

print()
print('=' * 92)
print('C. 望远镜一致性：Σ(逐 case dsoc) == overall dsoc（SOC 跨 case 连续）')
print('=' * 92)
s = sum(m['dsoc'] for m in cm)
assert abs(s - ov['dsoc']) < 1e-9, (s, ov['dsoc'])
assert abs(ov['soc_start'] - cm[0]['soc_start']) < 1e-12
assert abs(ov['soc_end'] - cm[-1]['soc_end']) < 1e-12
print('  Σdsoc=%.9f  overall dsoc=%.9f ✓  overall 首末取自首/末 case ✓' % (s, ov['dsoc']))

print()
print('=' * 92)
print('D. 方向性（核心）：净放电 ⇒ profit 被高估；净充电 ⇒ 被低估')
print('=' * 92)
cm_d, ov_d = run(-10.0)          # 强制放电（rated 10 MW）
cm_c, ov_c = run(+10.0)          # 强制充电
assert ov_d['dsoc'] < -1.0, ov_d['dsoc']
assert ov_d['profit_econ'] < ov_d['profit'], (ov_d['profit'], ov_d['profit_econ'])
assert ov_c['dsoc'] > 0.5, ov_c['dsoc']
assert ov_c['profit_econ'] > ov_c['profit'], (ov_c['profit'], ov_c['profit_econ'])
print('  强制放电(-10MW): dsoc=%+.4f MWh ⇒ profit %+.2f → profit_econ %+.2f (修正 %+.2f) ✓'
      % (ov_d['dsoc'], ov_d['profit'], ov_d['profit_econ'],
         ov_d['profit_econ'] - ov_d['profit']))
print('  强制充电(+10MW): dsoc=%+.4f MWh ⇒ profit %+.2f → profit_econ %+.2f (修正 %+.2f) ✓'
      % (ov_c['dsoc'], ov_c['profit'], ov_c['profit_econ'],
         ov_c['profit_econ'] - ov_c['profit']))
assert abs((ov_d['profit_econ'] - ov_d['profit']) - ov_d['dsoc'] * 580) < 1e-9
print('  修正量恰为 dsoc×580 ✓（反过来说明 profit 原口径确实漏掉了这部分）')

print()
print('=' * 92)
print('E. 静态契约：main 必须转发这 4 列 + 2 个 TB scalar（缺一个就是运行期 KeyError）')
print('=' * 92)
src = open('td3prrbb/td3prrbb_main.py', encoding='utf-8').read()
for frag in ("'soc_start': m['soc_start']", "'profit_econ': m['profit_econ']",
             "'profit_econ': overall['profit_econ']", "'dsoc': overall['dsoc']",
             "test9/overall/profit_econ", "test9/overall/dsoc"):
    assert frag in src, ('main 未接线', frag)
print('  main 行字典与 TB scalar 接线齐全 ✓')
print('  注: 历史 run 的 df_test9.csv 仍是 11 列, 新列从下一次 test9 事件起出现。')

print()
print('TEST OK')
