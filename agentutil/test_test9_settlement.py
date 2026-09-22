#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test9（45 步 9 case 固定场景探针）的**存货结算**回归 —— 不需要 MATLAB。

问题
----
`profit` 是**现金流口径**：`profit = Σ[P_G·pr·dt − 490·max(−pbrule,0)·dt]`，
`P_G = ptur − P_C + P_D`。储存在电池里的电其成本已扣（充电时 `P_G` 减少、收入减少），
价值却未认回 ⇒ 净充电时 profit 低估、净放电时高估（把期初存量卖掉记成利润）。
test9 每次固定从 SOC = 5.0 开始，而策略有强烈抽干倾向 ⇒ **系统性**偏差。

结算口径（只修指标，不动训练路径）
--------------------------------
    profit_econ = profit + ΔSOC · SOC_SETTLE_PR        SOC_SETTLE_PR = hc.PR_REF = 580
参考价**必须外生常数**：若取"当时电价"，每步充放收益逐项抵消（ΔSOC·(pr_ref−pr) ≡ 0）
⇒ 电池对目标贡献恒为 0 ⇒ 套利激励归零。

九方法**全部**已接入（本测试逐个真跑）。`nine_case_test` 是纯 Python，
只有 `env_test` 侧依赖 MATLAB，故用**桩 env + 桩 agent / 真 SocTracker + 真 RuleBasedBESS**。
"""
import importlib
import io
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np                                                  # noqa: E402

A_H = np.array([0.35, 0.2, 0.35, 0.4, 0.35, 0.0])   # 真实日志样本 (6 维)
MAX_E = 6.0
ETA = 0.98
PR_REF = 580.0


class _StubAgent:
    def __init__(self, a, nb_states=6):
        self.a = np.asarray(a, dtype=float).ravel().copy()
        self.is_training = True
        self.nb_states = nb_states      # 部分项目 (PER) 会按 nb_states 分支
        self.nb_actions = self.a.size

    def select_action(self, obs, *a, **k):
        return self.a.copy()

    def reset(self, obs=None, *a, **k):        # 部分项目 (TD3PER 等) 会调 reset
        self.s_t = obs
        return None


class _StubEnv:
    """只实现 nine_case_test 需要的三个方法；PowerOutput 用确定性公式代替 MATLAB。"""

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
                0.0, False, False, {'P_F': 0.0, 'pbrule': P_C - P_D, 'P_G': P_G})

    # 少数项目走 TestStep 老接口
    TestStep = ScenarioStep


# 九方法: (显示名, myfunc 模块, 下层类型 L=RL(agent_L) / T=规则(tracker), rule_bess 模块)
PROJS = [
    ('PER',         'myfunc',                          'L', 'rbb.rbb_rule_bess'),
    ('PRRBB',       'prrbb.prrbb_myfunc',              'L', 'prrbb.prrbb_rule_bess'),
    ('PRLE',        'prle.prle_myfunc',                'L', 'prrbb.prrbb_rule_bess'),
    ('TD3PER',      'td3per.td3per_myfunc',            'L', 'rbb.rbb_rule_bess'),
    ('TD3PRRBB',    'td3prrbb.td3prrbb_myfunc',        'L', 'td3prrbb.td3prrbb_rule_bess'),
    ('TD3PRLE',     'td3prle.td3prle_myfunc',          'L', 'td3prle.td3prle_rule_bess'),
    ('TD3PERRB',    'td3perrb.td3perrb_myfunc',        'T', 'td3perrb.td3perrb_rule_bess'),
    ('TD3PRRLRB',   'td3prrlrb.td3prrlrb_myfunc',      'T', 'td3prrlrb.td3prrlrb_rule_bess'),
    ('TD3LEPRRLRB', 'td3leprrlrb.td3leprrlrb_myfunc',  'T', 'td3leprrlrb.td3leprrlrb_rule_bess'),
]
NEED = ('soc_start', 'soc_end', 'dsoc', 'profit_econ')


def run(proj, pb_cmd=0.0):
    name, mod_name, kind, rb_name = proj
    mf = importlib.import_module(mod_name)
    assert float(mf.SOC_SETTLE_PR) == PR_REF, (name, mf.SOC_SETTLE_PR)
    rb = importlib.import_module(rb_name).RuleBasedBESS()
    if kind == 'L':
        low = _StubAgent([pb_cmd])
    else:
        low = importlib.import_module(rb_name).SocTracker()
    return mf.nine_case_test(_StubAgent(A_H), low, _StubEnv(), rb, seed=42)


print('=' * 96)
print('A. 九方法全部接入：列齐全 + profit_econ 算术 + 望远镜一致性')
print('=' * 96)
print('  %-13s %-6s %-10s %-11s %s' % ('方法', '引用价', '列齐全', '算术max|偏差|', 'Σdsoc − overall dsoc'))
ok, failed = [], []
for proj in PROJS:
    name = proj[0]
    try:
        cm, ov = run(proj)
        assert len(cm) == 9, len(cm)
        for m in cm:
            for k in NEED:
                assert k in m, ('case 缺列', k)
        for k in NEED:
            assert k in ov, ('overall 缺列', k)
        dev = max(abs(m['profit_econ'] - (m['profit'] + m['dsoc'] * PR_REF)) for m in cm)
        dev = max(dev, abs(ov['profit_econ'] - (ov['profit'] + ov['dsoc'] * PR_REF)))
        tel = abs(sum(m['dsoc'] for m in cm) - ov['dsoc'])
        assert dev < 1e-9 and tel < 1e-9, (dev, tel)
        ok.append(name)
        print('  %-13s %-6.1f %-10s %-11.2e %.2e ✓' % (name, PR_REF, '✓', dev, tel))
    except Exception as e:
        failed.append((name, '%s: %s' % (type(e).__name__, e)))
        print('  %-13s **失败** %s: %s' % (name, type(e).__name__, str(e)[:60]))
print()
print('  真跑通过 %d/9 : %s' % (len(ok), ', '.join(ok)))
if failed:
    print('  未通过:')
    for n, e in failed:
        print('     %-13s %s' % (n, e))

print()
print('=' * 96)
print('B. 方向性（用 TD3PRRBB 作代表）：净放电 ⇒ profit 高估；净充电 ⇒ 低估')
print('=' * 96)
d = [p for p in PROJS if p[0] == 'TD3PRRBB'][0]
_, ov_d = run(d, -10.0)
_, ov_c = run(d, +10.0)
assert ov_d['dsoc'] < -1.0 and ov_d['profit_econ'] < ov_d['profit'], ov_d['dsoc']
assert ov_c['dsoc'] > 0.5 and ov_c['profit_econ'] > ov_c['profit'], ov_c['dsoc']
print('  强制放电: dsoc=%+.4f ⇒ profit %+.2f → profit_econ %+.2f (修正 %+.2f) ✓'
      % (ov_d['dsoc'], ov_d['profit'], ov_d['profit_econ'], ov_d['profit_econ'] - ov_d['profit']))
print('  强制充电: dsoc=%+.4f ⇒ profit %+.2f → profit_econ %+.2f (修正 %+.2f) ✓'
      % (ov_c['dsoc'], ov_c['profit'], ov_c['profit_econ'], ov_c['profit_econ'] - ov_c['profit']))

print()
print('=' * 96)
print('C. 静态契约：9 个 main 都必须转发这 4 列 + 3 个 TB scalar（缺一个就是运行期 KeyError）')
print('=' * 96)
MAINS = ['PER_main0.py', 'prrbb/prrbb_main.py', 'prle/prle_main.py', 'td3per/td3per_main.py',
         'td3prrbb/td3prrbb_main.py', 'td3prle/td3prle_main.py', 'td3perrb/td3perrb_main.py',
         'td3prrlrb/td3prrlrb_main.py', 'td3leprrlrb/td3leprrlrb_main.py']
FRAGS = ("'soc_start': m['soc_start'], 'soc_end': m['soc_end'],",
         "'dsoc': m['dsoc'], 'profit_econ': m['profit_econ'],",
         "'soc_start': overall['soc_start'], 'soc_end': overall['soc_end'],",
         "'dsoc': overall['dsoc'], 'profit_econ': overall['profit_econ'],",
         "writer.add_scalar('test9/overall/profit_econ', overall['profit_econ'], step)")
for p in MAINS:
    t = io.open(p, encoding='utf-8').read()
    for f in FRAGS:
        assert f in t, (p, '未接线', f)
print('  9 个 main 的 5 个接线片段全部就位 ✓')

print()
print('=' * 96)
print('D. 无裸数字 & 单一真源')
print('=' * 96)
MFS = ['myfunc.py'] + [p[1].replace('.', '/') + '.py' for p in PROJS[1:]]
n_hc = 0
for p in MFS:
    t = io.open(p, encoding='utf-8').read()
    assert 'SOC_SETTLE_PR = float(hc.PR_REF)' in t, (p, '参考价不是单一真源')
    assert 'SOC_SETTLE_PR = 580' not in t and '* 580' not in t, (p, '出现裸数字 580')
    n_hc += 1
print('  %d 个 myfunc 的 SOC_SETTLE_PR 均 = float(hc.PR_REF)，无裸数字 ✓' % n_hc)

print()
print('TEST OK' if not failed else 'TEST PARTIAL — 见上文未通过项')
sys.exit(1 if failed else 0)
