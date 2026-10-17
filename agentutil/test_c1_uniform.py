#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""临时验证（用后删除）：uniform 口径下 9 个带 c1 的项目必须给出**逐位相同**的 c1。"""
import inspect
import importlib
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)                                         # 相对数据/输出路径以仓库根为基准
sys.path.insert(0, ROOT)                               # 项目模块 (hsrl_*, myfunc, util, td3*/...)
sys.path.insert(0, _HERE)                              # agentutil 内互相 import

import hsrl_cost as hc  # noqa: E402

CANDS = [
    ('prrbb.prrbb_myfunc', 'Cal_Rew_PRRBB'),
    ('prle.prle_myfunc', 'Cal_Rew_PRRBB'),
    ('td3prrbb.td3prrbb_myfunc', 'Cal_Rew_PRRBB'),
    ('td3prle.td3prle_myfunc', 'Cal_Rew_PRRBB'),
    ('td3prrlrb.td3prrlrb_myfunc', 'Cal_Rew_RLRB'),
    ('td3perrb.td3perrb_myfunc', 'Cal_Rew_RB'),
    ('td3per.td3per_myfunc', 'Cal_Rew_RB'),
    ('td3leprrlrb.td3leprrlrb_myfunc', 'Cal_Rew_RLRB'),
]


def call_c1(fn, pf, u=9.0, pr=580.0, E=3.0, pb=0.0, gap=0.5):
    pg_old = 5.0
    obs = np.array([pr, u, E, pg_old])
    obs2 = np.array([pr, u, E, pg_old + pf])
    act = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.0, pb])
    info = {'pbrule': pb, 'pb_original': pb, 'pbrulemin': -3.0, 'pbrulemax': 3.0,
            'clip_amount': 0.0, 'unfeasible_gap': gap, 'is_infeasible': False,
            'inevitable_violation': 0.0, 'soc': E}
    kw = dict(penalty_NN_H=0.0, time=1.0 / 60, ptur=5.0)
    sig = inspect.signature(fn).parameters
    if 'pb_cmd' in sig:
        kw['pb_cmd'] = pb
    if 'l_clip_over' in sig:
        kw['l_clip_over'] = 0.0
    if 'n_rem' in sig:                      # 下层 c2 = trackerr [MW] 需要块内剩余步数
        kw['n_rem'] = 3
    out = fn(obs, obs2, act, info, **kw)
    # 返回约定: PRRBB 族 = (r1, c1, r2, c2, details); RLRB 族 = (r1, c1, details)
    return float(out[1])


fns = []
for mod, name in CANDS:
    m = importlib.import_module(mod)
    fns.append((mod.split('.')[0], getattr(m, name)))

print('=' * 100)
print('A) uniform 模式：c1 必须 = P_F + K_V·(P_F−3)⁺（K_V=100），且 8 个项目逐位相同')
print('=' * 100)
hc.set_mode('uniform')
print('  {}'.format(hc.summary()))
PFS = [0.0, 0.5, 2.9, 3.0, 3.0001, 3.5, 5.0, 8.0]
bad = []
for pf in PFS:
    exp = hc.fluct_cost(pf)
    vals = [(nm, call_c1(fn, pf)) for nm, fn in fns]
    spread = max(v for _, v in vals) - min(v for _, v in vals)
    ok = spread < 1e-12 and abs(vals[0][1] - exp) < 1e-12
    if not ok:
        bad.append((pf, vals, exp))
    print('  P_F={:7.4f}  期望 {:8.4f}  实测一致={}  偏差={:.2e}'.format(pf, exp, spread < 1e-12, spread))
assert not bad, bad

print()
print('B) 预算自洽：P_F≡3 每步 ⇒ 5 步之和 = 15 = D_BLOCK')
s5 = sum(hc.fluct_cost(3.0) for _ in range(5))
assert abs(s5 - hc.D_BLOCK) < 1e-12, (s5, hc.D_BLOCK)
print('  Σ₅ fluct_cost(3.0) = {:.3f}  D_BLOCK = {:.3f}  ✔'.format(s5, hc.D_BLOCK))
print('  单次典型越限(0.85 MW)的额外消耗 = {:.2f} ⇒ {:.0f}% 的整块额度'
      .format(hc.K_V * 0.85, 100 * hc.K_V * 0.85 / hc.D_BLOCK))

print()
print('C) legacy 模式：各项目回到自己的旧公式（应互不相同）')
hc.set_mode('legacy')
print('  {}'.format(hc.summary()))
leg = {}
for pf in (1.0, 3.5):
    leg[pf] = {nm: call_c1(fn, pf) for nm, fn in fns}
    print('  P_F={:4.1f}: '.format(pf) +
          '  '.join('{}={:.3f}'.format(nm, v) for nm, v in sorted(leg[pf].items())))
# 已知旧公式: W_GAP=50 的项目 (prrbb/prle/td3prrbb) vs W_GAP=10 的项目
assert abs(leg[1.0]['prrbb'] - (0.1 * 1.0 + 50 * 0.5)) < 1e-9
assert abs(leg[1.0]['td3perrb'] - (0.1 * 1.0 + 10 * 0.5)) < 1e-9
assert abs(leg[3.5]['td3perrb'] - (10 * 0.5 + 0.1 * 3.5 + 10 * 0.5)) < 1e-9
print('  ✔ legacy 分支保留完好（W_GAP=50 与 W_GAP=10 两族都对得上旧公式）')

hc.set_mode('uniform')
print()
print('跨项目 c1 不变量测试：全部通过')
