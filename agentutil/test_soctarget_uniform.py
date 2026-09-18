#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下层 SOC 目标语义的**跨项目逐位等价**测试 —— 不需要 MATLAB。

三个"下层 RL + SOC 目标"项目（`td3per` / `td3prrbb` / `prle`）的下层口径应完全一致：

    a6 → soc_target = clip(3 + 5·a6, 0.5, 5.5)     (块级固定)
    r2 = −1 · clip_amount
    c2 = 10 · slack,  slack = max(0, |target−SOC'| − d_best)   (能力内追踪失败量)
    块末追踪偏好 = 10·max(0, |target−SOC_end| − 0.1)  (进 r1)

差异只允许出现在**上层**（td3per 非分段 / td3prrbb·prle 风价分段）与观测约定。
另断言 `td3per` 里的**本地实现**与单一真源 `hsrl_soctarget` 逐位一致（防漂移）。
"""
import importlib
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)                                         # 相对数据/输出路径以仓库根为基准
sys.path.insert(0, ROOT)                               # 项目模块 (hsrl_*, myfunc, util, td3*/...)
sys.path.insert(0, _HERE)                              # agentutil 内互相 import
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors='replace')
    except Exception:
        pass

import hsrl_soctarget as hst                            # noqa: E402

SOC_PROJECTS = {
    'td3per': ('td3per.td3per_myfunc', 'Cal_Rew_RB', 2),      # 蒸馏观测: 目标在第 3 维
    'td3prrbb': ('td3prrbb.td3prrbb_myfunc', 'Cal_Rew_PRRBB', 2),
    'prle': ('prle.prle_myfunc', 'Cal_Rew_PRRBB', 4),         # baseline 观测: 目标在末维
}

print('=' * 84)
print('1. 单一真源口径')
print('=' * 84)
assert hst.L_CLIP_COEF == 1.0 and hst.L_SOC_COEF == 10.0
assert hst.L_D_BLOCK == 0.2 and hst.L_D_BLOCK_MIN == 0.025
assert hst.L_COST_LIMIT == 0.1 and hst.L_EMA_LAM_LR == 1.0
# 与 ÷100 之前等价: 物理容忍度 0.02 MWh、对偶速度 10
assert abs(hst.L_D_BLOCK / hst.L_SOC_COEF - 20.0 / 1000.0) < 1e-12
assert abs(hst.L_EMA_LAM_LR * hst.L_SOC_COEF - 0.01 * 1000.0) < 1e-12
for a6, exp in ((-0.5, 0.5), (0.0, 3.0), (0.2, 4.0), (0.5, 5.5), (2.0, 5.5)):
    assert abs(hst.soc_target_from_signal(a6) - exp) < 1e-12, (a6, exp)
print('  a6→target 映射、L_CLIP_COEF=1 / L_SOC_COEF=10 / d_block_L=0.2 / η2=1.0 全部正确')
print('  d/c2 = {:.4g} MWh (与 ÷100 前一致) ; η2·c2 = {:g} (与 ÷100 前一致)'
      .format(hst.L_D_BLOCK / hst.L_SOC_COEF, hst.L_EMA_LAM_LR * hst.L_SOC_COEF))

print('=' * 84)
print('2. 三项目下层 r2/c2 逐位相同 (同一批输入)')
print('=' * 84)
mods = {p: importlib.import_module(m) for p, (m, _f, _i) in SOC_PROJECTS.items()}


def project_r2_c2(p, obs, obs2, act, rbb, ptur=20.0):
    """按**该项目自己的训练循环**算出 (r2, c2, soc_target, slack)。

    td3per 的 r2/c2 在 main 里算 (Cal_Rew_RB 只返回 r1/c1/details),
    td3prrbb / prle 的 r2/c2 在 myfunc 里算 —— 两种结构都必须给出同一个数。
    """
    mod, fn_name, _ = SOC_PROJECTS[p]
    fn = getattr(mods[p], fn_name)
    if p == 'td3per':
        _r1, _c1, d = fn(obs, obs2, act, rbb, penalty_NN_H=0.0, time=1.0 / 60, ptur=ptur)
        r2 = hst.lower_reward(rbb['clip_amount'])
        c2, info = hst.lower_cost_soc_track(
            d['soc_target'], float(obs[2]), float(obs2[2]),
            rbb['pbrulemin'], rbb['pbrulemax'], return_info=True)
        slack = 0.0 if info['infeasible_window'] else \
            max(0.0, info['d_next'] - info['d_best'])
        return r2, c2, d['soc_target'], slack
    _r1, _c1, r2, c2, d = fn(obs, obs2, act, rbb, penalty_NN_H=0.0,
                             time=1.0 / 60, ptur=ptur)
    assert abs(d['soc_slack'] - c2 / hst.L_SOC_COEF) < 1e-12, (p, d['soc_slack'], c2)
    return r2, c2, d['soc_target'], d['soc_slack']


rng = np.random.RandomState(0)
n_same = 0
for k in range(60):
    a6 = float(rng.uniform(-0.5, 0.5))
    soc_prev = float(rng.uniform(0.5, 5.5))
    delta = float(rng.uniform(-1.6, 1.6))
    soc_next = float(np.clip(soc_prev + delta, 0.5, 5.5))
    lo = float(rng.uniform(0.0, 5.0))
    hi = lo + float(rng.uniform(-0.5, 5.0))       # 允许不可行窗口 (lo > hi)
    clip = float(rng.uniform(0.0, 2.0))
    obs = np.array([float(rng.uniform(200, 900)), float(rng.uniform(3, 20)), soc_prev, 0.0])
    obs2 = np.array([obs[0], obs[1], soc_next, 0.0])
    rbb = dict(pbrule=float(rng.uniform(-3, 3)), unfeasible_gap=0.0, clip_amount=clip,
               pbrulemin=lo, pbrulemax=hi, is_infeasible=bool(lo > hi),
               inevitable_violation=False, soc=soc_next)
    act = np.array([0.3] * 5 + [a6, 1.0])
    vals = {}
    for p in SOC_PROJECTS:
        r2, c2, tgt, slack = project_r2_c2(p, obs, obs2, act, rbb)
        vals[p] = (r2, c2, tgt, slack)
        assert abs(r2 - hst.lower_reward(clip)) < 1e-12, (p, r2, clip)
        assert abs(c2 - hst.L_SOC_COEF * slack) < 1e-12, (p, c2)
    ref = vals['td3per']
    for p, v in vals.items():
        assert v == ref, (k, p, v, ref)
    n_same += 1
print('  {} 组随机输入 (含不可行窗口): td3per / td3prrbb / prle 的 (r2, c2, target, slack) 逐位相同'
      .format(n_same))

print('=' * 84)
print('3. 零成本动作 = 规则 SocTracker 的动作')
print('=' * 84)
for tag, (target, prev, lo, hi) in {
        'δ>0 可一步到位': (3.1, 3.0, 0.0, 10.0),
        'δ<0 放电': (2.9, 3.0, -10.0, 0.0),
        '窗口紧 (只能走 0.05 MWh)': (3.5, 3.0, 0.0, 3.0),
        '不可行窗口': (3.5, 3.0, 5.0, -5.0)}.items():
    gain = 0.98 * (1.0 / 60.0) if target > prev else (1.0 / 60.0) / 0.98
    pb_best = 0.0 if lo > hi else float(np.clip((target - prev) / gain, lo, hi))
    soc_rule = prev + gain * pb_best
    slack, info = hst.soc_track_slack(target, prev, soc_rule, lo, hi)
    assert abs(slack) < 1e-12, (tag, slack, info)
    slack2, info2 = hst.soc_track_slack(target, prev, prev, lo, hi)
    print('  {:<24s} 规则动作 slack={:.3g} | 完全不动 slack={:.4f}'
          .format(tag, slack, slack2))

print('=' * 84)
print('4. td3per 的本地实现 == 单一真源 (防漂移)')
print('=' * 84)
tf = mods['td3per']
grid = [(float(a), float(p), float(nn), 0.0, 5.0)
        for a in np.linspace(0.5, 5.5, 7) for p in np.linspace(0.5, 5.5, 7)
        for nn in np.linspace(0.5, 5.5, 7)]
for t, p, nn, lo, hi in grid:
    s1, i1 = tf.soc_track_slack(t, p, nn, lo, hi)
    s2, i2 = hst.soc_track_slack(t, p, nn, lo, hi)
    assert s1 == s2 and i1 == i2
    assert tf.lower_reward(0.37) == hst.lower_reward(0.37)
    assert tf.lower_cost_soc_track(t, p, nn, lo, hi) == hst.lower_cost_soc_track(t, p, nn, lo, hi)
    assert tf.soc_target_from_signal(0.13) == hst.soc_target_from_signal(0.13)
    assert tf.gap_penalty(t, nn) == hst.gap_penalty(t, nn)
print('  {} 组网格: soc_track_slack / lower_reward / lower_cost / target 映射 / gap_penalty 逐位一致'
      .format(len(grid)))

print('=' * 84)
print('5. 下层观测里的目标维')
print('=' * 84)
for p, (mod, _f, idx) in SOC_PROJECTS.items():
    m = importlib.import_module(mod)
    fn = getattr(m, 'build_L_obs_distilled')
    obsL = fn(np.array([580., 10., 3.0, 0.]), 10.0, np.array([0.3] * 5 + [0.2]),
              dict(pbrulemin=0.0, pbrulemax=3.0))
    assert abs(float(obsL[idx]) - 0.4) < 1e-12, (p, obsL, idx)
    print('  {:<9s} 目标维在 #{} = {:.3f}  (= soc_target/10 = 0.4)'.format(p, idx, obsL[idx]))

print('=' * 84)
print('TEST OK')
