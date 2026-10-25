#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""aifflor warmup 新口径契约：一半 LHS + 一半五维统一随机升降噪声。

判据（纯只读，不需要 MATLAB / FLORIS / torch）：
  1. `WarmupSchedule(wt_sched='unified')` 的块序：第 0 块 = `start`；奇数块 = 统一噪声
     （对整场偏移量 o 做 ±step 游走，五维同时取 `prior_i + o`）⇒ **五维同向、形状不变**；
     偶数块 = LHS 全局覆盖（[LHS_LO, LHS_HI]，五维相互独立）。
  2. 50/50：1440 块里 1 个起点块 + 719 个 LHS + 720 个统一噪声（块序号奇偶，精确）。
  3. 统一臂的随机步**是共享的**：相邻奇数块的动作差向量五个分量逐位相同（浮点级），
     且偏移量始终落在 `[lo−min(prior), hi−max(prior)]` ⇒ 五维恒在 [lo,hi]。
  4. 统一臂两端都能走到（整场升到顶 / 降到低），且不被中间的 LHS 块打断；
     同 seed 可复现；不消耗全局 numpy RNG 流。
  5. CLI：共享默认仍是 `lhs`；只有 `aifflor_main.py` 显式传 `wt_sched_default='unified'`；
     `build_from_args` 正确接线；信号维行为不受影响。
"""
import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)                                         # 相对数据/输出路径以仓库根为基准
sys.path.insert(0, ROOT)                               # 项目模块
sys.path.insert(0, _HERE)                              # agentutil 内互相 import

import hsrl_warmup as hw                                 # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors='replace')
    except Exception:
        pass

N_BLK = int(hw.WARMUP_STEPS) // 5                        # 7200 步 = 1440 块
PRIOR = np.asarray(hw.WT_PRIOR, dtype=float)
STEP = float(hw.WT_STEP)
LO, HI = float(hw.WT_LO), float(hw.WT_HI)
O_LO, O_HI = LO - float(PRIOR.min()), HI - float(PRIOR.max())


def _run(seed=7, n=N_BLK, price=None, **kw):
    s = hw.WarmupSchedule(5, 6, seed=seed, wt_sched='unified',
                          lhs_n_blocks=n, **kw)
    return s, np.array([s.action(price) for _ in range(n)], dtype=float)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------- 1) 块序与 50/50
print('--- 1) 块序与 50/50 ---')
sched, A = _run()
check(A.shape == (N_BLK, 6), A.shape)
check(np.allclose(A[0, :5], hw.WT_START) and A[0, 5] == 0.0,
      '第 0 块必须仍是起点 start / 信号 0: %r' % (A[0],))
n_odd = len(range(1, N_BLK, 2))          # 统一噪声块
n_even = len(range(2, N_BLK, 2))         # LHS 块
check(n_odd == 720 and n_even == 719, (n_odd, n_even))
check(sched._lhs_i == n_even,
      'LHS 调用次数应 = 偶数块数 %d（实测 %d）' % (n_even, sched._lhs_i))
check(sched._blk == N_BLK, sched._blk)
print('1440 块: 起点 1 + 统一噪声 %d + LHS %d' % (n_odd, n_even))

# ---------------------------------------------------------------- 2) 统一臂
print('--- 2) 五维统一随机升降 ---')
odd = A[1::2, :5]
# ① 形状不变：每块五个维相对 prior 的偏移**逐位相同**
offs = odd - PRIOR
check(np.max(np.ptp(offs, axis=1)) < 1e-12,
      '统一臂必须整场只用一个偏移 o（max ptp=%.3e）' % np.max(np.ptp(offs, axis=1)))
o = offs[:, 0]
check(o.min() >= O_LO - 1e-12 and o.max() <= O_HI + 1e-12,
      '偏移越界: [%r, %r] vs [%r, %r]' % (o.min(), o.max(), O_LO, O_HI))
check(odd.min() >= LO - 1e-12 and odd.max() <= HI + 1e-12,
      '统一臂动作必须始终在 [%.2f, %.2f]' % (LO, HI))
# ② 每步共享同一个 δ：相邻奇数块的差向量五分量相同
d = np.diff(odd, axis=0)
check(np.max(np.ptp(d, axis=1)) < 1e-12,
      '相邻统一块的步长必须五维相同（max ptp=%.3e）' % np.max(np.ptp(d, axis=1)))
step_i = d[:, 0]
check(np.all(np.abs(step_i) <= STEP + 1e-12),
      '单块偏移变化不得超过 step: %r' % np.abs(step_i).max())
check(np.all(np.isfinite(A)), '动作含 NaN/Inf')
# ③ 两端都要真正走到（整场升到顶/降到低），且不被 LHS 块打断
check(o.min() <= O_LO + 1e-9, '统一臂必须能降到偏移下界 (实测 min=%.4f)' % o.min())
check(o.max() >= O_HI - 1e-9, '统一臂必须能升到偏移上界 (实测 max=%.4f)' % o.max())
n_sat_lo = int(np.sum(np.isclose(o, O_LO)))
n_sat_hi = int(np.sum(np.isclose(o, O_HI)))
print('统一臂 720 块: o∈[%.4f, %.4f] (界 [%.4f, %.4f]) | 步长共享 ✓ | '
      '贴下界 %d 块 / 贴上界 %d 块' % (o.min(), o.max(), O_LO, O_HI, n_sat_lo, n_sat_hi))
# ④ 第 1 块必须 = prior ± step（从先验结构出发）
_d0 = float(A[1, 0] - PRIOR[0])
check(abs(abs(_d0) - STEP) < 1e-12, '第 1 块的共享步长应为 ±%.2f（实测 %r）' % (STEP, _d0))
check(np.max(np.abs(A[1, :5] - (PRIOR + _d0))) < 1e-12,
      '第 1 块应 = prior + 共享 δ: %r' % (A[1, :5],))

# ---------------------------------------------------------------- 3) LHS 臂
print('--- 3) 另一半仍是 LHS 全局覆盖 ---')
even = A[2::2, :5]
check(even.min() >= hw.LHS_LO - 1e-12 and even.max() <= hw.LHS_HI + 1e-12,
      'LHS 臂必须覆盖 [%.1f, %.1f]' % (hw.LHS_LO, hw.LHS_HI))
check(even.min() < 0.01 and even.max() > 0.49,
      'LHS 臂必须铺满全局盒（实测 [%.4f, %.4f]）' % (even.min(), even.max()))
cc = np.corrcoef(even.T)
off_r = cc[~np.eye(5, dtype=bool)]
check(np.abs(off_r).max() < 0.35,
      'LHS 臂五维应互相独立（max|r|=%.3f）' % np.abs(off_r).max())
print('LHS 臂 %d 块: 范围 [%.4f, %.4f] | 维间 max|r|=%.3f'
      % (even.shape[0], even.min(), even.max(), np.abs(off_r).max()))

# ---------------------------------------------------------------- 4) 可复现 / RNG 隔离
print('--- 4) 可复现性与 RNG 流隔离 ---')
_, A2 = _run(seed=7)
check(np.array_equal(A, A2), '同 seed 必须逐位可复现')
st0 = np.random.get_state()
_ = _run(seed=11, n=50)[1]
st1 = np.random.get_state()
check(st0[0] == st1[0] and np.array_equal(st0[1], st1[1]) and st0[2:] == st1[2:],
      'WarmupSchedule 不得消耗全局 numpy RNG 流')
# 信号维：fixed ⇒ 恒 0；price_walk ⇒ 300 上调 / 980 下调
_, A_fix = _run(seed=3, n=40, sig_mode='fixed')
check(np.all(A_fix[:, 5] == 0.0), 'sig_mode=fixed 时信号维应恒 0')
s_p = hw.WarmupSchedule(5, 6, seed=3, wt_sched='unified')
sig_up = np.array([s_p.action(300.0)[5] for _ in range(40)])
sig_dn = np.array([s_p.action(980.0)[5] for _ in range(80)])
check(np.all(np.diff(sig_up) >= -1e-12) and sig_up[-1] > 0.0,
      '低谷 300 应使信号上调: %r' % sig_up[-5:])
check(np.all(np.diff(sig_dn) <= 1e-12) and sig_dn[-1] < 0.0,
      '高峰 980 应使信号下调: %r' % sig_dn[-5:])
check(sig_up.max() <= hw.SIG_HI + 1e-12 and sig_dn.min() >= hw.SIG_LO - 1e-12,
      '信号维越界: [%r, %r]' % (sig_dn.min(), sig_up.max()))
print('同 seed 逐位相同；全局 RNG 状态不变；信号维方向/界 ✓')

# ---------------------------------------------------------------- 5) CLI 与接线
print('--- 5) CLI 默认与 aifflor 接线 ---')
ap = argparse.ArgumentParser()
hw.add_warmup_args(ap)
check(ap.parse_args([]).warmup_wt_sched == 'lhs',
      '共享默认必须仍是 lhs（其余 8 个项目行为不变）')
ap2 = argparse.ArgumentParser()
hw.add_warmup_args(ap2, wt_sched_default='unified')
d2 = ap2.parse_args([])
check(d2.warmup_wt_sched == 'unified', d2.warmup_wt_sched)
check(ap2.parse_args(['--warmup_wt_sched', 'lhs']).warmup_wt_sched == 'lhs',
      '显式 --warmup_wt_sched 必须能覆盖默认')


class _HStub:
    nb_turbine, nb_actions = 5, 6


ns = argparse.Namespace(**vars(d2), seed=7, rbb_rated_power=0.5, warmup=7200)
sched_b, rule_b, stats_b = hw.build_from_args(ns, _HStub(), None)
check(sched_b.wt_sched == 'unified' and sched_b.action(300.0).shape == (6,),
      'build_from_args 未接线到 unified')
check(sched_b._lhs_n == 7200 // 5, sched_b._lhs_n)
check(stats_b is None, '无 L agent ⇒ 不应有填入统计')
check(hw.WarmupSchedule(5, None, seed=1, wt_sched='unified').action().shape == (5,),
      'nb_actions=None ⇒ 只返回 5 维')
check(hw.WarmupSchedule(5, 6, wt_sched='unified', enabled=False).action() is None,
      'enabled=False ⇒ 回退旧随机臂 (None)')

# 静态接线：只有 aifflor_main 传 wt_sched_default='unified'
mains = []
for p in glob.glob(os.path.join(ROOT, '**', '*.py'), recursive=True):
    rel = os.path.relpath(p, ROOT).replace('\\', '/')
    if rel.startswith(('agentutil/', '.', 'libraries/', 'output/', 'runs/', '未命名/')):
        continue
    if rel == 'hsrl_warmup.py':
        continue
    try:
        txt = open(p, encoding='utf-8', errors='replace').read()
    except OSError:
        continue
    if 'add_warmup_args(' in txt:
        mains.append((rel, txt))
check(len(mains) >= 9, '调用 add_warmup_args 的 main 应 >= 9（实测 %d）' % len(mains))
unified_mains = [rel for rel, txt in mains
                 if "wt_sched_default='unified'" in txt]
check(unified_mains == ['aifflor/aifflor_main.py'],
      '应只有 aifflor_main 改默认（实测 %r）' % unified_mains)
for rel, txt in mains:
    if rel == 'aifflor/aifflor_main.py':
        continue
    txt_wo_doc = txt
    check('wt_sched_default' not in txt_wo_doc,
          '%s 不应传 wt_sched_default（共享默认保持 lhs）' % rel)
print('共享默认 lhs；仅 aifflor_main 传 unified（共扫描 %d 个入口）✓' % len(mains))

print('TEST OK')
