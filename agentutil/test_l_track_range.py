#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下层两区间 r2/c2 口径的设计不变量测试 —— 不需要 MATLAB。

被钉住的设计（`hsrl_soctarget`）：
    B  = [δ/(g·n_rem), δ/g]                      ← Soctrack 范围（摊还端 … 一步填平端）
    W  = [pbrulemin, pbrulemax]                  ← 平抑范围（rule 可行窗口）
    F  = B ∩ W                                   ← 参考集合；F≠∅ = 区间, F=∅ = 零宽单点 r*
    r2 = −1·|pb_cmd − pbrule|                    [MW]  越窗
    c2 = +1·dist(pb_applied, F) = +1·trackerr    [MW]  迟到/少做

不变量：
  1. 零成本集合 = F（区间）；带外 = 到最近端点的距离（线性）
  2. `trackerr(规则动作) ≡ 0`（含 F=∅ 的 best-effort 端点）
  3. `n_rem = 1` ⇒ B 退化为一点 ⇒ trackerr = 旧 slack/g（旧口径是本设计的特例）
  4. n_rem ↓ ⇒ 摊还端逐步收紧（deadline 时间表）
  5. r2 ≤ 0 ≤ c2，同系数 1、同单位 MW；`r2 + c2 = 0 ⇔ 不越窗 ∧ 落在 F 内`
  6. 平抑窗口不可行（逼吞吐）⇒ trackerr = 0（不可归责于下层）
  7. **F = ∅ ⇒ 参考集合是零宽单点**（无范围）：`ref_is_point=True`、`ref_lo == ref_hi == r*`
     且 r* 必是 W 的端点；只有 `pb == r*` 免费，两侧偏离都恰好 = |pb − r*|；
     越窗 2 MW 再夹回 r* 的"白费力"只吃 r2 ⇒ 严格劣于顶在 r*（唯一最优点）
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors='replace')
    except Exception:
        pass

import hsrl_soctarget as hst                            # noqa: E402

ETA_C = ETA_D = 0.98
DT = 1.0 / 60.0


def gain(delta):
    return ETA_C * DT if delta > 0 else DT / ETA_D


def soc_after(prev, pb, delta=None):
    """真正的 SOC 前向映射（充/放各用自己的 η）—— 与 applied_pb 严格互逆。"""
    dsoc = ETA_C * DT * max(float(pb), 0.0) - max(-float(pb), 0.0) * DT / ETA_D
    return prev + dsoc


def old_slack(target, prev, soc_next, lo, hi):
    """旧口径参考实现（仅用于第 3 条不变量对照）。"""
    delta = target - prev
    g = gain(delta)
    if lo > hi:
        return 0.0
    pb_best = float(np.clip(delta / g, lo, hi))
    d_best = abs(delta - g * pb_best)
    return max(0.0, abs(target - soc_next) - d_best)


rng = np.random.RandomState(7)
print('=' * 84)
print('1. 零成本集合 = F = B ∩ W；带外为到最近端点的距离')
print('=' * 84)
n_in = n_out = 0
for _ in range(4000):
    target = float(rng.uniform(0.5, 5.5))
    prev = float(rng.uniform(0.5, 5.5))
    lo = float(rng.uniform(-5.0, 2.0))
    hi = lo + float(rng.uniform(0.2, 8.0))
    n_rem = int(rng.randint(1, 6))
    delta = target - prev
    g = gain(delta)
    B_lo, B_hi = sorted([delta / g, delta / (g * n_rem)])
    F_lo, F_hi = max(B_lo, lo), min(B_hi, hi)
    if F_lo > F_hi:
        continue
    pb = float(rng.uniform(F_lo, F_hi))
    tr, info = hst.soc_track_range(target, prev, soc_after(prev, pb, delta), lo, hi, n_rem)
    assert abs(tr) < 1e-9, (tr, info)
    n_in += 1
    pb_out = F_hi + float(rng.uniform(0.0, 3.0))
    tr2, _ = hst.soc_track_range(target, prev, soc_after(prev, pb_out, delta), lo, hi, n_rem)
    assert abs(tr2 - (pb_out - F_hi)) < 1e-9, (tr2, pb_out, F_hi)
    n_out += 1
print('  区间内 {} 组 trackerr≡0；带外 {} 组 trackerr = |pb − 最近端点| 逐位成立'.format(n_in, n_out))

print('=' * 84)
print('2. trackerr(规则动作) ≡ 0（含 F=∅ 的 best-effort 端点）')
print('=' * 84)
n_rule = n_empty = 0
for _ in range(4000):
    target = float(rng.uniform(0.5, 5.5))
    prev = float(rng.uniform(0.5, 5.5))
    lo = float(rng.uniform(-5.0, 2.0))
    hi = lo + float(rng.uniform(0.2, 8.0))
    n_rem = int(rng.randint(1, 6))
    delta = target - prev
    g = gain(delta)
    pb_rule = float(np.clip(delta / g, lo, hi))
    tr, info = hst.soc_track_range(target, prev, soc_after(prev, pb_rule, delta), lo, hi, n_rem)
    assert abs(tr) < 1e-9, (tr, info)
    assert abs(info['pb_best'] - pb_rule) < 1e-12
    n_rule += 1
    n_empty += int(info['gap_empty'])
print('  {} 组: 规则动作 trackerr≡0（其中 F=∅ 的 {} 组走 best-effort 端点）'
      .format(n_rule, n_empty))

print('=' * 84)
print('3. n_rem = 1 ⇒ 退化为"一步填平"点（与旧 slack 口径同向时逐位一致）')
print('=' * 84)
mx_def = mx_old = mx_slack = mx_mwh = 0.0
n_old = 0
for _ in range(2000):
    target = float(rng.uniform(0.5, 5.5))
    prev = float(rng.uniform(0.5, 5.5))
    lo = float(rng.uniform(-5.0, 2.0))
    hi = lo + float(rng.uniform(0.2, 8.0))
    delta = target - prev
    g = gain(delta)
    # 只取与 δ 同向的施加功率（+0，即物理上"朝目标走"的样本）——反向样本在旧口径里
    # 用了"真实 SOC 残差之差"而新口径用"MW 距离"，两者定义不同（代码注释已说明）。
    pb = abs(float(rng.uniform(0.0, 8.0))) * (1.0 if delta > 0 else -1.0)
    pb = float(np.clip(pb, lo, hi)) if lo <= hi else 0.0
    sn = soc_after(prev, pb)
    tr, info = hst.soc_track_range(target, prev, sn, lo, hi, 1)
    pb_ref = 0.0 if lo > hi else float(np.clip(delta / g, lo, hi))
    mx_def = max(mx_def, abs(tr - abs(info['pb_applied'] - pb_ref)))
    if lo <= hi and lo <= delta / g <= hi:          # 未夹窗口 ⇒ 两口径定义重合
        mx_old = max(mx_old, abs(tr - old_slack(target, prev, sn, lo, hi) / g))
        # 诊断量 slack 必须等于旧口径的 MWh 量 (ΔSOC = g·pb ⇒ MWh = MW·g)
        mx_slack = max(mx_slack, abs(info['slack'] - old_slack(target, prev, sn, lo, hi)))
        n_old += 1
    mx_mwh = max(mx_mwh, abs(info['trackerr_mwh'] - tr * g))
print('  max|trackerr(n=1) − |pb_app − clip(δ/g,W)|| = {:.3e}'.format(mx_def))
print('  max|trackerr(n=1) − 旧 slack/g| = {:.3e}  (n={} 未夹窗口样本)'.format(mx_old, n_old))
print('  max|info[slack] − 旧 slack(MWh)| = {:.3e}  (同一批样本, 逐位)'.format(mx_slack))
print('  max|info[trackerr_mwh] − trackerr·g| = {:.3e}  (全样本; 换算方向已修)'.format(mx_mwh))
assert mx_def < 1e-9 and (n_old == 0 or mx_old < 1e-9)
assert mx_slack < 1e-9 and mx_mwh < 1e-12

print('=' * 84)
print('4. deadline 时间表: n_rem ↓ ⇒ 摊还端收紧（每步要求推进更多）')
print('=' * 84)
target, prev, lo, hi = 5.5, 3.0, -10.0, 10.0
delta = target - prev
for n_rem in (5, 4, 3, 2, 1):
    tr, info = hst.soc_track_range(target, prev, prev, lo, hi, n_rem)
    print('  n_rem={} : B=[{:7.2f}, {:7.2f}] MW  参考集合=[{:7.2f}, {:7.2f}]  完全不动 trackerr={:.2f} MW'
          .format(n_rem, info['band_lo'], info['band_hi'], info['ref_lo'], info['ref_hi'], tr))
    assert abs(info['band_hi'] - delta / gain(delta)) < 1e-12
    assert abs(info['band_lo'] - delta / (gain(delta) * n_rem)) < 1e-12

print('=' * 84)
print('5. 单位/符号/一致性: r2 ≤ 0 ≤ c2, 同系数 1 (MW)')
print('=' * 84)
assert hst.L_CLIP_COEF == hst.L_TRACK_COEF == 1.0
for clip in (0.0, 0.37, 2.5):
    assert hst.lower_reward(clip) == -clip
    assert hst.lower_reward(clip) <= 0
target, prev, lo, hi = 3.4, 3.0, 0.0, 2.0
pb_good = float(np.clip((target - prev) / gain(target - prev), lo, hi))
c2_good, info = hst.lower_cost_soc_track(target, prev, soc_after(prev, pb_good, target - prev),
                                         lo, hi, 3, return_info=True)
r2_good = hst.lower_reward(0.0)
assert abs(r2_good + c2_good) < 1e-12, (r2_good, c2_good)
print('  不越窗 ∧ 落在 F 内 ⇒ r2 + c2 = 0；系数同为 {:.0f}、单位同为 MW'
      .format(hst.L_CLIP_COEF))

print('=' * 84)
print('6. 平抑窗口不可行 ⇒ trackerr = 0（逼吞吐不可归责于下层）')
print('=' * 84)
tr, info = hst.soc_track_range(5.0, 3.0, 3.4, 5.0, -5.0, 3)
assert tr == 0.0 and info['infeasible_window']
print('  lo>hi 时 trackerr=0（infeasible_window={}）'.format(info['infeasible_window']))

print('=' * 84)
print('7. F=∅ ⇒ 参考集合是零宽单点（"只存在纯粹的一个竭力值"）：只有顶到它才免费')
print('=' * 84)
n_pt = n_samp = n_zero_nonref = n_idle = 0
mx_lin = 0.0
sum_idle = 0.0
for _ in range(6000):
    target = float(rng.uniform(0.5, 5.5))
    prev = float(rng.uniform(0.5, 5.5))
    lo = float(rng.uniform(-5.0, 2.0))
    hi = lo + float(rng.uniform(0.2, 8.0))
    n_rem = int(rng.randint(1, 6))
    _, info = hst.soc_track_range(target, prev, prev, lo, hi, n_rem)   # 只看几何
    if not info['gap_empty']:
        continue
    n_pt += 1
    r = info['ref_lo']
    assert info['ref_is_point'] and info['ref_hi'] == r, info          # 零宽 ⇒ 无范围
    assert abs(r - lo) < 1e-12 or abs(r - hi) < 1e-12, info            # r* 必是 W 的端点
    assert abs(info['pb_best'] - r) < 1e-12, (info['pb_best'], r)      # 规则动作 ≡ 竭力值
    # W 内扫点（含 r* 两侧）: trackerr 必须恰好 = |pb − r*|, 只有 pb == r* 才是 0
    for frac in (-1.0, -0.6, -0.3, -1e-9, 0.0, 1e-9, 0.3, 0.6, 1.0):
        pb = float(np.clip(r + frac * (hi - lo) * 0.3, lo, hi))
        tr, _ = hst.soc_track_range(target, prev, soc_after(prev, pb), lo, hi, n_rem)
        mx_lin = max(mx_lin, abs(tr - abs(pb - r)))
        n_samp += 1
        if abs(tr) < 1e-12 and abs(pb - r) > 1e-12:
            n_zero_nonref += 1                                          # 非 r* 却免费 ⇒ 有范围
    # "完全不动"（惰速）也要罚：这是 F=∅ 时单点语义的必然结果
    tr_idle, _ = hst.soc_track_range(target, prev, prev, lo, hi, n_rem)
    n_idle += 1
    sum_idle += tr_idle
assert n_zero_nonref == 0, n_zero_nonref
print('  {} 组 F=∅: ref_is_point 全 True, ref=[r*, r*] 且 r* 恒为 W 端点; '
      '{} 个扫描点 max|trackerr − |pb−r*|| = {:.3e}, 非 r* 却免费的点 = {}'
      .format(n_pt, n_samp, mx_lin, n_zero_nonref))
print('  同批状态"完全不动"平均 trackerr = {:.2f} MW（{} 组）—— 没尽力照罚'
      .format(sum_idle / max(1, n_idle), n_idle))

# 越窗白罚 (r2) + 单点免费 (c2) 合成的"帐篷": 唯一最优点就是竭力值 r*
n_tent = 0
for _ in range(6000):
    target = float(rng.uniform(0.5, 5.5))
    prev = float(rng.uniform(0.5, 5.5))
    lo = float(rng.uniform(-5.0, 2.0))
    hi = lo + float(rng.uniform(0.2, 8.0))
    n_rem = int(rng.randint(1, 6))
    _, info = hst.soc_track_range(target, prev, prev, lo, hi, n_rem)
    if not info['gap_empty']:
        continue
    r = info['ref_lo']
    # 顶在 r*: 指令不越窗 ⇒ clip_amount = 0, trackerr = 0 ⇒ 总代价 0
    assert abs(r - lo) < 1e-12 or abs(r - hi) < 1e-12
    assert abs(float(np.clip(r, lo, hi)) - r) < 1e-12
    # 朝窗外再推 2 MW: 施加值仍被夹回 r*(⇒ trackerr=0), 但指令越窗吃 r2 = −2
    over = lo - 2.0 if abs(r - lo) < 1e-12 else hi + 2.0
    pb_over = float(np.clip(over, lo, hi))
    assert abs(pb_over - r) < 1e-12
    tr_over, _ = hst.soc_track_range(target, prev, soc_after(prev, pb_over), lo, hi, n_rem)
    total_over = hst.lower_reward(abs(over - pb_over)) + hst.L_TRACK_COEF * tr_over
    assert abs(tr_over) < 1e-12 and total_over < -1.0, (tr_over, total_over)
    n_tent += 1
print('  {} 组: 顶在 r* ⇒ r2+c2 = 0；越窗 2 MW 再夹回 r* ⇒ r2+c2 ≤ −2（白费力，严格劣）'
      .format(n_tent))

print('=' * 84)
print('TEST OK')
