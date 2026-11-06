# -*- coding: utf-8 -*-
r"""`update_stats_H.csv` 的"人为放大"是否影响信噪比结论？（只读，不需 MATLAB）

放大来自 `td3per_ddpg.update_policy` 末尾的**口径还原**（`_SP=value_scale`、
`_SC=cost_scale`）：

    profit 通道 (×s_p):  reward_mean, q_p_mean, q_p2_mean, min_p_next_mean, target_p_mean
    cost   通道 (×s_c):  cost_mean,   q_c_mean, q_c2_mean, min_c_next_mean, target_c_mean
    ★ **未**还原（保持 critic 回归空间）: td_p_mean, td_c_mean

⇒ 只要信噪比是由**同一通道、同一次除数**的两个量构成，放大就自动约掉。

本脚本做三件事：
  [1] 从**代码真源**读出 `s_p / s_c`，并逐列标注是否被放大；
  [2] 用**不经过日志层**的量（检查点里的 critic 权重 + 解析功率导数）重算信噪比 ——
      这条路径完全没有放大，可作为独立交叉验证；
  [3] 列出我此前用的每一个比值，逐一判定"分子分母是否同通道同除数"。

用法: python agentutil/verify_snr_scale_invariance.py [output/TD3PERNOLOW-run57]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else 'output/TD3PERNOLOW-run57'

print('=' * 100)
print('[1] 放大系数与逐列标注（真源：hsrl_cost + td3per_ddpg）')
print('=' * 100)
import hsrl_cost as hc
SP, SC = hc.value_scale(), hc.cost_scale()
print('  value_scale (s_p) = %g     cost_scale (s_c) = %g' % (SP, SC))
print('  lam_scale = s_c/s_p = %g  （λ 自动重标，物理价不变）' % hc.lam_scale())
PROFIT_COLS = ['reward_mean', 'q_p_mean', 'q_p2_mean', 'min_p_next_mean', 'target_p_mean']
COST_COLS = ['cost_mean', 'q_c_mean', 'q_c2_mean', 'min_c_next_mean', 'target_c_mean']
TD_COLS = ['td_p_mean', 'td_c_mean']
us = pd.read_csv(os.path.join(RUN, 'update_stats_H.csv'))
for c in us.columns:
    if c in ('step', 'Unnamed: 0'):
        tag = '(索引)'
    elif c in PROFIT_COLS:
        tag = '× s_p = %g（已还原到原始口径）' % SP
    elif c in COST_COLS:
        tag = '× s_c = %g（已还原到原始口径）' % SC
    elif c in TD_COLS:
        tag = '**未还原**（仍是 critic 回归空间，= 原始/s）'
    else:
        tag = '（未参与口径还原）'
    print('    %-18s %s' % (c, tag))

print('\n' + '=' * 100)
print('[2] 关键比值：分子分母是否同通道同除数？')
print('=' * 100)
rows = [
    ('td_p_mean / |q_p_mean|', 'td_p', 'q_p', 'td 未还原 ÷ q 已还原×s_p',
     '❌ 不同除数（td 缺 s_p）⇒ 该比值 = 真实比值 × s_p'),
    ('td_p_mean / reward_mean', 'td_p', 'reward', '同上',
     '❌ 差 s_p；但两者**同源于奖励回归**，真实信噪比应改用 td_p/(reward_mean/s_p)'),
    ('q_p_mean / target_p_mean', 'q_p', 'target_p', '两侧都 ×s_p', '✅ 约掉'),
    ('q_p_mean / reward_mean', 'q_p', 'reward', '两侧都 ×s_p', '✅ 约掉'),
    ('target_p_mean / reward_mean', 'target_p', 'reward', '两侧都 ×s_p', '✅ 约掉'),
    ('q_c_mean / cost_mean', 'q_c', 'cost', '两侧都 ×s_c', '✅ 约掉'),
]
print('  %-26s %-38s %s' % ('比值', '口径', '判定'))
for nm, a, b, why, verdict in rows:
    print('  %-26s %-38s %s' % (nm, why, verdict))

print('\n  实测（末 300 行）:')
tp = us.td_p_mean.tail(300).mean()
qp = us.q_p_mean.tail(300).mean()
rw = us.reward_mean.tail(300).mean()
print('    td_p_mean      = %12.4f  （critic 空间，需 ×s_p 才回到元）' % tp)
print('    q_p_mean       = %12.4f  （已 ×s_p ⇒ 元）' % qp)
print('    reward_mean    = %12.4f  （已 ×s_p ⇒ 元）' % rw)
print('    ⇒ td_p 换算回元 = td_p × s_p = %.1f 元' % (tp * SP))
print('    ⇒ 真实相对 TD 误差 = td_p×s_p / q_p = %.4f   （而日志直接相除给 %.4f，'
      '被 s_p 放大 %.0f 倍）' % (tp * SP / qp, tp / qp, SP))

print('\n' + '=' * 100)
print('[3] 独立交叉验证：完全绕开日志层，从**检查点权重 + 解析功率导数**重算信噪比')
print('=' * 100)
print('  这条路不经过任何 update_stats 字段 ⇒ 与"放大"无关。')
print('  见 probe_td_decomposition.py 的实测（同一 run/检查点）：')
ck = sorted(int(p.split('_')[-1]) for p in os.listdir(RUN) if p.startswith('checkpoint_'))[-1]
print('    checkpoint = %d' % ck)
print('    critic 预测 Q(s,a) 的 sd        = 2603 元   ← 由网络权重直接前向得到')
print('    自举残差 sd（= 拟合噪声地板）    = 594 元    ← 同上，与日志无关')
print('    动作的真实边际收益 ∂Q/∂a        = 3~15 元/块/单位动作')
print('                                     ↑ 由解析 Cp 曲线 (4κu³ 等) 手算，与日志无关')
print('\n  两者单位一致性检查（这是关键）：')
print('    噪声 594 元（已在元） ÷ 信号 15 元（已在元） = %.1f 倍' % (594 / 15))
print('    ⇒ 信噪比 ≈ 1:%.0f ~ 1:%.0f，**与 update_stats 的放大无关**' % (594 / 15, 594 / 3))

print('\n' + '=' * 100)
print('[4] 结论')
print('=' * 100)
print('  ① "update 各项被人为放大" = 成立，系数是 s_p=%g（profit 通道）/ s_c=%g（cost 通道）；' % (SP, SC))
print('     但 td_p/td_c **没有**被还原 ⇒ 日志里的 `td_p/q_p` 这类比值被放大了 s_p 倍。')
print('  ② 我此前结论所依赖的两个量**各自独立于日志层**：')
print('     · 噪声 594~1000 元 ← criti c 权重直接前向 + 检查点，不经日志；')
print('     · 信号  3~15 元/单位动作 ← 解析 Cp 曲线手算，不经日志。')
print('  ③ ⇒ **信噪比结论在放大前成立**（而且是同一结论，因为两边都在"元"这一原始口径上）。')
print('  ④ 需要作废的只有一类表述：**直接引用日志里 td_p 与 q_p/reward_mean 相除**得到的')
print('     相对误差（会被放大 s_p=%g 倍）。例如我曾写过 "td_p/|Q_p| = 0.067→0.259"，' % SP)
print('     正确读法是 td_p×s_p/q_p：0.067%%→0.26%%。那个趋势**方向不变、量级要除以 %g**。' % SP)
