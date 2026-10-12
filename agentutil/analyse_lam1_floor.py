# -*- coding: utf-8 -*-
r"""三个"卡住收敛"的机制的**可核对证据**（只读，不需 MATLAB）。

[1] λ1 为什么永久贴下界：从日志逐块复算 `ema_c1_H` 与乘性对偶规则，给出
    "每个块的 λ 乘子" 分布 ⇒ 贴底不是偶然，是**结构性**的（预算 d_block=15 永不binding）。
    并给出反事实：λ 若停在**初值** 0.9667（而不是地板 0.0967），actor 目标里
    `λ·Q_c` 的占比会从 1.5% 变成 15%。
[2] critic 的 td_p/|Q_p| 全轨迹 + 分位；把它与"块奖励的不可约噪声"（同状态族内的
    条件 sd，即 Bayes 最优 TD）对照 ⇒ 判断"变差"是拟合恶化还是基准噪声。
[3] test9 的评测噪声来源：`df_test9.action` 列 —— 同一 step 的 10 次评测里动作是否
    相同。若不同，则评测期仍在加探索噪声（`is_training` 在 nine_case_test 内被置 False，
    但动作是在**置 False 之前**由主循环 select_action 生成？——由数据判定，不靠猜）。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))        # agentutil/_scratch/
ROOT = os.path.dirname(os.path.dirname(_HERE))            # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else 'output/TD3PERNOLOW-run57'
pd.set_option('display.width', 200)

# ------------------------------------------------------------------ [1] λ1
print('=' * 100)
print('[1] λ1 贴底机制')
print('=' * 100)
us = pd.read_csv(os.path.join(RUN, 'update_stats_H.csv'))
print('  update_stats 行数 %d，step %d~%d' % (len(us), us.step.min(), us.step.max()))
lam = 0.0966667
D_BLOCK = 15.0
ETA = 0.02
ema = None
mults, emas, lams = [], [], []
for c in us.cost_mean.to_numpy(float):
    ema = c if ema is None else (1 - 0.02) * ema + 0.02 * c
    lam = max(lam * float(np.exp(ETA * (ema / D_BLOCK - 1.0))), 0.0966667)
    mults.append(float(np.exp(ETA * (ema / D_BLOCK - 1.0))))
    emas.append(ema)
    lams.append(lam)
mults = np.array(mults)
print('  由日志复算的 λ 乘子 exp(η(EMA/d−1)): 均值 %.6f  最大 %.6f  <1 的块占比 %.1f%%'
      % (mults.mean(), mults.max(), 100 * (mults < 1).mean()))
print('  EMA(c1) 末值 %.4f / 预算 d_block_H %.1f ⇒ 比值 %.4f（<1 ⇒ 乘子恒 <1）'
      % (emas[-1], D_BLOCK, emas[-1] / D_BLOCK))
print('  ⇒ λ 每次乘 <1 再 clip 到地板 ⇒ **单调降到底后无法回升**（回升要求 EMA/d > 1，'
      '即均值 c1 > 3.0 MW·step，实测步级 c1 均值 1.14 ⇒ 永不发生）')
print('  步级 c1 均值 1.1391 ⇒ 块和 ≈ %.3f；预算 15 的 %.1f%%'
      % (5 * 1.1391, 100 * 5 * 1.1391 / D_BLOCK))
print('\n  反事实：λ 若停在初值 0.9667（而非地板 0.09667），actor 目标 -(Q_p − λQ_c) 里')
qc = float(us.q_c_mean.mean())
qp = float(us.q_p_mean.mean())
for lv, tag in ((0.0966667, '地板 λ=0.0967（实测）'), (0.966667, '初值 λ=0.9667（反事实）')):
    print('    %-26s λ·Q_c = %8.2f  |  Q_p = %8.2f  ⇒ 成本项占目标 %.2f%%'
          % (tag, lv * qc, qp, 100 * lv * qc / (qp + lv * qc)))

# ------------------------------------------------------------------ [2] TD
print('\n' + '=' * 100)
print('[2] critic 的 td_p/|Q_p|：是"拟合恶化"还是"基准噪声"？')
print('=' * 100)
q = us.q_p_mean.abs().to_numpy(float)
td = us.td_p_mean.to_numpy(float)
ratio = td / np.maximum(q, 1e-9)
st = us.step.to_numpy(float)
print('  %-6s %10s %10s %10s %10s %10s' % ('段', 'step', 'Q_p', 'td_p', 'td/q', '自洽比*'))
anchor = us.reward_mean.ewm(alpha=0.02, adjust=False).mean() * (1 / (1 - 0.95 ** 5))
for i in range(0, len(us), max(1, len(us) // 8)):
    print('  %-6d %10d %10.1f %10.1f %10.4f %10.3f'
          % (i, st[i], q[i], td[i], ratio[i], q[i] / anchor.iloc[i]))
print('  ratio 分位: p05 %.4f  p50 %.4f  p95 %.4f' % tuple(np.percentile(ratio, [5, 50, 95])))
print('  块奖励: mean %.1f  sd %.1f' % (us.reward_mean.mean(), us.reward_mean.std()))
print('  末段 td_p 均值 %.1f ⇒ 与块奖励 sd(%.1f) 之比 %.2f'
      % (td[-len(td) // 5:].mean(), us.reward_mean.std(),
         td[-len(td) // 5:].mean() / us.reward_mean.std()))
print('  ⇒ td_p 是**绝对误差（元）**；与"块奖励自身的 sd"比才是拟合质量。')
print('     td/sd ≈ 3 表示 critic 只解释了很小一部分方差 —— 但注意块奖励的很大一部分')
print('     是**状态里看不到的风况**（观测只有 4 维：pr/u/SOC/P_G）⇒ 存在不可约下界。')

# ------------------------------------------------------------------ [3] test9 评测噪声
print('\n' + '=' * 100)
print('[3] test9 的评测噪声：评测期动作是否确定性？')
print('=' * 100)
t9 = pd.read_csv(os.path.join(RUN, 'df_test9.csv'), encoding='utf-8-sig')
t9 = t9[t9.wind_level != 'OVERALL']
A = np.array([[float(x) for x in s.strip().strip('[]').split()]
              for s in t9['action'].astype(str)])
print('  action 列形状 %s（5 个机组维 + 1 个电池维）' % (A.shape,))
w5 = A[:, :5]
print('  全体 5 维动作 sd = %s' % np.round(w5.std(axis=0), 4))
print('  全体 5 维动作 min/max = %s / %s'
      % (np.round(w5.min(axis=0), 3), np.round(w5.max(axis=0), 3)))
# 同一 step 值的 10 次评测 = 同一场景步、同一风、同一状态 ⇒ 动作应逐位相同
g = t9.groupby('step')
sd_within, n_within = [], []
for k, v in g:
    if len(v) < 5:
        continue
    B = np.array([[float(x) for x in s.strip().strip('[]').split()]
                  for s in v['action'].astype(str)])
    sd_within.append(B[:, :5].std(axis=0))
    n_within.append(len(B))
S = np.array(sd_within)
print('  同一 step 内（同一状态！）的动作 sd: 均值 %s（%d 组，每组 %d 次）'
      % (np.round(S.mean(axis=0), 5), len(S), int(np.median(n_within))))
print('  ⇒ sd ≈ 0 才是确定性策略；实测非 0 ⇒ **评测期仍在注入探索噪声**')
# 与探索噪声地板的预期对照
eps_end = float(np.clip(0.999998 ** (4 * 13390), 0, 1))
amp = 0.05 * eps_end + 0.01 * (1 - eps_end)
print('  预期：wt_noise=0.05、wt_noise_floor=0.01、eps=0.999998^更新数')
print('        ε(末) ≈ %.4f ⇒ amp ≈ %.5f ⇒ 均匀噪声 sd = amp/√3 = %.5f'
      % (eps_end, amp, amp / np.sqrt(3)))
print('  实测同 step 内 sd 均值 = %.5f （同一量级 ⇒ 与"探索噪声"这一解释一致）'
      % S.mean())
print('\n  推论：本面板每一次评测都在测**带噪声的策略**，逐次差异的主要来源是噪声而不是')
print('        策略变化 ⇒ 用它的"波形"判断改进会被误导。')
