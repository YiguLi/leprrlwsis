# -*- coding: utf-8 -*-
r"""λ1 是否"真的在按对偶规则走"——**永久贴底的结构性证明**（只读，不需 MATLAB）。

判据（三件事分开）
------------------
1. **块级 c1 之和**（逐步日志按 `step//5` 聚合）—— H 层 `sum_c1` 的真值；
2. **`ema_c1_H`**（H 层对偶实际使用的量）与其预算 `d_block_H` 的比值 ——
   乘性对偶 `λ ← λ·exp(η(EMA/d − 1))` 只有在 `EMA/d > 1` 时才会把 λ 抬起来；
3. **λ 本身**：唯一值个数、是否等于 `lam_init` / `lam_min`。

⚠ `update_stats_H.cost_mean` **不是**块级 c1：那是 n-step 折叠后的量
（×`Σ_{k<n}γ^k`；n=5、γ=0.95 ⇒ 4.5244）。两者相差 4.6 倍是**正常的**，
见 `probe_cost_channel_scale.py` 的受控证明 —— 不要把那个倍数当 bug。

用法
----
    python agentutil/analyse_lam1_floor.py [output/TD3PERNOLOW-run57]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else 'output/TD3PERNOLOW-run57'
D_BLOCK = 15.0
ETA = 0.02
LOG = None
for _c in ('df_td3pernolow.csv', 'df_td3per.csv'):
    if os.path.exists(os.path.join(RUN, _c)):
        LOG = os.path.join(RUN, _c)
        break
if LOG is None:
    raise SystemExit('在 %s 下找不到逐步日志 df_*.csv' % RUN)

# ---- [1] 块级 c1（真值，来自逐步日志）----
acc = []
for ch in pd.read_csv(LOG, usecols=['step', 'c1', 'ema_c1_H', 'd_block_H', 'lam1'],
                      chunksize=200000, encoding='utf-8-sig', low_memory=False):
    acc.append(ch)
df = pd.concat(acc, ignore_index=True)
df['blk'] = df['step'] // 5
B = df.groupby('blk').agg(c1s=('c1', 'sum'), step=('step', 'min')).reset_index()

print('=' * 96)
print('λ1 贴底诊断: %s   (%s)' % (RUN, os.path.basename(LOG)))
print('=' * 96)
print('  df 行数 %d（step %d~%d）⇒ 块数 %d' % (len(df), df.step.min(), df.step.max(), len(B)))
print('\n[1] 块级 c1 之和（= H 层 sum_c1）')
print('    warmup 期（step≤1440） mean %8.4f  median %8.4f'
      % (B[B.step <= 1440].c1s.mean(), B[B.step <= 1440].c1s.median()))
post = B[B.step > 1440]
print('    策略期                 mean %8.4f  median %8.4f  p95 %8.4f  max %8.4f'
      % (post.c1s.mean(), post.c1s.median(), post.c1s.quantile(.95), post.c1s.max()))

print('\n[2] 对偶实际使用的量 ema_c1_H 与预算 d_block_H')
e = df.ema_c1_H.dropna()
print('    ema_c1_H: 末值 %8.4f  均值 %8.4f' % (e.iloc[-1], e.mean()))
print('    d_block_H 唯一值: %s' % np.unique(df.d_block_H.dropna().to_numpy()))
_ratio = e.iloc[-1] / D_BLOCK
print('    比值 ema/d: 末值 %.4f  ⇒ %s'
      % (_ratio,
         '**< 1 ⇒ 乘性对偶只会把 λ 往下压**' if _ratio < 1
         else '> 1 ⇒ 对偶会把 λ 抬起来'))

print('\n[3] λ1 本身')
lam = df.lam1.dropna()
print('    唯一值个数 %d，取值 %s' % (lam.nunique(), np.unique(lam.to_numpy())[:5]))
try:
    import hsrl_cost as hc
    kl = hc.lam_scale()
    print('    lam_scale=s_c/s_p=%.6g ⇒ λ_raw=λ1/kl=%.6g；物理区间 [%.4f, %.4f] 初值 %.4f'
          % (kl, lam.iloc[-1] / kl, hc.LAM_MIN, hc.LAM_MAX, hc.LAM_INIT))
    print('    判定: λ_raw == LAM_INIT ? %s   == LAM_MIN ? %s'
          % (abs(lam.iloc[-1] / kl - hc.LAM_INIT) < 1e-9,
             abs(lam.iloc[-1] / kl - hc.LAM_MIN) < 1e-9))
except Exception as ex:                        # noqa: BLE001
    print('    （未能 import hsrl_cost: %s）' % ex)

# ---- [4] 复算：若按规则走，λ 会怎样 ----
print('\n[4] 用 ema_c1_H 复算乘性对偶（λ 若真在按规则走会怎样）')
ema = None
mults = []
for x in e.to_numpy(float):
    ema = x if ema is None else (1 - ETA) * ema + ETA * x
    mults.append(float(np.exp(ETA * (ema / D_BLOCK - 1.0))))
mults = np.array(mults)
print('    乘子 exp(η(EMA/d−1)): 均值 %.6f  <1 的占比 %.1f%%  最大 %.6f'
      % (mults.mean(), 100 * (mults < 1).mean(), mults.max()))
print('    实测 λ1 末值 %.6g（恒定）' % lam.iloc[-1])
print('\n    ⇒ %s'
      % ('乘子均值 < 1 ⇒ 规则本身在压低 λ，贴底是结构性的（预算 d=15 相对实际成本过松）。'
         if mults.mean() < 1 else
         '乘子均值 ≥ 1 ⇒ **按规则 λ 应当上升**，但实测恒定 ⇒ λ 没有被对偶规则驱动，'
         '须查 `hc.h_settings()` 的注入点与 main 的 `agent_h.lam` 赋值。'))
