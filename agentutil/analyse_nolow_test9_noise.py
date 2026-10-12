# -*- coding: utf-8 -*-
r"""9 用例面板的**评测噪声 vs 策略改进**：把"同一 step 的重复评测"当成配对重复。

用法
----
    python agentutil/analyse_nolow_test9_noise.py [run ...] [--base output]

背景（本脚本要证的事）
--------------------
`nine_case_test(seed=42)` 的风/价序列每次调用都**逐位相同**（`RandomState(42)`），
且 `ScenarioReset` 是确定性的 ⇒ **若策略是确定性的，同一 step 处的重复评测应逐位相同**。
实测 run57：同一 step 的 10 次重复之间，45 步 turbine_profit 之和的 sd ≈ 570
（= 逐步 ±12.7 元、相对 3.5%）⇒ **评测带有策略噪声**（`nine_case_test` 期间
`agent.select_action()` 仍加 `explore_floor` 量级的探索噪声），因此：

  * test9 的"波形"（±10% 的逐检查点摆动）主要是**评测噪声**，不是策略变化；
  * 用它做检查点挑选/早停会把噪声当成改进；
  * 要测 <5% 的策略差别必须做**配对**（同 step 差分）或**确定性复算**（不带噪声）。

输出
----
[A] 每个 step 组的组内 sd（= 纯评测噪声，不随训练预算下降）
[B] 逐检查点均值 ± 噪声标准误 ⇒ 真改进 vs 噪声的分辨力
[C] 用组内重复做"配对差分"的趋势检验（比 raw 曲线灵敏 sqrt(n_rep) 倍）
"""
import argparse
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def find_run(spec, base='output'):
    if os.path.isdir(spec):
        return spec
    cand = os.path.join(base, spec)
    if os.path.isdir(cand):
        return cand
    hits = sorted([os.path.join(base, d) for d in os.listdir(base)
                   if os.path.isdir(os.path.join(base, d)) and spec in d],
                  key=lambda p: -os.path.getmtime(p))
    if not hits:
        raise SystemExit('找不到 run: %s' % spec)
    return hits[0]


def load(d):
    p = os.path.join(d, 'df_test9.csv')
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, encoding='utf-8-sig')


def block_sums(df, col, key='step'):
    """按 (episode-内 step 值) 把行切成 45 行的评测块，返回每块的列和。"""
    out = []
    i = 0
    n = len(df)
    while i + 45 <= n:
        out.append((df[key].iloc[i], float(df[col].iloc[i:i + 45].sum())))
        i += 45
    return pd.DataFrame(out, columns=[key, col])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*', default=['run57'])
    ap.add_argument('--base', default='output')
    ap.add_argument('--col', default='turbine_profit')
    args = ap.parse_args()

    for spec in args.runs:
        d = find_run(spec, args.base)
        df = load(d)
        if df is None:
            print('%s: 无 df_test9.csv' % d)
            continue
        print('\n' + '=' * 96)
        print('%s   (%d 行 = %d 次评测)' % (d, len(df), len(df) // 45))
        print('=' * 96)
        B = block_sums(df, args.col)
        # 同一 step 值 = 同一 episode 位置 ⇒ 10 次重复（episode 内 3 次评测）
        g = B.groupby('step')[args.col]
        cnt = g.size()
        sd_in = g.std()
        print('\n[A] 组内 sd（纯评测噪声；同 step 值 = 同 episode 位置，风序列逐位相同）')
        lab = {1920: 'ep 内第 1 次', 2400: 'ep 内第 2 次', 2880: 'ep 内第 3 次'}
        for k in sorted(sd_in.index):
            print('    step %-6d n=%2d  组内 sd = %8.1f  (%.2f%% of 均值 %.0f)   %s'
                  % (k, cnt[k], sd_in[k],
                     100 * sd_in[k] / B[args.col].mean(), B[args.col].mean(),
                     lab.get(int(k), '')))
        pooled = float(np.sqrt((sd_in ** 2 * (cnt - 1)).sum() / (cnt - 1).sum()))
        print('    合并（pooled）组内 sd = %.1f 元  ⇒ 单次评测的噪声 = ±%.1f 元'
              % (pooled, pooled))
        print('    ⚠ 若策略确定性，此 sd 应为 0 ⇒ 实测非 0 说明**评测时仍加探索噪声**')

        print('\n[B] 逐检查点均值 ± 噪声标准误（n≈10 次重复/检查点）')
        m = B.groupby('step')[args.col].mean()
        se = pooled / np.sqrt(cnt.mean())
        for k in m.index:
            print('    step %-6d 均值 %9.1f  ± %.1f (±%.2f%%)'
                  % (k, m[k], se, 100 * se / abs(m[k])))
        print('    ⇒ 两次检查点之间的"可分辨差异"下限 ≈ 2·√2·SE = %.0f 元 (%.2f%%)'
              % (2 * np.sqrt(2) * se, 100 * 2 * np.sqrt(2) * se / abs(m.mean())))

        print('\n[C] 趋势检验：raw vs 配对（组内去均值）')
        B['blk'] = np.arange(len(B)) // max(1, int(round(cnt.mean())))
        B['resid'] = B[args.col] - B.groupby('step')[args.col].transform('mean')
        for c, name in ((args.col, 'raw'), ('resid', '配对(去同 step 均值)')):
            x = np.arange(len(B))
            y = B[c].to_numpy(float)
            b, a = np.polyfit(x, y, 1)
            r = y - (a + b * x)
            s = np.sqrt((r ** 2).sum() / (len(B) - 2) / ((x - x.mean()) ** 2).sum())
            print('    %-22s 斜率 %+9.2f/评测块  t=%+6.2f'
                  % (name, b, b / s))
        print('    注: "配对"列已去掉同 step 的系统电平，剩下的是**同一 episode 位置上的**')
        print('        策略变化 + 噪声 ⇒ 斜率仍应按噪声标准误读，不要当改进幅度。')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
