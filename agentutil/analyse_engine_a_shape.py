# -*- coding: utf-8 -*-
"""受控检验：**两引擎对 `a` 的响应形状是否真的不同**（同一批随机数据，非参数）。

为什么需要它（内部矛盾）
------------------------
`analyse_engine_a_response.py` 的二次拟合给出：
    WFSim  a* = 0.185（二次项 −21.7，开口向下）
    FLORIS a* = 0.500（二次项 +8.57，开口向上 ⇒ 拟合的最优点是**极小点**）
但本项目独立实测（均匀 a 扫描、u=10、5 台 5D）FLORIS 均匀最优 ≈ **0.40**。
两者矛盾 ⇒ 要么二次形式不够、要么风速混淆。

本脚本用**非参数**办法消除混淆：取窄风速窗（u∈[8,12]），在该窗内按 `mean(a)` 分箱，
分别看两引擎的 P_farm 均值曲线。窄窗内 u³ 变化 ≤ 1.56³≈3.8 倍→仍有影响，故再做
"除以 u³ 归一化"的版本。

用法: F:\\anaconda\\envs\\wsis\\python.exe agentutil\\analyse_engine_a_response.py --shape
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

JSON_DEFAULT = os.path.join('output', 'engine_random_actions.json')


def shape_check(u, w, f, am, asd, u_lo, u_hi):
    m = (u >= u_lo) & (u <= u_hi) & np.isfinite(w) & np.isfinite(f) & (w > 1e-9)
    print('\n风速窗 u∈[%.1f, %.1f]，n=%d' % (u_lo, u_hi, m.sum()))
    if m.sum() < 8:
        print('  样本太少，跳过')
        return
    print('  %-14s %5s %12s %12s %10s %10s'
          % ('mean(a) 分箱', 'n', 'P_WFSim', 'P_FLORIS', '比值', 'P/u³(W)'))
    edges = np.arange(0.0, 0.5001, 0.05)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = m & (am >= lo) & (am < hi)
        if s.sum() == 0:
            continue
        print('  %4.2f-%-9.2f %5d %12.4f %12.4f %10.4f %10.5f'
              % (lo, hi, s.sum(), w[s].mean(), f[s].mean(),
                 f[s].mean() / w[s].mean(),
                 (w[s] / u[s] ** 3).mean()))
    # 归一化版本：P/u³（消掉风速一阶影响）
    print('  %-14s %5s %14s %14s' % ('', 'n', 'P_w/u³', 'P_f/u³'))
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = m & (am >= lo) & (am < hi)
        if s.sum() == 0:
            continue
        print('  %4.2f-%-9.2f %5d %14.5f %14.5f'
              % (lo, hi, s.sum(), (w[s] / u[s] ** 3).mean(), (f[s] / u[s] ** 3).mean()))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else JSON_DEFAULT
    d = json.load(open(path, encoding='utf-8'))
    u = np.asarray(d['wind'], dtype=float)
    A = np.asarray(d['actions'], dtype=float)
    w = np.asarray(d['p_wfsim'], dtype=float)
    f = np.asarray(d['p_floris'], dtype=float)
    am, asd = A.mean(axis=1), A.std(axis=1)
    print('数据: %s (mode=%s n=%d)' % (path, d.get('config', {}).get('mode'),
                                       d.get('config', {}).get('n')))
    print('=' * 84)
    print('受控响应形状检验（窄风速窗 + u³ 归一化，消除风速混淆）')
    print('=' * 84)
    for lo, hi in ((8, 12), (5, 9), (12, 18)):
        shape_check(u, w, f, am, asd, lo, hi)

    print('\n' + '=' * 84)
    print('结论判据')
    print('=' * 84)
    print('  * 若 P_w/u³ 与 P_f/u³ 在 mean(a) 上的**峰位明显不同**（或一个单调一个不单调），')
    print('    ⇒ 两引擎对 a 的响应形状不同 ⇒ 比值必然依赖动作 ⇒ 常数重标定无效。')
    print('  * 若两列峰位一致，只是比例不同 ⇒ 比值随 a 变化只是"幅度缩放"，')
    print('    那用"随 mean(a) 变的 k"就能修好。')


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    main()
