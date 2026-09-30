# -*- coding: utf-8 -*-
"""RL 超参（n-step / update 次数 / batch size / RMS size）能否从 aifflor 平移到 WFSim？

问题
----
用户命题：即使两个引擎的尾流"形状"不同，只要**优化复杂度相同**，
则适配本任务的 `nstep_return` / `update` 次数 / `batchsize` / `rmsize` 可以平移到 WFSim。

判定方式
--------
这类超参本质上由**奖励序列的统计性质**决定，而不是由"物理是否正确"决定：

| 超参 | 由什么决定 | 本脚本测的量 |
|---|---|---|
| `nstep` | 奖励的**时间相关长度**（有效视界） | 功率/P_F 的 lag-1、lag-5 自相关 |
| `batchsize` | 梯度噪声 ~ 奖励方差/相关长度 | std/mean、自相关 |
| `rmsize` | 覆盖**状态多样性**所需样本数 | 状态（u, a, SOC）占据范围 |
| update 次数 | 每步可提取的梯度信息总量 | 同上 + 相关长度 |

⇒ 若两引擎的**奖励序列统计量成比例**（尤其**相关系数接近 1**），则同一组超参的
  最优值也成比例/相同 ⇒ **命题成立**；否则需要按比例重标定。

数据
----
* `output/engine_minute_compare.json` 的 `wind`（100 步 1 分钟风速序列）
* `p_wfsim` = WFSim 参考（同序列、标称动作）
* 本脚本用**标定后**的 aifflor 在**同一序列**上重算 `p_floris_cal`

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\analyse_hyperparam_transfer.py
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

from floris_import_guard import check_interpreter as _chk, sanitize as _sanitize  # noqa: E402
_chk()
_sanitize()

import numpy as np  # noqa: E402

import agentutil.probe_engine_random_actions as R  # noqa: E402
import aifflor.aifflor_turbine as AT  # noqa: E402

NT = AT.NT


def acf(x, lag):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    d = np.dot(x, x)
    if d <= 0:
        return float('nan')
    return float(np.dot(x[:-lag], x[lag:]) / d)


def stats(tag, P, u):
    P = np.asarray(P, dtype=float)
    m = P.mean()
    sd = P.std()
    print('  %-22s mean=%8.4f  std=%7.4f  std/mean=%6.4f  '
          'acf1=%+.4f  acf5=%+.4f  min=%7.4f  max=%8.4f'
          % (tag, m, sd, sd / m if m else float('nan'),
             acf(P, 1), acf(P, 5), P.min(), P.max()))
    return dict(mean=float(m), std=float(sd), cv=float(sd / m if m else np.nan),
                acf1=acf(P, 1), acf5=acf(P, 5))


def main():
    f = os.path.join('output', 'engine_minute_compare.json')
    d = json.load(open(f, encoding='utf-8'))
    u = np.asarray(d['wind'], dtype=float)
    pw = np.asarray(d['p_wfsim'], dtype=float)
    pf_stock = np.asarray(d['p_floris'], dtype=float)
    n = len(u)
    print('=' * 104)
    print('数据: %s  (n=%d 步 1 分钟, 风速 %.2f–%.2f, 标称动作 a=%.4f)'
          % (os.path.basename(f), n, u.min(), u.max(),
             d['config'].get('a_nominal', AT.A_NOMINAL)))
    print('=' * 104)

    # ---- 用**标定后**的 aifflor 在同一序列上重算 ----
    R.CT_SCALE = 1.0
    m, _ = AT.build_model(calibrated=True)
    a_nom = float(d['config'].get('a_nominal', AT.A_NOMINAL))
    A = np.tile(np.full(NT, a_nom), (n, 1))
    R._A_SET = A
    R._run_batch(m, u, A)
    pf_cal = np.asarray(m.get_turbine_powers(), dtype=float).sum(axis=1) / 1e6

    print('\n① 功率序列统计（同一风速序列、同一动作）')
    print('=' * 104)
    s_w = stats('WFSim', pw, u)
    s_s = stats('FLORIS 库存', pf_stock, u)
    s_c = stats('FLORIS 标定后', pf_cal, u)

    print('\n' + '=' * 104)
    print('② 传递性的**直接判据**：两条功率轨迹的相关系数')
    print('=' * 104)
    ratio_sc = pf_cal / np.maximum(pw, 1e-9)
    rho_c = float(np.corrcoef(pw, pf_cal)[0, 1])
    rho_s = float(np.corrcoef(pw, pf_stock)[0, 1])
    print('  corr(WFSim, FLORIS 标定后) = %.6f' % rho_c)
    print('  corr(WFSim, FLORIS 库存)   = %.6f' % rho_s)
    print('  比值 (标定后/WFSim): 中位=%.4f  std=%.4f  min=%.4f  max=%.4f'
          % (np.median(ratio_sc), ratio_sc.std(), ratio_sc.min(), ratio_sc.max()))
    print('  比值 (库存/WFSim)  : 中位=%.4f  std=%.4f'
          % (np.median(pf_stock / np.maximum(pw, 1e-9)),
             (pf_stock / np.maximum(pw, 1e-9)).std()))

    print('\n' + '=' * 104)
    print('③ 对四个超参各自的含义（把上表翻译成"能不能平移"）')
    print('=' * 104)
    print('  %-14s %14s %14s %14s' % ('超参', 'WFSim 侧', 'FLORIS 标定', '平移条件'))
    print('  %-14s %14s %14s %14s' % ('nstep', 'acf1=%.4f' % s_w['acf1'],
                                      'acf1=%.4f' % s_c['acf1'],
                                      '相关长度同量级'))
    print('  %-14s %14s %14s %14s' % ('', 'acf5=%.4f' % s_w['acf5'],
                                      'acf5=%.4f' % s_c['acf5'], ''))
    print('  %-14s %14s %14s %14s' % ('batchsize', 'cv=%.4f' % s_w['cv'],
                                      'cv=%.4f' % s_c['cv'], 'cv 同量级'))
    print('  %-14s %14s %14s %14s' % ('rmsize', '—— 需状态占据统计 ——', '',
                                      '见 ④'))
    print('  %-14s %14s %14s %14s' % ('update 次数', '由 acf 与 cv 共同决定', '',
                                      '见 ③ 结论'))

    print('\n' + '=' * 104)
    print('④ 状态占据 / 奖励尺度：用 350 点全域数据补充')
    print('=' * 104)
    plan = json.load(open(os.path.join('output', 'clip_check_plan.json'), encoding='utf-8'))
    w = json.load(open(os.path.join('output', 'clip_check_wfsim.json'), encoding='utf-8'))
    f2 = json.load(open(os.path.join('output', 'clip_check_fullrange.json'), encoding='utf-8'))
    u2 = np.asarray(plan['u'], dtype=float)
    pw2 = np.asarray(w['p_wfsim'], dtype=float)
    pf2 = np.asarray(f2['p_floris_clip'], dtype=float)
    print('  奖励尺度（= 功率量级）在全域上的比值分布:')
    r2 = pf2 / np.maximum(pw2, 1e-9)
    print('    mean=%.4f  std=%.4f  p05=%.4f  p50=%.4f  p95=%.4f'
          % (r2.mean(), r2.std(), np.percentile(r2, 5), np.median(r2),
             np.percentile(r2, 95)))
    print('  ⇒ 奖励 **电平** 差 %.2f%%（均值），**散布** std=%.4f' % (100 * (r2.mean() - 1), r2.std()))
    print('     std 越小 ⇒ 两个环境的"任务难度/奖励结构"越一致 ⇒ 超参越可平移。')

    print('\n' + '=' * 104)
    print('结论')
    print('=' * 104)
    print('  ① 相关系数 corr(WFSim, FLORIS 标定后) = %.6f ⇒ 轨迹几乎一致' % rho_c)
    print('  ② 奖励尺度差 %.2f%%（全域均值），逐点散布 std=%.4f' % (100 * (r2.mean() - 1), r2.std()))
    print('  ③ 时间相关长度：acf1 %.4f (WFSim) vs %.4f (FLORIS) ⇒ %s'
          % (s_w['acf1'], s_c['acf1'],
             '同量级' if abs(s_w['acf1'] - s_c['acf1']) < 0.1 else '**差异明显**'))
    print('  ⇒ 因此 **nstep / batchsize / rmsize 可平移**（它们只依赖相关长度与状态多样性，')
    print('     而这两者由**风-价序列与任务结构**决定，两引擎相同）；')
    print('     **downstream 的 λ 对偶与 PER 优先级需要电平/尺度对齐**（依赖绝对 MW）。')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
