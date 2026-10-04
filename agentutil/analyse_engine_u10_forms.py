#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同风速 u=10 m/s、不同动作行为下 FLORIS vs WFSim 的差别**归类**（三种形态判定）。

用户提出的三形态判据
--------------------
形态一  系统性偏差：曲线形状对、整体偏高/偏低（或特定风向下固定偏差）⇒ 稳态误差，可校正。
形态二  缺时间滞后：风速突变后 WFSim 下游机组延迟数分钟响应（尾流输运时间），FLORIS 立即响应。
形态三  动态形态根本对不上：WFSim 有波动/蜿蜒/恢复过程，FLORIS 完全平滑 ⇒ 校正模型也难救。

本脚本回答「在**本项目实际使用的耦合**下，三形态各自是否存在、量级多大」。
数据全部是既有 artifact（**不需要 MATLAB**）：

* `output/clip_check_plan.json`      350 点执行计划（35 档风速 × 10 个随机行为）
* `output/clip_check_wfsim.json`     WFSim 参考（含逐台功率）—— MATLAB 实跑
* `output/clip_check_fullrange.json` FLORIS 侧（**生产口径**：`calibrated=True`）
* `output/u10_engine_forms.json`     本仓库探针 `probe_u10_engine_forms.py` 的产出（FLORIS 逐台 + 口径裁决）

u=10.0 恰好有 **10 个不同行为**（同一风速、10 组随机 a 向量，两引擎入参逐位相同）⇒
这正是「相同风速、不同行为」的受控对照；且**风速被钉死**，于是可以把既有 n=100 分析里
被风速混淆的「比值随动作漂移」单独分离出来。

⚠ 口径前提（**先读这条，否则量级会差 70%**）
-------------------------------------------
`documentation/engine_random_action_comparison.md` 报的 `FLORIS/WFSim = 1.67`、以及
"两引擎地形一个凹一个凸"等结论，来自 `output/engine_random_actions.json`。
`probe_u10_engine_forms.py` 已把它判定为 **FLORIS 随包默认尾流参数（stock）**，即
**非生产配置**：同一批 u=10 行为上 stock 给 `1.7164`，而生产口径 `calibrated=True` 给
`0.9892`，且后者**逐位复现** `clip_check_fullrange.json`（`max|Δ| = 0.000e+00`）。
⇒ 本文档所有"形态一"数字都按 **calibrated = 生产** 口径报，stock 只作对照。

形态判定的两类依据
------------------
* **数值类（形态一）**：本脚本从配对数据直接算。
* **结构性（形态二/三）**：由 WFSim 侧入口代码决定 —— `WFSim_port.m:53` 每次调用都跑一次
  完整 `WFSim(Wind,Control)`；`InitWFSim.m:14-17` 把流场重置为**均匀流**；windTunnel 场景
  `h=1, L=5`（`meshing.m:189-190`）⇒ 只推进 **5 s**；`startUniform=false`（`meshing.m:207`）
  ⇒ k=1 步用 `max_it=50` 迭代到"完全发展的尾流场"（`WFSim.m:103-107`），k>1 每步
  `max_it_dyn=1`（`WFSim.m:81`）。⇒ **每次调用都是从零重解的准稳态映射**，调用之间没有
  任何状态延续。故形态二（跨调用滞后）与形态三（内部波动/蜿蜒）在此耦合下**不可能出现**：
  日志里的一切时间结构都来自外生风速序列，而两引擎吃的是同一条序列。

用法: python agentutil/analyse_engine_u10_forms.py [--u 10.0] [--bin 0.5] [--json output/x.json]
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np                                                     # noqa: E402

#: 控制台默认是 cp1252/cp936，中文报告会 UnicodeEncodeError ⇒ 显式改 UTF-8。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

PLAN = os.path.join('output', 'clip_check_plan.json')
WREF = os.path.join('output', 'clip_check_wfsim.json')
FREF = os.path.join('output', 'clip_check_fullrange.json')
PROBE = os.path.join('output', 'u10_engine_forms.json')

#: WFSim 侧几何/口径（`meshing.m:180-190`、`WFSim_port.m:34`）：单排 5 台、间距 5D、D=100 m。
N_TURB = 5
D_ROTOR = 100.0
SPACING_D = 5.0
#: windTunnel 的积分时长（`meshing.m:189-190`: h=1, L=5）—— 用于与尾流输运时间尺度对照。
SIM_WINDOW_S = 5.0
#: 无尾流单机功率 @u=10（`Cp_eff = 0.95·4a(1−a)²`，a=1/3，ρ=1.2，A=π50²）—— 用于反推尾流效率。
def no_wake_single(u):
    cp = 0.95 * 4.0 * (1.0 / 3.0) * (1.0 - 1.0 / 3.0) ** 2
    return 0.5 * 1.2 * np.pi * (D_ROTOR / 2.0) ** 2 * cp * u ** 3 / 1e6


# ---------------------------------------------------------------- 工具
def load():
    plan = json.load(open(PLAN, encoding='utf-8'))
    w = json.load(open(WREF, encoding='utf-8'))
    f = json.load(open(FREF, encoding='utf-8'))
    out = {}
    for name, d in (('plan', plan), ('wfsim', w), ('floris', f)):
        out[name] = {'u': np.asarray(d['u'], dtype=float),
                     'A': np.asarray(d['A'], dtype=float)}
    out['wfsim']['p'] = np.asarray(w['p_wfsim'], dtype=float)
    out['wfsim']['pt'] = np.asarray(w['p_wfsim_turbine'], dtype=float)
    out['floris']['p'] = np.asarray(f['p_floris_clip'], dtype=float)
    return out


def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2)))


def ols(X, y):
    """最小二乘；返回 (coef, r2, resid, se)。`se` = 系数标准误（含 σ²(XᵀX)⁻¹）。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, p = X.shape
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    if n > p:
        s2 = ss_res / (n - p)
        try:
            cov = s2 * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        except np.linalg.LinAlgError:
            se = np.full(p, np.nan)
    else:
        se = np.full(p, np.nan)
    return coef, r2, resid, se


def best_scale(p_ref, p_tgt):
    """最小二乘 k 使 k·p_ref ≈ p_tgt；返回 (k, 相对残差 RMS%, 最大|残差|/均值%)。"""
    k = float(np.sum(p_ref * p_tgt) / np.sum(p_ref ** 2))
    res = p_tgt - k * p_ref
    m = float(np.mean(p_tgt))
    return k, 100.0 * rms(res) / m, 100.0 * np.max(np.abs(res)) / m


def format_table(rows, header, widths):
    lines = ['  ' + ' '.join(h.rjust(w) for h, w in zip(header, widths))]
    for r in rows:
        lines.append('  ' + ' '.join(str(v).rjust(w) for v, w in zip(r, widths)))
    return '\n'.join(lines)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description='u=10 下 FLORIS vs WFSim 的三形态归类')
    ap.add_argument('--u', type=float, default=10.0, help='目标风速（默认 10.0）')
    ap.add_argument('--bin', type=float, default=0.5,
                    help='辅面板半宽：取 |u−目标|≤bin（默认 0.5 ⇒ [9.5,10.5]）')
    ap.add_argument('--json', default=None, help='把结果落盘为 JSON')
    args = ap.parse_args()

    D = load()
    u, A = D['wfsim']['u'], D['wfsim']['A']
    pW, pWt = D['wfsim']['p'], D['wfsim']['pt']
    pF = D['floris']['p']

    # ------------------------------------------------------------ 1 对齐与口径
    print('=' * 96)
    print('1. 数据与口径')
    print('=' * 96)
    sameA = (np.max(np.abs(D['plan']['A'] - A)) == 0.0
             and np.max(np.abs(D['floris']['A'] - A)) == 0.0)
    sameU = (np.max(np.abs(D['plan']['u'] - u)) == 0.0
             and np.max(np.abs(D['floris']['u'] - u)) == 0.0)
    print('  n 总点数 = %d   风速 %g~%g m/s    三文件 u 逐位相同 = %s    A 逐位相同 = %s'
          % (len(u), u.min(), u.max(), sameU, sameA))
    print('  FLORIS 口径 = `AT.build_model(calibrated=True)`（标定尾流参数、**无**电平系数）'
          ' = `aifflor` 生产默认')
    print('  WFSim 口径   = `WFSim_port(u, 0, a[5], 0)`（MATLAB 实跑，本次不重算）')
    m0 = np.isclose(u, args.u)
    mb = np.abs(u - args.u) <= args.bin + 1e-9
    print('  u == %g 的受控点数 = %d（同风速、不同行为）    |u−%g| ≤ %g 的辅面板 = %d'
          % (args.u, int(m0.sum()), args.u, args.bin, int(mb.sum())))
    if int(m0.sum()) < 3:
        raise SystemExit('该风速下配对数不足，无法归类。')

    res = {'u_target': args.u, 'n_u_exact': int(m0.sum()), 'n_bin': int(mb.sum())}

    def panel(mask, tag, control_u):
        u_m, A_m = u[mask], A[mask]
        W, Wt, F = pW[mask], pWt[mask], pF[mask]
        n = int(mask.sum())
        ratio = F / W
        rel = 100.0 * (F - W) / W
        print()
        print('-' * 96)
        print('【%s】n = %d' % (tag, n))
        print('-' * 96)

        print('  A. 形态一：系统性偏差（同一风速、不同行为）')
        print('    %-4s %-38s %9s %9s %8s %9s'
              % ('#', '行为 a₁…a₅（a₁ 最上游）', 'P_WFSim', 'P_FLORIS', '比值', '相对差%'))
        for i in range(n):
            print('    %-4d [%s] %9.4f %9.4f %8.4f %+9.2f'
                  % (i + 1, ' '.join('%.3f' % v for v in A_m[i]), W[i], F[i], ratio[i], rel[i]))
        print('    %-4s %-38s %9.4f %9.4f %8.4f %+9.2f'
              % ('μ', '（均值）', W.mean(), F.mean(), ratio.mean(), rel.mean()))
        print('    %-4s %-38s %9s %9s %8.4f %9.2f'
              % ('σ', '（标准差）', '', '', ratio.std(ddof=1), rel.std(ddof=1)))
        print('    比值 min / max / 极差 = %.4f / %.4f / %.4f'
              % (ratio.min(), ratio.max(), ratio.max() / ratio.min()))
        print('    偏差方向：FLORIS %s WFSim，均值 %+.2f%%'
              % ('低于' if ratio.mean() < 1 else '高于', rel.mean()))

        k_f, r_f, mx_f = best_scale(W, F)
        k_w, r_w, mx_w = best_scale(F, W)
        print('    · 常数重标定 `P_FLORIS = k·P_WFSim`：k* = %.4f ⇒ 残差 RMS = %.3f%%'
              '（最大 %.2f%%）' % (k_f, r_f, mx_f))
        print('    · 反向 `P_WFSim = k·P_FLORIS`       ：k* = %.4f ⇒ 残差 RMS = %.3f%%'
              '（最大 %.2f%%）' % (k_w, r_w, mx_w))

        # 校正模型阶梯（全部在**本面板内**拟合 ⇒ in-sample，参数越多个数越多）
        print()
        print('  B. 校正模型阶梯（P_FLORIS ≈ f(a)·P_WFSim；同一风速下拟合，n=%d）' % n)
        mean_a = A_m.mean(axis=1)
        std_a = A_m.std(axis=1)
        X = np.column_stack([np.ones(n), A_m])
        if control_u:
            X = np.column_stack([X, u_m])
        ladder = []

        def fit(name, cols, y=None):
            # 目标是 **P_FLORIS**：模型形如 P_FLORIS = f(a)·P_WFSim ⇒ 设计列都乘 P_WFSim。
            Xd = np.column_stack(cols)
            yy = F if y is None else y
            c, r2, rs, _ = ols(Xd, yy)
            p = Xd.shape[1]
            ladder.append((name, p, 100.0 * rms(rs) / float(np.mean(F)),
                           100.0 * np.max(np.abs(rs)) / float(np.mean(F)), r2, c))

        fit('① 常数 k', [W])
        fit('② k + c·mean(a)', [W, W * mean_a])
        fit('③ + d·std(a)', [W, W * mean_a, W * std_a])
        fit('④ k + Σcⱼaⱼ（逐行为）', [W] + [W * A_m[:, j] for j in range(N_TURB)])
        print(format_table(
            [(nm, p, '%.2f%%' % a, '%.2f%%' % b, '%.4f' % r2)
             for nm, p, a, b, r2, _ in ladder],
            ['模型', '参数数', '残差RMS', '最大|残差|', 'R²'], [26, 6, 9, 10, 8]))
        print('      ⇒ 常数只能压到 %.2f%%；再往上加"随行为"的项，收益 %s'
              % (ladder[0][2],
                 ('显著（→ %.2f%%）' % ladder[3][2]) if ladder[3][2] < 0.6 * ladder[0][2]
                 else ('有限（→ %.2f%%）' % ladder[3][2])))
        print('      ⚠ n=%d、参数数最多 %d ⇒ ④ 是**在样本内**把行为自由度基本用掉，'
              '它的残差是"可达下界"，不是泛化误差。' % (n, ladder[3][1]))

        # 比值随行为漂移
        print()
        print('  C. 比值是否随行为漂移（本面板已消掉风速混淆）')
        c1, r2_1, _, se1 = ols(np.column_stack([np.ones(n), mean_a]), ratio)
        c2, r2_2, _, _ = ols(np.column_stack([np.ones(n), mean_a, std_a]), ratio)
        for j in range(N_TURB):
            cj, r2j, _, sej = ols(np.column_stack([np.ones(n), A_m[:, j]]), ratio)
            print('    · 比值 ~ 1 + a%d                : 斜率 %+.4f ± %.4f   R² = %.3f'
                  % (j + 1, cj[1], sej[1], r2j))
        print('    · 比值 ~ 1 + mean(a)          : 斜率 %+.4f ± %.4f   R² = %.3f'
              % (c1[1], se1[1], r2_1))
        print('    · 比值 ~ 1 + mean(a) + std(a) : R² = %.3f' % r2_2)

        # 幅值 vs 形状
        print()
        print('  D. 是"幅值差"还是"形状差"：P/u³ 归一化后的逐维斜率（其余维同时控制）')
        names = ['1'] + ['a%d' % (j + 1) for j in range(N_TURB)]
        Xs = np.column_stack([np.ones(n), A_m])
        got = {}
        for tag2, P in (('WFSim', W), ('FLORIS', F)):
            c, r2, _, se = ols(Xs, P / u_m ** 3, )
            got[tag2] = (c, se, r2)
            print('    %-7s P/u³ = %+.6f   (R² = %.4f)' % (tag2, c[0], r2))
            for j in range(N_TURB):
                print('             a%d: %+.6f ± %.6f' % (j + 1, c[j + 1], se[j + 1]))
        cW, seW, _ = got['WFSim']
        cF, seF, _ = got['FLORIS']
        print('    逐维斜率之差（FLORIS − WFSim）：')
        for j in range(N_TURB):
            dd = cF[j + 1] - cW[j + 1]
            se_d = float(np.hypot(seW[j + 1], seF[j + 1]))
            print('      a%d: %+.3e  (合成标准误 %.1e ⇒ %s)'
                  % (j + 1, dd, se_d,
                     '显著' if abs(dd) > 2 * se_d else '不显著'))
        sign_same = all(np.sign(cW[j + 1]) == np.sign(cF[j + 1]) for j in range(N_TURB))
        n_sig = sum(1 for j in range(N_TURB)
                    if abs(cF[j + 1] - cW[j + 1]) > 2 * float(np.hypot(seW[j + 1], seF[j + 1])))
        print('    5 维斜率符号一致 = %s；|差| > 2σ 的维数 = %d/%d'
              % (sign_same, n_sig, N_TURB))

        return {'tag': tag, 'n': n,
                'ratio_mean': float(ratio.mean()), 'ratio_std': float(ratio.std(ddof=1)),
                'ratio_min': float(ratio.min()), 'ratio_max': float(ratio.max()),
                'k_floris_over_wfsim': k_f, 'resid_const_pct': r_f, 'resid_const_max_pct': mx_f,
                'k_wfsim_over_floris': k_w, 'resid_const_rev_pct': r_w,
                'ladder': [dict(name=nm, nparam=p, rms_pct=a, max_pct=b, r2=r2)
                           for nm, p, a, b, r2, _ in ladder],
                'slope_ratio_on_mean_a': float(c1[1]), 'se_ratio_on_mean_a': float(se1[1]),
                'r2_ratio_on_mean_a': r2_1,
                'slope_wfsim': [float(x) for x in cW], 'se_wfsim': [float(x) for x in seW],
                'slope_floris': [float(x) for x in cF], 'se_floris': [float(x) for x in seF],
                'signs_agree': bool(sign_same), 'n_sig_dim_diff': int(n_sig),
                'p_wfsim': [float(x) for x in W], 'p_floris': [float(x) for x in F],
                'A': [[float(v) for v in row] for row in A_m],
                'p_wfsim_turbine': [[float(v) for v in row] for row in Wt]}

    r_exact = panel(m0, '受控面板：u = %g m/s 固定、%d 个不同行为' % (args.u, int(m0.sum())), False)
    r_bin = panel(mb, '辅面板：|u − %g| ≤ %g（n 更大，回归时控制 u）' % (args.u, args.bin), True)
    res['panel_exact'] = r_exact
    res['panel_bin'] = r_bin

    # ------------------------------------------------------------ 2 逐台结构
    print()
    print('=' * 96)
    print('2. 逐台结构：偏差由哪几台承担（FLORIS 逐台来自 %s）' % PROBE)
    print('=' * 96)
    if not os.path.exists(PROBE):
        print('  ⚠ 缺 %s ⇒ 跳过。先跑：' % PROBE)
        print('    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\probe_u10_engine_forms.py')
    else:
        pr = json.load(open(PROBE, encoding='utf-8'))
        A_p = np.asarray(pr['A'], dtype=float)
        Wt_p = np.asarray(pr['p_wfsim_turbine'], dtype=float)
        Ft_p = np.asarray(pr['p_floris_cal_turbine'], dtype=float)
        St_p = np.asarray(pr['p_floris_stock_turbine'], dtype=float)
        u_p = float(pr['u'])
        if A_p.shape != A[m0].shape or np.max(np.abs(A_p - A[m0])) > 0:
            print('  ⚠ 探针与 plan 的行为不一致 ⇒ 只用比例统计，不做逐行配对。')
        iso = no_wake_single(u_p)
        print('  无尾流单机功率 @u=%g = %.4f MW（5 台满发 = %.4f MW）'
              % (u_p, iso, 5 * iso))
        print()
        print('  %-8s %10s %10s %10s %8s %10s %8s'
              % ('机组', 'WFSim', 'FLORIS', '差值', '比值', '效率W', '效率F'))
        for j in range(N_TURB):
            w_j, f_j = Wt_p[:, j].mean(), Ft_p[:, j].mean()
            print('  T%-7d %10.4f %10.4f %+10.4f %8.4f %10.4f %8.4f'
                  % (j + 1, w_j, f_j, f_j - w_j, f_j / w_j, w_j / iso, f_j / iso))
        print('  %-8s %10.4f %10.4f %+10.4f %8.4f %10.4f %8.4f'
              % ('合计', Wt_p.sum(axis=1).mean(), Ft_p.sum(axis=1).mean(),
                 (Ft_p.sum(axis=1) - Wt_p.sum(axis=1)).mean(),
                 Ft_p.sum(axis=1).mean() / Wt_p.sum(axis=1).mean(),
                 Wt_p.sum(axis=1).mean() / (5 * iso), Ft_p.sum(axis=1).mean() / (5 * iso)))
        print()
        print('  逐台"同行为配对"的比值 P_F/P_W（每台对 10 个行为各算一次再平均，'
              '消掉各台 a 均值不同）：')
        Rt = Ft_p / Wt_p
        for j in range(N_TURB):
            print('    T%d: 比值均值 %.4f ± %.4f   （%s）'
                  % (j + 1, Rt[:, j].mean(), Rt[:, j].std(ddof=1),
                     'FLORIS 高' if Rt[:, j].mean() > 1 else 'FLORIS 低'))
        dec_w = (Wt_p[:, 0] / Wt_p[:, -1]).mean()
        dec_f = (Ft_p[:, 0] / Ft_p[:, -1]).mean()
        print('  "沿程衰减" T1/T5（逐行为算再平均）：WFSim %.3f  vs  FLORIS %.3f  '
              '⇒ 相差 %.1f×' % (dec_w, dec_f, dec_f / dec_w))
        print('  ⚠ 解读：风场**总和**只差 %.2f%%，但这是"T1 上 FLORIS 高出 %.0f%%"与'
              '"T3~T5 上 FLORIS 低 %.0f~%.0f%%"**相互抵消**的结果。'
              % (100 * (Ft_p.sum(axis=1).mean() / Wt_p.sum(axis=1).mean() - 1),
                 100 * (Rt[:, 0].mean() - 1),
                 100 * (1 - Rt[:, 2].mean()), 100 * (1 - Rt[:, 4].mean())))
        print('     ⇒ "总和对得上"不等于"模型对得上"：这是**形状差**，常数修不了。')
        print()
        print('  逐台绝对误差 |P_F − P_W| 的分布（占总绝对误差的份额）:')
        err = np.abs(Ft_p - Wt_p).mean(axis=0)
        for j in range(N_TURB):
            print('    T%d: %.4f MW  (%.1f%%)' % (j + 1, err[j], 100 * err[j] / err.sum()))
        print('  T1 与 T5 的"尾流效率"之比（相对 a=1/3 无尾流满发，含动作差异）：')
        print('    WFSim %.4f / %.4f = %.3f ;  FLORIS %.4f / %.4f = %.3f'
              % (Wt_p[:, 0].mean() / iso, Wt_p[:, -1].mean() / iso,
                 (Wt_p[:, 0] / Wt_p[:, -1]).mean(),
                 Ft_p[:, 0].mean() / iso, Ft_p[:, -1].mean() / iso,
                 (Ft_p[:, 0] / Ft_p[:, -1]).mean()))
        print('  ⚠ 该行把"动作差异"也混进来了（各台 a 不同），只作量级参考；'
              '干净的同行为配对比值见上面的 P_F/P_W。')
        st_mean = St_p.sum(axis=1).mean()
        print()
        print('  对照：stock（非生产）逐台合计 = %.4f MW ⇒ 比值 %.4f（生产口径 %.4f）'
              % (st_mean, st_mean / Wt_p.sum(axis=1).mean(),
                 Ft_p.sum(axis=1).mean() / Wt_p.sum(axis=1).mean()))
        res['per_turbine'] = dict(
            u=u_p, iso_single=float(iso),
            wfsim=[float(v) for v in Wt_p.mean(axis=0)],
            floris_cal=[float(v) for v in Ft_p.mean(axis=0)],
            floris_stock=[float(v) for v in St_p.mean(axis=0)],
            ratio_paired=[float(v) for v in Rt.mean(axis=0)],
            decay_wfsim=float(dec_w), decay_floris=float(dec_f),
            abs_err_share=[float(v) for v in err / err.sum()])

    # ------------------------------------------------------------ 3 形态二/三
    print()
    print('=' * 96)
    print('3. 形态二（缺时间滞后）与形态三（动态形态对不上）判定 —— 结构性证据')
    print('=' * 96)
    print('  耦合方式（生产路径）：')
    print('    env.py::PowerOutput  →  WFSim_port(u, 0.0, a[5], 0.0)  →  WFSim(Wind, Control)')
    print('  入口行为（逐行读代码）：')
    print('    · WFSim_port.m:53        每次调用执行一次**完整** WFSim(Wind,Control)')
    print('    · InitWFSim.m:14-17      流场被重置为**均匀流**（"no turbines present yet"）')
    print('    · meshing.m:189-190      windTunnel: h=1, L=5 ⇒ time=0:1:5，**只推进 %.0f s**'
          % SIM_WINDOW_S)
    print('    · meshing.m:207          startUniform=false ⇒ k=1 步 max_it=50 迭代到'
          '"完全发展的尾流场"')
    print('    · WFSim.m:81             max_it_dyn=1 ⇒ k>1 每步只解一次（松弛，非动力学推进）')
    print('  ⇒ 该入口是"**从零重解的准稳态映射**"：调用之间无状态延续')
    print('     ⇒ 形态二（跨调用滞后）在此耦合下**不可能出现**（无状态可滞后）；')
    print('     ⇒ 形态三（内部波动/蜿蜒/恢复序列）也**不可能出现**：')
    print('        - 无随机源：InitWFSim 每次同一均匀初值，无湍流脉动实现；')
    print('        - `turbul=true/turbModel=WFSim3` 是**平均场混合长度湍流模型**，')
    print('          它决定尾流恢复的**速率**，不产生时间上的波动序列；')
    print('        - 日志里的一切时间结构都来自**外生风速序列**，而两引擎吃同一条序列。')
    print()
    dx = SPACING_D * D_ROTOR
    t1, tf = dx / args.u, dx * (N_TURB - 1) / args.u
    print('  量级对照（u = %g m/s, 间距 %gD = %.0f m）：' % (args.u, SPACING_D, dx))
    print('    · 单跨尾流输运时间 5D/u = %.0f s；全风场 %.0f m / u = %.0f s'
          % (t1, dx * (N_TURB - 1), tf))
    print('    · 入口积分窗口只有 %.0f s ⇒ 若真靠"输运"，下游机组在窗口内根本来不及'
          % SIM_WINDOW_S)
    print('      感受上游尾流；它能给出合理亏损正说明靠 k=1 步 50 次迭代直接解出已发展的场。')
    print('  ⇒ 要让形态二/三出现，必须**换耦合**（调用间保留 sol、让 time 轴连续、逐时步改')
    print('     Wind/Control），而不是换引擎。')
    print('  ⇒ 可证伪的检验（本沙箱内 MATLAB 起不来，未执行）：'
          '`agentutil/probe_wfsim_memoryless.py`')
    print('     —— 同参数连调 20 次是否逐位相同、穿插别的 (u,a) 后能否复原、两实例共享')
    print('     engine 是否互不干扰。三节全过 ⇒ 无记忆，形态二/三被确证不存在。')
    res['transport'] = dict(dx_m=dx, t_5D_s=t1, t_farm_s=tf, sim_window_s=SIM_WINDOW_S)

    # ------------------------------------------------------------ 4 结论
    print()
    print('=' * 96)
    print('4. 结论表（相同风速 %g m/s、不同行为；数字为**生产口径 calibrated**）' % args.u)
    print('=' * 96)
    print('  %-4s %-16s %s' % ('形态', '是否存在', '依据 / 量级'))
    print('  %-4s %-16s %s' % ('一', '**是（主体）**',
                              '比值 %.4f±%.4f，方向 FLORIS %s；常数重标定残差 %.2f%%；'
                              '校正阶梯下界 %.2f%%'
                              % (r_exact['ratio_mean'], r_exact['ratio_std'],
                                 '偏低' if r_exact['ratio_mean'] < 1 else '偏高',
                                 r_exact['resid_const_pct'], r_exact['ladder'][3]['rms_pct'])))
    print('  %-4s %-16s %s' % ('二', '**否（本耦合下）**',
                              '入口每次重置 + %.0f s 积分 ⇒ 无状态、无从滞后；'
                              '输运尺度 %.0f s 被跳过' % (SIM_WINDOW_S, tf)))
    print('  %-4s %-16s %s' % ('三', '**否（本耦合下）**',
                              '无随机源、无湍流脉动实现；时间结构 100%% 外生'))
    print()
    print('  ⇒ 三形态里本项目**实际存在的只有形态一**。但注意它不是"纯电平差"：')
    print('     · 电平本身已经很近（%+.2f%%），可校正；' % (100 * (r_exact['ratio_mean'] - 1)))
    print('     · 剩下的 %.2f%% 常数残差里混着"随行为变化的增益"与"尾流沿程形状差"，'
          % r_exact['resid_const_pct'])
    print('       所以**固定一个系数修不干净**（要修需按行为/尾流状态建模）。')
    print('  ⇒ 形态二/三不是"差别大不大"，而是**在这条耦合路径上被结构性消掉了**：')
    print('     换成连续推进 WFSim 才会出现，届时才是真正的动态保真度问题。')
    print()
    print('  ⚠ 口径警告：`documentation/engine_random_action_comparison.md` 的 1.67× 是')
    print('     **stock（FLORIS 随包默认）**的结论（`probe_u10_engine_forms.py` 已裁决：')
    print('     同一批 u=10 行为上 stock = %.4f vs 生产 = %.4f）。引用时勿混。'
          % (1.7164, r_exact['ratio_mean']))

    if args.json:
        json.dump(res, open(args.json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print('\n  已落盘: %s' % args.json)


if __name__ == '__main__':
    main()
