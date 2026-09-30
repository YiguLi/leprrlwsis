# -*- coding: utf-8 -*-
"""用**标定后的 FLORIS** 重算 H 层约束口径的对齐度 —— 回答"超参数能否迁移"的收尾。

背景（两个前序实测）
--------------------
1. `analyse_engine_cost_scale.py`（FLORIS **默认参数**）：同一套 `c1` 口径下，
   越限率 1.00% → 8.00%（×8）、块和 3.998 → 13.233（**×3.31**）、
   块和÷预算 0.2665 → 0.8822 ⇒ 结论"要重标定 `s_c`"。
2. `calib_floris_to_wfsim.py`（**标定后**）：验证集 MAPE 66.9% → 4.45%，
   常数重标定后 67.9% → **1.315%**，比值 k* = 0.9868 ⇒ FLORIS 已贴近 WFSim。

⇒ 那 3.31× 是**默认参数**下的数字，必须用标定后的参数重算。本脚本就做这件事。

它算什么
--------
在**同一批验证集样本**上，对两个引擎分别构出逐条的出力序列，再按 `hsrl_cost` 口径算：

    P_F  = |P(t) − P(t−1)|                     （注意：这是**差分**，形状残差会被放大）
    c1   = P_F + K_V·(P_F − 3)⁺
    块和 = Σ₅ c1 ；块和÷d_block=15 ；越限率 P_F>3 ；超预算块占比

并给出"为了两通道平衡，等效的 `s_c` 应取多少"。

⚠ 验证集的样本顺序是 LHS 采样的顺序，**不是真实时间序列**，所以 `P_F` 在这里是
"相邻样本差分"的代理，不是物理分钟波动。它的作用是**两引擎之间的同口径对照**
（同一个差分序列，两边各算一次），不是绝对波动量级。

用法（不需要 MATLAB；数据已落盘）
--------------------------------
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\analyse_calibrated_cost_scale.py
    ... --theta 0.05 0.00107 0.10013 0.11351 1.94935 0.6235
"""
import argparse
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

import hsrl_cost as hc  # noqa: E402
import agentutil.calib_floris_to_wfsim as C  # noqa: E402
import agentutil.probe_engine_random_actions as R  # noqa: E402

BLOCK = 5
DATA_DEFAULT = os.path.join('output', 'calib_floris_to_wfsim.json')
# 用户标定跑出来的 θ*（--theta 可覆盖）
THETA_STAR = [0.05, 0.00107, 0.10013, 0.11351, 1.94935, 0.6235]


def cost_report(tag, P):
    """按 hsrl_cost 口径算约束量。P 为逐样本出力 [MW]（顺序 = 验证集顺序）。"""
    pf = np.abs(np.diff(P))
    if pf.size == 0:
        return None
    c1 = np.array([hc.fluct_cost(x) for x in pf])
    nb = len(c1) // BLOCK
    blocks = c1[:nb * BLOCK].reshape(nb, BLOCK).sum(axis=1) if nb else np.array([c1.sum()])
    viol = pf > hc.P_F_LIMIT
    return dict(tag=tag, n=len(pf), mean_p=P.mean(),
                pf_mean=float(pf.mean()), pf_p95=float(np.percentile(pf, 95)),
                pf_max=float(pf.max()),
                viol=float(viol.mean()), n_viol=int(viol.sum()),
                c1_mean=float(c1.mean()), c1_max=float(c1.max()),
                blk_mean=float(blocks.mean()),
                blk_over=float((blocks > hc.D_BLOCK).mean()))


def main():
    ap = argparse.ArgumentParser(description='用标定后的 FLORIS 重算约束口径',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument('--data', default=DATA_DEFAULT)
    ap.add_argument('--theta', type=float, nargs=6, default=THETA_STAR)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit('找不到 %s —— 先在普通终端跑 calib_floris_to_wfsim.py --gen'
                         % args.data)
    d = json.load(open(args.data, encoding='utf-8'))
    vu = np.asarray(d['val']['u'], dtype=float)
    vA = np.asarray(d['val']['A'], dtype=float)
    vW = np.asarray(d['val']['P_wfsim'], dtype=float)
    theta = np.asarray(args.theta, dtype=float)

    print('数据: %s（验证集 n=%d）' % (args.data, len(vu)))
    print('θ*  : %s' % dict(zip(['%s.%s' % (p[0], p[1]) for p in C.PARAMS],
                                [round(float(v), 5) for v in theta])))
    print('口径: c1 = P_F + %.0f·(P_F−%.0f)⁺  |  d_block = %.0f  |  s_p=%g  s_c(TD3)=%g'
          % (hc.K_V, hc.P_F_LIMIT, hc.D_BLOCK, hc.VALUE_SCALE, hc.COST_SCALE_TD3))

    # ---- 两个模型：随包默认（对照） vs 标定后（生产默认） ----
    R.register_op_model()
    m_stock = R.build_floris_model(stock=True)     # FLORIS 随包默认参数
    m_cal = R.build_floris_model(stock=False)      # 标定后的参数（现行默认）
    PF_cal = C.floris_eval(m_cal, vu, vA, theta=theta)
    # 顺便算随包默认（对照）。theta0 即随包默认值，显式施加到 stock 模型上。
    theta0 = np.array([p[4] for p in C.PARAMS], dtype=float)
    PF_def = C.floris_eval(m_stock, vu, vA, theta=theta0)

    print('\n' + '=' * 96)
    print('验证集上的出力对齐度')
    print('=' * 96)
    for tag, P in (('FLORIS 默认', PF_def), ('FLORIS 标定后', PF_cal)):
        r = P / vW
        ok = np.isfinite(r) & (vW > 1e-9)
        k = float(np.dot(vW[ok], P[ok]) / np.dot(vW[ok], vW[ok]))
        print('  %-14s 比值 mean %.4f  std %.4f  min %.4f  max %.4f   最优 k*=%.4f'
              % (tag, r[ok].mean(), r[ok].std(), r[ok].min(), r[ok].max(), k))

    rows = [cost_report('WFSim', vW),
            cost_report('FLORIS 默认', PF_def),
            cost_report('FLORIS 标定后', PF_cal)]
    rows = [r for r in rows if r]
    w = rows[0]
    if 100 * w['viol'] > 50:
        print('\n  ⚠⚠ **绝对量不可用**：验证集是 LHS 乱序采样，相邻样本的 u 与动作都不同 ⇒')
        print('     `P_F = |P(t)−P(t−1)|` 里绝大部分是"工况跳变"，不是物理分钟波动')
        print('     （实测 P_F 均值 %.2f MW，而真实分钟序列只有 ~0.69 MW）。' % w['pf_mean'])
        print('     越限率/块和÷预算/超预算% 因此失去物理意义，**只能用来做两引擎对照**。')
        print('     要拿绝对量请用真实分钟序列（`probe_engine_random_actions.py` 里')
        print('     让动作分块保持不变 + `--u-mode seq`）。')

    print('\n' + '=' * 96)
    print('H 层约束口径对照（同一批样本、同一差分顺序）')
    print('=' * 96)
    print('  %-16s %10s %10s %10s %10s %12s %10s'
          % ('', 'P_F均值', '越限率', 'c1均值', '块和', '块和÷15', '超预算%'))
    for r in rows:
        print('  %-16s %10.4f %9.2f%% %10.4f %10.4f %12.4f %9.1f%%'
              % (r['tag'], r['pf_mean'], 100 * r['viol'], r['c1_mean'],
                 r['blk_mean'], r['blk_mean'] / hc.D_BLOCK, 100 * r['blk_over']))

    w, fdef, fcal = rows[0], rows[1], rows[2]
    print('\n  相对 WFSim 的倍率:')
    print('    %-26s %10s %10s' % ('', '块和比', '越限率比'))
    for r, nm in ((fdef, 'FLORIS 默认'), (fcal, 'FLORIS 标定后')):
        print('    %-26s %10.4f %10.4f'
              % (nm, r['blk_mean'] / w['blk_mean'],
                 (r['viol'] / w['viol']) if w['viol'] > 0 else float('nan')))

    # ---- 等效 s_c ----
    print('\n' + '=' * 96)
    print('两通道平衡角度：等效的 s_c 该取多少')
    print('=' * 96)
    s_c0 = hc.COST_SCALE_TD3
    print('  设计目标（`hsrl_cost.py` 注释）：块奖励 r̄/s_p ≈ c̄/s_c（同量级）')
    for r, nm in ((w, '以 WFSim 为基准'), (fdef, 'FLORIS 默认'), (fcal, 'FLORIS 标定后')):
        need = s_c0 * (r['blk_mean'] / w['blk_mean'])
        print('    %-16s c̄/块=%.4f  ⇒ 等效 s_c = %7.1f  (基准 s_c=%g)'
              % (nm, r['blk_mean'], need, s_c0))
    print('\n  ⇒ 若"FLORIS 标定后"的等效 s_c 接近 %g，说明**标定已经把尺度差异吃掉**，' % s_c0)
    print('     超参数迁移的尺度障碍基本消失；若仍差 2 倍以上，则仍需为每个引擎单独标定。')

    # ---- 结论行 ----
    print('\n' + '=' * 96)
    print('结论')
    print('=' * 96)
    ratio_def = fdef['blk_mean'] / w['blk_mean']
    ratio_cal = fcal['blk_mean'] / w['blk_mean']
    print('  块和倍率：默认参数 ×%.3f  →  标定后 ×%.3f' % (ratio_def, ratio_cal))
    if abs(ratio_cal - 1) < 0.25:
        print('  ⇒ 标定后约束口径已对齐（±25% 内）⇒ **尺度类超参数可以跨引擎迁移**，')
        print('     只需按上表做一次常数重标定，不必为每个引擎重新调参。')
    else:
        print('  ⇒ 标定后约束口径仍未对齐 ⇒ 尺度类超参数**仍需为每个引擎单独标定**。')

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump(dict(theta=[float(x) for x in theta],
                       rows=rows, ratio_def=ratio_def, ratio_cal=ratio_cal),
                  open(args.json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print('\n已写入 %s' % args.json)


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    main()
