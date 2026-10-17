# -*- coding: utf-8 -*-
"""1.74× 的电平差对 `td3per` 的**约束口径**有多大影响（用实测两引擎序列直接算）。

背景（`hsrl_cost.py` 的口径，逐字复刻）
----------------------------------------
    P_F = |P_G(t) − P_G(t−1)|                    [MW]        ← 用**绝对**功率，与引擎无关地计算
    c1  = P_F + K_V·(P_F − 3)⁺ ,  K_V = 100      [MW·step]
    d_block = D_BLOCK = 3 MW × 5 步 = 15         [MW·step]   ← 块预算
    r1  = 售电收益 + W_GAP·追踪罚                 [元]
    VALUE_SCALE = 1000 (r1) , COST_SCALE_TD3 = 100 (c1)      ← 两通道进 critic 前各除一次

**问题**：`P_F`、`3 MW`、`d_block=15` 都是**绝对值**，而 FLORIS 的电平是 WFSim 的 1.74×。
本脚本量化这个 1.74× 把"同一套超参数"拧紧/拧松了多少。

用法: F:\\anaconda\\envs\\wsis\\python.exe agentutil\\analyse_engine_cost_scale.py
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

import hsrl_cost as hc  # noqa: E402   （单一真源：K_V / D_BLOCK / 两通道 scale）

JSON_DEFAULT = os.path.join('output', 'engine_minute_compare.json')
BLOCK = 5


def pf_series(P):
    """P_F 系列（P_F[0] 无前值 ⇒ 用 |P[1]-P[0]| 起步，与 env 首次 P_G_prev 口径一致）。"""
    d = np.abs(np.diff(P))
    return np.concatenate([[d[0]], d])


def report(tag, P):
    pf = pf_series(P)
    c1 = np.array([hc.fluct_cost(x) for x in pf])
    viol = pf > hc.P_F_LIMIT
    # 块和（长度 5 的整块）
    nb = len(c1) // BLOCK
    blocks = c1[:nb * BLOCK].reshape(nb, BLOCK).sum(axis=1)
    print('\n--- %s ---' % tag)
    print('  平均出力 %.4f MW | P_F 均值 %.4f | P_F p95 %.4f | P_F max %.4f'
          % (P.mean(), pf.mean(), np.percentile(pf, 95), pf.max()))
    print('  越限 P_F>3 的步数占比  %.4f%%   (%d/%d)'
          % (100 * viol.mean(), viol.sum(), len(pf)))
    print('  c1 = P_F + K_V(P_F−3)⁺ : 均值 %.4f | p95 %.4f | max %.4f  [MW·step]'
          % (c1.mean(), np.percentile(c1, 95), c1.max()))
    print('  块和 Σ₅c1              : 均值 %.4f | p95 %.4f | max %.4f   (预算 d_block=%.0f)'
          % (blocks.mean(), np.percentile(blocks, 95), blocks.max(), hc.D_BLOCK))
    print('  块和 / 预算            : 均值 %.4f | 超预算块占比 %.1f%%'
          % (blocks.mean() / hc.D_BLOCK, 100 * (blocks > hc.D_BLOCK).mean()))
    return dict(pf=pf, c1=c1, blocks=blocks, viol=float(viol.mean()),
                pf_mean=float(pf.mean()), c1_mean=float(c1.mean()),
                blk_mean=float(blocks.mean()), blk_p95=float(np.percentile(blocks, 95)),
                over=float((blocks > hc.D_BLOCK).mean()))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else JSON_DEFAULT
    d = json.load(open(path, encoding='utf-8'))
    w = np.asarray(d['p_wfsim'], dtype=float)
    f = np.asarray(d['p_floris'], dtype=float)
    u = np.asarray(d['wind'], dtype=float)

    print('=' * 92)
    print('口径 (hsrl_cost.py 单一真源): c1 = P_F + %.0f·(P_F−%.0f)⁺  |  d_block = %.0f'
          '  |  s_p = %g (r1)  s_c = %g (c1)'
          % (hc.K_V, hc.P_F_LIMIT, hc.D_BLOCK, hc.VALUE_SCALE, hc.cost_scale()))
    print('数据: %s  (n=%d, u 均值 %.3f)' % (path, len(u), u.mean()))
    print('=' * 92)

    rw = report('WFSim', w)
    rf = report('FLORIS', f)

    print('\n' + '=' * 92)
    print('1.74× 电平差把约束拧紧了多少（FLORIS / WFSim）')
    print('=' * 92)
    print('  P_F 均值比          %.4f' % (rf['pf_mean'] / rw['pf_mean']))
    print('  越限率比            %.4f   (%.4f%% → %.4f%%，即 ×%.2f)'
          % ((rf['viol'] / rw['viol']) if rw['viol'] > 0 else float('nan'),
             100 * rw['viol'], 100 * rf['viol'],
             (rf['viol'] / rw['viol']) if rw['viol'] > 0 else float('nan')))
    print('  c1 均值比           %.4f' % (rf['c1_mean'] / rw['c1_mean']))
    print('  块和均值比          %.4f   (%.4f → %.4f，预算恒为 %.0f)'
          % (rf['blk_mean'] / rw['blk_mean'], rw['blk_mean'], rf['blk_mean'], hc.D_BLOCK))
    print('  超预算块占比        %.1f%% → %.1f%%' % (100 * rw['over'], 100 * rf['over']))

    print('\n  两通道进 critic 的尺度（hsrl_cost 的设计目标是"块奖励同量级"）:')
    s_c = hc.COST_SCALE_TD3          # TD3 全系列实际用的值（见 hsrl_cost 末尾小节）
    print('    %-22s %14s %14s' % ('', 'WFSim', 'FLORIS'))
    print('    %-22s %14.4f %14.4f' % ('c̄/块', rw['blk_mean'], rf['blk_mean']))
    print('    %-22s %14.4f %14.4f' % ('c̄/块 ÷ s_c (s_c=%g)' % s_c,
                                       rw['blk_mean'] / s_c, rf['blk_mean'] / s_c))
    print('    ⇒ 同一条 c̄ 被 1.74× 放大后，cost 通道的回归目标也同比例变大；')
    print('      而 s_c = COST_SCALE_TD3 = %g 是在**旧口径**下定标的，'
          '两通道平衡随之漂移。' % s_c)
    print('      若要恢复该平衡，等效的 s_c 应 ≈ %.1f（= %g × %.2f），'
          '但 s_c 是用户超参数、不随引擎自动改。'
          % (s_c * (rf['blk_mean'] / rw['blk_mean']), s_c,
             rf['blk_mean'] / rw['blk_mean']))

    # 反事实：若把 FLORIS 的功率整体除 1.74 再按同一口径算，约束松紧是否回到 WFSim
    k = f.sum() / w.sum()
    rf_scaled = report('FLORIS（电平 ÷%.4f 后，模拟"重标定过"的臂）' % k, f / k)
    print('\n  ⇒ 重标定后越限率 %.4f%%（WFSim %.4f%%）—— 比值 %.3f'
          % (100 * rf_scaled['viol'], 100 * rw['viol'],
             (rf_scaled['viol'] / rw['viol']) if rw['viol'] > 0 else float('nan')))
    print('     块和/预算  %.4f（WFSim %.4f）'
          % (rf_scaled['blk_mean'] / hc.D_BLOCK, rw['blk_mean'] / hc.D_BLOCK))


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    main()
