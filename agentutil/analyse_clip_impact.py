# -*- coding: utf-8 -*-
"""`u<0` 的落点定位 + clip 影响的最终汇总（读 `clip_check_fullrange` 的原始结果重算）。

回答两个问题：
  ① `u<0` 出现在**哪台机组**、**哪个风速段**、**离中心多远**；
  ② `clip(u<0→0)` 对农场 / 逐台出力的影响分布。

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\analyse_clip_impact.py
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
PLAN = os.path.join('output', 'clip_check_plan.json')


def main():
    d = json.load(open(PLAN, encoding='utf-8'))
    us = np.asarray(d['u'], dtype=float)
    A = np.asarray(d['A'], dtype=float)
    n = len(us)

    R.CT_SCALE = 1.0
    m, _ = AT.build_model(calibrated=True)
    import yawflor.yawflor_turbine as YT

    neg_by_turb = np.zeros(NT, dtype=int)
    neg_by_pos = np.zeros((3, 3), dtype=int)
    min_u_all = np.full(n, np.nan)
    ue_c = np.zeros((n, NT))
    ue_n = np.zeros((n, NT))
    pt_c = np.zeros((n, NT))
    pt_n = np.zeros((n, NT))
    for i in range(n):
        R._A_SET = A[i:i + 1]
        m.set(wind_speeds=[float(us[i])], wind_directions=[YT.WIND_DIR_ALIGNED],
              turbulence_intensities=[YT.TI], yaw_angles=np.zeros((1, NT)))
        with R._solver_hook()():
            m.run()
        u = np.asarray(m.core.flow_field.u, dtype=float)[0]      # (NT,3,3)
        min_u_all[i] = float(u.min())
        neg = u < 0
        neg_by_turb += neg.sum(axis=(1, 2))
        neg_by_pos += neg.sum(axis=0)
        uc = np.clip(u, 0.0, None)
        ec = np.cbrt(np.mean(uc ** 3, axis=(1, 2)))
        en = np.real(np.cbrt(np.mean(u ** 3, axis=(1, 2))))
        ue_c[i], ue_n[i] = ec, en
        pw = np.asarray(m.get_turbine_powers(), dtype=float).reshape(-1)[:NT] / 1e6
        pt_c[i] = pw
        with np.errstate(invalid='ignore', divide='ignore'):
            rt = np.where(np.abs(en) > 1e-12, (ec / en) ** 3, np.nan)
        pt_n[i] = pw / np.where(np.abs(rt) > 1e-12, rt, np.nan)

    print('=' * 96)
    print('① u<0 的落点（%d 点 × %d 台 × 3×3 网格 = %d 个位置）' % (n, NT, n * NT * 9))
    print('=' * 96)
    tot_neg = int(neg_by_turb.sum())
    print('  负值总个数 = %d  (占全部位置的 %.4f%%)' % (tot_neg, 100 * tot_neg / (n * NT * 9)))
    print('  按机组:')
    for j in range(NT):
        print('     T%d: %5d 个  (%.2f%%)' % (j + 1, neg_by_turb[j],
                                            100 * neg_by_turb[j] / max(tot_neg, 1)))
    print('  按转子平面位置 (3×3 相对中心):')
    lab = ['上', '中', '下']
    for r_ in range(3):
        print('     %s排: %s' % (lab[r_], '  '.join('%6d' % v for v in neg_by_pos[r_])))
    print('  按风速段:')
    for lo, hi in ((3, 4), (4, 6), (6, 10), (10, 15), (15, 20.5)):
        mm = (us >= lo) & (us < hi)
        print('     [%4.1f,%4.1f): %2d 点，其中 min_u<0 的点 = %d'
              % (lo, hi, mm.sum(), int((min_u_all[mm] < 0).sum())))

    print('\n' + '=' * 96)
    print('② clip(u<0→0) 的影响')
    print('=' * 96)
    fc, fn = pt_c.sum(1), np.nansum(pt_n, 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        rf = np.where(fn != 0, fc / fn, np.nan)
    print('  农场总功率比 (clip/noclip)，全 %d 点:' % n)
    finite = rf[np.isfinite(rf)]
    print('     min=%.9f  中位=%.9f  max=%.9f' % (finite.min(), np.median(finite),
                                                 finite.max()))
    print('     |1−比| 最大 = %.4e  ⇒ clip 最多改变农场出力 **%.4f%%**'
          % (np.abs(1 - finite).max(), 100 * np.abs(1 - finite).max()))
    with np.errstate(invalid='ignore', divide='ignore'):
        rt = np.where(pt_n != 0, pt_c / pt_n, np.nan)
    rtf = rt[np.isfinite(rt)]
    print('  逐台功率比: |1−比| 最大 = %.4e  ⇒ 单台最多改变 **%.4f%%**'
          % (np.abs(1 - rtf).max(), 100 * np.abs(1 - rtf).max()))
    print()
    print('  说明：clip 的作用是把负速度抬到 0；负速度的绝对值通常只有 O(0.1) m/s，')
    print('        立方后 O(0.001)，在 3×3 的 Ue³ 均值里影响可忽略。')
    print('  最坏点 (min_u 最负) : u=%.2f, min_u=%.4f, 该点农场 |1−比|=%.3e'
          % (us[np.nanargmin(min_u_all)], np.nanmin(min_u_all),
             abs(1 - rf[np.nanargmin(min_u_all)])))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
