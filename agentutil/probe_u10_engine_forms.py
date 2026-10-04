# -*- coding: utf-8 -*-
"""u = 10 m/s、同一批行为下，把 FLORIS 的**两条口径链**与 WFSim 对齐钉死（不需 MATLAB）。

动机
----
仓库里有两份都自称"标定后"的 FLORIS artifact，但同一风速下的电平差 **1.67× vs 0.99×**
完全不同：

* `output/engine_random_actions.json`  → `FLORIS/WFSim` 均值 **1.6687**（= +66.9%）
* `output/clip_check_fullrange.json`   → u=10 处 `FLORIS/WFSim` 均值 **0.9892**（= −1.1%）

`documentation/engine_random_action_comparison.md` 全篇建立在 1.67 上，而
`aifflor/README.md`（u=10, a=1/3：FLORIS 3.294 vs WFSim 3.446）建立在 ~0.96 上。
两者不可能同时是"生产口径"。本探针判定**哪一条是 `aifflor` 的生产默认**：

* **路径 A（生产）** `AT.build_model(calibrated=True)` + `AT.eval_batch`
  —— 即 `aifflor_env.AifflorPhysics` 默认走的那条（`aifflor_main.py` 注释：
  calibrated=True 用标定尾流参数、**不加**电平系数）。
* **路径 B（非生产对照）** `AT.build_model(calibrated=False)` = FLORIS 随包默认尾流参数
  （`aifflor_turbine.py:128-129` 明说这是"只用电平系数模式"，须配 `power_scale=LEVEL_COEF`）。

判据（硬性）：路径 A 必须**逐位复现** `clip_check_fullrange.json` 的 `p_floris_clip`
（同一 plan、同一模型 ⇒ 应当 max|Δ| ≈ 0）。若成立，则 u=10 的"生产口径"电平差就是
**~1%**，而 1.67× 属于非生产配置，任何"形态一有多大"的结论都必须按口径分开报。

顺带产出 **FLORIS 逐台功率**（`clip_check_fullrange.json` 只存了风场总和），
这是逐台比对"偏差由哪几台承担"所必需的。

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\probe_u10_engine_forms.py [--u 10.0]
        [--json output/u10_engine_forms.json]
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

from floris_import_guard import check_interpreter as _chk, sanitize as _sanitize  # noqa: E402
_chk()
_sanitize()

import numpy as np                                                     # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import agentutil.probe_engine_random_actions as R                      # noqa: E402
import aifflor.aifflor_turbine as AT                                   # noqa: E402

NT = AT.NT
PLAN = os.path.join('output', 'clip_check_plan.json')
WREF = os.path.join('output', 'clip_check_wfsim.json')
FREF = os.path.join('output', 'clip_check_fullrange.json')
RAND = os.path.join('output', 'engine_random_actions.json')


def jsonable(o):
    """把含 tuple/ndarray 键值的东西转成 JSON 可序列化形式（`theta` 是 tuple 键的 dict）。"""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o



def main():
    ap = argparse.ArgumentParser(description='u=10 下 FLORIS 两条口径链 vs WFSim')
    ap.add_argument('--u', type=float, default=10.0)
    ap.add_argument('--json', default=os.path.join('output', 'u10_engine_forms.json'))
    args = ap.parse_args()

    plan = json.load(open(PLAN, encoding='utf-8'))
    w = json.load(open(WREF, encoding='utf-8'))
    f = json.load(open(FREF, encoding='utf-8'))
    us_all = np.asarray(plan['u'], dtype=float)
    A_all = np.asarray(plan['A'], dtype=float)
    sel = np.isclose(us_all, args.u)
    us, A = us_all[sel], A_all[sel]
    pW = np.asarray(w['p_wfsim'], dtype=float)[sel]
    pWt = np.asarray(w['p_wfsim_turbine'], dtype=float)[sel]
    pF_ref = np.asarray(f['p_floris_clip'], dtype=float)[sel]
    n = int(sel.sum())
    print('=' * 92)
    print('u = %g m/s 的行为数 = %d（来自 %s）' % (args.u, n, PLAN))
    print('=' * 92)

    # ---------------------------------------------------------- 路径 A（生产）
    m_cal, info_cal = AT.build_model(calibrated=True, verbose=True)
    print('[A] 生产口径: calibrated=True  theta 已施加=%s'
          % (info_cal.get('theta') is not None))
    farm_cal, turb_cal = AT.eval_batch(m_cal, us, A)
    farm_cal = np.asarray(farm_cal, dtype=float)
    turb_cal = np.asarray(turb_cal, dtype=float)

    # ---------------------------------------------------------- 路径 B（非生产）
    m_st, info_st = AT.build_model(calibrated=False, verbose=True)
    farm_st, turb_st = AT.eval_batch(m_st, us, A)
    farm_st = np.asarray(farm_st, dtype=float)
    turb_st = np.asarray(turb_st, dtype=float)
    lvl = float(AT.LEVEL_COEF)
    farm_st_lvl = farm_st * lvl

    # ---------------------------------------------------------- 判据
    print()
    print('-' * 92)
    print('判据：路径 A 是否逐位复现 clip_check_fullrange.json 的 p_floris_clip')
    print('-' * 92)
    d = np.abs(farm_cal - pF_ref)
    rel = d / np.abs(pF_ref)
    print('  max|Δ| = %.3e MW    max 相对差 = %.3e   %s'
          % (d.max(), rel.max(), '✓ 逐位一致（路径 A = 生产口径）' if rel.max() < 1e-9
             else '✗ 不一致 ⇒ 需查口径'))
    if rel.max() >= 1e-9:
        for i in range(n):
            print('    #%2d  A路径 %.6f   文件 %.6f   Δ %.3e'
                  % (i + 1, farm_cal[i], pF_ref[i], d[i]))

    # ---------------------------------------------------------- 三口径电平对照
    print()
    print('-' * 92)
    print('三种 FLORIS 口径在同一批 (u=%g, 行为) 上的电平（MW）' % args.u)
    print('-' * 92)
    print('  %-34s %10s %10s %10s' % ('口径', '均值', 'vs WFSim', '常数残差%'))
    rows = [('WFSim（MATLAB 参考）', pW),
            ('FLORIS A: calibrated（生产）', farm_cal),
            ('FLORIS B: stock（随包默认）', farm_st),
            ('FLORIS B×LEVEL_COEF=%.4f' % lvl, farm_st_lvl)]
    out_rows = {}
    for tag, P in rows:
        r = P / pW
        k = float(np.sum(pW * P) / np.sum(pW ** 2))
        res = 100.0 * float(np.sqrt(np.mean((P - k * pW) ** 2))) / float(np.mean(P))
        print('  %-34s %10.4f %10.4f %10.2f' % (tag, P.mean(), r.mean(), res))
        out_rows[tag] = dict(mean=float(P.mean()), ratio_mean=float(r.mean()),
                             ratio_std=float(r.std(ddof=1)), k=float(k), resid_pct=float(res))

    # ---------------------------------------------------------- 文档锚点核对
    print()
    print('-' * 92)
    print('文档锚点核对（u=%g, a≡1/3）' % args.u)
    print('-' * 92)
    A33 = np.full((1, NT), 1.0 / 3.0)
    p33_cal = float(np.ravel(AT.eval_batch(m_cal, [args.u], A33)[0])[0])
    p33_st = float(np.ravel(AT.eval_batch(m_st, [args.u], A33)[0])[0])
    print('  本探针算：calibrated = %.4f MW    stock = %.4f MW    stock×%.4f = %.4f MW'
          % (p33_cal, p33_st, lvl, p33_st * lvl))
    print('  仓库文档：WFSim a≡1/3 @u=10 = 3.4462 MW（`agentutil/README.md` '
          'probe_a_farm_structure，MATLAB 实测）')
    print('            FLORIS 标定 @u=10 = 3.294 MW（`aifflor/README.md` §52）')
    print('  ⇒ 本探针 calibrated 值 %s 与文档 3.294 一致'
          % ('✓' if abs(p33_cal - 3.294) / 3.294 < 0.05 else '✗ 差 %.1f%%'
             % (100 * (p33_cal / 3.294 - 1))))

    # ---------------------------------------------------------- 历史 artifact 归属
    print()
    print('-' * 92)
    print('两份历史 artifact 的口径归属')
    print('-' * 92)
    if os.path.exists(RAND):
        rj = json.load(open(RAND, encoding='utf-8'))
        pr = np.asarray(rj['p_floris'], dtype=float)
        wr = np.asarray(rj['p_wfsim'], dtype=float)
        rr = pr / wr
        print('  engine_random_actions.json : FLORIS/WFSim 均值 %.4f（+%.1f%%）'
              % (rr.mean(), 100 * (rr.mean() - 1)))
        print('      ⇒ 与 stock 口径的 +%.1f%% 同档 ⇒ **该 artifact 是 stock（非生产）**'
              % (100 * (out_rows['FLORIS B: stock（随包默认）']['ratio_mean'] - 1)))
        print('  clip_check_fullrange.json  : calibrated ⇒ **生产口径**')
        print('  ⚠ `documentation/engine_random_action_comparison.md` 的 1.67× 是 **stock**')
        print('     的结论，不能直接当"生产配置的形态一量级"。')

    res = dict(
        u=float(args.u), n=n,
        A=[[float(v) for v in row] for row in A],
        p_wfsim=[float(v) for v in pW],
        p_wfsim_turbine=[[float(v) for v in row] for row in pWt],
        p_floris_cal=[float(v) for v in farm_cal],
        p_floris_cal_turbine=[[float(v) for v in row] for row in turb_cal],
        p_floris_stock=[float(v) for v in farm_st],
        p_floris_stock_turbine=[[float(v) for v in row] for row in turb_st],
        p_floris_stock_times_level=[float(v) for v in farm_st_lvl],
        level_coef=lvl,
        verify_max_abs_delta=float(d.max()), verify_max_rel_delta=float(rel.max()),
        panel=out_rows,
        anchor_u10_a33=dict(calibrated=p33_cal, stock=p33_st, stock_times_level=p33_st * lvl),
        config=dict(nt=NT, floris_version=info_cal.get('floris_version'),
                    calibrated_params=info_cal.get('wake_params'),
                    theta=info_cal.get('theta'),
                    cp_eff=info_cal.get('cp_eff'), rho=info_cal.get('rho')),
    )
    json.dump(jsonable(res), open(args.json, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n  已落盘: %s' % args.json)


if __name__ == '__main__':
    main()
