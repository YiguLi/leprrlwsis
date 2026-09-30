# -*- coding: utf-8 -*-
"""方案 3：在 FLORIS 的组合步骤注入**带"持久性"的尾流叠加**（自建尾流模型）。

要解决的物理缺口
----------------
WFSim 是 `Ly = 4D` 的**封闭风洞**（Neumann 边界）：尾流无法侧向扩散 ⇒ 其等效效果是
"**亏损不衰减**"。而 FLORIS 的高斯尾流是**开放域**，`d ∝ 1/σ²` 必然随下游衰减
（σ 从 0.354D 增到 0.455D）。前面三轮扫描（CT 倍率 / 尾流增长率 / 幂平均指数）
都失败，因为**叠加规则无法表达"不衰减"**。

本脚本把"持久性"直接写进叠加规则
--------------------------------
标准 SOSFS：`u = hypot(wake_acc, d·u_initial) − wake_acc`，即 `d_total = √(Σd_i²)`。

新规则（两个旋钮）：

    d_SOSFS(new)  = hypot(w, d_new·u_initial) − w          # 原样保留单尾流形状
    d_persist     = ( (1−q)·w^p + q·(d_new·u_initial)^p )^(1/p)
                     ↑ 上游已累积的亏损按权重 (1−q) **保留**下来（这就是"持久性"）
    返回            max(d_old, d_SOSFS, d_persist)          # 单调不减 ⇒ 不衰减

    旋钮：`q ∈ [0,1]` 持久性权重（q=0 ⇒ 退化为原 SOSFS）；
          `p ∈ [1,2]` 幂平均指数（p=2 ⇒ 欧氏，p=1 ⇒ 线性）

物理含义：`q` 把"上游亏损"从"被同一因子稀释"改成"按比例留存"，
q→1 时亏损不再随下游衰减 —— 正是 WFSim 封闭域的等效行为。

判据（与前几轮一致）
--------------------
* 常数重标定后 MAPE ≤ 2%
* `u_min > 0` 且 `deficit_max < 0.9`

⚠ 风险声明：这是**偏离 FLORIS 标准公式的自建尾流模型**，用于论文需独立验证。

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\probe_persistent_combination.py
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
import agentutil.calib_floris_to_wfsim as C  # noqa: E402
import aifflor.aifflor_turbine as AT  # noqa: E402

DATA = os.path.join('output', 'calib_floris_to_wfsim.json')
NAME = 'probe-persistent'
GEOM = {'upwind': None, 'iwd': None}          # 由 build() 注入


def register(name=NAME):
    """注册"持久性"组合模型（继承 SOSFS 以复用 attrs 定义）。"""
    import floris.core.wake as W
    from floris.core.wake_combination import SOSFS

    if name in W.MODEL_MAP['combination_model']:
        return name

    class Persistent(SOSFS):
        """持久性叠加（**无 `max` 门**，直接替换 SOSFS）。

        第一版我在外面套了 `max(d_old, d_std, d_persist)`，实测那是 **no-op**：
        因为 `d_persist < d_std` 恒成立（trace 证据：T3 处 d_std=9.741 > d_persist=8.774），
        于是输出恒等于 `d_std`，24 行扫描结果完全相同。
        ⇒ 现在**直接**返回持久性混合，让它真正取代 SOSFS。
        """

        def function(self, wake_field, velocity_field):
            q = float(np.clip(R.PERSIST_Q, 0.0, 1.0))
            p = float(max(R.PERSIST_P, 1.0))
            w = np.clip(wake_field, 0.0, None)
            v = np.clip(velocity_field, 0.0, None)
            if q <= 0.0:
                return np.hypot(w, v)                    # 退化 = 原 SOSFS
            return ((1.0 - q) * w ** p + q * v ** p) ** (1.0 / p)

    Persistent.__name__ = 'Persistent'
    W.MODEL_MAP['combination_model'][name] = Persistent
    return name


def build(c_ct=1.0):
    """构造模型 + 注入"持久性"组合模型。

    ⚠ 几何（上游距离矩阵）在 `FlorisModel` 构造时**还不存在**（`grid.x_sorted` 为空），
    必须先用 `set()` 初始化一个点，`grid` 才会填充。故本函数最后做一次
    `m.set(...)` + 读几何 ⇒ 后续 `_run_batch` 会重新 `set`，不影响结果。
    """
    import floris.core.wake as W
    R.CT_SCALE = c_ct
    m, _ = AT.build_model(calibrated=True)
    wake = m.core.wake
    wake.model_strings['combination_model'] = NAME
    wake.combination_model = W.MODEL_MAP['combination_model'][NAME]()

    # ---- 先用一个点初始化网格，再读机组 x 坐标 ----
    m.set(wind_speeds=[10.0], wind_directions=[AT.WIND_DIR_ALIGNED],
          turbulence_intensities=[float(AT.TI)], yaw_angles=np.zeros((1, AT.NT)))
    g = m.core.grid
    xs = np.asarray(g.x_sorted, dtype=float)
    xs_t = xs[0, :, 0, 0] if xs.ndim == 4 else np.asarray(g.x).ravel()[:AT.NT]
    GEOM['upwind'] = np.clip(xs_t[:, None] - xs_t[None, :], 0.0, None)
    np.fill_diagonal(GEOM['upwind'], 0.0)
    return m


def evaluate(m, vu, vA, vW):
    R._A_SET = vA
    R._run_batch(m, vu, vA)
    pf = np.asarray(m.get_turbine_powers(), dtype=float).sum(axis=1) / 1e6
    mm = C.metrics(vW, pf)
    ui = np.asarray(m.core.flow_field.u_initial_sorted)
    uu = np.asarray(m.core.flow_field.u_sorted)
    dmax = max(float(np.nanmax((ui[:, j] - uu[:, j]) / np.maximum(ui[:, j], 1e-9)))
               for j in range(ui.shape[1]))
    return dict(mape_k=mm['mape_k'], u_min=float(np.nanmin(uu)),
                n_neg=int((uu < 0).sum()), deficit_max=dmax)


def main():
    d = json.load(open(DATA, encoding='utf-8'))
    vu = np.asarray(d['val']['u'], dtype=float)
    vA = np.asarray(d['val']['A'], dtype=float)
    vW = np.asarray(d['val']['P_wfsim'], dtype=float)
    print('验证集 n=%d (u 均值 %.2f, P_wfsim 均值 %.3f MW)' % (len(vu), vu.mean(), vW.mean()))
    register()
    print('已注册组合模型 %r：d = max(d_old, √-叠加, (1−q)·w^p ⊕ q·v^p)' % NAME)
    print('  旋钮 q = 持久性权重（0 ⇒ 退化为原 SOSFS）; p = 幂平均指数\n')

    q_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    p_list = [1.0, 1.5, 2.0]
    print('  %-8s %-6s %12s %13s %11s %8s %s'
          % ('q', 'p', '常数重标后', 'deficit_max', 'u_min', 'u<0', '判定'))
    rows = []
    for q in q_list:
        for p in p_list:
            R.PERSIST_Q, R.PERSIST_P = q, p
            try:
                r = evaluate(build(1.0), vu, vA, vW)
            except Exception as e:
                print('  %-8.2f %-6.2f  失败: %s' % (q, p, str(e)[:44]))
                continue
            r.update(q=q, p=p)
            fit = r['mape_k'] <= 0.02
            phys = (r['u_min'] > 0.0) and (r['deficit_max'] < 0.9)
            v = '**可用**' if (fit and phys) else ('拟合差' if not fit else '非物理')
            rows.append((r, fit, phys, v))
            print('  %-8.2f %-6.2f %11.3f%% %13.4f %11.4f %8d %s'
                  % (q, p, 100 * r['mape_k'], r['deficit_max'], r['u_min'],
                     r['n_neg'], v))

    good = [r for r, f, ph, _ in rows if f and ph]
    print('\n' + '=' * 92)
    print('结论')
    print('=' * 92)
    if rows:
        b = rows[0][0]
        print('  基准 (q=0 ⇒ 原 SOSFS): 常数重标定后 %.3f%%，deficit_max=%.3f, u_min=%.3f, u<0 %d'
              % (100 * b['mape_k'], b['deficit_max'], b['u_min'], b['n_neg']))
    if good:
        best = min(good, key=lambda r: r['mape_k'])
        print('  ✅ 找到 %d 个两全解；最优 q=%.2f p=%.2f ⇒ 常数重标定后 %.3f%%，'
              'deficit_max=%.3f, u_min=%.3f'
              % (len(good), best['q'], best['p'], 100 * best['mape_k'],
                 best['deficit_max'], best['u_min']))
        print('     ⇒ "持久性叠加"确实能在不缩窄尾流的前提下同时满足拟合与物理。')
    else:
        print('  ❌ 该 (q,p) 网格内仍无两全解 —— 说明"亏损不衰减"这一条修正')
        print('     还不足以复现 WFSim 的封闭域效果，需要更强的域约束（见 README）。')
    print('=' * 92)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
