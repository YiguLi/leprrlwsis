# -*- coding: utf-8 -*-
"""**标定后的 FLORIS 尾流参数（单一真源）** —— 用 WFSim 标定出来的默认值。

来源
----
`agentutil/calib_floris_to_wfsim.py` + `agentutil/calib_multistart_scan.py`
在 120 条标定 / 60 条验证样本（LHS，逐台 `a∈[0.025,0.5]`，风速 `uniform(3.5,25)`，
两引擎入参逐位相同）上做多种子 Nelder-Mead，得到的验证集最优解：

    validation: MAPE 4.448% | 常数重标定后 MAPE 1.098% | MedAPE 3.511% | k* = 0.9890
    对照: FLORIS 默认参数对 WFSim 的 MAPE = 66.891%（常数重标定后 67.946%）

⇒ 这两个模型在**开箱默认**下差 67%；用下面这组参数后差 **1.1%**。

用法（唯一入口）
----------------
    from floris_calibrated_defaults import apply_calibrated_wake_params
    m = <FlorisModel>
    apply_calibrated_wake_params(m)          # 施加标定参数
    apply_calibrated_wake_params(m, verbose=True)   # 顺便打印

⚠ 两点必须知道
--------------
1. **仅适用于本项目的 5 台 5D 单排 + `a` 通道 + `γ≡0` 配置**。换布局、换机型、
   或开偏航通道后需要重新标定。
2. **外推风险**：这组参数把尾流压得很窄，FLORIS 会报
   `Some velocities at the rotor are negative`（已实测，见
   `documentation/floris_calibration_to_wfsim.md` §3）。标定/验证集内无问题，
   但**未采样过的动作区域可能出 NaN** ⇒ 当训练环境用之前必须做 NaN 扫描。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---------------------------------------------------------------------------
# 标定后的参数值（2026 实测；顺序与 calib_floris_to_wfsim.PARAMS 一致）
#   key = (FLORIS 参数组, 参数名)
# ---------------------------------------------------------------------------
CALIBRATED_WAKE_PARAMS = {
    ('velocity_model', 'ka'): 0.05000,
    ('velocity_model', 'kb'): 0.00208,
    ('velocity_model', 'alpha'): 0.14456,
    ('velocity_model', 'beta'): 0.09134,
    ('turbulence_model', 'ai'): 1.65503,
    ('turbulence_model', 'constant'): 0.58031,
}

#: 标定前的 FLORIS 随包默认（仅用于对照/回归，不要用于生产）
STOCK_WAKE_PARAMS = {
    ('velocity_model', 'ka'): 0.38,
    ('velocity_model', 'kb'): 0.004,
    ('velocity_model', 'alpha'): 0.58,
    ('velocity_model', 'beta'): 0.077,
    ('turbulence_model', 'ai'): 0.8,
    ('turbulence_model', 'constant'): 0.5,
}

#: 标定质量（写进日志/论文时直接引用）
CALIBRATION_QUALITY = dict(
    val_mape=0.04448, val_mape_const_rescaled=0.01098, val_medape=0.03511,
    k_best=0.9890, stock_mape=0.66891, stock_mape_const_rescaled=0.67946,
    n_cal=120, n_val=60,
)


# 「标定脚本里的参数组名」→「本模块的参数组名」
_GROUP_ALIAS = {'velocity': 'velocity_model', 'turbulence': 'turbulence_model'}


def calibrated_theta():
    """按 `calib_floris_to_wfsim.PARAMS` 的顺序返回标定参数向量（6 个数）。"""
    from agentutil import calib_floris_to_wfsim as C
    return [float(CALIBRATED_WAKE_PARAMS[(_GROUP_ALIAS[p[0]], p[1])]) for p in C.PARAMS]


def apply_calibrated_wake_params(m, params=None, verbose=False):
    """把标定参数写进 `FlorisModel` 的 wake 参数表。返回生效后的字典。

    Parameters
    ----------
    m : FlorisModel
    params : dict or None   形如 `{('velocity_model','ka'): 0.05, ...}`；
                            None ⇒ 用 `CALIBRATED_WAKE_PARAMS`；
                            传 `STOCK_WAKE_PARAMS` 即恢复 FLORIS 随包默认。
    """
    params = CALIBRATED_WAKE_PARAMS if params is None else params
    vm = m.core.wake.model_strings['velocity_model']
    tm = m.core.wake.model_strings['turbulence_model']
    applied = {}
    for (group, name), val in params.items():
        model = vm if group == 'velocity_model' else tm
        path = ['wake', ('wake_velocity_parameters' if group == 'velocity_model'
                         else 'wake_turbulence_parameters'), model, name]
        m.set_param(path, float(val))
        applied['%s.%s' % (group, name)] = float(val)
    if verbose:
        print('[calibrated] 已施加标定尾流参数 (%s / %s): %s' % (vm, tm, applied))
    return applied


def describe():
    """一行说明（startup / 日志用）。"""
    q = CALIBRATION_QUALITY
    return ('[calibrated-defaults] FLORIS 尾流参数已按 WFSim 标定：'
            'MAPE %.3f%% → %.3f%%（常数重标定后 %.3f%% → %.3f%%，k*=%.4f）；'
            '标定/验证 n=%d/%d。⚠ 仅适用于 5 台 5D 单排 + a 通道 + γ≡0。'
            % (100 * q['stock_mape'], 100 * q['val_mape'],
               100 * q['stock_mape_const_rescaled'], 100 * q['val_mape_const_rescaled'],
               q['k_best'], q['n_cal'], q['n_val']))


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    print(describe())
    print('\n标定参数:')
    for (g, n), v in CALIBRATED_WAKE_PARAMS.items():
        print('  %-18s %-10s = %.5f   (随包默认 %.5f)'
              % (g, n, v, STOCK_WAKE_PARAMS[(g, n)]))
