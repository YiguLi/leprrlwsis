#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_yaw_official_example.py —— P0②：复现 FLORIS 官方示例并核对口径（只读，不需 MATLAB）。

为什么需要这个测试
------------------
P1 之后我们要用**自己写的**模型构造入口（`yawflor/yawflor_turbine.build_model`）
去跑偏航优化。在它之上做任何锚表之前，必须先证明**这个入口没有接错**：
它必须能逐位复现 FLORIS 官方发布的示例（同一台引擎、同一份配置、同一组结果）。

做法（两臂 + 一条对官方的核对）
------------------------------
* **臂 A（官方逐字）**：把官方 tag `v4.6.6` 的 `examples/inputs/gch.yaml` 原文
  落到 `agentutil/_scratch/`，用 `FlorisModel(<该文件>)` 装载，然后**逐字**执行官方
  示例 `examples/examples_control_optimization/001_opt_yaw_single_ws.py` 的模型部分
  （3 台 NREL 5MW、5D 间距 D=126 m、8 m/s、TI 0.06、GCH、风向 0→357 步长 3 的
  Serial-Refine 偏航优化）。绘图部分略去（与功率数值无关）。
* **臂 B（本地等价版）**：同一算例，但模型**只能**经
  `yawflor_turbine.build_model(nt=3, rotor_diameter=126, turbine_type='nrel_5MW', ...)`
  构造 ⇒ 与臂 A **逐位比较**基线/最优风场功率与最优偏航角。
  这一条直接回答"我们的配置有没有接错"：接错任何一项（布局/密度/参考高度/
  切变/TI/模型族）都会在这里显形。
* **对官方页面的核对（验收判据 ②，±1%）**：官方文档页
  `https://natlabrockies.github.io/floris/examples/examples_control_optimization/001_opt_yaw_single_ws.html`
  打印的表中，未对齐风向各行 `farm_power_baseline = farm_power_opt = 5.261863e6 W`
  ⇒ 断言 `max(farm_power_baseline)` 与该值相对偏差 ≤ 1%。
* **附带自检（防"背答案"）**：臂 A 的完整参数字典必须与随包
  `floris/default_inputs.yaml`（`name: GCH`）**只差算例项**
  （`layout_x/layout_y`、`turbulence_intensities`、`wind_speeds`、`wind_directions`、
  `reference_wind_height`、`turbine_type`）⇒ 说明我们内嵌的 YAML 没有被写错或漏项，
  也说明"取随包默认"这句话是可核对的。

运行
----
    python agentutil/test_yaw_official_example.py
    python agentutil/test_yaw_official_example.py --skip-opt      # 只标定/核对参数（快）
    python agentutil/test_yaw_official_example.py --json output/yawflor_official_example.json
"""
import argparse
import json
import os
import sys

import numpy as np

# ---- 路径 bootstrap（可被任意 CWD 调用）------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

os.environ.setdefault('MPLCONFIGDIR', os.path.join(ROOT, 'output', '_mplcache'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

# 官方页面的公开数值（验收判据 ②）；官方文档页表内未对齐风向行
PUBLISHED_UNALIGNED_W = 5.261863e6      # W；= 3 × NREL 5MW @ 8 m/s 无尾流
PUBLISHED_TOL = 0.01                    # ±1%

# 官方 tag v4.6.6 的 examples/inputs/gch.yaml 原文（值逐项转录；注释从简）。
# 来源：https://raw.githubusercontent.com/NatLabRockies/floris/v4.6.6/examples/inputs/gch.yaml
# 下面 _check_against_defaults() 会把它与随包 default_inputs.yaml 交叉核对。
GCH_YAML = """\
name: GCH
description: Three turbines using Gauss Curl Hybrid model
floris_version: v4
logging:
  console:
    enable: true
    level: WARNING
  file:
    enable: false
    level: WARNING
solver:
  type: turbine_grid
  turbine_grid_points: 3
farm:
  layout_x:
  - 0.0
  - 630.0
  - 1260.0
  layout_y:
  - 0.0
  - 0.0
  - 0.0
  turbine_type:
  - nrel_5MW
flow_field:
  air_density: 1.225
  reference_wind_height: -1
  turbulence_intensities:
  - 0.06
  wind_directions:
  - 270.0
  wind_shear: 0.12
  wind_speeds:
  - 8.0
  wind_veer: 0.0
  multidim_conditions:
    Tp: 2.5
    Hs: 3.01
wake:
  model_strings:
    combination_model: sosfs
    deflection_model: gauss
    turbulence_model: crespo_hernandez
    velocity_model: gauss
  enable_secondary_steering: true
  enable_yaw_added_recovery: true
  enable_active_wake_mixing: false
  enable_transverse_velocities: true
  wake_deflection_parameters:
    gauss:
      ad: 0.0
      alpha: 0.58
      bd: 0.0
      beta: 0.077
      dm: 1.0
      ka: 0.38
      kb: 0.004
    jimenez:
      ad: 0.0
      bd: 0.0
      kd: 0.05
  wake_velocity_parameters:
    cc:
      a_s: 0.179367259
      b_s: 0.0118889215
      c_s1: 0.0563691592
      c_s2: 0.13290157
      a_f: 3.11
      b_f: -0.68
      c_f: 2.41
      alpha_mod: 1.0
    gauss:
      alpha: 0.58
      beta: 0.077
      ka: 0.38
      kb: 0.004
    jensen:
      we: 0.05
  wake_turbulence_parameters:
    crespo_hernandez:
      initial: 0.1
      constant: 0.5
      ai: 0.8
      downstream: -0.32
"""

# gch.yaml 相对 default_inputs.yaml **允许**不同的项 = "算例项"（我们研究的问题）
CASE_KEYS = ('farm.layout_x', 'farm.layout_y', 'farm.turbine_type',
             'flow_field.turbulence_intensities', 'flow_field.wind_speeds',
             'flow_field.wind_directions', 'flow_field.reference_wind_height')
# 注意：air_density 1.225 == 随包默认；wind_shear 0.12 == 随包默认（不是差异项）

# 允许不同的**非算例**项（逐项给出理由，且必须满足"gch 侧缺失、defaults 侧有值"）：
#   * description/floris_version —— 纯元数据文字；
#   * multidim_conditions.{Tp,Hs} —— gch.yaml 多写的多维工况标签，只对
#     multi_dimensional_cp_ct 机型有意义（nrel_5MW 不是）；
#   * empirical_gauss / turbopark / turboparkgauss / wake_induced_mixing 的参数 ——
#     gch.yaml **没有为这些未使用的模型族写参数**，FLORIS 装载时补上自己的默认值。
#     它们只在切换模型族时才有意义（我们主臂用 gauss+crespo_hernandez+sosfs）。
META_KEYS = ('description', 'floris_version',
             'flow_field.multidim_conditions.Tp', 'flow_field.multidim_conditions.Hs')
UNUSED_MODEL_FAMILIES = ('empirical_gauss', 'turboparkgauss', 'turbopark',
                         'wake_induced_mixing')


def _ignorable_diff(key, a_val, b_val):
    """返回 (是否可忽略, 理由)；`None` 理由表示不可忽略。"""
    if key in CASE_KEYS:
        return True, '算例项'
    if key in META_KEYS:
        return True, '元数据/多维工况标签'
    if any(('.%s.' % fam) in key for fam in UNUSED_MODEL_FAMILIES):
        # 只有当 gch.yaml **根本没写**这一项（缺失 ⇒ 引擎补默认）时才算"未使用模型族"
        if a_val is None:
            return True, '未使用模型族（gch.yaml 未写该项）'
        return False, None
    return False, None


class Report:
    def __init__(self):
        self.rows = []
        self.fails = []

    def check(self, name, ok, detail=''):
        self.rows.append((name, bool(ok), detail))
        if not ok:
            self.fails.append(name)
        return bool(ok)

    def dump(self):
        w = max(len(r[0]) for r in self.rows) + 2
        print('| %-*s | %-6s | %s' % (w, '判据', '状态', '实测'))
        print('|%s|%s|%s' % ('-' * (w + 2), '-' * 8, '-' * 40))
        for name, ok, detail in self.rows:
            print('| %-*s | %-6s | %s' % (w, name, 'OK' if ok else '**FAIL**', detail))


def _flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = '%s.%s' % (prefix, k) if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _norm(v):
    """把 list / ndarray / numpy 标量归一成可比较的 Python 对象。"""
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_norm(x) for x in np.asarray(v).tolist()] if not isinstance(v, list) else [_norm(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return round(float(v), 12)
    if isinstance(v, float):
        return round(v, 12)
    return v


def _check_against_defaults(report):
    """内嵌 YAML vs 随包 default_inputs.yaml：只允许 CASE_KEYS 不同。"""
    from floris import FlorisModel
    yaml_path = os.path.join(_HERE, '_scratch', 'gch_official_v4.6.6.yaml')
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, 'w', encoding='utf-8') as fh:
        fh.write(GCH_YAML)

    a = _flatten(FlorisModel(yaml_path).get_param([]))
    b = _flatten(FlorisModel('defaults').get_param([]))
    diff = []
    n_ignored = 0
    for k in sorted(set(a) | set(b)):
        if _norm(a.get(k)) == _norm(b.get(k)):
            continue
        ok, _why = _ignorable_diff(k, a.get(k), b.get(k))
        if ok:
            n_ignored += 1
        else:
            diff.append((k, a.get(k), b.get(k)))
    report.check('内嵌 gch.yaml vs 随包 default_inputs.yaml：非算例项全一致',
                 not diff,
                 '%d 项差异（另有 %d 项可解释差异：算例项/元数据/未使用模型族）%s'
                 % (len(diff), n_ignored, (': ' + str(diff[:3])) if diff else ''))
    return yaml_path, a, b


def _run_arm_a(yaml_path):
    """臂 A：官方示例逐字复现（只去掉绘图）。"""
    from floris import FlorisModel, TimeSeries
    from floris.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR

    fmodel = FlorisModel(yaml_path)
    wind_directions = np.arange(0.0, 360.0, 3.0)
    time_series = TimeSeries(wind_directions=wind_directions, wind_speeds=8.0,
                             turbulence_intensities=0.06)
    fmodel.set(layout_x=[0.0, 5 * 126.0, 10 * 126.0], layout_y=[0.0, 0.0, 0.0],
               wind_data=time_series)
    df_opt = YawOptimizationSR(fmodel).optimize()
    return fmodel, df_opt


def _run_arm_b():
    """臂 B：本项目构造入口的**本地等价版**（同一算例，走 build_model）。"""
    from yawflor import yawflor_turbine as YT
    from floris import TimeSeries
    from floris.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR

    f2, info = YT.build_model(nt=3, rotor_diameter=126.0, spacing_d=5.0,
                             turbine_type='nrel_5MW', ti=0.06, wind_shear=0.12)
    wind_directions = np.arange(0.0, 360.0, 3.0)
    f2.set(wind_data=TimeSeries(wind_directions=wind_directions, wind_speeds=8.0,
                                turbulence_intensities=0.06))
    df_opt2 = YawOptimizationSR(f2).optimize()
    return f2, df_opt2, info


def main():
    ap = argparse.ArgumentParser(description='P0②：复现 FLORIS 官方 001_opt_yaw_single_ws')
    ap.add_argument('--skip-opt', action='store_true',
                    help='跳过 Serial-Refine 优化（只做参数核对；很快）')
    ap.add_argument('--json', default=None, help='结果落盘路径')
    args = ap.parse_args()

    import floris
    print('FLORIS %s' % floris.__version__)
    r = Report()

    # ---------- ① 内嵌 YAML 的忠实性（防写错/漏项）----------
    yaml_path, pa, pb = _check_against_defaults(r)
    print('内嵌官方 gch.yaml → %s' % yaml_path)

    # ---------- ② 臂 A vs 臂 B：参数级 ----------
    f2, info = None, None
    from yawflor import yawflor_turbine as YT          # noqa: E402
    f2, info = YT.build_model(nt=3, rotor_diameter=126.0, spacing_d=5.0,
                              turbine_type='nrel_5MW', ti=0.06, wind_shear=0.12)
    key_params = (['farm', 'layout_x'], ['farm', 'layout_y'], ['farm', 'turbine_type'],
                  ['flow_field', 'air_density'], ['flow_field', 'wind_shear'])
    same = all(_norm(pa['.'.join(k)]) == _norm(_norm2(f2, k)) for k in key_params)
    r.check('臂 B 关键参数 == 臂 A（layout/turbine_type/ρ/shear）', same)

    out = {'floris_version': floris.__version__,
           'arm_b_info': {k: (list(v) if isinstance(v, (list, tuple)) else v)
                          for k, v in info.items()}}

    if args.skip_opt:
        r.dump()
        print('\n(跳过 Serial-Refine)')
        _finish(r, args, out)
        return 1 if r.fails else 0

    # ---------- ③ 臂 A：官方示例 ----------
    fmodel, df = _run_arm_a(yaml_path)
    base = np.asarray(df['farm_power_baseline'], dtype=float)
    opt = np.asarray(df['farm_power_opt'], dtype=float)
    wd = np.asarray(df['wind_direction'], dtype=float)

    # 验收判据 ②：与官方页面打印值（未对齐风向 = 无尾流 = 3×单机）≤1%
    rel = abs(base.max() - PUBLISHED_UNALIGNED_W) / PUBLISHED_UNALIGNED_W
    r.check('官方页面锚 max(baseline) = %.6e W (±1%%)' % PUBLISHED_UNALIGNED_W,
            rel <= PUBLISHED_TOL, '实测 %.6e W（相对偏差 %.4f%%）' % (base.max(), rel * 100))

    i270 = int(np.argmin(np.abs(wd - 270.0)))
    yaw270 = np.asarray(df['yaw_angles_opt'].iloc[i270], dtype=float)
    r.check('270° 对齐工况：最优 ≥ 基线（转向确有收益）',
            opt[i270] >= base[i270] - 1e-6,
            'baseline=%.6f MW, opt=%.6f MW (+%.2f%%)'
            % (base[i270] / 1e6, opt[i270] / 1e6,
               100.0 * (opt[i270] / base[i270] - 1.0)))
    r.check('最优偏航角在官方优化器动作域内（|γ| ≤ 25°）',
            bool(np.all(np.abs(yaw270) <= 25.0 + 1e-9)),
            'γ* = %s' % np.round(yaw270, 3).tolist())

    # ---------- ④ 臂 B：本地等价版，与臂 A 逐位比较 ----------
    f2b, df2, info2 = _run_arm_b()
    base2 = np.asarray(df2['farm_power_baseline'], dtype=float)
    opt2 = np.asarray(df2['farm_power_opt'], dtype=float)
    r.check('基线功率 臂B ≡ 臂A（逐位）', bool(np.array_equal(base, base2)),
            'max|Δ| = %.3e W' % float(np.max(np.abs(base - base2))))
    r.check('最优功率 臂B ≡ 臂A（逐位）', bool(np.array_equal(opt, opt2)),
            'max|Δ| = %.3e W' % float(np.max(np.abs(opt - opt2))))
    yaw2 = np.vstack([np.asarray(x, dtype=float) for x in df2['yaw_angles_opt']])
    yaw1 = np.vstack([np.asarray(x, dtype=float) for x in df['yaw_angles_opt']])
    r.check('最优偏航角 臂B ≡ 臂A（逐位）', bool(np.array_equal(yaw1, yaw2)),
            'max|Δγ| = %.3e deg' % float(np.max(np.abs(yaw1 - yaw2))))

    out.update(dict(published_unaligned_w=PUBLISHED_UNALIGNED_W,
                    max_baseline_w=float(base.max()),
                    rel_dev=float(rel),
                    wd270_baseline_mw=float(base[i270] / 1e6),
                    wd270_opt_mw=float(opt[i270] / 1e6),
                    wd270_yaw_opt_deg=[float(x) for x in yaw270],
                    arm_b_equals_arm_a=bool(np.array_equal(base, base2)
                                            and np.array_equal(opt, opt2)
                                            and np.array_equal(yaw1, yaw2))))

    r.dump()
    print('\n270° 对齐：baseline = %.6f MW → opt = %.6f MW (+%.2f%%), γ* = %s'
          % (base[i270] / 1e6, opt[i270] / 1e6, 100 * (opt[i270] / base[i270] - 1),
             np.round(yaw270, 3).tolist()))
    _finish(r, args, out)
    return 1 if r.fails else 0


def _norm2(f, keys):
    """读 `f.get_param(keys)` 并归一（用于参数级比较）。"""
    return f.get_param(list(keys))


def _finish(r, args, out):
    print()
    if r.fails:
        print('❌ P0② 失败 %d 项：%s' % (len(r.fails), ', '.join(r.fails)))
    else:
        print('✅ P0② 全部通过（%d 项）——官方示例可复现、本项目构造入口无接错' % len(r.rows))
    if args.json:
        p = args.json if os.path.isabs(args.json) else os.path.join(ROOT, args.json)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w', encoding='utf-8') as fh:
            json.dump({'rows': [{'name': n, 'ok': o, 'detail': d} for n, o, d in r.rows],
                       'fails': r.fails, 'result': out},
                      fh, ensure_ascii=False, indent=2, default=str)
        print('结果已落盘：%s' % p)


if __name__ == '__main__':
    sys.exit(main())
