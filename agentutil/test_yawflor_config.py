#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_yawflor_config.py —— `yawflor` 新环境的**口径冻结契约**（只读，不需 MATLAB）。

为什么需要这个脚本
------------------
调研 11 篇已发表论文的结论是：**绝大多数口径项论文根本不报**
（FLORIS 版本号 4 篇经典论文只有 1 篇给；湍流模型 / 叠加模型 / 三个 `enable_*` 开关 /
空气密度 / 偏航速率 / 更新周期 / 稳态声明 几乎全部缺失，见
`agentutil/floris_reproducibility_survey.md`）。于是"我们的口径从哪来"不能靠引用回答，
只能靠**三条**回答：

1. **取引擎默认**：论文不报，恰恰因为它们用的是引擎默认值 —— 谁改了默认值就必须写
   （否则结果对不上、审稿人会问）。所以"模型口径"一行都取
   `floris/default_inputs.yaml`（`name: GCH`）与官方 `examples/inputs/gch.yaml` 的原值。
2. **算例设定自己定并声明**：布局 / D / 风速序列 / TI / ρ / 动作域 / 更新周期属于
   "我们研究的问题"，不属模型口径 —— 取本项目 WFSim 口径并在论文里写明。
3. **不能引用就量化**：每个未报告参数都做"偏离它值多少"的扫描（探针第 7 节的模型敏感性、
   `--wind-shear`、间距扫描等）。**结果不敏感的取默认即可；敏感的（尾流模型 ±50pp、
   切变 10pp、间距 36pp）必须主动交代并选默认。**

本脚本把第 1、2 条变成**可执行断言**：任何人改动口径都会在这里响警报，
并把当时的完整口径快照落盘（复现性artifact）。

用法
----
    python agentutil/test_yawflor_config.py            # 校验并打印表（任一不符 ⇒ 退出码 1）
    python agentutil/test_yawflor_config.py --json output/yawflor_config_snapshot.json
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

import probe_floris_yaw_structure as P                  # noqa: E402

# =============================================================================
# 冻结口径（单一真源）。`tier` 说明"凭什么定成这个值"：
#   官方默认 = 随包 default_inputs.yaml / 官方 gch.yaml / turbine_library 的原值
#   本项目   = WFSim 算例口径（meshing.m / WFSim_port.m），论文需写明
#   声明项   = 明确"不建模"，没有值可定，只有一句话
# =============================================================================
EXPECTED = {
    # ---- 引擎 ----
    'floris_version_prefix': ('4.6', '官方默认'),
    # ---- 模型口径（Tier 1）----
    'velocity_model': ('gauss', '官方默认'),
    'deflection_model': ('gauss', '官方默认'),
    'turbulence_model': ('crespo_hernandez', '官方默认'),
    'combination_model': ('sosfs', '官方默认'),
    'enable_secondary_steering': (True, '官方默认'),
    'enable_yaw_added_recovery': (True, '官方默认'),
    'enable_transverse_velocities': (True, '官方默认'),
    'enable_active_wake_mixing': (False, '官方默认'),
    'solver_type': ('turbine_grid', '官方默认'),
    'turbine_grid_points': (3, '官方默认'),
    'operation_model': ('cosine-loss', '官方默认'),
    'cosine_loss_exponent_yaw': (1.88, '官方默认（turbine_library 全部机型同值）'),
    'cosine_loss_exponent_tilt': (1.88, '官方默认'),
    # ---- 算例设定（Tier 2，本项目 WFSim 口径）----
    'air_density': (1.20, '本项目（meshing.m Rho=1.20；官方 1.225）'),
    'wind_shear': (0.12, '官方默认（gch.yaml / default_inputs.yaml）'),
    'ti': (0.06, '官方示例 gch.yaml'),
    'wind_direction': (270.0, '官方示例 gch.yaml'),
    'rotor_diameter': (100.0, '本项目（meshing.m Drotor=100）'),
    'hub_height': (100.0, '本项目（2D 模型无轮毂高，取 D）'),
    'TSR': (8.0, '官方参考机同值'),
    'ref_tilt': (0.0, '本项目（2D 无倾角）'),
    'layout_x': ([0.0, 500.0, 1000.0, 1500.0, 2000.0], '本项目（meshing.m: 5 台 5D 单排）'),
    'layout_y': ([0.0] * 5, '本项目（单排）'),
    'thrust_coefficient': (8.0 / 9.0, '本项目（= Cp/(1−a), a=1/3）'),
    'a_induced': (1.0 / 3.0, '本项目（Betz 单机最优点，固定）'),
    'power_scale': (0.95, '本项目（meshing.m powerscale=0.95）'),
    'gamma_max_deg': (25.0, '惯例（Fleming 2017 硬安全限 / King 2021 / Howland 2022）'),
    'power_table_max_u': (40.0, '本项目（必须 > env 守卫上界 25 m/s）'),
    # ---- 声明项（Tier 3）：没有值可定，只有决定 ----
    '_declared': ({
        'yaw_rate_max_deg_per_block': 'None（默认关；对照臂按 Wang 2026 开并报占空比）',
        'yaw_tau_s': '0（默认关）',
        'yaw_transport_delay': 'off（默认关；开启会使 env 有记忆，破坏 test9 的确定性判据）',
        'rated_power_cap': '无（WFSim 口径，power ∝ u³ 不截断）',
        'loads_fatigue': '不建模',
        'steady_state_statement': '准稳态；不建模偏航执行器动力学、尾流输运延迟与结构载荷',
    }, '声明项'),
}


class Report:
    def __init__(self):
        self.rows = []
        self.fails = []

    def check(self, name, actual, expected, tier, tol=0.0, note=''):
        if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple, np.ndarray)):
            ok = len(actual) == len(expected) and all(
                abs(float(a) - float(b)) <= tol for a, b in zip(actual, expected))
        elif isinstance(expected, float):
            ok = abs(float(actual) - float(expected)) <= max(tol, 1e-12)
        elif isinstance(expected, str) and name.startswith('floris_version'):
            ok = str(actual).startswith(expected)          # 版本号按前缀比对
        else:
            ok = actual == expected
        self.rows.append((name, actual if not isinstance(actual, np.ndarray) else list(actual),
                          expected, tier, ok, note))
        if not ok:
            self.fails.append(name)
        return ok

    def info(self, name, actual, tier, note=''):
        self.rows.append((name, actual, '—', tier, True, note))

    def dump(self):
        w = max(len(r[0]) for r in self.rows) + 2
        print('| %-*s | %-34s | %-30s | %-6s |' % (w, '口径项', '实测值', '冻结值', '状态'))
        print('|%s|%s|%s|%s|' % ('-' * (w + 2), '-' * 36, '-' * 32, '-' * 8))
        for name, actual, expected, tier, ok, _ in self.rows:
            a = str(actual)
            if len(a) > 34:
                a = a[:31] + '...'
            e = str(expected)
            if len(e) > 30:
                e = e[:27] + '...'
            print('| %-*s | %-34s | %-30s | %-6s |' % (w, name, a, e, 'OK' if ok else '**FAIL**'))


def main():
    ap = argparse.ArgumentParser(description='yawflor 口径冻结契约')
    ap.add_argument('--json', default=None, help='口径快照落盘路径（复现性 artifact）')
    args = ap.parse_args()

    import floris
    from floris import FlorisModel

    P.set_wind_shear(EXPECTED['wind_shear'][0])
    f, info = P.build_model()

    r = Report()
    gp = f.get_param

    # ---------- Tier 1：模型口径 ----------
    ver = floris.__version__
    r.check('floris_version', ver, EXPECTED['floris_version_prefix'][0], '官方默认',
            note='前缀比对（记全版本号 %s）' % ver)
    ms = gp(['wake', 'model_strings'])
    for k in ('velocity_model', 'deflection_model', 'turbulence_model', 'combination_model'):
        r.check('wake.%s' % k, ms[k], EXPECTED[k][0], EXPECTED[k][1])
    for k in ('enable_secondary_steering', 'enable_yaw_added_recovery',
              'enable_transverse_velocities', 'enable_active_wake_mixing'):
        r.check('wake.%s' % k, bool(gp(['wake', k])), EXPECTED[k][0], EXPECTED[k][1])
    sol = gp(['solver'])
    r.check('solver.type', sol['type'], EXPECTED['solver_type'][0], EXPECTED['solver_type'][1])
    r.check('solver.turbine_grid_points', int(sol['turbine_grid_points']),
            EXPECTED['turbine_grid_points'][0], EXPECTED['turbine_grid_points'][1])

    # ---------- Tier 2：算例设定 ----------
    ff = gp(['flow_field'])
    r.check('flow_field.air_density', float(gp(['flow_field', 'air_density'])),
            EXPECTED['air_density'][0], EXPECTED['air_density'][1], tol=1e-12)
    r.check('flow_field.wind_shear', float(gp(['flow_field', 'wind_shear'])),
            EXPECTED['wind_shear'][0], EXPECTED['wind_shear'][1], tol=1e-12)
    r.check('layout_x', gp(['farm', 'layout_x']), EXPECTED['layout_x'][0],
            EXPECTED['layout_x'][1], tol=1e-9)
    r.check('layout_y', gp(['farm', 'layout_y']), EXPECTED['layout_y'][0],
            EXPECTED['layout_y'][1], tol=1e-9)

    td = gp(['farm', 'turbine_type'])[0]
    pt = td['power_thrust_table']
    r.check('turbine.rotor_diameter', float(td['rotor_diameter']), EXPECTED['rotor_diameter'][0],
            EXPECTED['rotor_diameter'][1], tol=1e-9)
    r.check('turbine.hub_height', float(td['hub_height']), EXPECTED['hub_height'][0],
            EXPECTED['hub_height'][1], tol=1e-9)
    r.check('turbine.TSR', float(td['TSR']), EXPECTED['TSR'][0], EXPECTED['TSR'][1], tol=1e-9)
    r.check('turbine.operation_model', td['operation_model'], EXPECTED['operation_model'][0],
            EXPECTED['operation_model'][1])
    r.check('table.ref_air_density', float(pt['ref_air_density']), EXPECTED['air_density'][0],
            EXPECTED['air_density'][1], tol=1e-12)
    r.check('table.ref_tilt', float(pt['ref_tilt']), EXPECTED['ref_tilt'][0],
            EXPECTED['ref_tilt'][1], tol=1e-12)
    r.check('table.cosine_loss_exponent_yaw', float(pt['cosine_loss_exponent_yaw']),
            EXPECTED['cosine_loss_exponent_yaw'][0], EXPECTED['cosine_loss_exponent_yaw'][1],
            tol=1e-12)
    r.check('table.cosine_loss_exponent_tilt', float(pt['cosine_loss_exponent_tilt']),
            EXPECTED['cosine_loss_exponent_tilt'][0], EXPECTED['cosine_loss_exponent_tilt'][1],
            tol=1e-12)

    ws = np.asarray(pt['wind_speed'], dtype=float)
    pw = np.asarray(pt['power'], dtype=float)            # kW
    ct = np.asarray(pt['thrust_coefficient'], dtype=float)
    r.check('table.wind_speed 上界', float(ws.max()), EXPECTED['power_table_max_u'][0],
            EXPECTED['power_table_max_u'][1], tol=1e-9)
    r.check('table.Ct 恒值', [float(ct.min()), float(ct.max())],
            [EXPECTED['thrust_coefficient'][0], EXPECTED['thrust_coefficient'][0]],
            EXPECTED['thrust_coefficient'][1], tol=1e-12)
    # 功率表必须等于解析式 Cp_eff·½ρA·u³（kW）—— 这是"理想化 Betz 执行器盘"的定义
    ref_kw = P.cp_analytic(ws) * 1e3
    r.check('table.power ≡ 解析 Cp 表', float(np.max(np.abs(pw - ref_kw))), 0.0,
            '本项目（a≡1/3）', tol=1e-9, note='最大绝对偏差 kW')

    # ---------- 陷阱检查（静默失效类）----------
    # ① reference_wind_height 必须 == 轮毂高，否则切变下整场电平静默偏移
    r.check('flow_field.reference_wind_height == hub_height',
            float(gp(['flow_field', 'reference_wind_height'])), float(td['hub_height']),
            '陷阱①', tol=1e-9, note='不等 ⇒ shear 下功率静默偏移')
    # ② 流场密度必须 == 表参考密度，否则功率被静默缩放 (ρ/ρ_ref)
    r.check('air_density == table.ref_air_density',
            float(gp(['flow_field', 'air_density'])), float(pt['ref_air_density']),
            '陷阱②', tol=1e-12, note='不等 ⇒ P 静默 ×ρ/ρ_ref')
    # ③ 功率表必须覆盖 env 守卫区间 [3.5, 25]，否则 interp1d(fill_value=0) 把高风速截成 0
    r.check('table 覆盖 env 守卫上界 25 m/s', float(ws.max()) >= 25.0, True,
            '陷阱③', note='否则高风速被静默截为 0 出力')
    r.check('table 最小风速 ≤ 3.5', float(ws.min()) <= 3.5, True, '陷阱③')
    # ④ 单机标定：**必须分开两个口径**，否则会被自己的不变量骗到
    #    (a) 功率表本身 ≡ 解析 Cp 表（已在上面断言，偏差 0.0）—— 表的自变量是"转子等效风速"
    #    (b) shear = 0 时，隔离机组的功率 ≡ 轮毂风速的解析式（逐位级）
    #    (c) shear = 0.12（官方口径）时**不再逐位相等**：FLORIS 用 u_eff = cbrt(mean(u³))
    #        在转子盘上对切变剖面做平均，而解析式用的是轮毂风速 ⇒ 实测比值 0.99557。
    #        这不是错误，而是"轮毂风速 ≠ 转子等效风速"的物理事实 ⇒ 只报告不断言。
    f0, _ = P.build_model(nt=1, wind_shear=0.0)
    p0, _ = P.eval_batch(f0, [10.0], np.zeros((1, 1)))
    r.check('单机标定 vs 轮毂解析式（shear=0, u=10, MW）', float(p0[0]),
            float(P.cp_analytic(10.0)), '本项目', tol=1e-12)
    f1, _ = P.build_model(nt=1)
    p1, _ = P.eval_batch(f1, [10.0], np.zeros((1, 1)))
    ratio = float(p1[0] / P.cp_analytic(10.0))
    r.info('单机/轮毂解析式 比值（shear=0.12）', '%.6f（= u_eff/u_hub = %.6f）'
           % (ratio, ratio ** (1.0 / 3.0)), '物理',
           note='切变下转子等效风速 < 轮毂风速 ⇒ 只报告不断言')
    # ⑤ 确定性（无记忆）
    y = np.zeros((1, 5))
    a1, _ = P.eval_batch(f, [10.0], y)
    a2, _ = P.eval_batch(f, [10.0], y)
    r.check('可重复性（逐位相同）', bool(np.array_equal(a1, a2)), True, '契约')
    # ⑥ 已知数值锚（与 output/probe_yaw_shear012.log 对齐）：u=10, γ=0
    r.check('u=10 γ=0 总功率（MW）', float(a1[0]), 5.9423, '锚', tol=5e-4)

    r.info('动作域 γ_max', '±%.0f°' % EXPECTED['gamma_max_deg'][0], EXPECTED['gamma_max_deg'][1])
    r.info('更新周期', 'H 每 5 步 = 5 min（文献现场惯例 10 min~2.5 h）', '惯例')
    for k, v in EXPECTED['_declared'][0].items():
        r.info('声明.%s' % k, v, '声明项')

    r.dump()
    print()
    if r.fails:
        print('❌ 口径契约失败 %d 项：%s' % (len(r.fails), ', '.join(r.fails)))
    else:
        print('✅ 口径契约全部通过（%d 项）—— 改动任何口径都会在此触发' % len(r.rows))

    if args.json:
        out = args.json if os.path.isabs(args.json) else os.path.join(ROOT, args.json)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        snap = {
            'floris_version': ver,
            'rows': [{'name': n, 'actual': (list(a) if isinstance(a, np.ndarray) else a),
                      'expected': e, 'tier': t, 'ok': bool(ok)} for n, a, e, t, ok, _ in r.rows],
            'declared': EXPECTED['_declared'][0],
            'info': info,
        }
        with open(out, 'w', encoding='utf-8') as fh:
            json.dump(snap, fh, ensure_ascii=False, indent=2, default=str)
        print('口径快照已落盘：%s' % out)
    return 1 if r.fails else 0


if __name__ == '__main__':
    sys.exit(main())
