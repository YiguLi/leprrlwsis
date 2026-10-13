# -*- coding: utf-8 -*-
"""test9 的 **纯规则基线** —— 固定风机轴向因子 + RuleBasedBESS（SOC 目标恒 3.0）。

这条臂回答的问题
----------------
"**不用任何 RL 策略**，风机轴向因子钉在常数、BESS 交给规则，test9 九用例的
test9/overall/profit 是多少？" —— 即训练日志里 `df_test9.csv` 的 `OVERALL` 行、
TensorBoard 的 `test9/overall/profit` 所对应的那个标量，在**规则基线**上的取值。
任何 RL run 的 test9 overall profit 都必须相对这条线才有意义。

口径（逐项，全部复用生产代码，不重写）
------------------------------------
* 场景：`nine_case_test(seed=42, nolow=True)` —— 风速 LOW→MEDIUM→HIGH、段内电价
  LOW→MEDIUM→HIGH，每 case 5 步 = 1 个上层周期，共 45 步（与训练期 test9 同一场景）
* 上层动作：**固定** `a = [--a] × 5`（默认 0.33），不走 actor；
  由 `FixedActionAgent` 桩体提供（nolow 口径 ⇒ 5 维，无第 6 维 SOC 信号）
* 下层 BESS：**规则** —— `hst.rule_action(目标 3.0, SOC, 平抑窗口, n_rem)` ⇒
  "|δ| ≤ tol 时只平抑 / 否则全力朝 3.0 方向充放"，再由 `RuleBasedBESS.compute_pbrule`
  夹到平抑窗口 ∩ 物理窗口（`agentutil/test_nolow_branch.py` §A 钉住的语义）
* **SOC 目标恒 3.0**：nolow 分支里 `soc_target = soc_target_from_signal(0.0)`
  = `SOC_TARGET_CENTER` = 3.0 ⇒ "始终归为 soc=3"
* `--soc0` 可覆盖初始 SOC（默认 None = `env.ScenarioReset` 的 5.0）；
  即"起始 SOC 也钉在 3.0"这一读法可用 `--soc0 3` 复现
* 奖励/记录口径完全由 `nine_case_test` 给出：`profit = P_G·pr·dt − 490·max(−pbrule,0)·dt`
  （45 步求和 = overall profit；`profit_econ` 另按 `hc.PR_REF = 580` 结算 ΔSOC）
* `viol_derate_coef` 默认 0（与各 arm 的 `--viol_derate` 默认一致）
* 引擎：生产引擎 `env.WindBattery`（WFSim / MATLAB）—— **需 MATLAB**；
  `--stub-env` 为不需 MATLAB 的**自检**臂（数值非生产，仅验证接线）
* 纯前向：不经任何网络、不训练、不写 TensorBoard

输出（默认 `output/test9overallprofit/`）
----------------------------------------
  test9_cases.csv     9 个 case 的全部指标（含 profit / profit_econ / r1 分解）
  test9_overall.csv   45 步总量口径汇总（1 行）★ profit / profit_econ 主记录项
  test9_overall.json  同上 + 本次运行的元信息（动作值/引擎/seed/git commit/SOC 目标…）
  test_s_9case.csv    45 步逐步明细（动作、P_F、clip_amount、SOC、pb、上/下层奖励…）
  summary.txt         人读摘要（口径 + 关键读数 + profit 折日）
  events.out.tfevents.*  TensorBoard 事件（`--no-tensorboard` 关闭）

TensorBoard（`--tensorboard`，默认开）
--------------------------------------
输出目录本身就是 run 目录，tag 与各 `*_main.py` 训练期的写法**逐字一致**
（`test9/overall/{profit,profit_econ,r1,c1,r2,c2,gap_pen,cvm,pf_viol,dsoc,soc_end,
soc_track_err}` + 逐 case `test9/{WIND}_{PRICE}/{...}`）⇒

    tensorboard --logdir output

可把这条**基线臂**与各 RL run 的 `test9/overall/profit` 曲线放在同一张图里对比。
本臂没有训练 step ⇒ 默认在单点 `--tb-step 0` 落盘（可指定任意 step 以对齐某个 run）；
另记 main 未记的三个审计列 `turbine_profit / derate_loss / penalty_NN_H`。

用法
----
    python agentutil/run_test9_rule_baseline.py                      # 生产口径（WFSim）
    python agentutil/run_test9_rule_baseline.py --a 0.33 --soc0 3     # 起始 SOC 也钉 3.0
    python agentutil/run_test9_rule_baseline.py --stub-env           # 不需 MATLAB 的自检
    python agentutil/run_test9_rule_baseline.py --no-tensorboard     # 只出 CSV/JSON

⚠ TensorBoard 需要 `tensorboard` 包（本机在 `F:\\Anaconda\\envs\\wsis\\python.exe`）；
  缺包时用 `--no-tensorboard`，或换该解释器运行。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import importlib
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)                                         # 相对数据/输出路径以仓库根为基准
sys.path.insert(0, ROOT)                               # 项目模块
sys.path.insert(0, _HERE)                              # agentutil 内互相 import

import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402

import hsrl_soctarget as hst                           # noqa: E402  SOC 目标中心 = 3.0
import env as ENV                                      # noqa: E402  WindBattery（WFSim 路径）
from rbb.rbb_rule_bess import RuleBasedBESS            # noqa: E402  规则 BESS 单一真源

MAX_E = 6.0
NB_TURBINE = 5
TIME = 1.0 / 60.0
FLUCT_LIMIT = 3.0
RATED_POWER = 10.0
ETA_C = ETA_D = 0.98
SOC_TARGET = float(hst.SOC_TARGET_CENTER)              # = 3.0（nolow 分支恒定目标）

# 支持 nolow=True 语义的两个 myfunc（`nine_case_test(..., nolow=True)`）
PROJECTS = {
    'td3per': dict(label='TD3PER', myfunc='td3per.td3per_myfunc'),
    'td3perseg': dict(label='TD3PERSEG', myfunc='td3perseg.td3perseg_myfunc'),
}


# ---------------------------------------------------------------- 桩：固定动作上层
class FixedActionAgent:
    """恒定输出 5 维风机轴向因子的桩上层（nolow 口径：无第 6 维 SOC 信号）。

    只实现 `nine_case_test` 会调用的接口；`is_training` 会被 `nine_case_test` 置 False。
    """

    def __init__(self, action, nb_states=4):
        self.a = np.asarray(action, dtype=float).ravel().copy()
        self.nb_states = int(nb_states)
        self.nb_actions = int(self.a.size)
        self.is_training = False

    def select_action(self, obs=None, *args, **kwargs):
        return self.a.copy()

    def reset(self, obs=None, *args, **kwargs):
        self.s_t = obs

    def observe_jump(self, *args, **kwargs):
        pass

    def update_policy(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------- 桩环境（自检臂）
class StubEnv:
    """`--stub-env` 自检用：只实现 `nine_case_test` 需要的三个方法。

    ⚠ 物理是手写简化式（单机 Cp 线性代理），**不是** WFSim ⇒ 数值**不可**当生产读数，
    仅用于"接线是否通"的自检（与 `agentutil/test_nolow_branch.py` §C 同一桩体）。
    """

    def __init__(self, n_turbine=NB_TURBINE, E0=5.0):
        self.num_turbine = n_turbine
        self.max_E, self.min_E, self.eta, self.time = MAX_E, 0.0, ETA_C, TIME
        self.E0, self.E = E0, E0
        self._i, self._wind, self._price = 0, [], []

    def PowerOutput(self, u, act_wind):
        a = np.clip(np.asarray(act_wind, dtype=float).ravel()[:self.num_turbine], 0.0, 0.5)
        return float(a.sum() * 10.0 * min(max(float(u), 0.0), 12.0) / 12.0)

    def ScenarioReset(self, wind_seq, price_seq):
        self._wind, self._price, self._i = list(wind_seq), list(price_seq), 0
        self.E = self.E0
        pw = self.PowerOutput(self._wind[0], np.ones(self.num_turbine) * 0.3333)
        return np.array([float(self._price[0]), float(self._wind[0]), self.E, pw])

    def ScenarioStep(self, applied_action_H, act_bat, pw_tru):
        self._i = min(self._i + 1, len(self._wind) - 1)
        u, pr = float(self._wind[self._i]), float(self._price[self._i])
        pw2 = self.PowerOutput(u, applied_action_H)
        cmd = float(np.asarray(act_bat, dtype=float).ravel()[0])
        P_C, P_D = max(0.0, cmd), abs(min(cmd, 0.0))
        E_t = self.E + (P_C * self.eta - P_D / self.eta) * self.time
        if E_t > self.max_E:
            P_C = (self.max_E - self.E) / self.eta / self.time
            self.E = self.max_E
        elif E_t < self.min_E:
            P_D = (self.E - self.min_E) * self.eta / self.time
            self.E = self.min_E
        else:
            self.E = E_t
        P_G = float(pw_tru) - P_C + P_D
        return (pw2, np.array([pr, u, self.E, P_G]), np.array([P_C - P_D]),
                0.0, False, False, {'P_F': 0.0, 'pbrule': P_C - P_D, 'P_G': P_G})


def _git_head():
    try:
        p = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else ''
    except Exception:
        return ''


def _fmt(v, w=16, p=6):
    try:
        return '{:>{w}.{p}f}'.format(float(v), w=w, p=p)
    except (TypeError, ValueError):
        return '{:>{w}}'.format(str(v), w=w)


# ---------------------------------------------------------------- TensorBoard
# tag 清单与各 `*_main.py` 的训练期 test9 写法逐字一致 (见 td3per_main.py:665-689)
TB_OVERALL_TAGS = ['r1', 'c1', 'r2', 'c2', 'gap_pen', 'cvm', 'pf_viol',
                   'profit', 'profit_econ', 'dsoc', 'soc_end', 'soc_track_err']
TB_CASE_TAGS = ['r1', 'c1', 'r2', 'c2', 'gap_pen', 'cvm', 'pf_viol',
                'profit', 'soc_track_err']
TB_AUDIT_TAGS = ['turbine_profit', 'derate_loss', 'penalty_NN_H']   # main 未记的审计列


def write_tensorboard(out_dir, case_metrics, overall, step, seed, config_text):
    """把 test9 读数写成 TensorBoard 标量，返回事件文件路径（失败时返回 ''）。"""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:                                   # tensorboard 包缺失
        print('\n  ⚠ 跳过 TensorBoard（{}）；用 --no-tensorboard 可静默关闭'.format(exc))
        return ''
    writer = SummaryWriter(out_dir)
    writer.add_scalar('seed', int(seed), step)
    writer.add_text('config/run', config_text, step)
    for k in TB_OVERALL_TAGS + TB_AUDIT_TAGS:
        if k in overall:
            writer.add_scalar('test9/overall/{}'.format(k), float(overall[k]), step)
    for m in case_metrics:
        tag = 'test9/{}_{}'.format(m['wind_level'], m['price_level'])
        for k in TB_CASE_TAGS:
            if k in m:
                writer.add_scalar('{}/{}'.format(tag, k), float(m[k]), step)
    writer.flush()
    writer.close()
    files = sorted(glob.glob(os.path.join(out_dir, 'events.out.tfevents.*')))
    return files[-1] if files else ''

def main():
    ap = argparse.ArgumentParser(
        description='test9 纯规则基线：固定风机轴向因子 + RuleBasedBESS（SOC 目标恒 3.0）')
    ap.add_argument('--a', type=float, default=0.33,
                    help='固定的风机轴向因子（5 台相同；生产标称值 0.3333，用户口径 0.33）')
    ap.add_argument('--seed', type=int, default=42, help='test9 场景种子（生产 = 42）')
    ap.add_argument('--project', default='td3per', choices=sorted(PROJECTS),
                    help='提供 nine_case_test(nolow=True) 的项目模块（奖励/记录口径来源）')
    ap.add_argument('--soc0', type=float, default=None,
                    help='覆盖初始 SOC（默认 None = env.ScenarioReset 的 5.0）')
    ap.add_argument('--viol-derate', type=float, default=0.0,
                    help='越界乘性惩罚系数（生产默认 0 = 关闭；固定动作全在带内 ⇒ 恒 1.0）')
    ap.add_argument('--stub-env', action='store_true',
                    help='用简化桩环境代替 MATLAB/WFSim（自检用，数值非生产）')
    ap.add_argument('--tensorboard', dest='tensorboard', action='store_true', default=True,
                    help='写 TensorBoard 事件（默认开；tag 与各 main 的训练期 test9 逐字一致）')
    ap.add_argument('--no-tensorboard', dest='tensorboard', action='store_false',
                    help='不写 TensorBoard（缺 tensorboard 包时用）')
    ap.add_argument('--tb-step', type=int, default=0,
                    help='TensorBoard 落点 step（本臂无训练 ⇒ 默认单点 0；可指定以对齐某个 run）')
    ap.add_argument('--output', default=None,
                    help='输出目录（默认 output/test9overallprofit）')
    args_cli = ap.parse_args()

    spec = PROJECTS[args_cli.project]
    out_dir = os.path.abspath(args_cli.output or os.path.join('output', 'test9overallprofit'))
    os.makedirs(out_dir, exist_ok=True)
    action = [float(args_cli.a)] * NB_TURBINE

    print('=' * 92)
    print('  test9 纯规则基线（固定轴向因子 + RuleBasedBESS）')
    print('=' * 92)
    print('  上层动作 a          = {}（固定，不走 RL）'.format(action))
    print('  下层                = 规则 hst.rule_action(目标 {:.1f}) → RuleBasedBESS 夹窗'
          .format(SOC_TARGET))
    print('  初始 SOC            = {}'.format('默认（env.ScenarioReset = 5.0）'
                                             if args_cli.soc0 is None
                                             else '{:.3f}（--soc0 覆盖）'.format(args_cli.soc0)))
    print('  场景                = nine_case_test(seed={}, nolow=True)，45 步 / 9 case'
          .format(args_cli.seed))
    print('  口径来源            = {}.nine_case_test'.format(spec['myfunc']))
    print('  输出目录            = {}'.format(out_dir))

    # ---- 环境（生产 = WFSim/MATLAB；自检 = 桩）----
    myf = importlib.import_module(spec['myfunc'])
    engine = None
    if args_cli.stub_env:
        env_test = StubEnv(E0=(5.0 if args_cli.soc0 is None else float(args_cli.soc0)))
        engine_label = 'stub（自检；非生产数值）'
        print('\n  [--stub-env] 使用简化桩环境，不启动 MATLAB')
    else:
        import matlab.engine
        print('\n  启动 MATLAB 引擎 ...')
        t_start = _dt.datetime.now()
        engine = matlab.engine.start_matlab()
        print('  MATLAB 引擎就绪（{:.1f}s）；env_test = env.WindBattery（WFSim_port）'
              .format((_dt.datetime.now() - t_start).total_seconds()))
        env_test = ENV.WindBattery(MAX_E, NB_TURBINE, False, engine)
        engine_label = 'WFSim (MATLAB WFSim_port)'

    # ---- --soc0：把初始 SOC 也钉住（env.ScenarioReset 内部硬写 5.0）----
    if (args_cli.soc0 is not None) and (not args_cli.stub_env):
        _orig_reset = env_test.ScenarioReset

        def _reset_with_soc0(wind_seq, price_seq, _orig=_orig_reset,
                             _env=env_test, _soc=float(args_cli.soc0)):
            obs = _orig(wind_seq, price_seq)
            _env.E = _soc
            obs[2] = _soc
            return obs

        env_test.ScenarioReset = _reset_with_soc0

    # ---- 规则下层（参数与 td3per/td3perseg main 逐项一致）----
    rbb_system = RuleBasedBESS(max_E=MAX_E, min_E_factor=0.0, rated_power=RATED_POWER,
                               eta_c=ETA_C, eta_d=ETA_D, time=TIME,
                               fluctuation_limit=FLUCT_LIMIT)

    agent_H = FixedActionAgent(action, nb_states=4)

    # ---- 跑一次 test9 ----
    t0 = _dt.datetime.now()
    case_metrics, overall = myf.nine_case_test(
        agent_H, None, env_test, rbb_system, seed=args_cli.seed,
        test_csv=True, output=out_dir,
        phase_mode=0, phase_table=None,
        viol_derate_coef=args_cli.viol_derate,
        nolow=True)
    dt = (_dt.datetime.now() - t0).total_seconds()
    if engine is not None:
        engine.quit()

    # ---- 落盘 ----
    df_cases = pd.DataFrame(case_metrics)
    df_overall = pd.DataFrame([overall])
    df_cases.to_csv(os.path.join(out_dir, 'test9_cases.csv'), index=False)
    df_overall.to_csv(os.path.join(out_dir, 'test9_overall.csv'), index=False)

    # ---- TensorBoard（tag 与各 main 的训练期 test9 逐字一致）----
    tb_event = ''
    if args_cli.tensorboard:
        tb_event = write_tensorboard(
            out_dir, case_metrics, overall, int(args_cli.tb_step), int(args_cli.seed),
            config_text=('test9 纯规则基线 ｜ a = {} (固定) ｜ 下层 = RuleBasedBESS + '
                         'rule_action(目标 {:.1f}) ｜ 引擎 = {} ｜ seed = {} ｜ git = {}'
                         .format(action, SOC_TARGET, engine_label, args_cli.seed,
                                 _git_head()[:8])))
        if tb_event:
            print('\n  TensorBoard: {}（step = {}；tag = test9/overall/* 等，'
                  '与各 run 同口径 ⇒ `tensorboard --logdir output` 可直接对比）'
                  .format(tb_event, args_cli.tb_step))

    meta = {
        'experiment': 'test9overallprofit（纯规则基线）',
        'engine': engine_label,
        'project': args_cli.project,
        'label': spec['label'],
        'reference': '{}.nine_case_test(nolow=True)'.format(spec['myfunc']),
        'upper_action': action,
        'upper_action_source': 'fixed (no RL)',
        'lower_controller': 'RuleBasedBESS + hsrl_soctarget.rule_action',
        'soc_target': SOC_TARGET,
        'soc_start': (5.0 if args_cli.soc0 is None else float(args_cli.soc0)),
        'scenario_seed': int(args_cli.seed),
        'viol_derate': float(args_cli.viol_derate),
        'rbb': {'max_E': MAX_E, 'min_E_factor': 0.0, 'rated_power': RATED_POWER,
                'eta_c': ETA_C, 'eta_d': ETA_D, 'time': TIME,
                'fluctuation_limit': FLUCT_LIMIT},
        'tensorboard': {'written': bool(tb_event), 'step': int(args_cli.tb_step),
                        'event_file': os.path.basename(tb_event)},
        'git_commit': _git_head(),
        'timestamp': _dt.datetime.now().isoformat(timespec='seconds'),
        'elapsed_s': round(dt, 1),
        'overall': {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool)
                        else v) for k, v in overall.items()},
    }
    with open(os.path.join(out_dir, 'test9_overall.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    # ---- 打印 ----
    cols = ['wind_level', 'price_level', 'profit', 'profit_econ', 'dsoc',
            'soc_start', 'soc_end', 'r1', 'turbine_profit', 'derate_loss',
            'penalty_NN_H', 'c1', 'cvm', 'pf_viol', 'soc_track_err']
    cols = [c for c in cols if c in df_cases.columns]
    print('\n' + '=' * 120)
    print('  ① per-case 明细（seed={}，每个 case 5 步；profit/profit_econ/turbine_profit/'
          'derate_loss/penalty_NN_H 均为 5 步之和）'.format(args_cli.seed))
    print('=' * 120)
    with pd.option_context('display.width', 220, 'display.max_columns', 60,
                           'display.float_format', lambda v: '{:.4f}'.format(v)):
        print(df_cases[cols].to_string(index=False))

    lines = []
    lines.append('test9 纯规则基线（固定 a={} + RuleBasedBESS，SOC 目标恒 {:.1f}）'
                 .format(action, SOC_TARGET))
    lines.append('引擎 = {} ｜ 场景 seed = {} ｜ 口径 = {}.nine_case_test(nolow=True)'
                 .format(engine_label, args_cli.seed, spec['myfunc']))
    lines.append('初始 SOC = {} ｜ viol_derate = {} ｜ git = {}'
                 .format(meta['soc_start'], args_cli.viol_derate, meta['git_commit'][:8]))
    lines.append('')
    lines.append('★ test9/overall/profit（45 步总量，主记录项）')
    for k in ['profit', 'profit_econ', 'dsoc', 'soc_start', 'soc_end',
              'r1', 'turbine_profit', 'derate_loss', 'penalty_NN_H',
              'c1', 'cvm', 'pf_viol', 'soc_track_err']:
        if k in overall:
            lines.append('    {:>16s} = {}'.format(k, _fmt(overall[k])))
    lines.append('')
    lines.append('    profit 折算 = {} 元/天（×1440/45）'
                 .format(_fmt(float(overall['profit']) * 1440.0 / 45.0, p=3)))
    lines.append('    45 步 profit 校验: Σcase − overall = {}'
                 .format(_fmt(abs(float(df_cases['profit'].sum()) - float(overall['profit'])),
                               p=12)))
    lines.append('    耗时 = {:.1f}s ｜ 输出目录 = {}'.format(dt, out_dir))
    lines.append('    TensorBoard = {}'
                 .format('{}（step = {}）'.format(os.path.basename(tb_event), args_cli.tb_step)
                         if tb_event else '未写出'))
    txt = '\n'.join(lines)

    print('\n' + '=' * 120)
    print('  ② OVERALL（45 步总量口径）★ profit 主记录项')
    print('=' * 120)
    print('\n'.join(lines[4:]))

    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')

    print('\n  输出目录: {}'.format(out_dir))
    print('    test9_cases.csv / test9_overall.csv / test9_overall.json / '
          'test_s_9case.csv / summary.txt{}'.format(' / ' + os.path.basename(tb_event)
                                                    if tb_event else ''))


if __name__ == '__main__':
    main()
