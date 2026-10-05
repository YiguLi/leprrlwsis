# -*- coding: utf-8 -*-
"""在 **WFSim（MATLAB）引擎** 上，对一个已训练 run 跑一次生产口径的 9 用例测试（test9）。

动机
----
`*_main.py` 训练期每 480 步调一次 `nine_case_test(..., env_test=...)`，而 `env_test`
用的是**该项目自己的物理引擎**（`aifflor` = FLORIS，纯 Python）。本脚本把 `env_test`
换成 `env.WindBattery`（**WFSim / MATLAB**），其余接线与 `*_main.py` 的调用点逐项对齐
⇒ 得到"**同一策略、同一 test9 场景、不同引擎**"的读数，可直接与该项目 run 目录里
`df_test9.csv` 的 FLORIS 读数对照。

口径对齐（逐项，均取自生产代码，不重写）
--------------------------------------
* 场景：`nine_case_test(seed=42)` —— 风速 LOW→MEDIUM→HIGH，段内电价 LOW→MEDIUM→HIGH，
  每 case 5 步 = 1 个上层周期，共 45 步（与训练期 test9 同一场景）
* 相位：`Normalize_H(obs, rep_minute_for_price_level(pr), phase_mode, phase_table)`；
  `phase_mode` 由 `actor_6.pkl` 的观测维数自动探测（6 = 4 + 2 相位）
* 动作/奖励/下层：完全复用项目模块（`CheckAction` / `Cal_Rew_RB` / `hst.rule_action`
  经 `build_L_obs_distilled` + `lower_reward`/`lower_cost_soc_track`），未重写任何口径
* `viol_derate_coef` 默认 0（与当前各 arm 的 `--viol_derate` 默认一致）
* 隐层宽度从 `actor_6.pkl` / `actor_1.pkl` 反推（不写死）
* 纯前向：`is_training=False`，不更新网络、不写 TensorBoard
* 初始 SOC 由 `env.ScenarioReset` 决定（= 生产 test9 的 5.0），与五天线口径不同

⚠ 适用范围：只有**动作语义为 5 维轴向因子 `a∈[0,0.5]`** 的项目能这样换引擎
（`aifflor` / `td3per`）。`yawflor` 的动作是偏航角 `∈[-1,1]`，不能喂给 WFSim。

输出（默认 `output/<run 名>_test9_wfsim/`）
-----------------------------------------
  test9_cases.csv     9 个 case 的全部指标（含 profit / profit_econ / r1 三项分解）
  test9_overall.csv   45 步总量口径汇总（1 行）★ profit / profit_econ 主记录项
  test9_overall.json  同上（JSON，便于程序化引用，含 run/引擎/场景元信息）
  test_s_9case.csv    45 步逐步明细（动作、P_F、clip_amount、SOC、pb、上/下层奖励…）

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\run_test9_wfsim.py ^
        --run output/AIFFLOR-run59 --project aifflor --seed 42

    # 与生产训练同一场景的对照（同 run 的 FLORIS 读数）：
    #   output/AIFFLOR-run59/df_test9.csv 里 wind_level=price_level=OVERALL 的最后一行
"""
from __future__ import annotations

import argparse
import datetime as _dt
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
import torch                                           # noqa: E402

import hsrl_cost as hc                                 # noqa: E402  H 层 c1/λ 口径
import hsrl_soctarget as hst                           # noqa: E402  下层 r2/c2 单一真源
import env as ENV                                      # noqa: E402  WindBattery（WFSim 路径）
import matlab.engine                                   # noqa: E402
from rbb.rbb_rule_bess import RuleBasedBESS            # noqa: E402

MAX_E = 6.0
NB_TURBINE = 5
TIME = 1.0 / 60.0
FLUCT_LIMIT = 3.0
RATED_POWER = 10.0
ETA_C = ETA_D = 0.98

# 与 test_five_days.py 的 PROJECTS 注册表同形（只列**轴向因子**语义的项目）
PROJECTS = {
    'aifflor': dict(label='AIFFLOR',
                    agent=('aifflor.aifflor_ddpg', 'SP_TD3'),
                    myfunc='aifflor.aifflor_myfunc',
                    ref='aifflor/aifflor_main.py'),
    'td3per': dict(label='TD3PER',
                   agent=('td3per.td3per_ddpg', 'SP_TD3'),
                   myfunc='td3per.td3per_myfunc',
                   ref='td3per/td3per_main.py'),
}

# 相位观测维数 → phase_mode（与 test_five_days._DIM_TO_MODE 同表）
_DIM_TO_MODE = {0: 0, 2: 1, 5: 2}


def _git_head():
    try:
        p = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else ''
    except Exception:
        return ''


def _load_shape(path):
    sd = torch.load(path, map_location='cpu')
    if not isinstance(sd, dict):
        raise SystemExit('{}: 不是 state_dict'.format(path))
    for k in ('fc1.weight', '0.weight'):
        if k in sd:
            return tuple(int(v) for v in sd[k].shape)
    for v in sd.values():
        if hasattr(v, 'shape') and len(v.shape) == 2:
            return tuple(int(v) for v in v.shape)
    raise SystemExit('{}: 找不到 fc1.weight'.format(path))


def build_args(seed, hidden):
    """SP_TD3 构造函数读的 args 字段（口径 = 各 *_main.py 的当前默认值）。"""
    ns = argparse.Namespace()
    ns.hidden1, ns.hidden2 = int(hidden[0]), int(hidden[1])
    ns.rate = 0.01
    ns.prate = 0.0001
    ns.warmup = 200
    ns.discount = 0.95
    ns.bsize = 256
    ns.rmsize = 40000
    ns.window_length = 1
    ns.tau = 0.001
    ns.ou_theta = 0.15
    ns.ou_sigma = 0.2
    ns.ou_mu = 0.0
    ns.validate_episodes = 20
    ns.max_episode_length = 1440
    ns.validate_steps = 2000
    ns.output = 'output'
    ns.debug = False
    ns.init_w = 0.003
    ns.train_iter_H = 1000 * 1440
    ns.epsilon = 4000
    ns.seed = int(seed)
    ns.resume = 'default'
    ns.cuda = bool(torch.cuda.is_available())
    ns.lam = 3.0
    ns.lr_lam = 0.1
    ns.cost_single_limit = 1000.0
    ns.PER_alpha = 0.6
    ns.PER_beta = 0.4
    ns.PER_epsilon = 0.01
    ns.beta_increment_per_sampling = 3e-5
    ns.nb_turbine = NB_TURBINE
    return ns


def main():
    ap = argparse.ArgumentParser(description='WFSim 引擎上的 test9（9 用例）')
    ap.add_argument('--run', required=True, help='模型目录，如 output/AIFFLOR-run59')
    ap.add_argument('--project', default='aifflor', choices=sorted(PROJECTS))
    ap.add_argument('--seed', type=int, default=42, help='test9 场景种子（生产=42）')
    ap.add_argument('--viol-derate', type=float, default=0.0,
                    help='越界乘性惩罚系数（生产默认 0 = 关闭）')
    ap.add_argument('--output', default=None, help='输出目录（默认 output/<run>_test9_wfsim）')
    args_cli = ap.parse_args()

    run_dir = os.path.abspath(args_cli.run)
    if not os.path.isdir(run_dir):
        raise SystemExit('模型目录不存在: {}'.format(run_dir))
    spec = PROJECTS[args_cli.project]
    out_dir = os.path.abspath(args_cli.output or os.path.join(
        'output', '{}_test9_wfsim'.format(os.path.basename(run_dir.rstrip('\\/')))))
    os.makedirs(out_dir, exist_ok=True)

    # ---- 权重形状探测（隐层 + H 观测维数 → phase_mode）----
    a6 = os.path.join(run_dir, 'actor_6.pkl')
    a1 = os.path.join(run_dir, 'actor_1.pkl')
    for f in (a6, a1):
        if not os.path.isfile(f):
            raise SystemExit('缺少 {}'.format(f))
    nb_states_H = _load_shape(a6)[1]
    # 隐层两维都从 actor_6 的 fc1/fc2 读（同一个 state_dict）
    sd6 = torch.load(a6, map_location='cpu')
    hidden = (int(sd6['fc1.weight'].shape[0]), int(sd6['fc2.weight'].shape[0]))
    if nb_states_H - 4 not in _DIM_TO_MODE:
        raise SystemExit('actor_6 观测维数 {} 无法解释为 4+相位(0/2/5)'
                         .format(nb_states_H))
    phase_mode = _DIM_TO_MODE[nb_states_H - 4]

    print('=' * 92)
    print('  test9 @ WFSim 引擎   项目={}   模型={}'.format(spec['label'], run_dir))
    print('=' * 92)
    print('  actor_6 观测维数 = {} ⇒ phase_mode = {}（4+{}）；隐层 = {}'
          .format(nb_states_H, phase_mode, nb_states_H - 4, hidden))
    print('  参考实现: {}（本脚本只把 env_test 换成 WFSim，其余接线逐项对齐）'
          .format(spec['ref']))

    # ---- 引擎 & env（**唯一**与被测项目 main 不同的一处）----
    print('\n  启动 MATLAB 引擎 ...')
    engine = matlab.engine.start_matlab()
    print('  MATLAB 引擎就绪；env_test = env.WindBattery（WFSim_port）')
    env_test = ENV.WindBattery(MAX_E, NB_TURBINE, False, engine)

    # ---- 规则下层（与 *_main.py 同参）----
    rbb_system = RuleBasedBESS(max_E=MAX_E, min_E_factor=0.0, rated_power=RATED_POWER,
                               eta_c=ETA_C, eta_d=ETA_D, time=TIME,
                               fluctuation_limit=FLUCT_LIMIT)

    # ---- agents（kwargs 逐项对齐 aifflor_main / td3per_main 的当前接线）----
    import importlib
    mod_name, cls_name = spec['agent']
    AgentCls = getattr(importlib.import_module(mod_name), cls_name)
    myf = importlib.import_module(spec['myfunc'])
    ns = build_args(args_cli.seed, hidden)

    agent_H = AgentCls(nb_states=nb_states_H, nb_actions=6, args=ns,
                       lam_max=100.0, lam_min=3.0,
                       ema_lam=True, ema_alpha=0.02, ema_lam_lr=0.01,
                       record_update_stats=True,
                       policy_delay=2, policy_noise=0.1, noise_clip=0.3,
                       cost_critic_mode='max',
                       wt_noise=0.05, sig_noise_max=0.2, sig_noise_min=0.05)
    agent_H.lam = 5.0
    agent_H.value_scale = hc.value_scale()
    agent_H.cost_scale = hc.cost_scale()

    agent_L = AgentCls(nb_states=5, nb_actions=1, args=ns,
                       lam_max=hst.L_LAM_MAX, lam_min=hst.L_LAM_MIN,
                       ema_lam=True, ema_alpha=0.02, ema_lam_lr=hst.L_EMA_LAM_LR,
                       policy_delay=2, policy_noise=0.1, noise_clip=0.3,
                       cost_critic_mode='max')
    agent_L.lam = hst.L_LAM_INIT
    agent_L.cost_limit = hst.L_COST_LIMIT
    agent_L.value_scale = 1.0

    agent_H.load_weights(run_dir)
    agent_L.load_weights(run_dir)
    agent_H.is_training = False
    agent_L.is_training = False
    print('  权重加载完成（H/L），is_training=False（确定性、不更新）')

    # ---- 跑一次 test9 ----
    phase_table = myf.phase_daily_table() if (phase_mode >= 2 and
                                             hasattr(myf, 'phase_daily_table')) else None
    t0 = _dt.datetime.now()
    case_metrics, overall = myf.nine_case_test(
        agent_H, agent_L, env_test, rbb_system, seed=args_cli.seed,
        test_csv=True, output=out_dir,
        phase_mode=phase_mode, phase_table=phase_table,
        viol_derate_coef=args_cli.viol_derate)
    dt = (_dt.datetime.now() - t0).total_seconds()
    engine.quit()

    # ---- 落盘 ----
    df_cases = pd.DataFrame(case_metrics)
    df_overall = pd.DataFrame([overall])
    df_cases.to_csv(os.path.join(out_dir, 'test9_cases.csv'), index=False)
    df_overall.to_csv(os.path.join(out_dir, 'test9_overall.csv'), index=False)

    meta = {
        'run_dir': run_dir,
        'project': args_cli.project,
        'label': spec['label'],
        'engine': 'WFSim (MATLAB WFSim_port)',
        'reference_main': spec['ref'],
        'scenario_seed': int(args_cli.seed),
        'viol_derate': float(args_cli.viol_derate),
        'phase_mode': int(phase_mode),
        'nb_states_H': int(nb_states_H),
        'hidden': [int(hidden[0]), int(hidden[1])],
        'git_commit': _git_head(),
        'timestamp': _dt.datetime.now().isoformat(timespec='seconds'),
        'elapsed_s': round(dt, 1),
        'overall': {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool)
                        else v) for k, v in overall.items()},
    }
    with open(os.path.join(out_dir, 'test9_overall.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    # ---- 打印：9 case 明细（profit 在最前）----
    cols = ['wind_level', 'price_level', 'profit', 'profit_econ', 'dsoc',
            'soc_start', 'soc_end', 'r1', 'turbine_profit', 'derate_loss',
            'penalty_NN_H', 'c1', 'gap_pen', 'r2', 'c2', 'cvm', 'pf_viol',
            'soc_track_err']
    cols = [c for c in cols if c in df_cases.columns]
    print('\n' + '=' * 120)
    print('  ① per-case 明细（seed={}，5 步/case；profit/profit_econ/turbine_profit/'
          'derate_loss/penalty_NN_H 均为 5 步之和）'.format(args_cli.seed))
    print('=' * 120)
    with pd.option_context('display.width', 200, 'display.max_columns', 50,
                           'display.float_format', lambda v: '{:.4f}'.format(v)):
        print(df_cases[cols].to_string(index=False))

    print('\n' + '=' * 120)
    print('  ② OVERALL（45 步总量口径）★ profit 主记录项')
    print('=' * 120)
    for k in ['profit', 'profit_econ', 'dsoc', 'soc_start', 'soc_end',
              'r1', 'turbine_profit', 'derate_loss', 'penalty_NN_H',
              'c1', 'gap_pen', 'r2', 'c2', 'cvm', 'pf_viol', 'soc_track_err']:
        if k in overall:
            print('    {:>16s} = {:>18.6f}'.format(k, float(overall[k])))
    print('    {:>16s} = {}'.format('（45 步 profit 校验: Σcase = overall）',
                                    abs(float(df_cases['profit'].sum())
                                        - float(overall['profit'])) < 1e-9))
    print('    {:>16s} = {:.6f} 元/天外推（×1440/45）'
          .format('profit 折算', float(overall['profit']) * 1440.0 / 45.0))

    print('\n  输出目录: {}'.format(out_dir))
    print('    test9_cases.csv / test9_overall.csv / test9_overall.json / test_s_9case.csv')
    print('  耗时: {:.1f}s'.format(dt))


if __name__ == '__main__':
    main()
