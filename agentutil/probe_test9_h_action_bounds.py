# -*- coding: utf-8 -*-
"""test9 里 `penalty_NN_H` 到底来自**哪一维动作** —— 离线归因 (不需要 MATLAB)。

动机
----
`output/TD3PER-run96/df_test9.csv` 的 OVERALL 行满足一条**恒等式**:

    profit/45 − r1 == (penalty_NN_H + derate_loss)/45          (残差 ~1e-6, 已实测)

即 r1 的走低**只能**来自 `penalty_NN_H` 的增长 (derate_loss 恒 0, c=0)。
而 penalty 的来源只有一个: `CheckAction` 对**未压缩的 actor 输出**做的越界罚

    penalty_NN_H = 500·Σ_{j<5} viol_j(a_j; [0,0.5]) + 500·viol(a5; [−0.5,0.5])

上游 `model.Actor.forward` **故意不做 tanh/sigmoid 压缩** (最优解在边界上), 所以
动作可以真的跑到界外。本探针把 run 目录里的 actor 权重读出来, 在 **test9 的 45 步
观测轨道**上逐决策点复算动作, 把总越界量按维拆开 ⇒ 回答"是哪一维、多深"。

为什么不启动 MATLAB
------------------
H 层只从环境拿 4 元观测 (pr, u, E, P_G_prev)。其中 pr/u 由 test9 场景给定, E 由
下层电池按 `env.GetBatteryActions` 的物理 (纯 Python) 推进, 只有 P_G_prev 需要
风机功率。实测 run96 训练日志给出 **风场功率的单一标度**:

    ptur(u, a) ≈ KAPPA · u³ · C_p(a),   C_p = 4a(1−a)²,   KAPPA = 0.0065
    (三档风况分别 0.0065/0.0065/0.0067; 由 df_td3per 的 turbine_profit/(pr/60) 反解)

⇒ 用该代理功率即可复现观测轨道; 越界量对 ptur 的敏感度是**二阶**的 (a 的复算只通过
P_G_prev 这一维进入), 足够判"哪一维越界"。

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\probe_test9_h_action_bounds.py ^
        --run output/TD3PER-run96 --project td3per

输出 (打印 + 存 `output/<run>_test9_hbounds/`)
    h_actions.csv   45 步逐决策点: 6 维原始动作 / 越界量 / 分维罚 / 总罚
    summary.csv     按 case 汇总: 9 个 case 的罚来自哪一维
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np                                      # noqa: E402
import pandas as pd                                     # noqa: E402
import torch                                            # noqa: E402

from test_scenario import build_9case_scenario          # noqa: E402

KAPPA = 0.0065          # 风场功率标度 (见 docstring; 由训练日志反解)
MAX_E = 6.0
TIME = 1.0 / 60.0
ETA_C = ETA_D = 0.98
ACT_HI, ACT_LO = 0.5, 0.0
SIG_HI, SIG_LO = 0.5, -0.5
# 罚系数**从项目模块读** (td3per 系 = hc/常量 PENALTY_NN_H_COEF; run96 = 400),
# 不写死 —— 该系数历史上 200/400/500 都出现过, 写死会把归因的绝对量算错。
COEF_H = None           # main() 里按项目模块解析

# 与 *_main.py 的接线同参 (只影响 actor 构造, 不影响前向)
PROJECTS = {
    'td3per': dict(agent=('td3per.td3per_ddpg', 'SP_TD3'),
                   myfunc='td3per.td3per_myfunc'),
    'td3perrb': dict(agent=('td3perrb.td3perrb_ddpg', 'SP_TD3'),
                     myfunc='td3perrb.td3perrb_myfunc'),
}


def surrogate_power(u, a5, kappa=None):
    """ptur [MW] = KAPPA·u³·C_p(a) (逐台同 a ⇒ 用均值 C_p)。"""
    k = KAPPA if kappa is None else float(kappa)
    a = np.clip(np.asarray(a5, dtype=float), 0.0, 0.5)
    cp = 4.0 * a * (1.0 - a) ** 2
    return float(k * (u ** 3) * cp.mean())


def split_penalty(action, coef=None):
    """复刻 CheckAction 的越界罚 (系数从项目模块读), 返回 (总罚, 5 维分维罚, 信号维罚)。"""
    k = COEF_H if coef is None else float(coef)
    a = np.asarray(action, dtype=float).reshape(-1)
    per = np.zeros(5)
    for j in range(5):
        per[j] = k * (max(ACT_HI, a[j]) - ACT_HI + ACT_LO - min(ACT_LO, a[j]))
    sig = k * ((max(SIG_HI, a[5]) - SIG_HI) + (SIG_LO - min(SIG_LO, a[5])))
    return float(per.sum() + sig), per, float(sig)


def _check_coef_consistency(myf, coef):
    """用一次真实调用核对: 把 dim0 推到 0.5+1.0 ⇒ 罚应恰好 = coef·1.0。"""
    probe = np.array([1.5, 0.2, 0.2, 0.2, 0.2, 0.0])       # dim0 越界 1.0
    try:
        _, pen = myf.CheckAction(probe)
    except Exception as exc:                                # pragma: no cover
        return 'CheckAction 调用失败: %s' % exc
    return '实测罚 %.1f, 期望 %.1f ⇒ %s' % (
        float(pen), coef * 1.0, 'OK' if abs(float(pen) - coef) < 1e-6 else '不一致!')


def build_args(seed, hidden):
    import argparse as _ap
    ns = _ap.Namespace()
    ns.hidden1, ns.hidden2 = int(hidden[0]), int(hidden[1])
    ns.rate, ns.prate = 0.01, 0.0001
    ns.warmup, ns.discount, ns.bsize, ns.rmsize = 200, 0.95, 256, 40000
    ns.window_length, ns.tau = 1, 0.001
    ns.ou_theta, ns.ou_sigma, ns.ou_mu = 0.15, 0.2, 0.0
    ns.validate_episodes, ns.max_episode_length, ns.validate_steps = 20, 1440, 2000
    ns.output, ns.debug = 'output', False
    ns.init_w = 0.003
    ns.train_iter_H, ns.epsilon, ns.seed = 1000 * 1440, 4000, int(seed)
    ns.resume = 'default'
    ns.cuda = bool(torch.cuda.is_available())
    ns.lam, ns.lr_lam, ns.cost_single_limit = 3.0, 0.1, 1000.0
    ns.PER_alpha, ns.PER_beta, ns.PER_epsilon = 0.6, 0.4, 0.01
    ns.beta_increment_per_sampling = 3e-5
    ns.nb_turbine = 5
    return ns


def main():
    ap = argparse.ArgumentParser(description='test9 越界罚的逐维归因 (离线, 无 MATLAB)')
    ap.add_argument('--run', required=True, help='模型目录, 如 output/TD3PER-run96')
    ap.add_argument('--project', default='td3per', choices=sorted(PROJECTS))
    ap.add_argument('--seed', type=int, default=42, help='test9 场景种子 (生产=42)')
    ap.add_argument('--out', default=None, help='输出目录')
    ap.add_argument('--kappa', type=float, default=KAPPA,
                    help='代理功率标度 KAPPA (默认 %.4f; 用 --kappa_sweep 扫其敏感性)'
                         % KAPPA)
    ap.add_argument('--kappa_sweep', default=None,
                    help='逗号分隔的 KAPPA 列表: 只打印每个标度下的总罚 (敏感性)')
    cli = ap.parse_args()

    run_dir = os.path.abspath(cli.run)
    out_dir = os.path.abspath(cli.out or os.path.join(
        'output', '{}_test9_hbounds'.format(os.path.basename(run_dir.rstrip('\\/')))))
    os.makedirs(out_dir, exist_ok=True)

    a6 = os.path.join(run_dir, 'actor_6.pkl')
    sd = torch.load(a6, map_location='cpu')
    nb_states_H = int(sd['fc1.weight'].shape[1])
    hidden = (int(sd['fc1.weight'].shape[0]), int(sd['fc2.weight'].shape[0]))
    print('=' * 96)
    print('  test9 越界归因   run={}   actor_6: nb_states={} hidden={}'
          .format(run_dir, nb_states_H, hidden))
    print('=' * 96)
    if nb_states_H != 4:
        print('  ⚠ 观测维数 %d != 4 ⇒ 该 run 带相位特征, 本探针的 4 元观测轨道不适用;'
              ' 仍会按 4 维前向但结论需谨慎.' % nb_states_H)

    spec = PROJECTS[cli.project]
    AgentCls = getattr(importlib.import_module(spec['agent'][0]), spec['agent'][1])
    myf = importlib.import_module(spec['myfunc'])
    global COEF_H
    COEF_H = float(getattr(myf, 'PENALTY_NN_H_COEF', 500.0))
    print('  越界罚系数 PENALTY_NN_H_COEF = %s /单位 (读自 %s; CheckAction 核对: %s)'
          % (COEF_H, spec['myfunc'], _check_coef_consistency(myf, COEF_H)))
    ns = build_args(cli.seed, hidden)
    agent_H = AgentCls(nb_states=nb_states_H, nb_actions=6, args=ns,
                       lam_max=100.0, lam_min=3.0, ema_lam=True, ema_alpha=0.02,
                       ema_lam_lr=0.01, record_update_stats=False,
                       policy_delay=2, policy_noise=0.1, noise_clip=0.3,
                       cost_critic_mode='max', wt_noise=0.05,
                       sig_noise_max=0.2, sig_noise_min=0.05)
    agent_H.load_weights(run_dir)
    agent_H.is_training = False

    wind_seq, price_seq, case_ids, blocks = build_9case_scenario(seed=cli.seed)
    print('  场景: seed={}  45 步  风况 LOW(u<7) -> MEDIUM -> HIGH; 段内电价 300/580/980'
          .format(cli.seed))

    def rollout(kappa):
        """跑一遍 45 步 H 决策, 返回 9 行记录 (每行 = 一个决策点)。"""
        E = 5.0                                 # env.ScenarioReset 写死
        P_G_prev = surrogate_power(float(wind_seq[0]), np.full(5, 0.3333), kappa)
        out = []
        for step in range(len(wind_seq)):
            pr, u = float(price_seq[step]), float(wind_seq[step])
            if step % 5 == 0:
                obs = np.array([pr, u, E, P_G_prev])
                raw = np.asarray(agent_H.select_action(myf.Normalize(obs)),
                                 dtype=float).reshape(-1)
                pen, per, sig = split_penalty(raw)
                out.append(dict(step=step, bi=step // 5,
                                wind_level=case_ids[step][0],
                                price_level=case_ids[step][1],
                                pr=pr, u=u, E_decision=E, P_G_prev=P_G_prev,
                                **{'a%d' % j: raw[j] for j in range(6)},
                                **{'v%d' % j: per[j] / COEF_H for j in range(5)},
                                v_sig=sig / COEF_H, penalty=pen,
                                **{'pen%d' % j: per[j] for j in range(5)},
                                pen_sig=sig))
                act = raw
            # ---- 推进环境一步 (env.ScenarioStep 的等价实现) ----
            # ⚠ 本探针**不重建下层 BESS 策略**: 那需要 S_ddpg + RBB 的完整接线。
            #    只关心 H 动作越界, 故取 pb ≡ 0 (电池不动作) ⇒ E 恒定 5.0、P_G = ptur。
            #    对越界结论的影响: E 只通过 obs[2] 进入 actor, 而实测 run96 的测试期
            #    dsoc ≈ −0.05 MWh / 45 步 ⇒ E 全程几乎不动, 该近似无害。
            P_G_prev = surrogate_power(u, np.clip(act[:5], 0.0, 0.5), kappa)
        return out

    if cli.kappa_sweep:
        print('\n[0] 代理功率标度敏感性 —— KAPPA 只影响观测轨道, 越界结论必须稳健')
        print('    (对照: run96 实测 test9 的 9 个决策点罚之和 ≈ 500~750, '
              '见 df_test9.csv 的 penalty_NN_H)')
        print('    %-9s %10s   %s' % ('KAPPA', '总罚', '  '.join('pen%d' % j for j in range(5))))
        for kv in [float(x) for x in cli.kappa_sweep.split(',')]:
            d2 = pd.DataFrame(rollout(kv))
            print('    %-9.4f %10.2f   %s' % (
                kv, d2['penalty'].sum(),
                '  '.join('%5.1f' % d2['pen%d' % j].sum() for j in range(5))))

    print('\n  代理功率: ptur = %.4f·u³·C_p(a)  (C_p = 4a(1-a)²)' % cli.kappa)
    rows = rollout(cli.kappa)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'h_actions.csv'), index=False)

    # ---- 汇总 ----
    tot = df['penalty'].sum()
    print('\n[1] 45 步中每次决策的越界罚 (9 次决策, 5 步块首)')
    print(df[['step', 'wind_level', 'price_level', 'u', 'E_decision',
              'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'penalty']].to_string(index=False))
    print('\n[2] 分维归因 (全 9 个决策点合计)')
    print('  总罚 = %.2f  (CheckAction 口径 %.0f/单位)' % (tot, COEF_H))
    for j in range(5):
        share = 100.0 * df['pen%d' % j].sum() / tot if tot else 0.0
        print('    dim%d: 罚 %8.2f (%5.1f%%)   动作范围 [%+.4f, %+.4f]   越界步数 %d/9'
              % (j, df['pen%d' % j].sum(), share, df['a%d' % j].min(),
                 df['a%d' % j].max(), int((df['v%d' % j] > 1e-9).sum())))
    print('    sig : 罚 %8.2f (%5.1f%%)   动作范围 [%+.4f, %+.4f]   越界步数 %d/9'
          % (df['pen_sig'].sum(),
             100.0 * df['pen_sig'].sum() / tot if tot else 0.0,
             df['a5'].min(), df['a5'].max(), int((df['v_sig'] > 1e-9).sum())))

    print('\n[3] 按 case 汇总 (每个 case = 1 次决策)')
    g = df.groupby(['wind_level', 'price_level']).agg(
        u=('u', 'mean'), penalty=('penalty', 'sum'),
        **{('pen%d' % j): ('pen%d' % j, 'sum') for j in range(5)},
        pen_sig=('pen_sig', 'sum'))
    with pd.option_context('display.width', 200):
        print(g.round(2).to_string())
    g.to_csv(os.path.join(out_dir, 'summary.csv'))

    dom = int(np.argmax([df['pen%d' % j].sum() for j in range(5)])) \
        if tot > 0 else -1
    print('\n[判定] 主导越界维 = %s ; 总罚 %.2f'
          % (('dim%d' % dom) if dom >= 0 else '(无越界)', tot))
    print('  ⚠ 只作**定性**归因: 本探针用代理功率 (电池不动作) 复现观测轨道, 与真机'
          ' test9 的绝对罚值可差数倍; 判"哪一维"稳健 (κ 扫 0.004~0.010 结论不变),')
    print('    判"多深"需在真引擎上带 --log_actions 重跑。')
    print('  输出目录: %s  (h_actions.csv / summary.csv)' % out_dir)


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    main()
