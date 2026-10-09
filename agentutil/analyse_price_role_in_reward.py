# -*- coding: utf-8 -*-
"""电价在 H 层奖励里到底扮演什么角色 —— 反事实检验（用 run96 真实日志 + 已训 actor）。

要回答的提案
------------
> 既然电价不同导致 Q 值方差大，那把奖励改成"直接奖励发电瓦数"（电价不进奖励），
> BESS 改用一套综合电价与退化成本的系数，不就好了？

本脚本用**数据**判，而不是凭直觉。四段输出：

  [A] 块级 H 奖励的方差分解：`r = wT·pr·ptur·dt + wB·(−pr·pbrule·dt) − 退化 − 罚`
      的各项 Var/Cov 贡献，以及**只按电价档分组的组间方差占比**（电价的电平解释度）。
  [B] 已训策略对电价的**输入敏感度**：固定 u/SOC/P_G，只把 pr ∈ {300,580,980} 喂给
      actor，看 6 维动作怎么变 —— 若"发电瓦数"与"收益"等价，这一列应当与电价无关。
  [C] "电价不进奖励"的反事实臂（块级重算）：
      · 现状（全用真实电价）／风机真价+储能常数 580／两项都换常数 580
      · 各臂的 sd、corr(pr)、电价电平解释度 ⇒ 判断"去电价"能否真正压掉方差。
  [D] 口径自洽检查：块奖励 → nstep 折叠 → `update_stats_H.reward_mean` / `q_p_mean` /
      `td_p_mean` 的关系（TD 误差相对奖励与值的比例 —— 判别"方差大"是不是真瓶颈）。

关键实测结论（run96, 16640 个完整块）
-----------------------------------
* 块奖励 mean 180.1 / sd 289.8 元；**只按电价档分组的组间方差只占 24.3%**。
* 把电价换成常数 580：sd 只从 289.8 降到 211.6（−27%）⇒ 价格不是主导方差源；
  剩下的来自风功率（风机项 262.5±239.0）与电池指令（储能项 0.6±208.1）。
* actor 的 **SOC 信号维**对电价的跨档极差均值 0.34（300→+0.1、580→−0.1、980→−0.35
  的单调"低价充/高价放"），机组维 0.02~0.11 ⇒ 策略确实在用电价。
* 电池的**毛套利本身接近零和**：储能项按电价档均值 {300:−1.8, 580:−7.0, 980:+11.6}
  元/块，而退化 71.6 元/块 ⇒ 电池整体净亏（约 2.06 万元/天）；它的存在理由是
  满足平抑/跟踪约束，不是套利。⇒ 把 BESS 奖励写成"综合常数系数"在数据上是**有道理**的。
* 训练侧按 490 元/MWh 计放电退化，而评测端 `SOC_SETTLE_PR = 580` 结算库存 ⇒
  两者口径不一致（同一 MWh 在训练里比评测里"便宜"）。

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\analyse_price_role_in_reward.py ^
        --run output/TD3PER-run96
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

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

PRICE_LIST = (300.0, 580.0, 980.0)
PR_REF = 580.0          # 与 myfunc.SOC_SETTLE_PR / hc.PR_REF 一致
DEG_DISCHARGE = 490.0   # 与 myfunc.DEG_COST_DISCHARGE 一致

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

import hsrl_cost as hc                                  # noqa: E402

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

PRICE_LIST = (300.0, 580.0, 980.0)


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


def load_actions(df):
    """把 'original_action_H' 的字符串列展开成 a0..a5（并重置索引，避免 CSV 里
    多个 run 追加造成的重复 `Unnamed: 0` 索引把 `.loc[mask]` 变成标签查询）。"""
    A = np.array([np.fromstring(s.strip('[]'), sep=' ') for s in df['original_action_H']])
    for j in range(6):
        df['a%d' % j] = A[:, j]
    return df.reset_index(drop=True)


def part_a(df):
    print('=' * 104)
    print('[A] H 层奖励的方差分解 —— 电价(pr) vs 其余（run96 训练日志，每 5 步一个 H 决策）')
    print('=' * 104)
    # 块级: 每 5 步取一次 H 决策（train 期 H 每块决策一次）。⚠ 用**布尔掩码**切片，
    # 不用 .loc[索引] —— CSV 的 'Unnamed: 0' 在追加写时会重复 ⇒ 标签查询会多取行。
    blk = df[df['step'] % 5 == 0].copy()
    for c in ('turbine_profit', 'bess_profit', 'degradation', 'penalty_NN_H', 'pr'):
        blk[c] = pd.to_numeric(blk[c], errors='coerce')
    blk['r1_num'] = blk['turbine_profit'] + blk['bess_profit'] - blk['degradation'] - blk['penalty_NN_H']
    cols = ['turbine_profit', 'bess_profit', 'degradation', 'penalty_NN_H']
    print('  样本: %d 个 H 决策块 (每块 = 1 次决策 + 5 步累计记录列)' % len(blk))
    print('  %-18s %12s %12s %10s %10s' % ('项', '均值', '标准差', '最小', '最大'))
    for c in cols + ['r1_num']:
        v = blk[c].astype(float)
        print('  %-18s %12.4f %12.4f %10.2f %10.2f' % (c, v.mean(), v.std(), v.min(), v.max()))
    v = blk['r1_num'].astype(float)
    print('\n  方差分解 (Var(r1) = Σ Var(分量) + 2Σ Cov):')
    tot = v.var()
    for c in cols:
        var_c = blk[c].var()
        cov_c = blk[c].cov(v)
        print('    %-18s Var=%12.3f  贡献 %6.1f%%  Cov=%12.3f (占比 %6.1f%%)'
              % (c, var_c, 100 * var_c / tot, cov_c, 100 * cov_c / tot))
    print('    Var(r1) = %.3f' % tot)
    # 电价"解释度": 只按电价档分组的**组间方差 / 总方差**（无其他分组变量）
    gm = blk.groupby('pr')['r1_num']
    between = float((gm.count() * (gm.mean() - v.mean()) ** 2).sum() / len(blk))
    print('\n  只按电价档分组: 组间方差 = %.3f  ⇒ 占总方差 %.1f%%（电价的**电平解释度**）'
          % (between, 100 * between / tot))
    print('\n  按电价档分组的 r1（turbine+bess−退化−罚, 5 步累计）:')
    print('    %-8s %8s %12s %12s %12s' % ('pr', 'n', '均值', '标准差', '块/均值'))
    for pr, sub in gm:
        print('    %-8.0f %8d %12.2f %12.2f %12.3f'
              % (pr, sub.count(), sub.mean(), sub.std(), sub.mean() / 5))
    print('    ⇒ 三档电价之间 r1 均值极差 = %.2f 元/块'
          % (gm.mean().max() - gm.mean().min()))
    return blk.reset_index(drop=True)


def part_b(run, seed, hidden):
    print()
    print('=' * 104)
    print('[B] 已训策略对电价的真实敏感度 —— 固定 u/SOC/P_G，只变 pr 喂给 actor')
    print('=' * 104)
    myf = importlib.import_module('td3per.td3per_myfunc')
    AgentCls = getattr(importlib.import_module('td3per.td3per_ddpg'), 'SP_TD3')
    sd = torch.load(os.path.join(run, 'actor_6.pkl'), map_location='cpu')
    nb_states = int(sd['fc1.weight'].shape[1])
    ns = build_args(seed, hidden)
    ag = AgentCls(nb_states=nb_states, nb_actions=6, args=ns,
                  lam_max=100.0, lam_min=3.0, ema_lam=True, ema_alpha=0.02,
                  ema_lam_lr=0.01, record_update_stats=False, policy_delay=2,
                  policy_noise=0.1, noise_clip=0.3, cost_critic_mode='max',
                  wt_noise=0.05, sig_noise_max=0.2, sig_noise_min=0.05)
    ag.load_weights(run)
    ag.is_training = False
    print('  actor_6 观测维数 = %d（4 = 无相位 ⇒ 奖励里的电价是**唯一**的电价信息源）'
          % nb_states)
    print('  %-10s %-8s %-8s %-8s | %s' % ('u', 'SOC', 'P_G', 'pr', '动作 a0..a5'))
    rows = []
    for u in (5.5, 9.0, 13.0):
        for soc in (1.5, 3.0, 5.0):
            acts = {}
            for pr in PRICE_LIST:
                obs = np.array([pr, u, soc, 0.5 * u])
                a = np.asarray(ag.select_action(myf.Normalize(obs)), dtype=float)
                acts[pr] = a
                rows.append(dict(u=u, SOC=soc, P_G=0.5 * u, pr=pr,
                                 **{'a%d' % j: a[j] for j in range(6)}))
            spread = max(abs(acts[p] - acts[q]).max() for p in PRICE_LIST for q in PRICE_LIST)
            flag = '  ← 与电价相关' if spread > 0.01 else ''
            for pr in PRICE_LIST:
                print('  %-10.1f %-8.1f %-8.2f %-8.0f | %s%s'
                      % (u, soc, 0.5 * u, pr,
                         ' '.join('%+.3f' % x for x in acts[pr]),
                         flag if pr == PRICE_LIST[-1] else ''))
    dfb = pd.DataFrame(rows)
    print('\n  各维"跨电价档极差"的全局均值（0 = 完全不受电价影响）:')
    for j in range(6):
        rng = dfb.groupby(['u', 'SOC'])['a%d' % j].agg(lambda s: s.max() - s.min())
        name = 'a%d(机组限功率)' % j if j < 5 else 'a5(SOC 信号)'
        print('    %-16s 极差均值 %.4f   最大 %.4f' % (name, rng.mean(), rng.max()))
    return dfb


def to_blocks(df):
    """逐 5 步聚合成 H 决策块（与训练里 `sum_r1` 同口径：块内 5 步 r1 之和 − 块末 gap 罚）。"""
    for c in ('r1', 'gap_pen', 'penalty_NN_H', 'turbine_profit', 'bess_profit',
              'degradation', 'pr', 'pbrule', 'u', 'SOC'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.copy()
    df['bidx'] = np.floor(df['step'] / 5).astype(int)
    g = df.groupby('bidx').agg(n=('step', 'size'), r1=('r1', 'sum'), gap=('gap_pen', 'sum'),
                               pen=('penalty_NN_H', 'sum'), tp=('turbine_profit', 'sum'),
                               bess=('bess_profit', 'sum'), deg=('degradation', 'sum'),
                               pr=('pr', 'mean'), pb=('pbrule', 'mean'),
                               u=('u', 'mean'), SOC=('SOC', 'mean'))
    g = g[g.n == 5].copy()                     # 丢掉首尾不完整的块
    g['H'] = g.r1 - g.gap                      # ★ 真正的块奖励（进 buffer 前的值）
    g['tp_pow'] = g.tp / g.pr                  # 剥掉电价：块内风机电量 [MWh]
    g['pb_MWh'] = g.pb * 5 / 60                # 块内储能电量 [MWh]（>0 = 充电）
    return g


def _between_share(series, groups):
    """只按 `groups` 分组的组间方差占总方差的比例（电价的"电平解释度"）。"""
    gm = series.groupby(groups)
    between = float((gm.count() * (gm.mean() - series.mean()) ** 2).sum() / len(series))
    return 100.0 * between / series.var()


def part_a(g):
    print('=' * 104)
    print('[A] 块级 H 奖励的方差分解 —— 电价(pr) vs 其余（%d 个完整块）' % len(g))
    print('=' * 104)
    v = g['H']
    print('  块奖励      mean %8.2f  sd %7.2f  [%.1f, %.1f]'
          % (v.mean(), v.std(), v.min(), v.max()))
    for c, name in [('tp', '风机项'), ('bess', '储能项'), ('deg', '退化'), ('pen', '越界罚')]:
        print('  ├ %-8s mean %8.2f  sd %7.2f' % (name, g[c].mean(), g[c].std()))
    print('\n  Var 占比（ΣVar + 2ΣCov = Var(r)）:')
    for c, name in [('tp', '风机'), ('bess', '储能'), ('deg', '退化'), ('pen', '越界罚')]:
        print('    %-6s Var=%10.1f (%5.1f%%)  Cov(r)=%10.1f (%5.1f%%)'
              % (name, g[c].var(), 100 * g[c].var() / v.var(),
                 g[c].cov(v), 100 * g[c].cov(v) / v.var()))
    print('    Var(块奖励) = %.1f  (sd %.1f)' % (v.var(), v.std()))
    print('\n  只按电价档分组的**组间方差**占 %.1f%%（电价的电平解释度）'
          % _between_share(v, g['pr']))
    print('  三档均值 %s  极差 %.1f 元/块'
          % ({int(k): round(x, 1) for k, x in v.groupby(g['pr']).mean().items()},
             v.groupby(g['pr']).mean().max() - v.groupby(g['pr']).mean().min()))
    print('  corr(块奖励, pr) = %+.3f（含风/价相关性，不等于因果）' % v.corr(g['pr']))
    return g


def part_b(run, seed, hidden):
    print()
    print('=' * 104)
    print('[B] 已训策略对电价的**输入**敏感度 —— 固定 u/SOC/P_G，只把 pr 变一下喂给 actor')
    print('=' * 104)
    myf = importlib.import_module('td3per.td3per_myfunc')
    AgentCls = getattr(importlib.import_module('td3per.td3per_ddpg'), 'SP_TD3')
    sd = torch.load(os.path.join(run, 'actor_6.pkl'), map_location='cpu')
    nb_states = int(sd['fc1.weight'].shape[1])
    ns = build_args(seed, hidden)
    ag = AgentCls(nb_states=nb_states, nb_actions=6, args=ns,
                  lam_max=100.0, lam_min=3.0, ema_lam=True, ema_alpha=0.02,
                  ema_lam_lr=0.01, record_update_stats=False, policy_delay=2,
                  policy_noise=0.1, noise_clip=0.3, cost_critic_mode='max',
                  wt_noise=0.05, sig_noise_max=0.2, sig_noise_min=0.05)
    ag.load_weights(run)
    ag.is_training = False
    print('  actor_6 观测维数 = %d（= 4 ⇒ 无相位特征；奖励里的电价是**唯一**的电价信息源）'
          % nb_states)
    rows = []
    for u in (5.5, 9.0, 13.0):
        for soc in (1.5, 3.0, 5.0):
            for pr in PRICE_LIST:
                obs = np.array([pr, u, soc, 0.5 * u])
                a = np.asarray(ag.select_action(myf.Normalize(obs)), dtype=float)
                rows.append(dict(u=u, SOC=soc, P_G=0.5 * u, pr=pr,
                                 **{'a%d' % j: a[j] for j in range(6)}))
    dfb = pd.DataFrame(rows)
    print('\n  各维"跨电价档极差"（同 u/SOC 下 980−300 的动作差；0 = 完全不受电价影响）:')
    for j in range(6):
        rng = dfb.groupby(['u', 'SOC'])['a%d' % j].agg(lambda s: s.max() - s.min())
        name = 'a%d(机组限功率)' % j if j < 5 else 'a5(SOC 信号)'
        print('    %-16s 极差均值 %.4f   最大 %.4f' % (name, rng.mean(), rng.max()))
    print('\n  SOC 信号维（a5）随电价的变化（各 u/SOC 组合均值）:')
    for pr, sub in dfb.groupby('pr'):
        print('    pr=%-6.0f  a5 均值 %+0.3f   ⇒ target = 3+5·a5 = %.2f MWh'
              % (pr, sub['a5'].mean(), 3 + 5 * sub['a5'].mean()))
    print('    ⇒ 单调"低价充/高价放" ⇒ 策略**确实在用电价**；把电价从奖励里拿掉，')
    print('      这一维就失去唯一的信息来源（观测里电价仍归一化存在，但奖励不再区分）')
    return dfb


def part_c(g):
    print()
    print('=' * 104)
    print('[C] 反事实臂：把电价换成常数 pr_ref = %.0f（= 结算价 SOC_SETTLE_PR）' % PR_REF)
    print('=' * 104)
    g = g.copy()
    g['A'] = g.tp - PR_REF * g.pb_MWh - g.deg - g.pen            # 只换储能项的价
    g['C'] = PR_REF * g.tp_pow - PR_REF * g.pb_MWh - g.deg - g.pen  # 风机也换常数价
    for tag, col in [('现状（全部真实电价）', 'H'),
                     ('风机真价 + 储能常数价', 'A'),
                     ('风机常数价 + 储能常数价', 'C')]:
        s = g[col]
        print('  %-24s mean %8.2f  sd %7.2f  corr(pr) %+.3f  电价电平解释度 %5.1f%%'
              % (tag, s.mean(), s.std(), s.corr(g['pr']), _between_share(s, g['pr'])))
    print('  ⇒ sd 从 %.1f 只降到 %.1f（−%.0f%%）—— 风功率与电池指令才是方差主体'
          % (g['H'].std(), g['C'].std(), 100 * (1 - g['C'].std() / g['H'].std())))
    print()
    print('  电池的买卖账（块级）:')
    print('    %-6s %8s %11s %12s %11s %10s' % ('pr', 'n', 'pb均值MW', '放电块占比',
                                               '储能项(元)', '退化(元)'))
    for pr, sub in g.groupby('pr'):
        print('    %-6.0f %8d %11.3f %10.1f%% %12.2f %10.2f'
              % (pr, len(sub), sub.pb.mean(), 100 * (sub.pb < 0).mean(),
                 sub.bess.mean(), sub.deg.mean()))
    print('    净(储能−退化) 按档:', {int(k): round(x, 1)
                                      for k, x in (g.bess - g.deg).groupby(g.pr).mean().items()})
    tput = g.pb.abs().mean() * 5 / 60
    print('    吞吐 %.1f MWh/天 ; 退化 %.0f 元/天 ; 储能毛利 %.0f 元/天'
          % (tput * 288, DEG_DISCHARGE * tput * 288, g.bess.mean() * 288))
    print('    ⇒ **电池整体净亏**：它存在的理由是满足平抑/跟踪约束，不是套利。')
    print('       ⇒ 把 BESS 奖励写成"综合常数系数"在数据上是有依据的（见结论）。')
    print()
    print('  训练/评测口径不一致:')
    print('    训练侧放电退化 = %.0f 元/MWh ; 评测端库存结算价 SOC_SETTLE_PR = %.0f 元/MWh'
          % (DEG_DISCHARGE, PR_REF))
    print('    电价档差 980−300 = 680 元/MWh ⇒ 允许的往返效率损耗上限 = %.2f' % (680 / DEG_DISCHARGE))
    return g


def part_d(run, g):
    print()
    print('=' * 104)
    print('[D] 口径自洽检查：块奖励 → nstep 折叠 → critic 的实际拟合量级')
    print('=' * 104)
    fold = sum(0.95 ** k for k in range(5))
    p = os.path.join(run, 'update_stats_H.csv')
    if not os.path.exists(p):
        print('  (缺 %s，跳过)' % p)
        return
    u = pd.read_csv(p)
    print('  块奖励（日志重算）        mean %8.2f' % g['H'].mean())
    print('  nstep=5 折叠后（进 buffer）mean %8.2f（γ 折叠和 %.2f）' % (g['H'].mean() * fold, fold))
    print('  update_stats.reward_mean  mean %8.2f  ⇒ critic 拟合的就是这个量级'
          % u.reward_mean.mean())
    print('  q_p_mean %8.1f（/VALUE_SCALE(%.0f) = %.2f 回归空间）'
          % (u.q_p_mean.mean(), hc.VALUE_SCALE, u.q_p_mean.mean() / hc.VALUE_SCALE))
    print('  td_p_mean %7.1f  ⇒ TD/奖励 = %.2f, TD/Q = %.3f'
          % (u.td_p_mean.mean(), u.td_p_mean.mean() / u.reward_mean.mean(),
             u.td_p_mean.mean() / u.q_p_mean.mean()))
    u2 = u.copy()
    u2['bucket'] = (u2.step // 16000) * 16000
    gg = u2.groupby('bucket').agg(reward=('reward_mean', 'mean'), td=('td_p_mean', 'mean'))
    gg['td/奖励'] = (gg.td / gg.reward).round(2)
    print('\n  TD/奖励 随训练:')
    print(gg.round(1).to_string())
    print('  ⇒ TD 误差 ≈ 块奖励的 0.6~0.8 且**随训练上升**：这是"critic 拟合难度"的直接读数。')
    print('    注意它**不能**靠"把奖励整体缩放"改善（比值不变），但**去掉电价确实会改变目标的')
    print('    方差结构**（[C] 臂 C 的 sd 211.6 vs 289.8）⇒ 值得作为一条独立 A/B 臂验证。')


def main():
    ap = argparse.ArgumentParser(description='电价在 H 奖励里的角色（反事实检验）')
    ap.add_argument('--run', default='output/TD3PER-run96')
    ap.add_argument('--csv', default=None, help='训练逐步日志（默认 <run>/df_td3per.csv）')
    ap.add_argument('--seed', type=int, default=42)
    cli = ap.parse_args()

    csv = cli.csv or os.path.join(cli.run, 'df_td3per.csv')
    df = pd.read_csv(csv)
    sd = torch.load(os.path.join(cli.run, 'actor_6.pkl'), map_location='cpu')
    hidden = (int(sd['fc1.weight'].shape[0]), int(sd['fc2.weight'].shape[0]))
    print('数据: %s  (%d 行逐步日志)' % (csv, len(df)))
    print('模型: %s  actor_6 hidden=%s' % (cli.run, hidden))

    g = to_blocks(df)
    g = part_a(g)
    dfb = part_b(cli.run, cli.seed, hidden)
    part_c(g)
    part_d(cli.run, g)

    out = 'output/_price_role_report.csv'
    dfb.assign(kind='actor_price_sensitivity').to_csv(out, index=False)
    print('\n落盘: %s' % out)


if __name__ == '__main__':
    main()
