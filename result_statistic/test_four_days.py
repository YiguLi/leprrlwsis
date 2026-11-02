#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四固定风速日的统一评测（env 式：发出状态 → 接收动作 → 推进环境 → 记录结果）。

本脚本只做**测试**，不做任何训练
--------------------------------
* 网络只前向：`agent.is_training=False` ⇒ `select_action` 的探索噪声恒为 0；
* 不调用 `update_policy` / `observe_jump` / PER 采样 / lam 对偶 / curriculum；
* 不写 replay、不存 checkpoint、不更新 target 网络。

"任务与 env 相同"具体指什么
--------------------------
评测循环就是 `env.py::WindBattery` 的那套契约，逐步照搬（不另立一套物理）：

    obs  = env.reset()                 # [pr, u, E, P_G_prev]
    obs2, ... = env.step(H, L, pw_tru) # → (pw_tru2, obs2, applied_L, penalty_L,
                                       #    done, done_L, info{P_F, P_G})
    pw   = env.PowerOutput(u, a)       # 风机总出力（WFSim / FLORIS 后端）
    pr   = env.CalPrice()              # 峰谷平动态电价 300/580/980
    u    = env.CalWindSpeed()          # 按 `env.wind_day` 推进到下一分钟

`env.reset()` 之后本脚本把 `env.wind_day` 换成**固定的那一天**（并重置
`count_wind` / `u` / `P_G_prev` / `E`），其余全部交给生产实现：
`test_five_days.ProjectRunner.run_day()` 逐方法复刻各自 main 的训练时推理路径
（H 每 5 步、L 每步、RBB 硬裁剪、残余动作 / 规则 SocTracker、piecewise 权重、
相位观测），并按 (模型 × 天) 返回记录。**不重写推理路径** ⇒ 不会与生产口径漂移。

只测以下四天（`FOUR_DAYS`，来自 `agentutil/select_4_wind_days.py`）
------------------------------------------------------------------
    d1 低风速·低方差   iloc=24    日均 8.968 m/s   std 1.666   CV 0.186
    d2 低风速·高方差   iloc=21    日均 8.950 m/s   std 2.542   CV 0.284
    d3 中高风速·低方差 iloc=164   日均 11.974 m/s  std 1.415   CV 0.118
    d4 中高风速·高方差 iloc=2     日均 11.288 m/s  std 3.253   CV 0.288

每次运行都**核验**这四列仍在原统计值附近（`--no-verify-days` 可跳过）：
风速数据被换过就直接报错，而不是悄悄换一天。

记录什么
--------
    <out>/four_day_days.csv      每 (方法 × run × 天) 一行：三项指标 + SOC/充放电/峰值
    <out>/four_day_steps.csv     每步明细（可 --no-save-steps 关闭）
    <out>/four_day_summary.csv   每 (方法 × run) 的四天均值/标准差/合计 ★主表
    <out>/four_day_by_day.csv    每天 × 每方法（跨 run 均值）—— 看"哪种风况谁更好"
    <out>/four_day_ranking.csv   按指标优先级排序：越限次数 → 越限总量 → 利润
    <out>/tensorboard/           可选（--no-tensorboard 关闭）

三项指标（与 `test_five_days.py` 完全一致，便于横向对照）
    viol_count      日内功率波动越限次数   Σ 1[P_F > 3.0]
    viol_excess_mw  越限总量              Σ max(P_F − 3, 0)   [MW]
    profit_rmb      利润                  Σ (P_G·pr − 490·max(−pb,0))·Δt   [元]

用法
----
    # 一个方法一组模型（完整四天 × 1440 步；WFSim 约 0.3~1 s/步 ⇒ 4 天约 0.5~1.6 h/模型）
    python result_statistic/test_four_days.py --td3per output/TD3PER-run19

    # 多方法、多组模型（同一 KEY 重复 = run1/run2/...）
    python result_statistic/test_four_days.py \\
        --td3per output/TD3PER-run18 output/TD3PER-run19 \\
        --aifflor output/AIFFLOR-run15

    # 冒烟：每天只跑 60 步（指标只统计这 60 步，用于验证管线）
    python result_statistic/test_four_days.py --td3per output/TD3PER-run19 --episode-length 60

    # 只看计划，不加载模型、不推进仿真
    python result_statistic/test_four_days.py --td3per output/TD3PER-run19 --dry-run

    # 列出支持的方法
    python result_statistic/test_four_days.py --list-projects

环境
----
* 需要**项目自己的解释器**（`anaconda3/envs/llmenhancedwsis`，torch 2.12）；
* 走 WFSim 的方法需要 MATLAB 引擎；纯 FLORIS 方法（`yawflor`）不需要。受限沙箱里
  MATLAB 起不来（named pipe 被禁），请在普通终端运行。

目录约定
--------
本目录是用户在根目录显式指定的（`result_statistic/`），是 `AGENTS.md`
"项目无关脚本一律进 `agentutil/`" 的**例外**；脚本仍遵守 `agentutil/` 的另一条硬约定：
**项目代码不得反向 import 本目录**（本项目代码确实没有）。
"""
import argparse
import os
import sys
import time

# ---------------------------------------------------------------------------
# 路径 bootstrap（可被任意 CWD / 任意路径调用）
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))     # result_statistic/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)                                         # 相对数据/输出路径以仓库根为基准
sys.path.insert(0, ROOT)                               # 项目模块

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors='replace')
    except Exception:
        pass

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

import test_five_days as T                                        # noqa: E402

# ---------------------------------------------------------------------------
# 四固定风速日（单一真源 = agentutil/select_4_wind_days.py 的输出）
# ---------------------------------------------------------------------------
#: (tag, 展示名, iloc 列位置, 列名, 日均, 日内 std, CV)
#: 统计值来自 `python agentutil/select_4_wind_days.py`（ddof=0），用于**核验**
#: 风速池没被换过；不是拿来画图的常量。
FOUR_DAYS = (
    ('d1_low_lowcv',   'd1 低风速·低方差',   24,  '26',  8.968494, 1.666237, 0.185788),
    ('d2_low_highcv',  'd2 低风速·高方差',   21,  '23',  8.949579, 2.541804, 0.284014),
    ('d3_high_lowcv',  'd3 中高风速·低方差', 164, '169', 11.973994, 1.414746, 0.118152),
    ('d4_high_highcv', 'd4 中高风速·高方差', 2,   '1',   11.288473, 3.252644, 0.288139),
)

MEAN_TOL = 0.02        # 日均风速核验容差 (m/s)
STD_TOL = 0.02         # std 核验容差 (m/s)

METRICS = T.DAY_METRICS            # ('viol_count', 'viol_excess_mw', 'profit_rmb')
METRIC_CN = {'viol_count': '越限次数', 'viol_excess_mw': '越限总量(MW)',
             'profit_rmb': '利润(元)'}


def _w(s):
    """显示宽度：CJK 全角字符占 2 列（否则表头会错位）。"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in str(s))


def _pad(s, width, right=False):
    """按**显示宽度**补齐（`str.format` 只按 len 补，中文表头会歪）。"""
    s = str(s)
    gap = max(0, int(width) - _w(s))
    return (' ' * gap + s) if right else (s + ' ' * gap)


def _row(cells, widths):
    """按显示宽度排一行：首列左对齐，其余右对齐。"""
    return '  ' + ' '.join(_pad(c, w, right=(i > 0))
                           for i, (c, w) in enumerate(zip(cells, widths)))


# ===========================================================================
# 四天：载入 + 核验
# ===========================================================================
def load_days(verify=True):
    """返回 [(tag, label, iloc, col, series)]；verify 时核验统计值未被换过。"""
    df, _pool = T.load_wind_pool()
    days, errs = [], []
    for tag, label, iloc, col, mean_ref, std_ref, cv_ref in FOUR_DAYS:
        if iloc < 1 or iloc >= df.shape[1]:
            raise SystemExit('iloc=%d 超出风速池列范围 [1, %d)' % (iloc, df.shape[1]))
        s = df.iloc[:, iloc].reset_index(drop=True)
        m, sd = float(s.mean()), float(s.std(ddof=0))
        cv = sd / m if m else float('nan')
        ok = (abs(m - mean_ref) <= MEAN_TOL) and (abs(sd - std_ref) <= STD_TOL)
        if not ok:
            errs.append('  %-14s iloc=%3d 期望 日均 %.3f/std %.3f，实际 %.3f/%.3f'
                        % (tag, iloc, mean_ref, std_ref, m, sd))
        days.append(dict(tag=tag, label=label, iloc=int(iloc), col=str(col),
                         series=s, mean=m, std=sd, cv=cv, ok=ok))
    if errs and verify:
        raise SystemExit('风速池已变化（这四天的统计值对不上）：\n' + '\n'.join(errs) +
                         '\n  ⇒ 恢复原数据，或用 --no-verify-days 强制按 iloc 跑。')
    return df, days


def print_days(days):
    print('\n' + '=' * 96)
    print('  本次只测这四天（固定，不随机抽）：')
    print('=' * 96)
    W = (16, 6, 6, 9, 8, 8, 8, 8)
    print(_row(['tag', 'iloc', '列名', '日均', 'std', 'CV', 'min', 'max'], W))
    print('  ' + '-' * 92)
    for d in days:
        s = d['series']
        print(_row([d['tag'], d['iloc'], d['col'], '%.3f' % d['mean'],
                    '%.3f' % d['std'], '%.4f' % d['cv'],
                    '%.2f' % float(s.min()), '%.2f' % float(s.max())], W))


# ===========================================================================
# 汇总 / 打印
# ===========================================================================
def summarise(df_days):
    """每 (project, run) 的四天均值 ± std + 合计。"""
    g = df_days.groupby(['project', 'run'], as_index=False)
    out = g.agg(days=('day_idx', 'count'),
                viol_count_mean=('viol_count', 'mean'),
                viol_count_std=('viol_count', 'std'),
                viol_count_sum=('viol_count', 'sum'),
                viol_excess_mw_mean=('viol_excess_mw', 'mean'),
                viol_excess_mw_std=('viol_excess_mw', 'std'),
                viol_excess_mwh_sum=('viol_excess_mwh', 'sum'),
                profit_rmb_mean=('profit_rmb', 'mean'),
                profit_rmb_std=('profit_rmb', 'std'),
                profit_rmb_sum=('profit_rmb', 'sum'),
                soc_end_mean=('soc_end', 'mean'),
                max_PF_max=('max_PF', 'max'))
    # 指标优先级：越限次数 → 越限总量 → 利润（与项目 README 的主指标顺序一致）
    out = out.sort_values(['viol_count_mean', 'viol_excess_mw_mean', 'profit_rmb_mean'],
                          ascending=[True, True, False]).reset_index(drop=True)
    out['rank'] = np.arange(1, len(out) + 1)
    return out


def by_day_table(df_days):
    """每天 × 每方法（跨 run 均值）。"""
    agg = (df_days.groupby(['project', 'day_tag', 'day_idx'], as_index=False)
           .agg(viol_count=('viol_count', 'mean'),
                viol_excess_mw=('viol_excess_mw', 'mean'),
                profit_rmb=('profit_rmb', 'mean')))
    return agg


def print_report(df_days, df_sum, days):
    """表格一律走 `_row()`：中文表头按显示宽度补齐，数值列右对齐。"""
    print('\n' + '=' * 96)
    print('  [A] 逐日明细（每 方法 × 天）')
    print('=' * 96)
    W_A = (22, 3, 16, 8, 12, 13, 9)
    print(_row(['方法/run', '#', '天', '越限次数', '越限总量(MW)', '利润(元)', '末SOC'], W_A))
    print('  ' + '-' * 92)
    for _, r in df_days.iterrows():
        print(_row(['%s/run%d' % (r['project'], int(r['run'])), int(r['day_idx']),
                    r['day_tag'], '%d' % int(r['viol_count']),
                    '%.4f' % float(r['viol_excess_mw']), '%.2f' % float(r['profit_rmb']),
                    '%.3f' % float(r['soc_end'])], W_A))

    print('\n' + '=' * 96)
    print('  [B] 四天汇总（均值 ± 标准差；排名按 越限次数 → 越限总量 → 利润）')
    print('=' * 96)
    W_B = (4, 24, 17, 22, 23, 15)
    print(_row(['名次', '方法/run', '越限次数', '越限总量(MW)', '利润(元)', '四天利润合计'], W_B))
    print('  ' + '-' * 92)
    for _, r in df_sum.iterrows():
        print(_row([int(r['rank']), '%s/run%d' % (r['project'], int(r['run'])),
                    '%.1f±%.1f' % (r['viol_count_mean'], r['viol_count_std']),
                    '%.4f±%.4f' % (r['viol_excess_mw_mean'], r['viol_excess_mw_std']),
                    '%.2f±%.2f' % (r['profit_rmb_mean'], r['profit_rmb_std']),
                    '%.2f' % r['profit_rmb_sum']], W_B))

    print('\n' + '=' * 96)
    print('  [C] 每天 × 每方法（跨 run 均值）—— 看"哪种风况谁更好"')
    print('=' * 96)
    cols = [(p, int(r)) for (p, r) in df_sum[['project', 'run']].itertuples(index=False)]
    pivot_v = df_days.pivot_table(index='day_tag', columns=['project', 'run'],
                                  values='viol_count', aggfunc='mean')
    pivot_p = df_days.pivot_table(index='day_tag', columns=['project', 'run'],
                                  values='profit_rmb', aggfunc='mean')
    order = [d['tag'] for d in days]
    heads = ['%s/run%d' % (p, r) for (p, r) in cols]
    W = max(14, max(len(h) for h in heads) + 2)
    for title, piv, fmt in (('越限次数', pivot_v, '{:>%d.1f}' % W),
                            ('利润(元)', pivot_p, '{:>%d.2f}' % W)):
        print('\n  【%s】' % title)
        print('  ' + _pad('天', 16) + ''.join(('{:>%d}' % W).format(h) for h in heads))
        print('  ' + '-' * (16 + W * len(heads)))
        for tag in order:
            if tag not in piv.index:
                continue
            row = piv.loc[tag]
            cells = ''.join(fmt.format(float(row[(p, r)])) if (p, r) in row.index
                            else ' ' * W for (p, r) in cols)
            print('  ' + _pad(tag, 16) + cells)

    print('\n  说明: 越限 = Σ 1[P_F > 3.0 MW]；越限总量 = Σ max(P_F−3,0) [MW]；')
    print('        利润 = Σ (P_G·pr − 490·max(−pb,0))·Δt，Δt = 1/60 h；')
    print('        四天只是 4 个样本，均值±std 的 std 是**跨天**离散度，别当置信区间用。')


# ===========================================================================
# 确定性 stub 引擎（仅用于无 MATLAB 环境的管线自检）
# ===========================================================================
class _StubWFSim:
    """`WFSim_port(u, phi, a, phi2, want_full)` → `{'out': W}` 的确定性替代。

    单机 `Cp = 4a(1−a)²` + 简化串行尾流亏损（与 `agentutil/test_five_days_regression.py`
    的 stub 同式）。它**不需要物理正确**，只需要确定、有限、对动作敏感 —— 这样策略
    轨迹、储能窗口、越限计数都会被真正走到，用来验证"状态→动作→推进→记录"这条管线。
    ⚠ `--stub-engine` 下的**指标数值没有物理意义**，不能当成 WFSim 读数引用。

    ⚠ `env.py::PowerOutput` 传 5 个实参（第 5 个 `want_full=False`，见 env.py 的注释），
    旧 stub 只收 4 个会被 `except:` 吞成 UnboundLocalError ⇒ 这里必须收下并忽略。
    """

    RHO = 1.20
    AREA = float(np.pi * 50.0 ** 2)

    def __init__(self):
        self.calls = 0

    def WFSim_port(self, u, phi, a, phi2, want_full=False):
        u0 = float(np.asarray(u).reshape(-1)[0])
        acts = [float(x) for x in np.asarray(a).reshape(-1)][:5]
        self.calls += 1
        if not (3.5 <= u0 <= 25):
            return {'out': 0.0}
        total, deficit = 0.0, 0.0
        for aj in acts:
            aj = min(max(aj, 0.0), 0.5)
            cp = 4.0 * aj * (1.0 - aj) ** 2
            total += 0.95 * 0.5 * self.RHO * self.AREA * cp * (u0 * (1.0 - deficit)) ** 3
            deficit = min(0.45, max(deficit, 0.12 * cp + 0.30 * deficit))
        return {'out': total}

    def quit(self):
        return None


def install_stub_engine():
    """把 `matlab.engine.start_matlab` 换成 stub，并让 `T.ensure_engine()` 复用它。

    必须在构造 `ProjectRunner`（它会 import `*_myfunc`，而 myfunc 在**模块顶层**调用
    `matlab.engine.start_matlab()`）之前调用。
    """
    import matlab.engine
    stub = _StubWFSim()
    matlab.engine.start_matlab = lambda *a, **kw: stub
    T._ENGINE = stub                       # T.ensure_engine() 见到非 None 就直接返回
    return stub


# ===========================================================================
# 主流程
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description='四固定风速日评测（只推理不训练；env 式状态-动作-推进-记录）',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # ---- 与 test_five_days 同形的模型路径参数（每个方法可给多组 = run1/run2/...）----
    for key in T.PROJECT_ORDER:
        p.add_argument('--' + key, nargs='+', default=None, metavar='MODEL_DIR',
                       help='%s 模型目录；给多个 = 多组模型' % T.PROJECTS[key]['label'])
    p.add_argument('--models', nargs='*', default=None, metavar='KEY=PATH',
                   help='另一种写法，同一 KEY 可重复以给多组模型')
    p.add_argument('--projects', nargs='*', default=None, choices=T.PROJECT_ORDER,
                   help='只测这些方法')
    p.add_argument('--max-runs', type=int, default=0, help='每方法最多取前 N 组（0=不限）')
    p.add_argument('--list-projects', action='store_true', help='列出支持的方法后退出')

    # ---- 测试规模 ----
    p.add_argument('--episode-length', type=int, default=1440,
                   help='每天步数（默认 1440 = 完整一天；调小仅用于冒烟）')
    p.add_argument('--soc-init', choices=['uniform', 'train'], default='uniform',
                   help='初始 SOC：uniform=统一 --soc0（默认 0.25，跨方法公平）；'
                        'train=镜像各方法训练时的重置方式（td3 系为 U(0.5,5.5)）')
    p.add_argument('--soc0', type=float, default=0.25,
                   help='--soc-init uniform 时的初始 SOC [MWh]（默认 0.25 = env.reset() 的 E）')
    p.add_argument('--viol-limit', type=float, default=3.0, help='波动越限阈值 MW（默认 3.0）')
    p.add_argument('--deg-cost', type=float, default=490.0,
                   help='放电退化成本 元/MWh（默认 490）')

    # ---- 运行环境 ----
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--seed', type=int, default=42, help='种子（只影响 --soc-init train 的抽样）')
    p.add_argument('--phase-feat', type=int, default=None, choices=[0, 1, 2],
                   help='相位特征模式（默认从 actor_6.pkl 自动探测）')
    p.add_argument('--seg-method', default=None,
                   choices=['expert', 'percentile', 'kmeans'],
                   help='覆盖 piecewise 臂（如 td3perseg）的分段方法')

    # ---- 输出 ----
    p.add_argument('--output', default=os.path.join('output', 'result_statistic'),
                   help='输出目录（默认 output/result_statistic）')
    p.add_argument('--no-save-steps', action='store_true', help='不写每步明细 CSV')
    p.add_argument('--no-tensorboard', action='store_true', help='不写 TensorBoard')
    p.add_argument('--no-verify-days', action='store_true',
                   help='跳过"四天统计值仍匹配"的核验（数据被换过时仍按 iloc 跑）')
    p.add_argument('--stub-engine', action='store_true',
                   help='用**确定性 stub 引擎**代替 WFSim（无需 MATLAB，仅用于管线自检；'
                        '指标数值无物理意义，不可引用）')
    p.add_argument('--dry-run', action='store_true',
                   help='只打印计划与四天清单，不加载权重、不推进仿真')
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_projects:
        print('支持的方法（key: label）:')
        for key in T.PROJECT_ORDER:
            spec = T.PROJECTS[key]
            print('  {:<14s} {:>14s}   后端 = {}'.format(
                key, spec['label'],
                'FLORIS（纯 Python）' if spec.get('matlab') is False else 'WFSim（MATLAB）'))
        return 0

    # ---- 1. 四天（固定）----
    df_wind, days = load_days(verify=not args.no_verify_days)
    print('=' * 96)
    print('  四固定风速日评测（只推理，不训练）')
    print('=' * 96)
    print_days(days)

    # ---- 2. 模型路径（无参数直接退出，避免交互式卡住）----
    explicit = any(getattr(args, k, None) for k in T.PROJECT_ORDER) or args.models
    if not explicit:
        raise SystemExit('未提供任何模型路径。示例：\n'
                         '  python result_statistic/test_four_days.py '
                         '--td3per output/TD3PER-run19\n'
                         '用 --list-projects 看支持的方法。')
    paths = T.resolve_model_paths(args)
    keys = list(paths.keys())
    n_runs = sum(len(v) for v in paths.values())
    n_ep = n_runs * len(days)
    print('\n将测试 %d 个方法、共 %d 组模型: %s'
          % (len(keys), n_runs,
             ', '.join(T.PROJECTS[k]['label'] for k in keys)))
    print('计划: %d 组模型 × %d 天 = %d 个 episode（每 episode %d 步）→ 合计 %d 步'
          % (n_runs, len(days), n_ep, args.episode_length, n_ep * args.episode_length))
    print('口径: 初始 SOC = %s | 越限阈值 %.1f MW | 退化成本 %.0f 元/MWh'
          % ('%.3f MWh（--soc-init uniform）' % args.soc0
             if args.soc_init == 'uniform' else '训练重置（--soc-init train）',
             args.viol_limit, args.deg_cost))
    wf = [k for k in keys if T.PROJECTS[k].get('matlab', True)]
    if wf:
        n_wf_steps = n_ep * args.episode_length
        print('⚠ 其中 %d 个方法走 WFSim（约 0.3~1 s/步）⇒ 本计划约 %.1f~%.1f 小时'
              % (len(wf), n_wf_steps * 0.3 / 3600.0, n_wf_steps * 1.0 / 3600.0))

    if args.dry_run:
        print('\n[dry-run] 不加载权重、不推进仿真。')
        return 0

    # ---- 3. 把本次口径写进 test_five_days 的模块常量（它按全局读取）----
    #      ProjectRunner / run_day 直接读这些模块级常量，故必须先设。
    T.EPISODE_LENGTH = int(args.episode_length)
    T.FLUCT_LIMIT = float(args.viol_limit)
    T.BESS_DEG_RMB_PER_MWH = float(args.deg_cost)
    # uniform 臂的初始 SOC：默认 0.25 MWh = `env.reset()` 的 E（与 max_E=0.5 同量级，
    # 不会像 3.0 那样在第一分钟被 GetBatteryActions 夹到 max_E 并计入 penalty_L）。
    T.SOC_INIT = float(args.soc0)

    # ---- 4. 设备 ----
    import torch
    if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()):
        device = torch.device('cuda:0')
        print('设备: GPU %s' % torch.cuda.get_device_name(device))
    else:
        device = torch.device('cpu')
        print('设备: CPU')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)          # 纯前向

    # ---- 5. MATLAB 引擎（只有走 WFSim 的方法才起）----
    engine = None
    if args.stub_engine:
        engine = install_stub_engine()
        print('\n' + '!' * 96)
        print('  ⚠ --stub-engine：WFSim 已被**确定性 stub 物理**取代（无需 MATLAB）。')
        print('    本次结果只用于验证"状态 → 动作 → 推进 → 记录"管线，指标数值无物理意义。')
        print('!' * 96)
    elif wf:
        engine = T.ensure_engine()
    else:
        print('  跳过 MATLAB 引擎（所选方法全部为纯 Python 后端）')

    # ---- 6. 输出 ----
    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)
    days_csv = os.path.join(out_dir, 'four_day_days.csv')
    steps_csv = os.path.join(out_dir, 'four_day_steps.csv')
    sum_csv = os.path.join(out_dir, 'four_day_summary.csv')
    by_day_csv = os.path.join(out_dir, 'four_day_by_day.csv')
    rank_csv = os.path.join(out_dir, 'four_day_ranking.csv')
    if os.path.isfile(steps_csv):
        os.remove(steps_csv)

    writer = None
    if not args.no_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(out_dir, 'tensorboard'))
        except Exception as exc:                           # noqa: BLE001
            print('[警告] tensorboard 不可用(%s)，只写 CSV' % exc)

    # ---- 7. 主循环：每 (方法 × run) 加载一次，跑完四天 ----
    all_days, step_rows_total = [], 0
    t_all = time.time()
    try:
        for key in keys:
            spec = T.PROJECTS[key]
            label = spec['label']
            for run_idx, model_dir in enumerate(paths[key]):
                run_no = run_idx + 1
                model_name = os.path.basename(model_dir.rstrip('\\/'))
                print('\n' + '-' * 96)
                print('  方法 %s (%s)  第 %d/%d 组模型  <- %s'
                      % (label, key, run_no, len(paths[key]), model_dir))
                print('-' * 96)
                run_args = T._build_args(device, args.seed,
                                         T._detect_hidden(model_dir, 6))
                if args.seg_method:
                    run_args.seg_method = args.seg_method
                runner = T.ProjectRunner(key, model_dir, device, run_args, engine,
                                         phase_feat=args.phase_feat,
                                         soc_init=args.soc_init,
                                         rng=np.random.RandomState(args.seed),
                                         verbose=True)
                run_days = []
                for i, d in enumerate(days):
                    day_idx = i + 1
                    t0 = time.time()
                    day, rows = runner.run_day(d['series'], day_idx, d['iloc'], d['col'])
                    day.update(dict(run=run_no, model=model_name, model_dir=model_dir,
                                    day_tag=d['tag'], day_label=d['label']))
                    all_days.append(day)
                    run_days.append(day)
                    print('    %-16s (iloc=%3d, 日均 %5.2f m/s): 越限 %4d 次, '
                          '越限总量 %8.4f MW, 利润 %11.2f 元, 末SOC %5.3f  [%.0fs]'
                          % (d['tag'], d['iloc'], d['mean'], day['viol_count'],
                             day['viol_excess_mw'], day['profit_rmb'],
                             day['soc_end'], time.time() - t0))

                    # 逐步明细（带 project/run/model/day 列，便于逐日回看）
                    if not args.no_save_steps and rows:
                        df_rows = pd.DataFrame(rows)
                        df_rows.insert(0, 'run', run_no)
                        df_rows.insert(1, 'model', model_name)
                        df_rows.insert(2, 'day_tag', d['tag'])
                        df_rows.to_csv(steps_csv, mode='a',
                                       header=(step_rows_total == 0), index=False)
                        step_rows_total += len(df_rows)

                    if writer is not None:
                        for m in METRICS:
                            writer.add_scalar('%s/%s' % (label, m), float(day[m]), day_idx)
                        for r in rows:
                            writer.add_scalar('%s/%s/P_F' % (label, d['tag']),
                                              float(r['P_F']), int(r['step']))
                            writer.add_scalar('%s/%s/SOC' % (label, d['tag']),
                                              float(r['SOC']), int(r['step']))

                # 每跑完一组模型就把该组的逐日表落盘（中途崩了也不丢）
                pd.DataFrame(all_days).to_csv(days_csv, index=False,
                                              encoding='utf-8-sig')
    finally:
        if writer is not None:
            writer.close()
        if engine is not None:
            T.quit_engine()

    if not all_days:
        print('\n没有产生任何结果。')
        return 1

    # ---- 8. 汇总落盘 ----
    df_days = pd.DataFrame(all_days)
    df_sum = summarise(df_days)
    df_by_day = by_day_table(df_days)

    df_days.to_csv(days_csv, index=False, encoding='utf-8-sig')
    df_sum.to_csv(sum_csv, index=False, encoding='utf-8-sig')
    df_by_day.to_csv(by_day_csv, index=False, encoding='utf-8-sig')
    df_sum[['rank', 'project', 'run', 'viol_count_mean', 'viol_excess_mw_mean',
            'profit_rmb_mean', 'profit_rmb_sum']].to_csv(
        rank_csv, index=False, encoding='utf-8-sig')

    print('\n' + '=' * 96)
    print('  四天清单（固定）: %s' % ', '.join(
        '%s=iloc%d' % (d['tag'], d['iloc']) for d in days))
    print_report(df_days, df_sum, days)

    print('\n输出文件:')
    print('  逐日明细   : %s' % days_csv)
    if not args.no_save_steps:
        print('  逐步明细   : %s (%d 行)' % (steps_csv, step_rows_total))
    print('  四天汇总   : %s' % sum_csv)
    print('  每天×方法  : %s' % by_day_csv)
    print('  排名       : %s' % rank_csv)
    print('  总耗时     : %.1f min' % ((time.time() - t_all) / 60.0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
