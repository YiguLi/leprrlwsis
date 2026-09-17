#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""critic 基线初始化 的 A/B 驱动（同 seed 短程对照）
==================================================
两个臂（唯一差别是 critic 输出层 bias 是否用 warmup 观测初始化）：

  base : --critic_bias_init none      （旧行为，Q 从 ~0.1 起步）
  bias : --critic_bias_init estimate  （b = r̄_block/(1−γ)，在首次策略更新前生效）

两臂共用同一 `--seed`、同一 `--max_steps`、同一份数据/环境 ⇒ 只改一个开关，
随机数消费结构不变（不动抽样次数），是同 seed 配对最干净的形式。

用法
----
    python ab_critic_bias.py --project td3perrb --seed 101 --steps 20000
    python ab_critic_bias.py --project td3per --steps 40000 --python <解释器路径>
    python ab_critic_bias.py --dry-run            # 只打印将要执行的命令

跑完会自动用 analyse_q_scale.transient_stats 对比两侧的
「尺度瞬态占比」与「末段相对 TD 误差」，并给出 episode 收益的末段均值。
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors='replace')
    except Exception:
        pass

import test_five_days as T                      # 复用 PROJECTS 注册表(import 不启动 MATLAB)
from analyse_q_scale import transient_stats

ARMS = [('base', 'none'), ('bias', 'estimate')]


def build_cmd(py, main, seed, steps, out_parent, arm_mode):
    cmd = [py, main, '--seed', str(seed), '--critic_bias_init', arm_mode]
    if steps and steps > 0:
        cmd += ['--max_steps', str(steps)]
    cmd += ['--output', out_parent]
    return cmd


def tail_mean(path, col, frac=0.1):
    df = pd.read_csv(path)
    if col not in df or len(df) == 0:
        return float('nan')
    k = max(1, int(len(df) * frac))
    return float(df[col].iloc[-k:].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='td3perrb', choices=sorted(T.PROJECTS.keys()))
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--steps', type=int, default=20000,
                    help='每臂训练步数（0=用 main 的默认预算 1.44M）')
    ap.add_argument('--output', default=os.path.join('output', '_ab_bias'))
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    main_py = T.PROJECTS[a.project]['ref']
    env_label = T.PROJECTS[a.project]['label']
    print('项目 {} → main={} (env={})'.format(a.project, main_py, env_label))
    print('seed={}  steps={}  输出根={}'.format(a.seed, a.steps or '默认(1.44M)', a.output))

    runs = {}
    for arm, mode in ARMS:
        parent = os.path.join(a.output, arm)
        cmd = build_cmd(a.python, main_py, a.seed, a.steps, parent, mode)
        print('\n[{}] {}'.format(arm, ' '.join(cmd)))
        if a.dry_run:
            continue
        os.makedirs(parent, exist_ok=True)
        rc = subprocess.call(cmd)                 # 继承 stdio，不抓管道
        if rc != 0:
            print('[{}] 退出码 {}，终止'.format(arm, rc))
            return rc
        hits = sorted(glob.glob(os.path.join(parent, '*', 'update_stats_H.csv')))
        runs[arm] = os.path.dirname(hits[0]) if hits else None

    if a.dry_run:
        print('\n(dry-run 结束，未执行)')
        return 0

    print('\n' + '=' * 100)
    print('对照结果（同一 seed={}，唯一差别 = critic bias 是否初始化）'.format(a.seed))
    print('=' * 100)
    rows = []
    for arm, _ in ARMS:
        d = runs.get(arm)
        if not d:
            print('[{}] 未找到输出'.format(arm))
            continue
        st = transient_stats(os.path.join(d, 'update_stats_H.csv'))
        ep = sorted(glob.glob(os.path.join(d, 'df_episode_*.csv')))
        prof = tail_mean(ep[0], 'episode_profit') if ep else float('nan')
        rows.append((arm, st, prof))
        print('\n[{}] {}'.format(arm, d))
        if st:
            print('   |Q_p| 终值={:.0f}  达50%={:.3f}预算  达90%={:.3f}预算  相对TD(末10%)={:.3f}'
                  .format(st['q_final'], st['f50'], st['f90'], st['r_last']))
            print('   |Q_p| 曲线={}'.format(st['curve']))
        print('   末段 episode_profit 均值={:.1f}'.format(prof))

    if len(rows) == 2 and rows[0][1] and rows[1][1]:
        b, s = rows[0][1], rows[1][1]
        print('\n' + '-' * 100)
        print('判据: 达90%预算 {:.3f} → {:.3f} ({:+.1%}) | 达50%预算 {:.3f} → {:.3f} ({:+.1%})'
              .format(b['f90'], s['f90'], s['f90'] - b['f90'],
                      b['f50'], s['f50'], s['f50'] - b['f50']))
        print('      相对TD(末10%) {:.3f} → {:.3f} | episode_profit(末段) {:.1f} → {:.1f}'
              .format(b['r_last'], s['r_last'], rows[0][2], rows[1][2]))
        if s['f90'] < b['f90'] - 0.02:
            print('      ⇒ 瞬态显著缩短（预期方向），可继续加长跑并核对最终指标')
        else:
            print('      ⇒ 瞬态未见明显改善：检查 --critic_bias_init 是否生效(启动日志会打印)、'
                  '或 warmup 样本是否足够')
    return 0


if __name__ == '__main__':
    sys.exit(main())
