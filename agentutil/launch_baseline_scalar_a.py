#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""launch_baseline_scalar_a.py —— aifflor 的 baseline 臂（--scalar_wt）一键启动。

刻意设计的 baseline（与 aifflor 默认臂的**全部差异**只有三条）：

  1. `--c1_smooth 0`        c1 用**非平滑**硬折点：c1 = |ΔP| + 20·(|ΔP|−3)⁺
  2. `--critic_arch v1`     神经网络结构 = PER 的 fc1→fc4→cat(a)→fc2→fc5→fc3
                            （hidden1/2 = 16/100，与 PER_main0 的默认逐位相同）
  3. `--scalar_wt 1`        五台风机的轴向因子**全部由动作第 0 维设定**（a_i = a_0）；
                            第 1..4 维存在但无用（投影线性 ⇒ 对它们的梯度恒为 0）。
                            第 6 维 g（块总放电量）仍是独立自由度。

其余全部沿用 aifflor 当前默认（HSRL、nstep=5、γ=0.95、utd_h=4、
rmsize=40000、PER α=0.6/β=0.4、warmup=7200、wt_noise=0.05→0、s_p=1000/s_c=100、
λ∈[2,50]、d_block=15、FLORIS 环境、加噪风速池）。

⚠ 2026-11-03 起 aifflor 的 `--bsize`/`--rate` 默认改成 **32 / 0.01**
（tau 仍 0.001），本 launcher 不固定这三项 ⇒ **现在跑 baseline 会自动用新默认**。
要复现 2026-11-03 之前的历史 baseline（256 / 0.003），加：
`--extra --bsize 256 --rate 0.003`。

用法：
    python agentutil/launch_baseline_scalar_a.py                  # 跑满（默认输出目录）
    python agentutil/launch_baseline_scalar_a.py --max-steps 300  # 冒烟
    python agentutil/launch_baseline_scalar_a.py --dry-run        # 只打印命令
    python agentutil/launch_baseline_scalar_a.py --extra --seed 1 # 额外透传（放最后）

契约测试：python agentutil/test_aifflor_scalar_wt.py
"""
import argparse
import os
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

PY = sys.executable

# ---- 三条 baseline 差异（改这里就是改 baseline 的定义）----
BASELINE_FLAGS = [
    '--c1-mode', 'uniform',
    '--c1_smooth', '0',
    '--critic_arch', 'v1',
    '--hidden1', '16', '--hidden2', '100', '--init_w', '0.003',
    '--scalar_wt', '1',
]
COMMON_FLAGS = [
    '--hsrl',
    '--max_episode_length', '1440',
    '--seed', '0',
    '--progress', '1',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join('output', 'BASELINE-SCALAR-V1'))
    ap.add_argument('--wind', default='FFC_speed_noise_1.csv',
                    help='风速池（默认 = run3/V6 用的加噪池）')
    ap.add_argument('--max-steps', type=int, default=0, help='0 = 跑满（args 默认预算）')
    ap.add_argument('--no-clean', action='store_true',
                    help='保留旧输出目录（默认会清空，避免 main 追加写造成混数据）')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--extra', nargs=argparse.REMAINDER, default=[],
                    help='额外透传给 aifflor_main 的参数（**必须放命令行最后**）')
    a = ap.parse_args()

    cmd = [PY, os.path.join('aifflor', 'aifflor_main.py')] + COMMON_FLAGS + \
          ['--output', a.out] + BASELINE_FLAGS + list(a.extra or [])
    if a.max_steps > 0:
        cmd += ['--max_steps', str(a.max_steps)]

    env = dict(os.environ)
    env['HSRL_WIND_CSV'] = a.wind
    env['PYTHONIOENCODING'] = 'utf-8'

    print('=' * 78)
    print('aifflor baseline（--scalar_wt）：神经网络=PER 结构、c1=非平滑、五台共用 a[0]')
    print('=' * 78)
    print('  风速池      : %s' % a.wind)
    print('  输出目录    : %s' % a.out)
    print('  与默认臂差异: --c1_smooth 0 | --critic_arch v1 (16/100) | --scalar_wt 1')
    print('  其余        : aifflor 默认（HSRL / nstep5 / γ0.95 / utd_h4 / bsize256 / ...）')
    print('  命令        :')
    print('    ' + ' '.join(cmd))
    print('=' * 78)
    if a.dry_run:
        return 0

    if not a.no_clean and os.path.exists(a.out):
        shutil.rmtree(a.out, ignore_errors=True)
        print('[clean] 已清空旧目录 %s' % a.out)
    rc = subprocess.call(cmd, env=env)
    print('[done] rc=%d  产物: %s' % (rc, a.out))
    return rc


if __name__ == '__main__':
    sys.exit(main())
