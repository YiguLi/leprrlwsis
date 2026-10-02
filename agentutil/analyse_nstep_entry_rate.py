# -*- coding: utf-8 -*-
"""实测 n 步滑窗下的**条目数 vs 环境步**比例，核对 `--rmsize` 的口径注释。

要回答
------
`td3per_main.py` 的 `--rmsize` 注释写：
    "覆盖环境步 = rmsize × 5 × nstep"   ⇒ 20000 条 × 5 × 5 = **500k 步**
而 `td3per_ddpg.py:524-531` 的滑窗实现注释写：
    "滑窗 stride-1 ⇒ 条目数 ≈ 块数"      ⇒ 20000 条 = 20000 块 = **100k 步**

两者差 5 倍。本脚本实测：
  * `memory._storage` 每条对应的环境步数
  * `_pending` 队列长度（应为 n-1 = 4，即"只浪费 n-1 条"）
  * episode 末尾部结算是否补齐（无浪费）

依据
----
滑窗 stride-1：每来一个新块（5 环境步）就结算队首一条
  ⇒ 每 5 环境步产出 1 条  ⇒ **1 条 = 5 环境步**
  ⇒ 覆盖环境步 = rmsize × 5（**与 nstep 无关**）

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\analyse_nstep_entry_rate.py --steps 3000
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

from floris_import_guard import check_interpreter as _chk, sanitize as _sanitize  # noqa: E402
_chk()
_sanitize()

import numpy as np  # noqa: E402

ST = {'samples': [], 'pend': []}


def install():
    """插桩 H 层 `observe_jump`。

    ⚠ 只能跑 `aifflor`（纯 Python）：`td3per_main.py` 在 import 期就
    `matlab.engine.start_matlab()`（`td3per_main.py:93`），沙箱下起不来。
    但两条路径的**滑窗实现逐字相同**（`aifflor_ddpg.py` 是 `td3per_ddpg.py` 的克隆），
    所以用 `aifflor` 测得的产出率对 `td3per` 同样成立。
    """
    import aifflor.aifflor_ddpg as D
    orig = D.SP_TD3.observe_jump

    def spy(self, *a, **kw):
        r = orig(self, *a, **kw)
        if self.nb_actions == 6:          # 只看 H 层
            ST['samples'].append(len(self.memory._storage))
            ST['pend'].append(len(getattr(self, '_pending', [])))
        return r

    D.SP_TD3.observe_jump = spy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--script', default=os.path.join('aifflor', 'aifflor_main.py'))
    ap.add_argument('--module', default='aifflor.aifflor_main')
    args, _u = ap.parse_known_args()

    install()
    import runpy
    print('=' * 96)
    print('跑 %s，--max_steps %d，实测 H 层条目产出率' % (args.script, args.steps))
    print('（滑窗实现与 td3per 逐字相同，故结论同样适用于 td3per）')
    print('=' * 96)
    sys.argv = [os.path.join(ROOT, args.script), '--max_steps', str(args.steps)]
    try:
        runpy.run_module(args.module, run_name='__main__')
    except SystemExit:
        pass

    s = np.asarray(ST['samples'], dtype=float)
    p = np.asarray(ST['pend'], dtype=float)
    n_obs = len(s)
    print('\n' + '=' * 96)
    print('结果')
    print('=' * 96)
    if n_obs < 10:
        print('  样本太少 (observe_jump 调用 %d 次)' % n_obs)
        return 0
    print('  observe_jump 被调用 %d 次（= H 层观察到的块数，1 块 = 5 环境步）' % n_obs)
    print('  memory 条目数：首 %d → 末 %d（峰值 %d）' % (s[0], s[-1], s.max()))
    print('  _pending 长度：min %d, max %d, 末 %d（nstep-1 = 4 ⇒ "只浪费 n-1 条"）'
          % (p.min(), p.max(), p[-1]))
    print()
    # 条目产出率：每条对应多少块
    blocks = n_obs
    entries = s[-1]
    print('  条目/块 = %.4f   ⇒ 每 %.3f 个块产出 1 条 (理论 stride-1 = 1.000 条/块)'
          % (entries / blocks, blocks / max(entries, 1)))
    print('  环境步/条 = %.3f   (理论 = 5.0 环境步/条，因为 1 块 = 5 步)'
          % (5.0 * blocks / max(entries, 1)))
    print()
    print('  外推到 --rmsize=20000：')
    print('    覆盖环境步 = 20000 × %.3f = **%.0f 步**'
          % (5.0 * blocks / max(entries, 1), 20000 * 5.0 * blocks / max(entries, 1)))
    print('    对照 `--rmsize` 注释的 "rmsize × 5 × nstep" = 20000 × 5 × 5 = 500000 步')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
