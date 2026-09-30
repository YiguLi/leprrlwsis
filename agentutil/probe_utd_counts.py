# -*- coding: utf-8 -*-
"""探针：验证 `--utd_h/--utd_l` **真的**控制了每块的 `update_policy` 调用次数。

用法（须用本项目解释器）
------------------------
    python agentutil/probe_utd_counts.py aifflor/aifflor_main.py \
        --warmup 200 --max_steps 260 --utd_h 4 --utd_l 6

做法
----
`runpy` 以 `__main__` 跑目标 main（参数原样透传），但先把 `update_policy` 包一层
**只计数、不改行为**的包装；H/L 用 `nb_actions` 区分（>1 = H 层 6 维动作，=1 = L 层）。

判据
----
    H 调用次数 == utd_h × 块数
    L 调用次数 == utd_l × 块数
（块数 = 更新门 `step > warmup` 且 `episode_steps % 5 == 4` 的次数）

⚠ 只对 **aifflor / yawflor** 这类无 MATLAB 依赖的 main 可直接用；td3per 系列的
`import matlab.engine` 会拉起 MATLAB，本探针不为此负责。
"""
import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

for _s in (sys.stdout, sys.stderr):                    # 先修流，否则重定向时中文报表会崩
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError, OSError):
        pass

from floris_import_guard import check_interpreter, sanitize        # noqa: E402
check_interpreter()
sanitize()

import aifflor.aifflor_ddpg as D                                   # noqa: E402

CNT = {'H': 0, 'L': 0}


def _wrap(orig):
    def f(self, *a, **k):
        CNT['H' if int(getattr(self, 'nb_actions', 1)) > 1 else 'L'] += 1
        return orig(self, *a, **k)
    return f


D.SP_TD3.update_policy = _wrap(D.SP_TD3.update_policy)

if len(sys.argv) < 2:
    print(__doc__)
    raise SystemExit(2)
TARGET = sys.argv[1]
sys.argv = [TARGET] + sys.argv[2:]

_rc = 0
try:
    runpy.run_path(TARGET, run_name='__main__')
except SystemExit as _e:
    _rc = int(_e.code or 0)

print('=' * 72)
print('[probe] update_policy 调用次数: H=%d  L=%d  (exit=%d)' % (CNT['H'], CNT['L'], _rc))
if CNT['H'] and CNT['L']:
    print('[probe] H/L 比值 = %.4f  (应与 --utd_h/--utd_l 一致)' % (CNT['H'] / CNT['L']))
elif CNT['H']:
    print('[probe] 只统计到 H 层调用（该 arm 无 L 层?）')
else:
    print('[probe] ⚠ 没有统计到任何 update_policy 调用（warmup 是否过大 / 运行是否提前失败）')
print('=' * 72)
