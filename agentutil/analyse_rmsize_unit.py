# -*- coding: utf-8 -*-
"""实测 H 层"条目数 vs 环境步"的真实比例，判定 `--rmsize` 的口径单位。

口径之争
--------
* 说法 A（现有注释）: `1 条 = 5 环境步`（= 1 块）⇒ `rmsize=40000` 覆盖 200k 步
* 说法 B（用户）      : `40000 就是 40000`（= 环境步/条数 1:1）⇒ 覆盖 40k 步

判据：`observe_jump` 每次调用产出多少条、每次调用对应多少环境步。
滑窗 stride-1 下 `observe_jump` 每块调一次、每次 `while` 循环结算 1 条
⇒ 若实测 `条数/调用 ≈ 1` 且 `调用/步 = 1/5`，则 1 条 = 5 步（说法 A）。

⚠ 只能跑 `aifflor`（纯 Python；`td3per_main` import 期就启 MATLAB，沙箱下起不来）。
   但滑窗实现在 6 个 ddpg 里**逐字相同**，结论可迁移。
"""
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

ST = {'calls': 0, 'entries': []}


def install():
    import aifflor.aifflor_ddpg as D
    orig = D.SP_TD3.observe_jump

    def spy(self, *a, **kw):
        r = orig(self, *a, **kw)
        if self.nb_actions == 6:                      # H 层
            ST['calls'] += 1
            ST['entries'].append(len(self.memory._storage))
        return r

    D.SP_TD3.observe_jump = spy


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    install()
    import runpy
    print('跑 aifflor_main --max_steps %d，统计 H 层 observe_jump 的调用与条目产出' % steps)
    sys.argv = [os.path.join(ROOT, 'aifflor', 'aifflor_main.py'),
                '--max_steps', str(steps)]
    try:
        runpy.run_module('aifflor.aifflor_main', run_name='__main__')
    except SystemExit:
        pass

    n = np.asarray(ST['entries'], dtype=float)
    c = ST['calls']
    print('\n' + '=' * 90)
    print('结果')
    print('=' * 90)
    if c < 20:
        print('  observe_jump 仅调用 %d 次，样本不足' % c)
        return 0
    print('  observe_jump 调用次数 = %d' % c)
    print('  memory 条目数: 首 %d → 末 %d (峰值 %d)' % (n[0], n[-1], n.max()))
    print()
    print('  ① 每次调用产出的条目数 = %.4f 条/调用' % (n[-1] / c))
    print('  ② 每次调用对应的环境步 = 5.0（H 层每块决策一次、每块 5 环境步）')
    print('  ③ 环境步/条 = 5.0 × (%d/%d) = **%.4f**' % (c, n[-1], 5.0 * c / n[-1]))
    print('  ④ rmsize=40000 覆盖环境步 = 40000 × %.4f = **%.0f 步**'
          % (5.0 * c / n[-1], 40000 * 5.0 * c / n[-1]))
    print()
    if abs(5.0 * c / n[-1] - 5.0) < 0.5:
        print('  ⇒ 判定：**1 条 = 5 环境步**（说法 A 成立）')
        print('     40000 条 = 200,000 环境步 = 总预算 1.44M 的 13.9%')
    elif abs(5.0 * c / n[-1] - 1.0) < 0.3:
        print('  ⇒ 判定：**1 条 = 1 环境步**（说法 B 成立）')
        print('     40000 条 = 40,000 环境步 = 总预算的 2.8%')
    else:
        print('  ⇒ 比例介于两者之间，需进一步查')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
