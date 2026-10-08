# -*- coding: utf-8 -*-
"""正式口径批量改写：episode 起始 SOC → **固定 3.0 MWh**（`--soc_reset_low/high` 默认 3/3）。

用户规格（2026-10-08）："将 soc 起始点设为 3"。旧口径是所有 main 在 episode 重置时
从 **U[0.5, 5.5]** 随机（实测冒烟抽到 4.6695），会让前若干块的收益/退化都花在"SOC 拉回中性"
上、并引入与策略无关的方差。

三种写法（逐文件精确命中，任一不符整体不写盘）：
  A. 9 个 main 硬编码两行（注释 + `reset_E = float(np.random.uniform(0.5, 5.5))`）
     → 改成读 `args.soc_reset_low/high`（默认 3/3 ⇒ 固定 3；传 0.5/5.5 复现旧口径）。
  B. `td3prle` / `td3prrbb` 的局部常量 `soc_reset_low, soc_reset_high = 0.5, 5.5`
     → 改成读 args。
  C. `aifflorpave` 不进本表（由 `port_aifflorpave.py` 从 aifflor_main 重新派生）。

用法::

    python agentutil/port_soc_start.py [--apply|--check|--revert]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

A_TARGETS = [
    'aifflor/aifflor_main.py',
    'singleaif/singleaif_main.py',
    'td3per/td3per_main.py',
    'td3perrb/td3perrb_main.py',
    'td3perseg/td3perseg_main.py',
    'td3prrlrb/td3prrlrb_main.py',
    'td3leprrlrb/td3leprrlrb_main.py',
    'yawflor/yawflor_main.py',
]
B_TARGETS = ['td3prle/td3prle_main.py', 'td3prrbb/td3prrbb_main.py']

A_OLD = ('            # SOC 随机化: 每个 episode(1440步) 从 U[0.5,5.5] 重新采样\n'
         '            reset_E = float(np.random.uniform(0.5, 5.5))\n')
A_NEW = ('            # episode 起始 SOC: **正式口径 = 固定 3.0 MWh**\n'
         '            #   (旧口径 = 每 episode 从 U[0.5,5.5] 随机 ⇒ 传 '
         '--soc_reset_low 0.5 --soc_reset_high 5.5 复现)\n'
         '            reset_E = float(np.random.uniform(float(args.soc_reset_low),\n'
         '                                              float(args.soc_reset_high)))\n')

B_OLD = '    soc_reset_low, soc_reset_high = 0.5, 5.5\n'
B_NEW = ('    # episode 起始 SOC: **正式口径 = 固定 3.0**（默认来自 --soc_reset_low/high）\n'
         '    soc_reset_low, soc_reset_high = float(args.soc_reset_low), float(args.soc_reset_high)\n')

TABLE = [(f, A_OLD, A_NEW, 'A') for f in A_TARGETS] + \
        [(f, B_OLD, B_NEW, 'B') for f in B_TARGETS]


def _do(text, path, idx):
    for f, old, new, kind in TABLE:
        if f != path:
            continue
        a, b = (old, new) if idx == 0 else (new, old)
        n = text.count(a)
        return (text.replace(a, b, 1) if n == 1 else None), n, kind
    return None, 0, '?'


def main():
    apply = '--apply' in sys.argv
    check = '--check' in sys.argv
    rev = '--revert' in sys.argv
    ok = True
    for f, _o, _n, _k in TABLE:
        disk = open(f, encoding='utf-8').read()
        if rev:
            new, n, kind = _do(disk, f, 1)
            if new is None:
                print('  %-34s **反向未命中(%d)**' % (f, n))
                ok = False
                continue
            open(f, 'w', encoding='utf-8', newline='').write(new)
            print('  %-34s 已回滚' % f)
            continue
        if check:
            r, n1, kind = _do(disk, f, 1)
            if r is None:
                print('  %-34s **反向未命中(%d)**' % (f, n1))
                ok = False
                continue
            rep, n2, _ = _do(r, f, 0)
            good = (rep == disk)
            ok = ok and good
            print('  %-34s %s 往返逐位一致=%s（形状 %s）' % (f, 'OK ' if good else '**失败**', good, kind))
            continue
        new, n, kind = _do(disk, f, 0)
        good = (new is not None)
        ok = ok and good
        print('  %-34s %s 形状 %s 命中 %d' % (f, 'OK  ' if good else '**失败**', kind, n))
        if good and apply:
            open(f, 'w', encoding='utf-8', newline='').write(new)
    print(('--check: 全部可逆 ✅' if ok else '--check: **FAIL**') if check else
          ('✅ 已写盘（aifflorpave 需重新派生）' if apply else
           ('✅ 已回滚' if rev else '(dry-run)')))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
