# -*- coding: utf-8 -*-
"""正式口径批量改写：warmup 期上层动作调度**传入电价** ⇒ 第 6 维（SOC 目标信号）按价格游走。

用户口径（2026-10-08 修正版）::

    SOC 目标生成逻辑 = 电价驱动随机游走：300 上调 / 580 不动 / 980 下调（0.3–0.5 MWh，随机）；
    下层 = 调用规则 BESS 组件**完全追踪**该目标（Δ≡0 ⇒ trackerr≡0）。

实现落点：`WarmupSchedule.action(price)` —— 目标语义 `target = 3 + 5·a6` ⇒
"目标步幅 0.3–0.5 MWh" ⇔ "信号步幅 0.06–0.10"（已在 `hsrl_warmup` 里实现）。
本脚本只把 12 个 main 的调用行从 `action()` 改成 `action(float(observation[0]))`
（`aifflorpave` 不进本表 —— 由 `port_aifflorpave.py` 从 aifflor_main 重新派生）。

锚点在 13 个 main 里**逐字相同**，命中数必须精确为 1，任一不符整体不写盘。

用法::

    python agentutil/port_warmup_sig_walk.py [--apply|--check|--revert]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

TARGETS = [
    'aifflor/aifflor_main.py',
    'singleaif/singleaif_main.py',
    'td3per/td3per_main.py',
    'td3perrb/td3perrb_main.py',
    'td3perseg/td3perseg_main.py',
    'td3prle/td3prle_main.py',
    'td3prrbb/td3prrbb_main.py',
    'td3prrlrb/td3prrlrb_main.py',
    'td3leprrlrb/td3leprrlrb_main.py',
    'yawflor/yawflor_main.py',
    'prle/prle_main.py',
    'prrbb/prrbb_main.py',
]

# ⚠ 缩进必须与 main 里的真实写法逐字一致（16 空格 —— 该行在 `if step <= args.warmup:` 内，
#   而那个 if 在 `if episode_steps % 5 == 0:` 内）。用 12 空格会**子串命中**到行中间，
#   把新块插进注释里 ⇒ IndentationError（本脚本第一版已踩过）。
OLD = ('                _wa = warmup_sched.action()'
       '          # None ⇒ 对照臂 (--warmup_sched random)\n')
NEW = ('                # 传电价 ⇒ 第 6 维(SOC 目标信号)按价格游走(300 上调/580 不动/980 下调);\n'
       '                # 下层用规则组件**完全追踪**该目标(Δ≡0 ⇒ trackerr≡0)。\n'
       '                # --warmup_sig fixed 可退回"信号恒 0"的旧口径。\n'
       '                _wa = warmup_sched.action(float(observation[0]))'
       '   # None ⇒ 对照臂 (--warmup_sched random)\n')


def _sub(text, a, b):
    n = text.count(a)
    return (text.replace(a, b, 1) if n == 1 else None), n


def main():
    apply = '--apply' in sys.argv
    check = '--check' in sys.argv
    rev = '--revert' in sys.argv
    ok = True
    for f in TARGETS:
        disk = open(f, encoding='utf-8').read()
        if rev:
            new, n = _sub(disk, NEW, OLD)
            if new is None:
                print('  %-32s **反向未命中(%d)**' % (f, n))
                ok = False
                continue
            open(f, 'w', encoding='utf-8', newline='').write(new)
            print('  %-32s 已回滚' % f)
            continue
        if check:
            r, n1 = _sub(disk, NEW, OLD)
            if r is None:
                print('  %-32s **反向未命中(%d)**' % (f, n1))
                ok = False
                continue
            rep, n2 = _sub(r, OLD, NEW)
            good = (rep == disk)
            ok = ok and good
            print('  %-32s %s 往返逐位一致=%s' % (f, 'OK ' if good else '**失败**', good))
            continue
        new, n = _sub(disk, OLD, NEW)
        good = (new is not None)
        ok = ok and good
        print('  %-32s %s 命中 %d' % (f, 'OK  ' if good else '**失败**', n))
        if good and apply:
            open(f, 'w', encoding='utf-8', newline='').write(new)
    print(('--check: 全部可逆 ✅' if ok else '--check: **FAIL**') if check else
          ('✅ 已写盘（aifflorpave 需重新派生）' if apply else
           ('✅ 已回滚' if rev else '(dry-run)')))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
