# -*- coding: utf-8 -*-
"""正式口径契约：`warmup = --bsize × --warmup_bsize_mult`（默认 4）+ **warmup 期零更新**。

判据（逐 main + 共享真源）：
  ① 12 个正式 main（+ 派生臂 aifflorpave）的 `--warmup` 默认是 `None`（WARMUP_AUTO），
     且都调用 `<hwarm>.resolve_warmup(args)`；
  ② `resolve_warmup` 的算术：未传 ⇒ mult×bsize；显式传 ⇒ 尊重显式值（可复现历史 run）；
  ③ **零更新契约**：每个 `update_policy()` 调用点都必须位于（更浅缩进的）
     `if step > args.warmup:` 守卫块内；
  ④ `hsrl_agent_init.BiasInitTracker.on_block_end` 在 warmup 期发现
     `agent.update_count != 0` 时**抛错**，为 0 时不抛；
  ⑤ `first_trigger_step(bsize×mult)` 落在块末（≡4 mod 5）且 = warmup 本身（当 warmup≡4 mod 5）。

用法::  python agentutil/test_warmup_bsize_uniform.py     # exit 0 = PASS
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import hsrl_warmup as hwarm            # noqa: E402
import hsrl_agent_init as hinit        # noqa: E402

MAINS = [
    'td3per/td3per_main.py', 'td3perseg/td3perseg_main.py', 'td3perrb/td3perrb_main.py',
    'td3prrlrb/td3prrlrb_main.py', 'td3prrbb/td3prrbb_main.py', 'td3prle/td3prle_main.py',
    'td3leprrlrb/td3leprrlrb_main.py', 'singleaif/singleaif_main.py',
    'aifflor/aifflor_main.py', 'yawflor/yawflor_main.py',
    'prle/prle_main.py', 'prrbb/prrbb_main.py',
    'aifflorpave/aifflorpave_main.py',      # 派生臂，应自动继承
]
FAIL = []


def check(cond, msg):
    print(('  OK   ' if cond else '  FAIL ') + msg)
    if not cond:
        FAIL.append(msg)


class _A(object):
    def __init__(self, bsize, warmup=None, mult=0.0):
        self.bsize, self.warmup, self.warmup_bsize_mult = bsize, warmup, mult


class _Agent(object):
    def __init__(self, uc):
        self.update_count = uc


print('=' * 84)
print('① --warmup 默认 WARMUP_AUTO + resolve 调用（逐 main）')
for f in MAINS:
    s = open(f, encoding='utf-8').read()
    lines = s.splitlines()
    ai = [i for i, l in enumerate(lines) if "add_argument('--warmup'," in l]
    win = '\n'.join(lines[ai[0]:ai[0] + 4]) if ai else ''
    # `--warmup` 默认必须是 None（= WARMUP_AUTO）；标记 WARMUP_AUTO 只在 resolve 行上
    # （`default=200,` 那行**故意不加行内注释**：单行写法的 main 会被注释吞掉 type=int）。
    has_auto = ('default=None' in win) and ('default=200' not in win)
    alias = re.search(r'^import hsrl_warmup as (\w+)', s, re.M)
    has_call = bool(alias) and ('%s.resolve_warmup(args)' % alias.group(1)) in s
    check(has_auto and has_call, '%-34s auto=%s resolve=%s' % (f, has_auto, has_call))

print('=' * 84)
print('② resolve_warmup 算术（**固定 1440 步，与 --bsize 解耦**）')
check(hwarm.WARMUP_STEPS == 1440, 'WARMUP_STEPS = 1440（一个 episode/一天）')
a = hwarm.resolve_warmup(_A(256), verbose=False)
check(a.warmup == 1440, 'bsize=256, 未传 ⇒ 1440（实测 %d）' % a.warmup)
b = hwarm.resolve_warmup(_A(64), verbose=False)
check(b.warmup == 1440, 'bsize=64, 未传 ⇒ **同样 1440**（与 bsize 解耦，实测 %d）' % b.warmup)
c = hwarm.resolve_warmup(_A(256, warmup=200), verbose=False)
check(c.warmup == 200, '显式 --warmup 200 ⇒ 200（最高优先，实测 %d）' % c.warmup)
d = hwarm.resolve_warmup(_A(256, mult=4.0), verbose=False)
check(d.warmup == 1024, '旧口径 --warmup_bsize_mult 4 ⇒ 1024（实测 %d）' % d.warmup)
e = hwarm.resolve_warmup(_A(512), verbose=False)
check(e.warmup == 1440, 'bsize=512 ⇒ 仍 1440（实测 %d；此时 H 缓冲 288<512 会告警）' % e.warmup)

print('=' * 84)
print('③ 零更新契约：update_policy() 必须在更浅缩进的 `if step > args.warmup:` 块内')
for f in MAINS:
    s = open(f, encoding='utf-8').read().splitlines()
    bad = []
    for i, l in enumerate(s):
        if 'update_policy()' not in l or l.strip().startswith('#'):
            continue
        ind = len(l) - len(l.lstrip())
        ok = False
        for j in range(i - 1, max(-1, i - 20), -1):
            g = s[j]
            gi = len(g) - len(g.lstrip())
            if g.strip().startswith('if step > args.warmup:') and gi < ind:
                ok = True
                break
        if not ok:
            bad.append(i + 1)
    check(not bad, '%-34s 未受保护调用点: %s' % (f, bad if bad else '无'))

print('=' * 84)
print('④ BiasInitTracker：warmup 期 update_count!=0 ⇒ 抛错；==0 ⇒ 不抛')
t = hinit.BiasInitTracker(mode='none', warmup=100, verbose=False)
t.n_warmup_blocks = 0
raised = False
try:
    t.on_block_end(50, 1.0, 0.1, _Agent(3))
except AssertionError:
    raised = True
check(raised, 'warmup 期 update_count=3 ⇒ AssertionError')
t2 = hinit.BiasInitTracker(mode='none', warmup=100, verbose=False)
t2.n_warmup_blocks = 0
raised2 = False
try:
    t2.on_block_end(50, 1.0, 0.1, _Agent(0))
except AssertionError:
    raised2 = True
check((not raised2) and t2.n_warmup_blocks == 1, 'update_count=0 ⇒ 不抛且计数 +1（%d）' % t2.n_warmup_blocks)

print('=' * 84)
print('⑤ first_trigger_step 落在块末')
for w in (1000, 1024, 256, 200):
    ts = hinit.first_trigger_step(w)
    check(ts % 5 == 4 and ts >= w, 'warmup=%-5d ⇒ trigger=%-5d (≡4 mod 5 ✓)' % (w, ts))

print('=' * 84)
print('TEST PASSED' if not FAIL else '**TEST FAILED**: %s' % FAIL)
sys.exit(0 if not FAIL else 1)
