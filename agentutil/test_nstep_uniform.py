# -*- coding: utf-8 -*-
"""六个 TD3 方法的 **H 层 n 步机制**跨项目一致性测试 (批次 2 验收)。

对每个项目真跑移植后的 `observe_jump`/`_emit_nstep`/`_push_transition`
(绕过 `__init__` 直接装配最小实例 + 桩 memory), 断言:
  1. `nstep=1` ⇒ 每条直接落库 (与历史 1-step 口径同形), 不产生折叠。
  2. `nstep=3` 滑窗: 喂 6 个块 (r=1..6, 第 6 块 done=True) ⇒ 恰好 6 条折叠转移;
     r_sum = Σγ^k r_k (γ=0.5), 条数与起点符合"每个块都是某条转移的起点"。
  3. boot = γ^n; 窗口内出现 done ⇒ boot=0 且存 done=1 (截断语义)。
  4. 两个 PRLE 项目: 折叠块的 context **顺序拼接** (供 relabel 按块 γ 加权重算);
     非 PRLE 项目不传 context。

不需 MATLAB。
"""

import importlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

PROJS = [
    ('td3per',      'td3per.td3per_ddpg',           'SP_TD3',           False),
    ('td3perrb',    'td3perrb.td3perrb_ddpg',       'SP_TD3',           False),
    ('td3prrlrb',   'td3prrlrb.td3prrlrb_ddpg',     'SP_TD3',           False),
    ('td3prrbb',    'td3prrbb.td3prrbb_ddpg',       'SP_TD3',           False),
    ('td3prle',     'td3prle.td3prle_ddpg',         'SP_TD3_PRLE',      True),
    ('td3leprrlrb', 'td3leprrlrb.td3leprrlrb_ddpg', 'SP_TD3_LEPRRLRB',  True),
]
GAM, N = 0.5, 3
FAIL = []


def check(cond, msg):
    print(('  [OK]   ' if cond else '  [FAIL] ') + msg)
    if not cond:
        FAIL.append(msg)


class StubMem:
    """最小 memory: 只记 append 的顺序与参数。"""

    def __init__(self):
        self.data = []
        self._next_idx = 0

    def append(self, s, a, r, c, s2, d, context=None):
        self.data.append((s, a, r, c, s2, d, context))
        self._next_idx += 1


def make_agent(mod_name, cls_name, nstep, is_prle):
    cls = getattr(importlib.import_module(mod_name), cls_name)
    ag = cls.__new__(cls)                     # 绕过 __init__ (不需要 torch/args)
    ag.nstep = max(1, int(nstep))
    ag.target_agg = 'min'
    ag.discount = GAM
    ag._pending = []
    ag._boot = np.full(64, float(GAM), dtype=np.float32)
    ag.memory = StubMem()
    ag.is_training = True
    ag._prle = is_prle
    return ag


def feed(ag, r_list, done_at_last=True, contexts=None):
    n = len(r_list)
    for i, r in enumerate(r_list):
        d = done_at_last and (i == n - 1)
        s = np.array([i, i, i, i], dtype=float)
        a = np.array([i], dtype=float)
        if ag._prle:
            ctx = (contexts[i] if contexts else [{'k': i}])
            ag.observe_jump(s, a, float(r), float(r) * 0.1, s + 1, d, context=ctx)
        else:
            ag.observe_jump(s, a, float(r), float(r) * 0.1, s + 1, d)


print('=' * 78)
for k, mod, cls, is_prle in PROJS:
    print('\n=== %s (%s%s) ===' % (k, cls, ', PRLE' if is_prle else ''))
    # ---- 1) nstep=1: 直接落库, 无折叠 ----
    ag1 = make_agent(mod, cls, 1, is_prle)
    feed(ag1, [1, 2, 3, 4], done_at_last=False)
    ok1 = (len(ag1.memory.data) == 4 and len(ag1._pending) == 0
           and all(t[2] == float(i + 1) for i, t in enumerate(ag1.memory.data)))
    check(ok1, 'nstep=1 ⇒ 4 条直接落库、r 逐个原样、无 pending (得 %d 条)' % len(ag1.memory.data))

    # ---- 2/3) nstep=3 滑窗 + boot + done 截断 ----
    ag3 = make_agent(mod, cls, N, is_prle)
    feed(ag3, [1, 2, 3, 4, 5, 6], done_at_last=True)
    got_r = [round(t[2], 10) for t in ag3.memory.data]
    got_b = [round(float(ag3._boot[i]), 10) for i in range(len(ag3.memory.data))]
    got_d = [t[5] for t in ag3.memory.data]
    got_s = [float(t[0][0]) for t in ag3.memory.data]
    exp_r = [2.75, 4.5, 6.25, 8.0, 8.0, 6.0]
    exp_b = [0.125, 0.125, 0.125, 0.0, 0.0, 0.0]
    exp_s = [0, 1, 2, 3, 4, 5]      # 喂入时状态用 [i,i,i,i] (i 从 0 起) ⇒ 起点 0..5
    check(got_r == exp_r, '滑窗 r_sum = Σγ^k r_k: %s (期望 %s)' % (got_r, exp_r))
    check(got_b == exp_b, 'boot = γ^n, 窗口内含 done ⇒ 0: %s (期望 %s)' % (got_b, exp_b))
    check(got_s == exp_s, '每条转移的起点覆盖每个块 (stride-1): %s (期望 %s)' % (got_s, exp_s))
    check(got_d == [0, 0, 0, 1, 1, 1], 'done 截断语义: %s' % got_d)
    check(len(ag3._pending) == 0, 'episode 末 pending 清空')

    # ---- 4) context (PRLE) ----
    agc = make_agent(mod, cls, N, is_prle)
    ctxs = [[{'k': i, 'sub': j} for j in range(2)] for i in range(6)]
    feed(agc, [1, 2, 3, 4, 5, 6], done_at_last=True, contexts=ctxs)
    if is_prle:
        c0 = agc.memory.data[0][6]
        ok_ctx = (c0 == ctxs[0] + ctxs[1] + ctxs[2])
        check(ok_ctx, 'context 顺序拼接 (块0|块1|块2, 共 %d 子步)' % (len(c0) if c0 else -1))
    else:
        check(all(t[6] is None for t in agc.memory.data), '非 PRLE: 不传 context')

print('\n' + '=' * 78)
if FAIL:
    print('失败 %d 项:' % len(FAIL))
    for f in FAIL:
        print('  - ' + f)
    sys.exit(1)
print('全部通过 ✓  (六个 TD3 方法的 H 层 n 步机制已同源)')
