# -*- coding: utf-8 -*-
"""把 td3per 的 **H 层 n 步回报** 同步到其余五个 TD3 方法 (批次 2)。

移植内容:
  1. `*_ddpg.py`: 加 `nstep`/`target_agg` kwarg (默认 1/'min' = 旧口径逐位不变);
     `__init__` 里的 n 步块 (滑窗 `_pending` + `_boot` bootstrap 系数);
     `update_policy` 的 target 聚合 (`--target_agg`) 与按条 `_disc`;
     `_push_transition`/`_emit_nstep` + 滑窗版 `observe_jump`。
  2. `*_main.py`: `--nstep` (默认 5) / `--target_agg` (默认 min) 开关 + 构造接线;
     `--rmsize` 默认 80000 → 20000 (与 td3per 口径一致)。⚠ 本行原写"n=5 时 1 条 = 5n
     环境步" —— 那是分块聚合口径的遗留说法: 滑窗 stride-1 下 1 条 = 1 块 = 5 环境步,
     覆盖环境步 = rmsize x 5, **与 n 无关** (见 README 的 --rmsize 行)。
  3. 两个 PRLE 项目 (`td3prle`/`td3leprrlrb`) 的 `observe_jump(context=...)` 变体:
     折叠块的 context **拼接**保留, 供 LLM 阈值变更时的 relabel 重算;
     并让 `PRLEBuffer.relabel_all(..., discount=gamma)` **按块**加 γ 权重
     (默认 discount=1.0 ⇒ 与旧行为逐位相同), main 调用处传 `args.discount`。

安全: 每个变换要求"作用域内恰好命中 1 次", 任一不满足则**整体不写盘**;
默认 dry-run, `--apply` 才写。块的正文从 `td3per/td3per_ddpg.py` 逐字提取。

用法: python agentutil/port_nstep.py [--apply]
"""

import argparse
import ast
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

# (项目, ddpg 类名, 签名尾行(是否带 wt_noise), main 里 H 构造的末行, 是否 PRLE(context))
TARGETS = [
    ('td3perrb',    'SP_TD3',           'wt',  'sig_noise_min=args.sig_noise_min)', False),
    ('td3prrlrb',   'SP_TD3',           'wt',  'sig_noise_min=args.sig_noise_min)', False),
    ('td3prrbb',    'SP_TD3',           'cc',  'cost_critic_mode=args.cost_critic_mode)', False),
    ('td3prle',     'SP_TD3_PRLE',      'cc',  'cost_critic_mode=args.cost_critic_mode)', True),
    ('td3leprrlrb', 'SP_TD3_LEPRRLRB',  'cc',  'cost_critic_mode=args.cost_critic_mode)', True),
]

SIG_WT = '                 wt_noise=0.05, sig_noise_max=0.2, sig_noise_min=0.05):'
SIG_CC = "                 cost_critic_mode='max'):"
INSERT_SIG = ",\n                 nstep=1, target_agg='min'):"
A1_BLOCK = ('        # A1: terminal 语义修复 —— bootstrap 当且仅当非终止转移\n'
            '        target_q_batch_p = profit_batch + \\\n'
            '            self.discount * (1 - terminal_batch) * min_p_next\n'
            '        target_q_batch_c = cost_batch + \\\n'
            '            self.discount * (1 - terminal_batch) * min_c_next\n')
OBS_CLEAN_HEAD = '    def observe_jump(self, s_t, a_t, r_t, c_t, s_t1, done):\n'
OBS_PRLE_HEAD = '    def observe_jump(self, s_t, a_t, r_t, c_t, s_t1, done, context=None):\n'
RMSIZE_OLD = "    parser.add_argument('--rmsize',\n                        default=80000,"
RMSIZE_NEW = "    parser.add_argument('--rmsize',\n                        default=20000,"
BSIZE_ANCHOR = "    parser.add_argument('--bsize', default=128, type=int,"

# PRLE 变体 (context 随折叠块拼接; 供 relabel_all 按块 γ 加权重算)
OBS_PRLE_BLOCK = '''    # ---- n 步回报: 写入侧 (PRLE 变体: context 随折叠块拼接, 供 relabel 用) ----
    def _push_transition(self, s_t, a_t, r_t, c_t, s_t1, done, boot, context=None):
        """写一条转移进 memory, 记录它的 bootstrap 系数 boot 与 relabel context。

        context 来自被折叠的 n 个块的子步上下文**顺序拼接** ⇒ relabel_all 按块
        γ 加权重算时得到与训练目标一致的折扣和 (见 *_buffer.relabel_all(discount=...))。
        """
        idx = self.memory._next_idx
        self.memory.append(s_t, a_t, r_t, c_t, s_t1, done, context=context)
        if self._boot is not None and 0 <= idx < len(self._boot):
            self._boot[idx] = boot

    def _emit_nstep(self, start, n_used):
        """把 _pending[start : start+n_used] 折叠成一条 n 步转移 (含 context 拼接)。"""
        seg = self._pending[start:start + n_used]
        gam = float(self.discount)
        r_sum = 0.0
        c_sum = 0.0
        trunc = False
        ctx_all = []
        for k, item in enumerate(seg):
            _s, _a, _r, _c, _s2, _d, _ctx = item
            r_sum += (gam ** k) * float(_r)
            c_sum += (gam ** k) * float(_c)
            if _ctx:
                ctx_all.extend(_ctx)
            if _d:
                trunc = True
                break
        boot = 0.0 if trunc else gam ** n_used
        self._push_transition(seg[0][0], seg[0][1], r_sum, c_sum, seg[-1][4],
                              1.0 if trunc else float(seg[-1][5]), boot,
                              context=(ctx_all or None))

    def observe_jump(self, s_t, a_t, r_t, c_t, s_t1, done, context=None):
        if not self.is_training:
            return
        if self.nstep <= 1:
            self.memory.append(s_t, a_t, r_t, c_t, s_t1, done, context=context)
            return
        self._pending.append((s_t, a_t, r_t, c_t, s_t1, done, context))
        # ---- 滑窗 (stride-1): 每来一个新块结算**队首**那条, 队列长度稳定在 n-1 ----
        #   分块版只有 1/n 的块状态成为训练输入; 滑窗版每个块都是某条转移的起点
        #   ⇒ 状态覆盖 = 块数 (标准做法: Rainbow/Ape-X 的 n_step wrapper)。
        while len(self._pending) >= self.nstep:
            self._emit_nstep(0, self.nstep)
            self._pending.pop(0)
        if done:
            k = len(self._pending)
            for i in range(k):
                self._emit_nstep(i, k - i)
            self._pending = []

'''

REL_ALL_OLD_C = '    def relabel_all(self, relabel_r_fn, relabel_c_fn=None):'
REL_ALL_NEW_C = ('    def relabel_all(self, relabel_r_fn, relabel_c_fn=None, discount=1.0):')
REL_LOOP_OLD = """            new_reward = 0.0
            for sub in ctx:
                new_reward += relabel_r_fn(
                    sub['u'],
                    sub['pr'],
                    sub['turbine_raw'],
                    sub['bess_raw'],
                    sub['degradation'],
                    sub['penalty_NN_H'],
                )
"""
REL_LOOP_NEW = """            # n 步折叠后 ctx 覆盖多个块 (每 block_size 个子步为一块): 按块加 γ^k
            #   discount=1.0 (默认, 单块 context) 时权重恒 1 ⇒ 与旧行为逐位相同。
            _blk = int(getattr(self, 'block_size', 5) or 5)
            _gam = float(discount)
            new_reward = 0.0
            for _i, sub in enumerate(ctx):
                new_reward += (_gam ** (_i // _blk)) * relabel_r_fn(
                    sub['u'],
                    sub['pr'],
                    sub['turbine_raw'],
                    sub['bess_raw'],
                    sub['degradation'],
                    sub['penalty_NN_H'],
                )
"""
REL_COST_OLD = """            new_cost = cost
            if relabel_c_fn is not None:
                new_cost = 0.0
                for sub in ctx:
                    new_cost += relabel_c_fn(
                        sub['u'], sub['P_F'],
                        sub.get('unfeasible_gap', 0.0))
"""
REL_COST_NEW = """            new_cost = cost
            if relabel_c_fn is not None:
                new_cost = 0.0
                for _i, sub in enumerate(ctx):
                    new_cost += (_gam ** (_i // _blk)) * relabel_c_fn(
                        sub['u'], sub['P_F'],
                        sub.get('unfeasible_gap', 0.0))
"""
REL_CALL_OLD = ('n_relabeled = agent_H.memory.relabel_all(\n'
                '                        relabel_r1_substep, relabel_c1_substep)')
REL_CALL_NEW = ('n_relabeled = agent_H.memory.relabel_all(\n'
                '                        relabel_r1_substep, relabel_c1_substep,\n'
                '                        discount=float(args.discount))')

# td3leprrlrb 的 relabel 循环签名不同 (r1 多 SOC, c1 多 gap_pen)
REL_LOOP_LE_OLD = """            new_reward = 0.0
            for sub in ctx:
                new_reward += relabel_r_fn(
                    sub['u'],
                    sub['pr'],
                    sub['turbine_raw'],
                    sub['bess_raw'],
                    sub['degradation'],
                    sub['penalty_NN_H'],
                    sub.get('SOC', 3.0),
                )
"""
REL_LOOP_LE_NEW = """            _blk = int(getattr(self, 'block_size', 5) or 5)
            _gam = float(discount)
            new_reward = 0.0
            for _i, sub in enumerate(ctx):
                new_reward += (_gam ** (_i // _blk)) * relabel_r_fn(
                    sub['u'],
                    sub['pr'],
                    sub['turbine_raw'],
                    sub['bess_raw'],
                    sub['degradation'],
                    sub['penalty_NN_H'],
                    sub.get('SOC', 3.0),
                )
"""
REL_COST_LE_OLD = """            new_cost = cost
            if relabel_c_fn is not None:
                new_cost = 0.0
                for sub in ctx:
                    new_cost += relabel_c_fn(
                        sub['u'],
                        sub['P_F'],
                        sub.get('unfeasible_gap', 0.0),
                        sub.get('gap_pen', 0.0),
                    )
"""
REL_COST_LE_NEW = """            new_cost = cost
            if relabel_c_fn is not None:
                new_cost = 0.0
                for _i, sub in enumerate(ctx):
                    new_cost += (_gam ** (_i // _blk)) * relabel_c_fn(
                        sub['u'],
                        sub['P_F'],
                        sub.get('unfeasible_gap', 0.0),
                        sub.get('gap_pen', 0.0),
                    )
"""
REL_LOOP_BY_PROJ = {'td3prle': (REL_LOOP_OLD, REL_LOOP_NEW),
                    'td3leprrlrb': (REL_LOOP_LE_OLD, REL_LOOP_LE_NEW)}
REL_COST_BY_PROJ = {'td3prle': (REL_COST_OLD, REL_COST_NEW),
                    'td3leprrlrb': (REL_COST_LE_OLD, REL_COST_LE_NEW)}


def read(p):
    with open(p, encoding='utf-8') as f:
        return f.read()


def write(p, s):
    with open(p, 'w', encoding='utf-8', newline='') as f:
        f.write(s)


def func_range(src, name):
    """作用域区间: 支持顶层函数/类, 也支持**类方法** (任意缩进)。

    类: `^class NAME` → 下一个顶层 class/def。
    方法: 任意缩进的 `def NAME(` → 下一个**同缩进**的 def/class。
    """
    m = re.search(r'^class %s\b' % re.escape(name), src, re.M)
    if m:
        nxt = re.search(r'^(?:class |def )', src[m.end():], re.M)
        return (m.start(), m.end() + (nxt.start() if nxt else len(src) - m.end()))
    m = re.search(r'^([ \t]*)def %s\s*\(' % re.escape(name), src, re.M)
    if not m:
        return None
    indent = m.group(1)
    pat = re.compile(r'^(?:class |%sdef )' % re.escape(indent), re.M) if indent \
        else re.compile(r'^(?:class |def )', re.M)
    nxt = pat.search(src, m.end())
    return (m.start(), nxt.start() if nxt else len(src))


class Patcher:
    def __init__(self, path, dry):
        self.path = path
        self.dry = dry
        self.src = read(path)
        self.n_ok = 0
        self.fails = []

    def _apply(self, old, new, label, scope=None, first=False):
        if scope:
            rng = func_range(self.src, scope)
            if rng is None:
                self.fails.append('%s: 找不到函数 %s' % (label, scope))
                return
            a, b, seg = rng[0], rng[1], self.src[rng[0]:rng[1]]
        else:
            a, b, seg = 0, len(self.src), self.src
        n = seg.count(old)
        if n == 0 or (n != 1 and not first):
            self.fails.append('%s: 锚点在 %s 命中 %d 次 (要求 %d)'
                              % (label, scope or '<全局>', n, 1 if not first else '>=1'))
            return
        i = seg.index(old)
        if self.dry:
            print('    [OK] %-24s %s' % (label, ('(%s)' % scope) if scope else ''))
        seg2 = seg[:i] + new + seg[i + len(old):]
        self.src = self.src[:a] + seg2 + self.src[b:]
        self.n_ok += 1

    def rep(self, old, new, label, scope=None, first=False):
        self._apply(old, new, label, scope, first)

    def ins_after(self, anchor, ins, label, scope=None):
        self._apply(anchor, anchor + ins, label, scope)

    def save(self):
        if self.dry or self.fails:
            return
        try:
            ast.parse(self.src)
        except SyntaxError as e:
            self.fails.append('%s: 写盘前 AST 失败: %s' % (self.path, e))
            return
        write(self.path, self.src)


def blocks_from_td3per():
    """从 td3per/td3per_ddpg.py 逐字提取四个块。"""
    src = read('td3per/td3per_ddpg.py')

    def seg(start_pat, end_pat, inclusive_end=True):
        i = src.index(start_pat)
        j = src.index(end_pat, i) + len(end_pat)
        if inclusive_end:
            j = src.index('\n', j) + 1
        return src[i:j]

    b_init = seg('        # ---- H 层 n 步回报',
                 '                      if self.nstep > 1 else None)')
    b_tagg = seg("        if self.target_agg == 'mean':",
                 '            min_p_next = self._pair_min(q_p1_next, q_p2_next)')
    b_disc = seg('        # A1: terminal 语义修复', '            _disc * (1 - terminal_batch) * min_c_next')
    b_obs = seg('    # ---- n 步回报: 写入侧', '            self._pending = []')
    return b_init, b_tagg, b_disc, b_obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    dry = not args.apply

    B_INIT, B_TAGG, B_DISC, B_OBS = blocks_from_td3per()
    # main 的 --nstep/--target_agg CLI 块 (逐字提取, 从注释头到 target_agg 语句末行)
    msrc = read('td3per/td3per_main.py')
    i = msrc.index("    parser.add_argument('--nstep'")
    i = msrc.rindex('    # ----', 0, i)                      # 注释头
    j = msrc.index("parser.add_argument('--target_agg'", i)
    j = msrc.index('\n', msrc.index("')", msrc.index("'(min = TD3 原口径", j))) + 1
    B_CLI = msrc[i:j]
    print('块大小: init=%d tagg=%d disc=%d obs=%d cli=%d\n'
          % (len(B_INIT), len(B_TAGG), len(B_DISC), len(B_OBS), len(B_CLI)))

    all_fails = []
    for proj, cls, sigkind, ctor_tail, is_prle in TARGETS:
        print('=' * 90)
        print('=== %s (%s, PRLE=%s) ===' % (proj, cls, is_prle))
        dd = '%s/%s_ddpg.py' % (proj, proj)
        mn = '%s/%s_main.py' % (proj, proj)

        p = Patcher(dd, dry)
        # D1 签名
        sig_old = SIG_WT if sigkind == 'wt' else SIG_CC
        p.rep(sig_old, sig_old[:-2] + INSERT_SIG, '签名 kwarg', scope=cls)
        # D2 init 块
        p.ins_after('        self.depsilon = 1.0 / args.epsilon\n', '\n' + B_INIT,
                    'init n 步块', scope=cls)
        # D3 target 聚合
        p.rep('        min_p_next = self._pair_min(q_p1_next, q_p2_next)\n', B_TAGG,
              'update_policy target_agg', scope='update_policy')
        # D4 按条 _disc
        p.rep(A1_BLOCK, B_DISC, 'update_policy _disc', scope='update_policy')
        # D5 observe_jump 三件套
        if is_prle:
            rng = func_range(p.src, cls)
            seg_src = p.src[rng[0]:rng[1]]
            m = re.search(r'    def observe_jump\(self, s_t, a_t, r_t, c_t, s_t1, done, context=None\):\n(?:.*\n)*?\n', seg_src)
            if not m:
                p.fails.append('observe_jump(PRLE): 未匹配')
            else:
                p.rep(m.group(0), OBS_PRLE_BLOCK, 'observe_jump PRLE 变体', scope=cls)
        else:
            old = ('    def observe_jump(self, s_t, a_t, r_t, c_t, s_t1, done):\n'
                   '        if self.is_training:\n'
                   '            self.memory.append(s_t, a_t, r_t, c_t, s_t1, done)\n')
            p.rep(old, B_OBS, 'observe_jump 三件套', scope=cls)

        q = Patcher(mn, dry)
        # M1 CLI 块插到 --bsize 之前
        q._apply(BSIZE_ANCHOR, B_CLI + BSIZE_ANCHOR, 'CLI --nstep/--target_agg')
        # M2 rmsize 默认
        q.rep(RMSIZE_OLD, RMSIZE_NEW, '--rmsize 默认 20000')
        # M3 构造接线
        q.rep(ctor_tail, ctor_tail[:-1] + ',\n' + ' ' * (len(ctor_tail) - len(ctor_tail.lstrip()))
              + 'nstep=args.nstep, target_agg=args.target_agg)', '构造接线 nstep/target_agg', first=True)
        # M4 PRLE: relabel 传 discount
        if is_prle:
            q.rep(REL_CALL_OLD, REL_CALL_NEW, 'relabel_all 传 discount')

        for pt in (p, q):
            for f in pt.fails:
                print('    [FAIL] %s: %s' % (os.path.basename(pt.path), f))
                all_fails.append('%s/%s' % (pt.path, f))
            if not pt.fails:
                pt.save()
                print('    ---- %s: %d 个变换全部命中%s'
                      % (os.path.basename(pt.path), pt.n_ok, '' if dry else ' (已写盘)'))

        # 两个 PRLE 项目的缓冲: relabel_all 按块加 γ
        if is_prle:
            bf = '%s/%s_buffer.py' % (proj, proj)
            r = Patcher(bf, dry)
            r.rep(REL_ALL_OLD_C, REL_ALL_NEW_C, 'relabel_all 签名')
            lo, ln = REL_LOOP_BY_PROJ[proj]
            co, cn = REL_COST_BY_PROJ[proj]
            r.rep(lo, ln, 'relabel_all γ 加权(reward)')
            r.rep(co, cn, 'relabel_all γ 加权(cost)')
            for f in r.fails:
                print('    [FAIL] %s: %s' % (os.path.basename(bf), f))
                all_fails.append('%s/%s' % (bf, f))
            if not r.fails:
                r.save()
                print('    ---- %s: %d 个变换全部命中%s'
                      % (os.path.basename(bf), r.n_ok, '' if dry else ' (已写盘)'))

    print('\n' + '=' * 90)
    if all_fails:
        print('未通过 %d 项 (未写盘):' % len(all_fails))
        for f in all_fails:
            print('  - ' + f)
        return 1
    print('全部命中%s' % (' (dry-run)' if dry else ' (已写盘)'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
