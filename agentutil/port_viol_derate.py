# -*- coding: utf-8 -*-
"""把 td3per 的**奖励侧**改动移植到其余五个 TD3 方法 (批次 1)。

移植内容 (仅奖励侧; n 步机制属批次 2):
  1. `violation_amount` / `violation_derate` + 常量  —— **从 td3per_myfunc.py 逐字提取**后插入,
     保证六个方法的 helper 完全同源。
  2. r1 函数 (`Cal_r1_continuous` / `Cal_r1_PRRBB`) 加 `turbine_derate=1.0` 并乘在
     `turbine_profit` 上。
  3. 包装函数 (`Cal_Rew_RB` / `Cal_Rew_RLRB` / `Cal_Rew_PRRBB`) 透传 `turbine_derate`。
  4. `nine_case_test`: 加 `viol_derate_coef=None`; 先留**原始**动作再 CheckAction;
     按块算 `_derate`; 传进包装函数; 逐步累加 `turbine_profit`/`derate_loss`/`penalty_NN_H`;
     per_case / case_metrics / overall 三处加键。
  5. main: `--viol_derate` 开关 (默认 5.0) + 启动打印 + 块首算系数 + 传进奖励调用与
     `nine_case_test` + df_test9 两个行块各加 3 列。

设计:
  * **绝不盲改**: 每个变换都要求"在指定作用域内恰好命中 1 次", 否则整体中止 (不写盘)。
  * 作用域用**顶层函数范围**限定 (避免 td3prrbb/td3prle 里同名字符串出现在别处)。
  * 默认 dry-run: 只打印每个变换的命中情况与上下文; `--apply` 才写盘。

用法:
    python agentutil/port_viol_derate.py            # dry-run
    python agentutil/port_viol_derate.py --apply    # 写盘
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

# 目标: (项目名, r1 函数名, 包装函数名, 是否多目标 r1/c1/r2/c2, main 里 df_test9 的写法)
TARGETS = [
    ('td3perrb',    'Cal_r1_continuous', 'Cal_Rew_RB',    False, 'append'),
    ('td3prrlrb',   'Cal_r1_PRRBB',      'Cal_Rew_RLRB',  False, 'append'),
    ('td3leprrlrb', 'Cal_r1_PRRBB',      'Cal_Rew_RLRB',  False, 'append'),
    ('td3prrbb',    'Cal_r1_PRRBB',      'Cal_Rew_PRRBB', True,  'concat'),
    ('td3prle',     'Cal_r1_PRRBB',      'Cal_Rew_PRRBB', True,  'concat'),
]

HELPER_MARK = '# ---- 越界乘性惩罚 (--viol_derate)'
CLI_MARK = "    # ---- 越界乘性惩罚 ----"

# 逐项目锚点覆盖 (默认锚点不适用的地方)
TEST9_SIG = {
    'default':      '                   phase_mode=0, phase_table=None):',
    'td3leprrlrb':  '                   phase_feat=0, soc_tol_upper=SOC_TOL):',
}
# per_case 初始化: 默认走正则; td3prrbb 的收尾行带注释, 用显式锚点
PER_CASE_EXPLICIT = {
    'td3prrbb': ("                    'soc_start': None, 'soc_end': None}",
                 "                    'soc_start': None, 'soc_end': None,\n"
                 "                    'turbine_profit': 0.0, 'derate_loss': 0.0,"
                 " 'penalty_NN_H': 0.0}"),
}
# overall 里 profit 行的显式锚点 (默认是 float(sum(m['profit'] ...)))
OVERALL_PROFIT = {
    'td3prrbb': "        'profit': _profit_tot,  # 45 步 profit 之和（现金流口径）\n",
    'td3prle':  "        'profit': float(sum(m['profit'] for m in case_metrics)),  # 45 步 profit 之和\n",
}
# main 里从本方法 myfunc 导入的「含 nine_case_test」那一行 (用于追加 viol_derate 接线)
IMPORT_LINE = {
    'td3perrb':    '    Cal_Rew_RB, gap_penalty, soc_target_from_signal, nine_case_test,',
    'td3prrlrb':   '    Cal_Rew_RLRB, gap_penalty, soc_target_from_signal, nine_case_test,',
    'td3leprrlrb': '    Cal_Rew_RLRB, gap_penalty, soc_target_from_signal, nine_case_test,',
    'td3prrbb':    '    Cal_Rew_PRRBB, nine_case_test,',
    'td3prle':     '    Cal_Rew_PRRBB, nine_case_test,',
}


def read(p):
    with open(p, encoding='utf-8') as f:
        return f.read()


def write(p, s):
    with open(p, 'w', encoding='utf-8', newline='') as f:
        f.write(s)


def func_range(src, name):
    """顶层 `def name(` 的 [start, end) 字符区间 (end = 下一个顶层 def / 文件尾)。"""
    m = re.search(r'^def %s\s*\(' % re.escape(name), src, re.M)
    if not m:
        return None
    nxt = re.search(r'^def ', src[m.end():], re.M)
    return (m.start(), m.end() + (nxt.start() if nxt else len(src) - m.end()))


def last_line_indent(s):
    """取多行片段**最后一行**的缩进 (不能用 re.match(r'\\s*', s): 片段以 \\n 开头会连换行一起吃掉)。"""
    last = s.split('\n')[-1]
    return last[:len(last) - len(last.lstrip())]


class Patcher:
    def __init__(self, path, dry):
        self.path = path
        self.dry = dry
        self.src = read(path)
        self.n_ok = 0
        self.fails = []

    def _apply(self, old, new, where, scope, label):
        """在 scope 内把 old 换成 new; 要求恰好 1 次。"""
        if scope:
            rng = func_range(self.src, scope)
            if rng is None:
                self.fails.append('%s: 找不到函数 %s' % (label, scope))
                return
            a, b = rng
            seg = self.src[a:b]
        else:
            a, b, seg = 0, len(self.src), self.src
        n = seg.count(old)
        if n != 1:
            self.fails.append('%s: 锚点在 %s 内命中 %d 次 (要求 1)\n        锚点: %r'
                              % (label, scope or '<全局>', n, old[:110]))
            return
        i = seg.index(old)
        ctx = seg[max(0, i - 60):i + len(old) + 40].replace('\n', '\n        ')
        if self.dry:
            print('    [OK] %-26s @%s%s' % (label, where,
                  '' if not scope else ' (%s)' % scope))
            print('         …%s…' % ctx)
        seg2 = seg[:i] + new + seg[i + len(old):]
        self.src = self.src[:a] + seg2 + self.src[b:]
        self.n_ok += 1

    def rep(self, old, new, label, scope=None):
        self._apply(old, new, 'replace', scope, label)

    def ins_before(self, anchor, ins, label, scope=None):
        self._apply(anchor, ins + anchor, 'insert-before', scope, label)

    def ins_after(self, anchor, ins, label, scope=None):
        self._apply(anchor, anchor + ins, 'insert-after', scope, label)

    def save(self):
        if self.dry or self.fails:
            return
        try:
            ast.parse(self.src)
        except SyntaxError as e:
            self.fails.append('%s: 写盘前 AST 解析失败: %s' % (self.path, e))
            return
        write(self.path, self.src)


def helper_block():
    """从 td3per_myfunc.py 逐字提取 helper 块 (保证六方法同源)。"""
    src = read('td3per/td3per_myfunc.py')
    i = src.index(HELPER_MARK)
    j = src.index('def Cal_r1_continuous(')
    blk = src[i:j]
    assert 'def violation_derate(' in blk and 'def violation_amount(' in blk, '提取失败'
    return blk.rstrip('\n') + '\n\n\n'


def cli_block():
    """从 td3per_main.py 逐字提取 --viol_derate 的 CLI 块。"""
    src = read('td3per/td3per_main.py')
    i = src.index(CLI_MARK)
    j = src.index("parser.add_argument('--viol_derate'", i)
    j = src.index('\n', src.index('VIOL_DERATE_COEF)', j)) + 1
    return src[i:j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='真正写盘 (默认 dry-run)')
    args = ap.parse_args()
    dry = not args.apply

    HB = helper_block()
    CLI = cli_block()
    print('helper 块 %d 字符 / CLI 块 %d 字符 (从 td3per 提取)\n' % (len(HB), len(CLI)))

    all_fails = []
    for proj, r1fn, rewfn, multi, csvmode in TARGETS:
        print('=' * 92)
        print('=== %s (r1=%s, wrap=%s, 多目标=%s, test9=%s) ===' % (proj, r1fn, rewfn, multi, csvmode))
        print('=' * 92)
        mf = '%s/%s_myfunc.py' % (proj, proj)
        mn = '%s/%s_main.py' % (proj, proj)

        # ---------------- myfunc ----------------
        p = Patcher(mf, dry)
        p.ins_before('def %s(pr, ptur, pbrule, penalty_NN_H, wind_level, price_level,' % r1fn,
                     HB, 'helper 块')
        p.rep('                 SOC=3.0, time=1.0 / 60):',
              '                 SOC=3.0, time=1.0 / 60, turbine_derate=1.0):',
              'r1 签名', scope=r1fn)
        p.rep('    return turbine_profit + bess_profit - degradation - penalty_NN_H',
              '    return (turbine_profit * turbine_derate + bess_profit - degradation\n'
              '            - penalty_NN_H)', 'r1 乘系数', scope=r1fn)
        # 包装函数签名: 末行是 pb_cmd=None): 或 ptur=None, n_rem=None):
        wrap_tail = ('                 pb_cmd=None):' if not multi
                     else '                   penalty_NN_H=0.0, time=1.0 / 60, ptur=None, n_rem=None):')
        p.rep(wrap_tail, wrap_tail[:-2] + ', turbine_derate=1.0):', '包装签名', scope=rewfn)
        # 包装函数内 r1 调用: 加 turbine_derate 透传
        rng = func_range(p.src, rewfn)
        seg = p.src[rng[0]:rng[1]]
        m = re.search(r'(r1 = %s\(pr_old, ptur, pbrule, penalty_NN_H,\n\s+wind_level, price_level, SOC=SOC_decision, time=time)\)'
                      % re.escape(r1fn), seg)
        if not m:
            p.fails.append('包装 r1 调用: 未匹配')
        else:
            last = m.group(1).split('\n')[-1]
            ind = last[:len(last) - len(last.lstrip())]
            p.rep(m.group(0), m.group(1) + ',\n' + ind + 'turbine_derate=turbine_derate)',
                  '包装 r1 调用', scope=rewfn)
        # nine_case_test
        p.rep(TEST9_SIG.get(proj, TEST9_SIG['default']),
              TEST9_SIG.get(proj, TEST9_SIG['default'])[:-2] + ', viol_derate_coef=None):',
              'test9 签名', scope='nine_case_test')
        p.rep('            action_H = agent_H.select_action(norm_obs)\n'
              '            action_H, penalty_NN_H = CheckAction(action_H)',
              '            action_H_raw = agent_H.select_action(norm_obs)\n'
              '            action_H, penalty_NN_H = CheckAction(action_H_raw)\n'
              '            _derate = violation_derate(action_H_raw, viol_derate_coef)',
              'test9 留原始动作+系数', scope='nine_case_test')
        # 包装调用尾部加 turbine_derate=_derate
        rng = func_range(p.src, 'nine_case_test')
        if rng:
            seg = p.src[rng[0]:rng[1]]
            m = re.search(r'(\n\s+ptur=ptur,[^\n]*)\n(\s*\))', seg)
            if m:
                ind = last_line_indent(m.group(1))
                p.rep(m.group(0), m.group(1) + '\n' + ind + 'turbine_derate=_derate,' + '\n' + m.group(2),
                      'test9 传系数', scope='nine_case_test')
            else:
                p.fails.append('test9 传系数: 未匹配 ptur=ptur 行')
        # per_case init 加 3 键
        if proj in PER_CASE_EXPLICIT:
            old, new = PER_CASE_EXPLICIT[proj]
            p.rep(old, new, 'per_case 三键', scope='nine_case_test')
        else:
            rng = func_range(p.src, 'nine_case_test')
            seg = p.src[rng[0]:rng[1]]
            m = re.search(r"(per_case = \{i: \{[^}]*?)(\}\s*\n\s*for i in range)", seg)
            if m:
                p.rep(m.group(0),
                      m.group(1) + ",\n                    'turbine_profit': 0.0, 'derate_loss': 0.0,"
                                   " 'penalty_NN_H': 0.0" + m.group(2),
                      'per_case 三键', scope='nine_case_test')
            else:
                p.fails.append('per_case 三键: 未匹配 per_case 初始化')
        # 逐步累加 (插在 pc['r1'] += r1 后)
        p.rep("        pc['r1'] += r1\n",
              "        pc['r1'] += r1\n"
              "        # r1 分解三项 (审计: profit/N - r1 == (罚 + 乘性削减)/N)\n"
              "        pc['turbine_profit'] += details['turbine_profit']\n"
              "        pc['derate_loss'] += details['turbine_profit'] * (1.0 - _derate)\n"
              "        pc['penalty_NN_H'] += penalty_NN_H\n",
              '逐步累加三量', scope='nine_case_test')
        # case_metrics 加 3 键
        rng = func_range(p.src, 'nine_case_test')
        seg = p.src[rng[0]:rng[1]]
        m = re.search(r"('profit': pc\['profit'\],[^\n]*\n)", seg)
        if m:
            p.rep(m.group(0), m.group(0) +
                  "            'turbine_profit': pc['turbine_profit'],\n"
                  "            'derate_loss': pc['derate_loss'],\n"
                  "            'penalty_NN_H': pc['penalty_NN_H'],\n", 'case_metrics 三键',
                  scope='nine_case_test')
        else:
            p.fails.append('case_metrics 三键: 未匹配 profit 行')
        # overall 加 3 键
        rng = func_range(p.src, 'nine_case_test')
        seg = p.src[rng[0]:rng[1]]
        anchor_op = OVERALL_PROFIT.get(proj)
        if anchor_op is None:
            m = re.search(r"('profit': float\(sum\(m\['profit'\] for m in case_metrics\)\),\n)", seg)
            anchor_op = m.group(0) if m else None
        if anchor_op:
            p.rep(anchor_op, anchor_op +
                  "        'turbine_profit': float(sum(m['turbine_profit'] for m in case_metrics)),\n"
                  "        'derate_loss': float(sum(m['derate_loss'] for m in case_metrics)),\n"
                  "        'penalty_NN_H': float(sum(m['penalty_NN_H'] for m in case_metrics)),\n",
                  'overall 三键', scope='nine_case_test')
        else:
            p.fails.append('overall 三键: 未匹配 overall profit 行')

        # ---------------- main ----------------
        q = Patcher(mn, dry)
        il = IMPORT_LINE[proj]
        ind_il = last_line_indent(il)
        if il + '\n' not in q.src:
            q.fails.append('import 接线: 未匹配导入行')
        else:
            q.rep(il + '\n', il + '\n' + ind_il + 'violation_derate, VIOL_DERATE_COEF,\n',
                  'import 接线')
        q.ins_before("    parser.add_argument('--window_length', default=1, type=int, help='')",
                     CLI, 'CLI --viol_derate')
        q.ins_after('                applied_action_H, penalty_NN_H = CheckAction(original_action_H)\n',
                    '            viol_derate = violation_derate(original_action_H, args.viol_derate)\n',
                    '块首算系数')
        m = re.search(r'(\n\s+ptur=ptur,[^\n]*)\n(\s*\))', q.src)
        if m:
            ind = last_line_indent(m.group(1))
            q.rep(m.group(0), m.group(1) + '\n' + ind + 'turbine_derate=viol_derate,' + '\n' + m.group(2),
                  '奖励调用传系数')
        else:
            q.fails.append('奖励调用传系数: 未匹配 ptur=ptur 行')
        m = re.search(r'(\n\s+phase_mode=phase_mode, phase_table=phase_table)\)', q.src)
        if m:
            q.rep(m.group(0), m.group(1) + ',\n                    viol_derate_coef=args.viol_derate)',
                  'test9 调用传系数')
        else:
            m = re.search(r'(\n\s+phase_feat=phase_feat, soc_tol_upper=soc_tol_upper)\)', q.src)
            if m:
                q.rep(m.group(0), m.group(1) + ',\n                    viol_derate_coef=args.viol_derate)',
                      'test9 调用传系数')
            else:
                q.fails.append('test9 调用传系数: 未匹配')
        # df_test9 两个行块各加 3 键
        pat_m = re.search(r"(\n(\s+)'profit': m\['profit'\],[^\n]*\n)", q.src)
        pat_o = re.search(r"(\n(\s+)'profit': overall\['profit'\],[^\n]*\n)", q.src)
        if pat_m:
            ind = pat_m.group(2)
            q.rep(pat_m.group(1), pat_m.group(1) +
                  "%s'turbine_profit': m['turbine_profit'],\n"
                  "%s'derate_loss': m['derate_loss'],\n"
                  "%s'penalty_NN_H': m['penalty_NN_H'],\n" % (ind, ind, ind), 'df_test9 per-case 三列')
        else:
            q.fails.append('df_test9 per-case 三列: 未匹配')
        if pat_o:
            ind = pat_o.group(2)
            q.rep(pat_o.group(1), pat_o.group(1) +
                  "%s'turbine_profit': overall['turbine_profit'],\n"
                  "%s'derate_loss': overall['derate_loss'],\n"
                  "%s'penalty_NN_H': overall['penalty_NN_H'],\n" % (ind, ind, ind),
                  'df_test9 OVERALL 三列')
        else:
            q.fails.append('df_test9 OVERALL 三列: 未匹配')

        for pt in (p, q):
            for f in pt.fails:
                print('    [FAIL] %s: %s' % (os.path.basename(pt.path), f))
                all_fails.append('%s/%s' % (pt.path, f))
            if not pt.fails:
                pt.save()
                print('    ---- %s: %d 个变换全部命中%s' %
                      (os.path.basename(pt.path), pt.n_ok, '' if dry else ' (已写盘)'))

    print('\n' + '=' * 92)
    if all_fails:
        print('未通过 %d 项 (未写盘):' % len(all_fails))
        for f in all_fails:
            print('  - ' + f)
        return 1
    print('全部 %d 个项目的变换均命中%s' % (len(TARGETS), ' (dry-run, 未写盘)' if dry else ' (已写盘)'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
