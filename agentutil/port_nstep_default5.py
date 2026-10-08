# -*- coding: utf-8 -*-
"""把全方法 `--nstep` 的 CLI 默认值 10 -> 5（含 help 文案与契约测试断言）。

动机
----
`--nstep`（H 层 n 步回报，滑窗 stride-1）在代码里的 CLI 默认是 10，而
根 `README.md` 的默认值表、`agentutil/README.md`、`agentutil/test_nstep_h.py`
的说明文字都已经按 **5 = 生产口径** 写；`agentutil/test_phase_uniform.py` 则按
代码现状断言 **10**。⇒ 现在把代码统一到 5，并把那条断言一起改。

改动内容（每个**带 `--nstep`** 的 `*_main.py` 恰好 3 处）
--------------------------------------------------------
1. `parser.add_argument('--nstep', default=10, type=int,`  ->  `default=5`
2. help 文案 `(默认 10 = 生产口径; ... 建议 10)` -> `(默认 5 = 生产口径; ... 建议 5)`
3. `parser.add_argument('--nstep', ...)` 前的注释头 `---- 默认值 = 10 (生产口径) ----`
   -> `---- 默认值 = 5 (生产口径) ----`

外加 `agentutil/test_phase_uniform.py` 的契约断言
`"'--nstep', default=10"` -> `"'--nstep', default=5"`（含打印文案里的"默认 10"）。

⚠ `nstep` 只经由 CLI 默认值生效（`*_ddpg.py` 的构造签名默认仍是 1 ⇒ `nstep=1`
与改动前逐位一致）。因此**显式传 `--nstep N` 的运行逐位不变**（传 5/10 都能复现
各自的历史 run），本脚本只翻默认值。

自动识别"哪些 main 真带 `--nstep`"
--------------------------------
按 `parser.add_argument('--nstep'` 是否存在筛选（`prrbb` / `prle` / `rbb` /
`未命名/*` 本来就没有这个开关 ⇒ 明确跳过并打印，不算失败）。

安全
----
* 默认 **dry-run**，`--apply` 才写盘；
* 每个文件要求其全部锚点**恰好命中 1 次**，任一文件不符 ⇒ **整体不写盘**；
* `--check` 只校验"已处于目标状态"（供改动后回归用）；`--revert` 反向还原；
* 保留原文件换行风格（`newline=''` 读写，CRLF/LF 各自保持）。

用法
----
    python agentutil/port_nstep_default5.py            # dry-run
    python agentutil/port_nstep_default5.py --apply
    python agentutil/port_nstep_default5.py --check
    python agentutil/port_nstep_default5.py --revert --apply
"""

import argparse
import glob
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

MAIN_GLOB = '*/*_main.py'
CONTRACT = 'agentutil/test_phase_uniform.py'

# 匹配判定 nstep 的 CLI 声明 (行内允许缩进/换行风格差异)
RE_HAS_NSTEP = re.compile(r"parser\.add_argument\('--nstep'")
# 锚点 1: `--nstep` 声明里的 CLI 默认值
RE_DEF = re.compile(r"(--nstep', default=)(\d+)(,)")
# 锚点 2/3: 声明之后 3 行内的 help 文案 "(默认 N = 生产口径" 与 "建议 N)"
RE_HELP_DEF = re.compile(r"(\(默认 )(\d+)( = 生产口径)")
RE_HELP_TIP = re.compile(r"(历史口径; 建议 )(\d+)(\))")
# 锚点 4: 声明**之前**最近的那行注释头 "---- 默认值 = N (生产口径) ----"
RE_CMT = re.compile(r"(#   ---- 默认值 = )(\d+)( \(生产口径\) ----)")
# 契约测试: 断言行的 default=N + 其下一行打印文案里的"默认 N"
RE_CONTRACT = re.compile(r"('--nstep', default=)(\d+)")
RE_CONTRACT_TXT = re.compile(r"(: --nstep 默认 )(\d+)( /)")
# 窗口: 注释头必须紧邻 `--nstep` 声明（该语句上一行）—— 实测它与锚点相距 332
# 字符（中间隔着 5 行"依据"注释）⇒ 回看 500 字符足够且不会碰到更远的 `--rmsize`
# 同类注释头（那个在 3000+ 字符之外）。aifflor 全文件共 8 个同名"默认值"块。
CMT_LOOKBACK_CHARS = 500


def read(path):
    """读成 `\\n` 统一文本。⚠ 必须归一化 `\\r\\n`：本仓库 `core.autocrlf=true`，
    工作区是 CRLF，而锚点字面量只含 `\\n` —— 不归一化会让匹配落在第一个 `\\n` 处
    （曾把整份文件内容插进 `--nstep` 块，文件写坏）。"""
    with open(path, encoding='utf-8', newline='') as fh:
        return fh.read().replace('\r\n', '\n').replace('\r', '\n')


def write(path, text):
    """写回：按工作区约定还原 CRLF（`core.autocrlf=true` ⇒ 工作区 CRLF, 索引 LF）。"""
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        fh.write(text.replace('\n', '\r\n'))


def sub_exactly_one(text, regex, label, new_value, lo=0, hi=None, must_equal=None):
    """把 text[lo:hi] 窗口内 regex 的**数值捕获组**改成 new_value。

    窗口内必须**恰好命中 1 次**（`must_equal` 非 None 时，还要求被捕获的数值等于它，
    否则记 0 命中 —— `--check` 用它判断"是否仍处于旧状态"），否则返回
    (原文本, (label, 命中数))。窗口只影响搜索范围，替换在原文本上进行
    （不截断/不重建全文），因此 CRLF/LF 与缩进原样保留。
    """
    hi = len(text) if hi is None else hi
    matches = list(regex.finditer(text, lo, hi))
    if must_equal is not None:
        matches = [m for m in matches
                   if any(g.isdigit() and int(g) == int(must_equal) for g in m.groups())]
    if len(matches) != 1:
        return text, (label, len(matches))
    m = matches[0]
    # 逐组重建: 数字组写成 new_value, 其余原样保留。
    # ⚠ `last` 必须从 **0** 起算（首段 = text[0:m.start()] 即整段前缀），
    #   结尾用 text[last:]（= m.end() 之后的尾部）。这里写错过两次：
    #   一次 last=m.start() ⇒ 前缀被丢掉；一次结尾用 text[m.end():] 且 last 从 0 起
    #   ⇒ 前缀被重复拼接、文件撑大约一倍。`_apply` 里的长度断言专拦这两类。
    out, last = [], 0
    for gi in range(1, m.re.groups + 1):
        s, e = m.span(gi)
        out.append(text[last:s])
        out.append(str(new_value) if text[s:e].isdigit() else text[s:e])
        last = e
    out.append(text[last:])
    return ''.join(out), (label, 1)


def main():
    ap = argparse.ArgumentParser(
        description='全方法 --nstep 默认值 10 -> 5（含 help 文案与契约断言）',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true', help='真正写盘（默认 dry-run）')
    ap.add_argument('--revert', action='store_true', help='反向还原（5 -> 10）')
    ap.add_argument('--check', action='store_true', help='只校验已处于目标状态、不写盘')
    args = ap.parse_args()
    if args.check and (args.apply or args.revert):
        raise SystemExit('--check 不能与 --apply/--revert 同用')

    old_v, new_v = (5, 10) if args.revert else (10, 5)

    # ---- 收集目标: 只有真正带 --nstep 的 main (+ 契约测试) ----
    targets, skipped = [], []
    for p in sorted(glob.glob(MAIN_GLOB)):
        src = read(p)
        (targets if RE_HAS_NSTEP.search(src) else skipped).append(p)
    targets.append(CONTRACT)

    direction = '还原 5 -> 10' if args.revert else '迁移 10 -> 5'
    mode = 'CHECK' if args.check else ('APPLY' if args.apply else 'DRY-RUN')
    print('=' * 100)
    print('  --nstep 默认值 %s   [%s]   带 nstep 的 main %d 个 + 契约测试 1 个'
          % (direction, mode, len(targets) - 1))
    print('  跳过（本就没有 --nstep）: %s' % (', '.join(skipped) if skipped else '(无)'))
    print('=' * 100)

    edits = [('CLI default', RE_DEF), ('help 默认', RE_HELP_DEF),
             ('help 建议', RE_HELP_TIP), ('注释头', RE_CMT)]
    pending, failures = [], []
    for path in targets:
        src = read(path)
        marks, ok = [], True

        def _apply(text, rx, label, new_value):
            """在合适窗口内替换: 以 `--nstep` 声明为锚, 前找注释头 / 后找 help。"""
            nonlocal ok
            anchor = RE_HAS_NSTEP.search(text)
            if anchor is None:
                marks.append('%s:无锚 ✗' % label)
                ok = False
                return text
            if rx is RE_CMT:                      # 注释头紧邻声明**之前**
                lo, hi = max(0, anchor.start() - CMT_LOOKBACK_CHARS), anchor.start()
            else:                                  # help 在声明**之后**
                lo, hi = anchor.start(), len(text)
            new, (lab, n) = sub_exactly_one(text, rx, label, new_value, lo, hi)
            # 安全断言: 每次替换只动个位数字符 (10->5), 长度变化必须极小 ——
            #   曾经因为重建逻辑写错把整份前缀重复拼接 (文件撑大近一倍), 这里硬拦。
            if n == 1 and abs(len(new) - len(text)) > 64:
                marks.append('%s:长度异常(%+d) ✗' % (label, len(new) - len(text)))
                ok = False
                return text
            marks.append('%s:%d%s' % (lab, n, '' if n == 1 else ' ✗(期望 1)'))
            if n != 1:
                ok = False
            return new

        if path == CONTRACT:
            out = src
            for label, rx in (('契约 default', RE_CONTRACT), ('契约 文案', RE_CONTRACT_TXT)):
                out, (lab, n) = sub_exactly_one(out, rx, label, new_v)
                marks.append('%s:%d%s' % (lab, n, '' if n == 1 else ' ✗(期望 1)'))
                if n != 1:
                    ok = False
        else:
            out = src
            for label, rx in edits:
                out = _apply(out, rx, label, new_v)

        if args.check:
            # 已处于目标状态 ⇔ 锚点仍停在 old_v (即 old_v 在锚点位模式上还能命中)
            if path == CONTRACT:
                hits = sum(sub_exactly_one(src, rx, 'x', new_v, must_equal=old_v)[1][1]
                           for rx in (RE_CONTRACT, RE_CONTRACT_TXT))
            else:
                hits = 0
                for _, rx in edits:
                    anchor = RE_HAS_NSTEP.search(src)
                    lo, hi = ((max(0, anchor.start() - CMT_LOOKBACK_CHARS), anchor.start())
                              if rx is RE_CMT else (anchor.start(), len(src)))
                    hits += sub_exactly_one(src, rx, 'x', new_v, lo, hi, must_equal=old_v)[1][1]
            rev_ok = (hits == 0)
            print('  [%s] %-44s %s' % ('OK ' if rev_ok else 'FAIL', path,
                                       '已处于目标状态' if rev_ok else '仍是旧状态(命中 %d)' % hits))
            if not rev_ok:
                failures.append(path)
            continue
        changed = out != src
        print('  [%s] %-44s %s%s' % ('OK ' if ok else 'FAIL', path, ' | '.join(marks),
                                     '' if changed else '   (无变化)'))
        if ok:
            pending.append((path, out, changed))
        else:
            failures.append(path)

    if args.check:
        print('\n结论: %s' % ('全部已达标' if not failures
                              else '仍有 %d 个文件未达标' % len(failures)))
        return 1 if failures else 0
    if failures:
        print('\n✗ 有 %d 个文件锚点不匹配 ⇒ **整体不写盘**（安全中止）' % len(failures))
        return 1
    if not args.apply:
        print('\n(dry-run) 可写改动 %d 个文件；加 --apply 落盘。'
              % sum(1 for _, _, c in pending if c))
        return 0
    n = 0
    for path, out, changed in pending:
        if changed:
            write(path, out)
            n += 1
    print('\n✓ 已写盘 %d 个文件（无变化的已跳过）' % n)
    return 0


if __name__ == '__main__':
    sys.exit(main())
