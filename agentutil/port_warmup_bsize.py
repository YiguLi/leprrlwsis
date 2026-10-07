# -*- coding: utf-8 -*-
"""正式口径批量改写：`--warmup` 默认 → 自动（= `--bsize × --warmup_bsize_mult`）。

用户要求（原话）："正式方法中，添加正式修改：使用 --bsize*4 steps 进行 warmup。
warmup 期间不进行任何网络更新。"

本脚本只做两处**机械**改写（每个 main 各 1 处 + 1 行插入）：
  A. `--warmup` 的 `default=200,` → `default=None,   # WARMUP_AUTO ...`
     （None ⇒ 交给 `hsrl_warmup.resolve_warmup` 解析成 `bsize × warmup_bsize_mult`）
  B. 在 `args = parser.parse_args()` 之后插入
     `    args = <hwarm 别名>.resolve_warmup(args)   # WARMUP_AUTO ...`
     （显式 `--warmup N` 仍然优先 ⇒ 复现历史 run 传 `--warmup 200`）

"warmup 期零网络更新"**不需要新行为**：13 个 main 的 `update_policy()` 调用点本就全部
落在 `if step > args.warmup:` 块内（已逐处核对）。本次把它升级为**运行期断言**
（`hsrl_agent_init.BiasInitTracker.on_block_end`：warmup 期每个块末检查
`agent.update_count == 0`，违反即抛错；触发点打印"warmup 期 N 块 / 0 次更新"）。

`--check` 做**反向还原**：把两处新文本换回旧文本必须逐位等于原文件
⇒ "只差这两处"的机器可读证明。

用法::

    python agentutil/port_warmup_bsize.py            # dry-run（打印命中次数）
    python agentutil/port_warmup_bsize.py --apply
    python agentutil/port_warmup_bsize.py --check
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

TARGETS = [
    'td3per/td3per_main.py',
    'td3perseg/td3perseg_main.py',
    'td3perrb/td3perrb_main.py',
    'td3prrlrb/td3prrlrb_main.py',
    'td3prrbb/td3prrbb_main.py',
    'td3prle/td3prle_main.py',
    'td3leprrlrb/td3leprrlrb_main.py',
    'singleaif/singleaif_main.py',
    'aifflor/aifflor_main.py',
    'yawflor/yawflor_main.py',
    'prle/prle_main.py',
    'prrbb/prrbb_main.py',
    # 注: `rbb/rbb_main.py` **不在**本表 —— 它既不 import hsrl_warmup、也不使用
    #     BiasInitTracker（遗留独立基线，且未在 test_five_days 注册）；给它加口径
    #     需要先引入整套 warmup 基建，超出本次"正式方法"范围。
]

OLD_DEFAULT = 'default=200,'
# ⚠ 不要在这里加行内注释：有的 main 把 `--warmup` 写成单行
#   `add_argument('--warmup', default=200, type=int, help=...)`
#   ⇒ 行内注释会把后面的 `type=int,` 一起注释掉（本脚本第二版踩过）。纯值替换 + 窗口定位即可。
NEW_DEFAULT = 'default=None,'
PARSE = 'args = parser.parse_args()'
RESOLVE_TMPL = 'args = {alias}.resolve_warmup(args)   # WARMUP_AUTO: 正式口径 warmup=bsize*mult, 期零网络更新'


def _sub_line(text, old_core, new_core, tag, log, window=None):
    """按**行内容**替换（保留缩进与行尾），old_core 需在该行内唯一出现。

    `new_core` 含换行时，其**后续行沿用被匹配行的缩进**（否则插入到函数体里会
    IndentationError —— 本脚本第一版就踩了这个坑）。
    """
    lines = text.splitlines(keepends=True)
    idx = [i for i, l in enumerate(lines) if old_core in l]
    if window is not None:
        idx = [i for i in idx if window[0] <= i <= window[1]]
    if len(idx) != 1:
        log.append((tag, len(idx)))
        return text
    i = idx[0]
    indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    new_block = new_core.replace('\n', '\n' + indent)
    lines[i] = lines[i].replace(old_core, new_block, 1)
    log.append((tag, 1))
    return ''.join(lines)


def patch(text, path, log):
    # ---- A. --warmup 的 default ----
    lines = text.splitlines(keepends=True)
    anchor = [i for i, l in enumerate(lines) if "add_argument('--warmup'," in l]
    if len(anchor) != 1:
        log.append(('%s :: --warmup anchor' % path, len(anchor)))
        return None, log
    a = anchor[0]
    text = _sub_line(text, OLD_DEFAULT, NEW_DEFAULT, '%s :: A default' % path, log,
                     window=(a, min(a + 4, len(lines) - 1)))
    # ---- B. resolve 调用 ----
    m = re.search(r'^import hsrl_warmup as (\w+)', text, re.M)
    if not m:
        log.append(('%s :: hwarm 别名' % path, 0))
        return None, log
    alias = m.group(1)
    text = _sub_line(text, PARSE, PARSE + '\n' + RESOLVE_TMPL.format(alias=alias),
                     '%s :: B resolve 插入' % path, log)
    return text, log


def revert(text, path, log):
    lines = text.splitlines(keepends=True)
    keep, n_del = [], 0
    for l in lines:
        if 'WARMUP_AUTO: 正式口径' in l:
            n_del += 1
            continue
        keep.append(l)
    log.append(('%s :: 反向 B' % path, n_del))
    text = ''.join(keep)
    # 反向 A 也要**限定在 --warmup 的窗口内**（generic 的 'default=None,' 别处也可能有）
    lines = text.splitlines(keepends=True)
    anchor = [i for i, l in enumerate(lines) if "add_argument('--warmup'," in l]
    if len(anchor) != 1:
        log.append(('%s :: 反向 A anchor' % path, len(anchor)))
        return text
    a = anchor[0]
    return _sub_line(text, NEW_DEFAULT, OLD_DEFAULT, '%s :: 反向 A' % path, log,
                     window=(a, min(a + 4, len(lines) - 1)))


def main():
    apply = '--apply' in sys.argv
    check = '--check' in sys.argv
    do_revert = '--revert' in sys.argv
    ok = True
    if do_revert:
        for path in TARGETS:
            disk = open(path, encoding='utf-8').read()
            rev = revert(disk, path, [])
            open(path, 'w', encoding='utf-8', newline='').write(rev)
        print('✅ 已回滚 %d 个 main（去掉 resolve 调用、恢复 default=200）' % len(TARGETS))
        return 0
    for path in TARGETS:
        if not os.path.isfile(path):
            print('  %-34s **缺文件**' % path)
            ok = False
            continue
        disk = open(path, encoding='utf-8').read()
        if check:
            # 正确语义：磁盘上是"已改写版" ⇒ 先反向还原、再正向改写，必须逐位回到磁盘版
            # （往返证明）；且还原结果里必须重新出现旧口径、且不含新标记。
            rev = revert(disk, path, [])
            rep, _ = patch(rev, path, [])
            ok_round = (rep == disk)
            ok_marks = (OLD_DEFAULT in rev) and ('WARMUP_AUTO' not in rev)
            good = ok_round and ok_marks
            ok = ok and good
            print('  %-34s %s  往返逐位一致=%s | 还原后含旧 default=200=%s | 还原后无新标记=%s' % (
                path, 'OK  ' if good else '**失败**', ok_round, OLD_DEFAULT in rev,
                'WARMUP_AUTO' not in rev))
            continue
        src = disk
        new, log = patch(src, path, [])
        hits = [n for _t, n in log]
        good = (new is not None) and all(n == 1 for n in hits)
        ok = ok and good
        print('  %-34s %s  %s' % (path, 'OK  ' if good else '**失败**',
                                  '  '.join('%.0f' % n for n in hits)))
        if not good:
            for t, n in log:
                print('        %-40s 命中 %d' % (t, n))
            continue
        if apply:
            open(path, 'w', encoding='utf-8', newline='').write(new)
    if check:
        print('--check: %s' % ('全部可逆 ✅' if ok else '**FAIL**'))
        return 0 if ok else 1
    print('(dry-run)' if not apply else '✅ 已写盘 %d 个 main' % len(TARGETS))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
