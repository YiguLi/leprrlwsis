#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""给"没有输出简化开关"的 12 个 `*_main.py` 接上 `run_io`（aifflor 同款）。

每个 main 插入 4 处（幂等；默认 dry-run，`--apply` 才写盘）：

1. `from util import *` 之后：`import run_io`
2. `--hidden1` 之前：`--log_file` / `--progress` / `--progress_every` 三个参数
   （**默认 `--log_file 0` = 行为与改造前逐位相同**；`--log_file 1` 才把详细输出写进
   `<output>/train.log`、控制台只留进度行）
3. `args.output = get_output_folder(...)` 之后：`run_io.setup(...)`
4. 训练循环的 `step += 1` 之后：`run_io.tick(step)`

用法::

    python agentutil/port_run_io.py            # dry-run
    python agentutil/port_run_io.py --apply
"""
import argparse
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)

MAINS = [
    'td3per/td3per_main.py',
    'td3prrbb/td3prrbb_main.py',
    'td3prle/td3prle_main.py',
    'td3prrlrb/td3prrlrb_main.py',
    'td3perrb/td3perrb_main.py',
    'td3leprrlrb/td3leprrlrb_main.py',
    'td3perseg/td3perseg_main.py',
    'prle/prle_main.py',
    'prrbb/prrbb_main.py',
    'singleaif/singleaif_main.py',
    'yawflor/yawflor_main.py',
    'aifflorpave/aifflorpave_main.py',
]

ARGS_BLOCK = """    parser.add_argument('--log_file', default=0, type=int,
                        help='1 = 把逐行详细输出写入 <output>/train.log，控制台只留 --progress 进度行'
                             '（与 aifflor 同款；默认 0 = 输出与改造前逐位相同）')
    parser.add_argument('--progress', default=1, type=int,
                        help='1 = 每 --progress_every 步打印一行进度（步/秒 + ETA）；'
                             '只在 --log_file 1 时生效（默认 1）')
    parser.add_argument('--progress_every', default=500, type=int,
                        help='进度行间隔（环境步）')
"""

SETUP_BLOCK = """    # ---- 输出简化（aifflor 同款）：--log_file 1 ⇒ 详细输出进 train.log ----
    run_io.setup(args.output,
                 log_file=int(getattr(args, 'log_file', 0) or 0),
                 progress=int(getattr(args, 'progress', 1) or 1),
                 every=int(getattr(args, 'progress_every', 500) or 500),
                 total=0)          # 真正的总步数在 train(...) 调用前用 run_io.set_total 设
"""

TOTAL_RE = re.compile(r'^(\s+)(train|train_prle|train_prrbb)\(args\.train_iter_H,')


def patch(path):
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
    text = ''.join(lines)
    notes = []
    if 'import run_io' in text:
        return ['  [skip] 已接 run_io']

    # 1) import
    for i, line in enumerate(lines):
        if line.strip() == 'from util import *':
            lines.insert(i + 1, '\nimport run_io   # 输出简化（--log_file/--progress，aifflor 同款）\n')
            notes.append('  import run_io (after line %d)' % (i + 1))
            break
    else:
        return ['  [ERR] 找不到 `from util import *`']

    # 2) parser args
    for i, line in enumerate(lines):
        if "add_argument('--hidden1'" in line:
            lines.insert(i, ARGS_BLOCK)
            notes.append('  cli: --log_file/--progress/--progress_every (before line %d)' % (i + 1))
            break
    else:
        return notes + ['  [ERR] 找不到 --hidden1 锚点']

    # 3) setup after get_output_folder
    for i, line in enumerate(lines):
        if 'get_output_folder(args.output, args.env)' in line:
            lines.insert(i + 1, SETUP_BLOCK)
            notes.append('  output: run_io.setup (after line %d)' % (i + 1))
            break
    else:
        return notes + ['  [ERR] 找不到 get_output_folder 锚点']

    # 4) tick after step += 1
    hits = [i for i, line in enumerate(lines) if re.match(r'^(\s+)step \+= 1\s*$', line)]
    if len(hits) != 1:
        return notes + ['  [ERR] `step += 1` 命中 %d 次（预期 1）' % len(hits)]
    i = hits[0]
    indent = re.match(r'^(\s+)', lines[i]).group(1)
    lines.insert(i + 1, indent + 'run_io.tick(step)\n')
    notes.append('  loop: run_io.tick(step) (after line %d)' % (i + 1))

    # 5) set_total 在 train(...) 调用之前（此时 train_iter_H 已按 max_steps/max_episode_length 定稿）
    hits = [i for i, line in enumerate(lines) if TOTAL_RE.match(line)]
    if len(hits) != 1:
        return notes + ['  [ERR] train(...) 调用锚点命中 %d 次（预期 1）' % len(hits)]
    i = hits[0]
    indent = TOTAL_RE.match(lines[i]).group(1)
    lines.insert(i, indent + 'run_io.set_total(int(args.train_iter_H))\n')
    notes.append('  total: run_io.set_total (before line %d)' % (i + 1))
    return notes, ''.join(lines)


def main():
    ap = argparse.ArgumentParser(description='给 mains 接 run_io（输出简化）')
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    n = 0
    for rel in MAINS:
        path = os.path.join(ROOT, rel)
        print('=' * 70)
        print(rel)
        res = patch(path)
        if isinstance(res, tuple):
            notes, new = res
        else:
            notes, new = res, None
        for x in notes:
            print(x)
        if new is not None and a.apply:
            with open(path, 'w', encoding='utf-8', newline='') as f:
                f.write(new)
            n += 1
    print('=' * 70)
    print('%s：%d 个文件' % ('APPLIED' if a.apply else 'DRY-RUN', n))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
