#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""把 v7 结构变成**所有方法的默认网络**（一次性迁移；默认 dry-run，`--apply` 才写盘）。

背景
----
用户 2026-11-04 口径："v7 是极为成功的结构，将所有方法的神经网络结构改为默认 v7"。
v7 的定义（见 `documentation/critic_arch_v7.md`）：

    critic: cat(state, action) → 400 → 300 → 400 → 1     （model.V7_DIMS）
    actor : state → 400 → 300 → 400 → action              （model.V7_ACTOR_DIMS）

本脚本只改**老方法**（没有 `--critic_arch` 开关的 9 个算法：prle / singleaif /
td3leprrlrb / td3perrb / td3perseg / td3prle / td3prrbb / td3prrlrb / yawflor），
每个项目两处：

1. `<pkg>_ddpg.py`：
   * `from model import (Actor, Critic)` → 追加 `V7_ACTOR_DIMS`；
   * 在 `net_cfg = {...}` 之后插入"结构解析块"（`actor_cfg` / `critic_cfg`，
     `arch = getattr(args, 'critic_arch', 'v1')`，v7 时 actor 用 400/300）；
   * 把 `Actor(... **net_cfg)` → `**actor_cfg`、`Critic(... **net_cfg)` → `**critic_cfg`。
2. `<pkg>_main.py`：在 `--hidden1` 之前插入 `--critic_arch`（**default='v7'**）。

⚠ 刻意**不动**回落值：`getattr(args, 'critic_arch', 'v1')` 仍是 v1。
   这样 `agentutil/` 的探针、`test_five_days.py` 等"手工构造 args"的脚本
   不会被静默改义（它们拿到的仍是历史网络）；只有真正用 CLI 起训练时才默认 v7。

幂等：文件里已出现 `--critic_arch` / `actor_cfg` 就跳过并报告。

用法::

    python agentutil/port_v7_default.py            # 只打印将要做的改动
    python agentutil/port_v7_default.py --apply    # 写盘
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)

#: (包名, 主脚本, ddpg 模块)
PROJECTS = [
    ('prle', 'prle_main.py', 'prle_ddpg.py'),
    ('singleaif', 'singleaif_main.py', 'singleaif_ddpg.py'),
    ('td3leprrlrb', 'td3leprrlrb_main.py', 'td3leprrlrb_ddpg.py'),
    ('td3perrb', 'td3perrb_main.py', 'td3perrb_ddpg.py'),
    ('td3perseg', 'td3perseg_main.py', 'td3perseg_ddpg.py'),
    ('td3prle', 'td3prle_main.py', 'td3prle_ddpg.py'),
    ('td3prrbb', 'td3prrbb_main.py', 'td3prrbb_ddpg.py'),
    ('td3prrlrb', 'td3prrlrb_main.py', 'td3prrlrb_ddpg.py'),
    ('yawflor', 'yawflor_main.py', 'yawflor_ddpg.py'),
]

NET_BLOCK = """        # ---- 网络结构（--critic_arch）: 2026-11-04 起**默认 v7** ----
        #   v7 = critic cat(state,action)→400→300→400→1 + actor state→400→300→400→action。
        #   ⚠ 回落仍是 v1：**不传 args.critic_arch 的探针/评测脚本**保持历史网络（逐位不变）。
        #   actor 只在 **H 层**（nb_actions>1）跟着放大；L 层是 1 维动作，
        #   保持 --hidden1/--hidden2（与 aifflor 口径一致，避免无谓放大 100 倍）。
        _arch = str(getattr(args, 'critic_arch', 'v1') or 'v1')
        actor_cfg = dict(net_cfg)
        critic_cfg = dict(net_cfg)
        critic_cfg['arch'] = _arch
        if _arch == 'v7' and self.nb_actions > 1:
            actor_cfg['hidden1'], actor_cfg['hidden2'] = V7_ACTOR_DIMS
"""

MAIN_ARG = """    parser.add_argument('--critic_arch', default='v7',
                        choices=['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7'],
                        help='critic/actor 结构（2026-11-04 起默认 **v7**）: '
                             'v7 = critic cat(s,a)→400→300→400→1 + actor s→400→300→400→a；'
                             'v1 = 历史 16/100（复现旧 run / 加载旧 checkpoint 用）')
"""


def find_net_cfg_end(lines):
    """返回 `net_cfg = {` 的闭合 `}` 的行号（0-based），找不到返回 None。"""
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('net_cfg = {'):
            start = i
            break
    if start is None:
        return None
    depth = 0
    for j in range(start, len(lines)):
        depth += lines[j].count('{') - lines[j].count('}')
        if depth == 0 and j > start:
            return j
    return None


def patch_ddpg(path):
    with open(path, encoding='utf-8') as f:
        text = f.read()
    lines = text.splitlines(keepends=True)
    changes = []
    if 'actor_cfg' in text:
        return ['  [skip] 已有 actor_cfg（已迁移？）']

    # 1) model import
    n = 0
    for i, line in enumerate(lines):
        if line.strip() == 'from model import (Actor, Critic)':
            lines[i] = line.replace('from model import (Actor, Critic)',
                                    'from model import (Actor, Critic, V7_ACTOR_DIMS)')
            changes.append('  import: + V7_ACTOR_DIMS (line %d)' % (i + 1))
            n += 1
    if n == 0:
        return ['  [ERR] 找不到单行 `from model import (Actor, Critic)`，跳过']

    # 2) net block
    end = find_net_cfg_end(lines)
    if end is None:
        return ['  [ERR] 找不到 net_cfg = {...} 块，跳过']
    lines.insert(end + 1, NET_BLOCK)
    changes.append('  net block: 插在 net_cfg 闭合行 %d 之后' % (end + 1))

    # 3) constructor kwargs
    n_actor = n_critic = 0
    for i, line in enumerate(lines):
        if '**net_cfg' not in line:
            continue
        if 'Actor(' in line:
            lines[i] = line.replace('**net_cfg', '**actor_cfg')
            n_actor += 1
        elif 'Critic(' in line:
            lines[i] = line.replace('**net_cfg', '**critic_cfg')
            n_critic += 1
    left = [i + 1 for i, line in enumerate(lines) if '**net_cfg' in line]
    if left:
        changes.append('  [ERR] 还有未替换的 **net_cfg: 行 %s' % left)
    changes.append('  kwargs: Actor→actor_cfg ×%d, Critic→critic_cfg ×%d'
                   % (n_actor, n_critic))
    return changes, ''.join(lines)


def patch_main(path):
    with open(path, encoding='utf-8') as f:
        text = f.read()
    if "'--critic_arch'" in text:
        return ['  [skip] 已有 --critic_arch']
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if "add_argument('--hidden1'" in line:
            lines.insert(i, MAIN_ARG)
            return ['  cli: --critic_arch（default v7）插在 line %d 之前' % (i + 1)], ''.join(lines)
    return ['  [ERR] 找不到 --hidden1 锚点，跳过']


def main():
    ap = argparse.ArgumentParser(description='把 v7 设为所有方法的默认网络')
    ap.add_argument('--apply', action='store_true', help='真正写盘（默认 dry-run）')
    a = ap.parse_args()

    n_change = 0
    for pkg, main_name, ddpg_name in PROJECTS:
        mpath = os.path.join(ROOT, pkg, main_name)
        dpath = os.path.join(ROOT, pkg, ddpg_name)
        print('=' * 78)
        print('%s' % pkg)
        for tag, path, fn in (('ddpg', dpath, patch_ddpg), ('main', mpath, patch_main)):
            if not os.path.isfile(path):
                print('  [ERR] 缺文件 %s' % path)
                continue
            res = fn(path)
            if isinstance(res, tuple):
                changes, new_text = res
            else:
                changes, new_text = res, None
            for c in changes:
                print('  %-5s %s' % (tag, c.strip()))
            if new_text is not None and a.apply:
                with open(path, 'w', encoding='utf-8', newline='') as f:
                    f.write(new_text)
                n_change += 1
    print('=' * 78)
    print('%s：%d 个文件被修改' % ('APPLIED' if a.apply else 'DRY-RUN', n_change))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
