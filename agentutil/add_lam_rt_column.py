#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 9 个有"块末汇总 → update_stats_H.csv"的主脚本里补一列 `lam_rt`。

`lam_rt = λ_actor·q_c/q_p`（见 `hsrl_cost.lam_rt`）是**唯一**同时满足
"约束项到底占利润多少"与"不随 `--value_scale`/`--cost_scale` 变"的判据
（两边的 s 各自约掉）。原来的表里没有这一列，读者只能手工还原两个通道的
scale 才能算出来 —— 这正是被取消的"按通道还原"留下的坑。

不改任何训练路径：只在 `_mean`（块末 2 批均值 dict）里多一个键。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

MAINS = [
    'aifflor/aifflor_main.py',
    'aifflorpave/aifflorpave_main.py',
    'PER_main0.py',
    'prle/prle_main.py',
    'prrbb/prrbb_main.py',
    'singleaif/singleaif_main.py',
    'td3leprrlrb/td3leprrlrb_main.py',
    'td3per/td3per_main.py',
    'td3perrb/td3perrb_main.py',
    'td3perseg/td3perseg_main.py',
    'td3prle/td3prle_main.py',
    'td3prrbb/td3prrbb_main.py',
    'td3prrlrb/td3prrlrb_main.py',
    'yawflor/yawflor_main.py',
]

OLD = ("                    _mean = {k: float(np.mean([s[k] for s in _stats]))\n"
       "                             for k in _stats[0]}\n")

NEW = ("                    _mean = {k: float(np.mean([s[k] for s in _stats]))\n"
       "                             for k in _stats[0]}\n"
       "                    # scale-free 判据: 约束项占利润的比 (λ_actor·q_c/q_p; 两通道\n"
       "                    # 的 s 各自约掉) —— 手工折算 scale 才能算的量, 现在直接落盘。\n"
       "                    _mean['lam_rt'] = hc.lam_rt(\n"
       "                        agent_H.lam, _mean.get('q_c_mean', 0.0),\n"
       "                        _mean.get('q_p_mean', 0.0))\n"
       "                    # 让这张表**自描述空间**: 所有 q_*/td_*/target_* 都在回归空间\n"
       "                    # (不折算), 分析脚本要用原始口径时按这两列自行换算。\n"
       "                    _mean['value_scale'] = float(getattr(agent_H, 'value_scale', 1.0) or 1.0)\n"
       "                    _mean['cost_scale'] = float(\n"
       "                        getattr(agent_H, 'cost_scale', None)\n"
       "                        or getattr(agent_H, 'value_scale', 1.0) or 1.0)\n")


def main():
    check = '--check' in sys.argv
    miss = []
    for f in MAINS:
        if not os.path.exists(f):
            miss.append((f, 'MISSING'))
            continue
        with open(f, encoding='utf-8') as fh:
            src = fh.read()
        n = src.count(OLD)
        if n != 1:
            miss.append((f, 'anchor x%d' % n))
            continue
        if "'lam_rt'" in src:
            miss.append((f, 'already present'))
            continue
        if not check:
            with open(f, 'w', encoding='utf-8', newline='') as fh:
                fh.write(src.replace(OLD, NEW))
        print('  %-38s ✓' % f)
    if miss:
        print('\n✗ 未处理：')
        for m in miss:
            print('   %-38s %s' % m)
        sys.exit(1)
    print('\n✓ %d 个 main %s' % (len(MAINS), '检查通过' if check else '已写入'))


if __name__ == '__main__':
    main()
