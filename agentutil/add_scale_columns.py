#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在上一批 `lam_rt` 接线之后，补写 `value_scale` / `cost_scale` 两列。

动机：`update_stats_H.csv` 里的 q_*/td_*/target_* 现在**都在回归空间**（取消折算的
直接后果）。分析脚本若需要原始口径（例如把 `q_c` 与块级 `c1` 和 `d_block=15` 对比，
或与历史 run 的图对齐），必须知道当初的 s —— 让表**自描述**比让读者去猜/去翻启动
日志可靠。两列都是常数（run 内不变）。

幂等：已含 value_scale 键的行会跳过。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

MAINS = [
    'aifflor/aifflor_main.py', 'aifflorpave/aifflorpave_main.py', 'PER_main0.py',
    'prle/prle_main.py', 'prrbb/prrbb_main.py', 'singleaif/singleaif_main.py',
    'td3leprrlrb/td3leprrlrb_main.py', 'td3per/td3per_main.py',
    'td3perrb/td3perrb_main.py', 'td3perseg/td3perseg_main.py',
    'td3prle/td3prle_main.py', 'td3prrbb/td3prrbb_main.py',
    'td3prrlrb/td3prrlrb_main.py', 'yawflor/yawflor_main.py',
]

ANCHOR = ("                    _mean['lam_rt'] = hc.lam_rt(\n"
          "                        agent_H.lam, _mean.get('q_c_mean', 0.0),\n"
          "                        _mean.get('q_p_mean', 0.0))\n")

ADD = ANCHOR + (
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
        with open(f, encoding='utf-8') as fh:
            src = fh.read()
        if "'value_scale']" in src:
            miss.append((f, 'already present'))
            continue
        if src.count(ANCHOR) != 1:
            miss.append((f, 'anchor x%d' % src.count(ANCHOR)))
            continue
        if not check:
            with open(f, 'w', encoding='utf-8', newline='') as fh:
                fh.write(src.replace(ANCHOR, ADD))
        print('  %-38s ✓' % f)
    if miss:
        print('\n未处理：')
        for m in miss:
            print('   %-38s %s' % m)
        sys.exit(1)
    print('\n✓ %d 个 main %s' % (len(MAINS), '检查通过' if check else '已写入'))


if __name__ == '__main__':
    main()
