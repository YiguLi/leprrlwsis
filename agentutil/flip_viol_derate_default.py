# -*- coding: utf-8 -*-
"""把 `--viol_derate` 的默认值从 5.0 改回 **0 = 关闭** (run14 实测该机制有害)。

只改默认值 + 文档/注释, **不动机制实现** (开关、helper、分解三列、全部测试都保留),
以便日后在解决"奖励电平非平稳"后做受控实验。

证据 (run14 vs run13, 同 step 0-12k; 两者只差 --viol_derate):
  块奖励 116.8 -> 29.4 (-5x) | 越界率 24.2% -> 38.2% (振荡 18%~68%)
  罚/步 5.07 -> 10.27, 削减/步 0 -> 9.46, 净 r1/步 23.35 -> 5.88 (坏窗口 -14.6)
  Qbar 3.8 未跟随 (锚 4~9.9 -> 2.1~3.0) => 自洽比 0.4~0.8 -> **1.25~1.78 (高估)**
  |dQp/da|/Qbar 0.85 -> 2.08 | 策略抖动 rms(1) 0.122 -> 0.360
  机制: 乘性罚让奖励电平依赖策略自身越界率 => 非平稳 => 与固定 s_p/warmup 基线失配 => 正反馈

用法: python agentutil/flip_viol_derate_default.py [--apply]
"""

import argparse
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

SIX = ['td3per', 'td3perrb', 'td3prrlrb', 'td3prrbb', 'td3prle', 'td3leprrlrb']

CONST_OLD = 'VIOL_DERATE_COEF = 5.0     # c; <=0 关闭 (返回 1.0 ⇒ 与历史路径逐位相同)\n'
CONST_NEW = ('VIOL_DERATE_COEF = 0.0     # c; <=0 关闭 (返回 1.0 ⇒ 与历史路径逐位相同)\n'
             '#   ⚠ **默认 0 = 关闭**: c=5.0 试行过一次 (run14) 并**实测有害** ——\n'
             '#     乘性罚让奖励电平依赖"策略自身的越界率" ⇒ 电平非平稳 (块奖励 117->29,\n'
             '#     净 r1 掉到负), 而 critic 的 Qbar 因 warmup 基线 + 固定 s_p 不跟随\n'
             '#     ⇒ 自洽比 0.4~0.8 变 1.25~1.78 (高估约 2x) ⇒ actor 追虚高 Q ⇒ 越界更多\n'
             '#     ⇒ 正反馈 (越界率 18%~78% 振荡, 策略抖动 3 倍)。要重新启用必须**先**解决\n'
             '#     "奖励电平非平稳" (见 README 的 --viol_derate 条目)。\n')

CLI_OLD = ("    #   复现历史 run1~run13 请显式传 --viol_derate 0 (该路径逐位不变: f 恰为 1.0)。\n"
           "    parser.add_argument('--viol_derate', default=VIOL_DERATE_COEF, type=float,\n"
           "                        help='越界乘性惩罚系数 c: 风机收益 × exp(-c·Σ前5维越界量) '\n"
           "                             '(默认 %.1f = 生产口径; 0 = 关闭, 逐位复现历史 run)' % VIOL_DERATE_COEF)\n")
CLI_NEW = ("    #   ⚠ **默认 0 = 关闭** (run14 实测 c=5.0 有害: 奖励电平被策略自身越界率绑架\n"
           "    #     ⇒ critic 高估约 2x ⇒ 正反馈; 详见 README 的 --viol_derate 条目)。\n"
           "    #   传 0 = 逐位复现历史 run1~run14。\n"
           "    parser.add_argument('--viol_derate', default=VIOL_DERATE_COEF, type=float,\n"
           "                        help='越界乘性惩罚系数 c: 风机收益 × exp(-c·Σ前5维越界量) '\n"
           "                             '(默认 %.1f = **关闭**; 5.0 = 已证伪的旧口径, 见 README run14 分析)'\n"
           "                             % VIOL_DERATE_COEF)\n")


def patch(path, pairs, dry):
    with open(path, encoding='utf-8') as f:
        src = f.read()
    ok = True
    for old, new, label in pairs:
        n = src.count(old)
        if n != 1:
            print('    [FAIL] %s: %s 命中 %d 次 (要求 1)' % (os.path.basename(path), label, n))
            ok = False
            continue
        src = src.replace(old, new)
    if ok and not dry:
        with open(path, 'w', encoding='utf-8', newline='') as f:
            f.write(src)
    if ok:
        print('    [OK]   %s: %d 处替换%s' % (os.path.basename(path), len(pairs), '' if dry else ' (已写盘)'))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    dry = not args.apply
    allok = True
    for k in SIX:
        allok &= patch('%s/%s_myfunc.py' % (k, k),
                       [(CONST_OLD, CONST_NEW, 'VIOL_DERATE_COEF 5.0->0.0')], dry)
        allok &= patch('%s/%s_main.py' % (k, k),
                       [(CLI_OLD, CLI_NEW, '--viol_derate CLI 注释/help')], dry)
    print('\n%s' % ('全部命中%s' % (' (dry-run)' if dry else ' (已写盘)') if allok else '有未命中项, 未写盘'))
    return 0 if allok else 1


if __name__ == '__main__':
    sys.exit(main())
