# -*- coding: utf-8 -*-
r"""一次性迁移：BESS 全局口径 `10 MW / 6 MWh` → `0.5 MW / 0.5 MWh`。

用户口径（2026-10）
------------------
  * 最大充放电 = **0.5 MW**（旧 10 MW）
  * 最大充能（容量）= **0.5 MWh**（旧 6 MWh，SOC 工作区 [0.5, 5.5]）
  * `SOC_TOL` = **0.01 MWh**（旧 0.1）

派生缩放（容量 12× ⇒ SOC 工作区必须同步，否则目标 3.0 > 容量 0.5 会物理不可行）
------------------------------------------------------------------------
      SOC_TARGET_CENTER   3.0      → 0.25
      SOC_TARGET_SPAN     5.0      → 0.4166666666666667   (= 5/12)
      SOC_TARGET_MIN/MAX  0.5,5.5  → 0.041666666666666664, 0.4583333333333333

不做本次替换的（用户未定，保持原值并在此登记）
----------------------------------------------
  * `L_CLIP_CAP = 20.0`   —— 只由功率窗口 ±rated 的跨度决定（新为 ±0.5 ⇒ 跨度 1.0）
  * `L_D_BLOCK / L_D_BLOCK_MIN / L_COST_LIMIT`（MW·min 量纲）
  * `hc.D_BLOCK=15`、λ 区间、`W_GAP`、`PENALTY_*`
  * `SOC_TOL` 的 **help 文本**与 docstring 中的数字（只改真正的赋值行）

安全机制：每一组替换都要求**命中数精确匹配**，任一不符 ⇒ 整体不写盘（dry-run 打印差异）。
用法：
    python agentutil/migrate_bess_scale.py            # dry-run，只打印
    python agentutil/migrate_bess_scale.py --apply
"""
import argparse
import glob
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)

SKIP_DIRS = ('agentutil', '__pycache__', 'output', '未命名', 'libraries', '.git')

# (模式, 替换, 期望命中数或 None=不限, 说明)
RULES = [
    # ---- 1. rated_power 10 → 0.5 ----
    (r"--rbb_rated_power', default=10\.0", "--rbb_rated_power', default=0.5", 16,
     'CLI 默认 rbb_rated_power'),
    (r"rated_power=10\.0", "rated_power=0.5", 12,
     'RuleBasedBESS/SocTracker/rule_action 构造默认（含 test_simple.py）'),
    (r"RATED_POWER = 10\.0", "RATED_POWER = 0.5", 3,
     'hsrl_warmup / 评测脚本的模块常量'),
    # ---- 2. 容量 6 → 0.5 ----
    (r"max_E = 6\.", "max_E = 0.5", 21, '各 main 的 max_E'),
    (r"MAX_E = 6\.", "MAX_E = 0.5", 2, '评测脚本的 MAX_E'),
    (r"max_E=6\.", "max_E=0.5", 34, '函数签名/构造默认里的 max_E（含 Normalize/Original）'),
    # ---- 3. CheckAction 功率上限 10 → 0.5（两种空格写法）----
    (r"charge ?= ?10\.", "charge=0.5", 38,
     'CheckAction 的物理功率上限（另 2 处在 output/ 一次性探针里，不在改动面）'),
    # ---- 4. SOC_TOL 0.1 → 0.01（只改赋值行，不动 help/docstring）----
    (r"^SOC_TOL = 0\.1", "SOC_TOL = 0.01", 9, '各 myfunc 与 hsrl_cost 的 SOC_TOL 赋值'),
    # ---- 5. SOC 目标映射（12× 缩放）----
    (r"^SOC_TARGET_CENTER = 3\.0", "SOC_TARGET_CENTER = 0.25", 10, 'SOC 目标中心'),
    (r"^SOC_TARGET_SPAN = 5\.0", "SOC_TARGET_SPAN = 0.4166666666666667", 10, 'SOC 目标跨度'),
    (r"SOC_TARGET_MIN, SOC_TARGET_MAX = 0\.5, 5\.5",
     "SOC_TARGET_MIN, SOC_TARGET_MAX = 0.041666666666666664, 0.4583333333333333",
     9, 'SOC 目标上下限（显式版本）'),
    # ---- 6. 起始 SOC（12× 缩放）----
    (r"^SOC_START = 3\.0", "SOC_START = 0.25", 1, 'hsrl_warmup 起始 SOC'),
    (r"^SOC_TARGET = 3\.0", "SOC_TARGET = 0.25", 1, 'hsrl_warmup warmup 期目标'),
    (r"self\.E = 3\.", "self.E = 0.25", 1, 'env.reset 起始 SOC'),
]


def files():
    """收集待改文件：跳过 output/ 与 agentutil/ 等目录，但**保留**根目录 /
    项目子目录下的 `test_*.py`（它们也带 BESS 常量）。"""
    out = []
    for f in glob.glob('**/*.py', recursive=True):
        parts = f.replace('\\', '/').split('/')
        if any(p in SKIP_DIRS for p in parts):
            continue
        out.append(f)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    total = 0
    grand = {}
    for pat, rep, want, desc in RULES:
        hits = []
        rx = re.compile(pat, re.M)
        # 用 finditer 计数（与替换同一套遍历语义；findall 在带捕获组时语义不同，历史上踩过）
        n_file = {}
        for f in files():
            t = open(f, encoding='utf-8').read()
            n = sum(1 for _ in rx.finditer(t))
            if n:
                hits.append((f, n))
                n_file[f] = n
        n_all = sum(n for _, n in hits)
        status = 'OK '
        if want is not None and n_all != want:
            status = '❌ '
        print('%s %-38s 命中 %3d (期望 %s)  %s' % (status, desc, n_all,
                                                   want if want is not None else '任意',
                                                   ', '.join('%s:%d' % (f.replace(chr(92), '/'), n)
                                                             for f, n in hits[:6])))
        print('%s %-38s 命中 %3d (期望 %s)  %s' % (status, desc, n_all,
                                                   want if want is not None else '任意',
                                                   ', '.join('%s:%d' % (f.replace(chr(92), '/'), n)
                                                             for f, n in hits[:6])))
        if want is not None and n_all != want:
            print('   ⇒ 命中数不符，**跳过该组**（避免误伤）。逐文件明细：')
            for f, n in hits:
                print('        %-46s %d' % (f.replace(chr(92), '/'), n))
            continue
        grand[desc] = n_all
        total += n_all
        if args.apply:
            for f, _ in hits:
                t = open(f, encoding='utf-8').read()
                open(f, 'w', encoding='utf-8', newline='\n').write(rx.sub(rep, t))
    print('\n合计 %d 处%s' % (total, '（已写盘）' if args.apply else '（dry-run，未写盘）'))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
