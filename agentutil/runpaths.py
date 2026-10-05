# -*- coding: utf-8 -*-
"""run 路径解析的**单一真源**（agentutil 内部互用；**项目代码不得 import 本模块**）。

为什么需要它
------------
早期探针把 run 前缀与日志名都写死了：

    d = os.path.join('output', 'TD3PER-' + name)          # ✗ 换个前缀就崩
    pd.read_csv(os.path.join(d, 'df_td3per.csv'), ...)     # ✗ aifflor 是 df_aifflor.csv

⇒ 在 `AIFFLOR-run*` / `SINGLEAIF-run*` / `YAWFLOR-run*` 上一律报
`找不到 run: AIFFLOR-run62`。本模块把它统一成：

* **定位目录**：显式目录 → `output/<name>` → `output/*-<name>`（任意前缀，含历史 `TD3PER-`）
* **定位日志表**：按**列**识别（`original_action_H` + `turbine_profit`），而不是按文件名命名习惯
  —— 一个 run 目录里同时躺着 `df_<tag>.csv`（逐步，目标表）、
  `df_episode_<tag>.csv`（逐 episode 汇总）、`df_test9.csv`（9 用例），
  只按 `df_*.csv` 通配会挑错。

用法::

    from runpaths import resolve_run, find_log_csv
    d = resolve_run('AIFFLOR-run62')          # 或 resolve_run('output/AIFFLOR-run62')
    log = find_log_csv(d)
"""
import glob
import os

import pandas as pd

#: 逐步日志表必须同时含这两列（用来把"逐步表"与 episode 汇总 / test9 表区分开）
REQUIRED_COLS = ('original_action_H', 'turbine_profit')


def find_log_csv(d, required=REQUIRED_COLS):
    """返回 run 目录 `d` 里的**逐步日志表**路径；找不到返回 `None`。

    优先历史名 `df_td3per.csv`（老 run 保持逐位不变），否则在 `df_*.csv` 里挑**真正含所需列**者。
    """
    if not d or not os.path.isdir(d):
        return None
    p = os.path.join(d, 'df_td3per.csv')
    if os.path.exists(p):
        return p
    for cand in sorted(glob.glob(os.path.join(d, 'df_*.csv'))):
        base = os.path.basename(cand)
        if base.startswith(('df_episode', 'df_test9', 'df_update')):
            continue
        try:
            head = pd.read_csv(cand, nrows=1)
        except Exception:
            continue
        if all(c in head.columns for c in required):
            return cand
    return None


def resolve_run(name, required=REQUIRED_COLS, verbose=False):
    """`run62` / `output/AIFFLOR-run62` / 任意目录 → run 目录（且确认有逐步日志表）。

    候选顺序（命中即返回）：显式目录 → `output/<name>` → `output/TD3PER-<name>`
    → `output/*-<name>` → `output/*-<name>-*`。全部落空则 `SystemExit` 并列出试过的候选。
    """
    cands = []
    if os.path.isdir(str(name)):
        cands.append(str(name))
    cands += [os.path.join('output', str(name)), os.path.join('output', 'TD3PER-' + str(name))]
    cands += sorted(glob.glob(os.path.join('output', '*-' + str(name))))
    cands += sorted(glob.glob(os.path.join('output', '*-' + str(name) + '-*')))
    for d in cands:
        if find_log_csv(d, required):
            if verbose:
                print('  [run] %s → %s' % (name, d))
            return d
    raise SystemExit('找不到 run: %s\n  试过: %s\n  提示: 直接给目录（如 output/AIFFLOR-run62）最稳'
                     % (name, ', '.join(cands[:6])))
