#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""日志表**增量落盘**的回归测试 —— 不需要 MATLAB。

背景（实测）：主循环里"只增不减"的整表 `df` 若每 50 步做一次全量 `df.to_csv(path)`，
单次写盘量 = O(累积行数) ⇒ **迭代数越大越卡**，表现为"每 50 整数倍 step 卡一下"，
且卡顿随训练步数加剧（run9 实测该表到 67k 步已 63 MB；`td3prle` 的注释里记着同一条诊断）。
改为"首段覆盖写 + 其后只追加新行"后，每次写盘量 = O(新增行) = 常数，整轮总写盘量 O(N)。

两节：
  A. **静态**（6 个 TD3 main）：主表与 `update_stats_H.csv` 的落盘点必须存在"追加"路径；
     裸的全量 `df.to_csv(` 只允许出现在 (i) `*_flushed_rows == 0` 守卫下的首段覆盖写，
     或 (ii) 文件末尾的收尾安全写。
  B. **功能**（`util.flush_csv_append`）：增量写盘与一次性全量写盘**逐位一致**；
     每次落盘只追加增量（写盘量不随累积行数增长）；幂等；首段覆盖旧文件；
     中途落盘的文件始终可读（崩溃最多丢一个落盘间隔的行，与旧语义一致）。
"""
import os
import re
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors='replace')
    except Exception:
        pass

from util import flush_csv_append                                    # noqa: E402

TD3_MAINS = ['td3per/td3per_main.py', 'td3prrbb/td3prrbb_main.py',
             'td3prle/td3prle_main.py', 'td3prrlrb/td3prrlrb_main.py',
             'td3perrb/td3perrb_main.py', 'td3leprrlrb/td3leprrlrb_main.py']
# 三种已存在的增量实现（同一语义）: rows-列表 / df 切片计数 / 共享 helper
INC = re.compile(r"_flush_csv\(|flush_csv_append\(|mode='a'|iloc\[[^\]]*flushed")


def src(p):
    return open(p, encoding='utf-8').read()


print('=' * 84)
print('A. 静态: 6 个 TD3 main 的落盘点必须是增量的 (无 O(N) 全量重写)')
print('=' * 84)
for p in TD3_MAINS:
    t = src(p)
    L = t.split('\n')
    # ① 主帧的"每 50 步"落盘点必须存在追加路径
    i = next((k for k, l in enumerate(L) if re.search(r'if step % 50 == 0', l)), -1)
    assert i >= 0, (p, '未找到 50 步落盘点')
    blk = '\n'.join(L[i:i + 14])
    assert INC.search(blk), (p, '50 步落盘点没有追加路径 ⇒ 仍是 O(N) 全量重写 (50 倍数卡顿源)', blk[:200])
    # ② update_stats_H.csv 的每块落盘点同理
    n_upd = 0
    for k, l in enumerate(L):
        if 'update_stats_H.csv' in l:
            n_upd += 1
            ctx = '\n'.join(L[max(0, k - 4):k + 5])
            assert INC.search(ctx), (p, 'update_stats_H.csv 落盘无追加路径 (行 %d)' % (k + 1))
    assert n_upd >= 1, (p, '未找到 update_stats_H.csv 落盘点')
    # ③ 裸全量 df.to_csv 只允许: 首段覆盖写(带 ==0 守卫) 或 循环外的收尾安全写
    #    循环内的写缩进 ≥12 空格; 收尾 `if use_csv:` 块里的写缩进 ≤8 ⇒ 用缩进判定
    bare = [k for k, l in enumerate(L) if re.match(r'\s*df\.to_csv\(', l)]
    for k in bare:
        guarded = any(re.search(r'==\s*0\b', x) for x in L[max(0, k - 3):k])
        indent = len(L[k]) - len(L[k].lstrip())
        assert guarded or indent <= 8, \
            (p, '第 %d 行有裸全量 df.to_csv (缩进 %d, 既非首段也非收尾)' % (k + 1, indent))
    print('  %-32s 50 步落盘=增量 ✓  update_stats=%d 处(增量) ✓  裸全量写=%d 处(首段/收尾) ✓'
          % (p, n_upd, len(bare)))

print('=' * 84)
print('B. 功能: util.flush_csv_append —— 增量 ≡ 全量, 且写盘量不随累积行数增长')
print('=' * 84)
tmp = tempfile.mkdtemp(prefix='csvinc_')
try:
    N, STEP = 2000, 50
    rng = np.random.RandomState(0)
    big = pd.DataFrame({'step': np.arange(N), 'v': rng.rand(N),
                        'txt': ['row-%d' % i for i in range(N)]})
    p_inc = os.path.join(tmp, 'inc.csv')
    p_full = os.path.join(tmp, 'full.csv')
    big.to_csv(p_full)
    full_bytes = os.path.getsize(p_full)
    written, prev, deltas = 0, 0, []
    for k in range(1, N + 1):
        if k % STEP == 0:
            written = flush_csv_append(p_inc, big.iloc[:k].reset_index(drop=True), written)
            assert written == k, (written, k)
            # 中途文件始终可读, 且恰好是已落盘的行
            got = pd.read_csv(p_inc)
            assert len(got) == k and list(got['step']) == list(range(k)), (len(got), k)
            sz = os.path.getsize(p_inc)
            deltas.append(sz - prev)
            prev = sz
    # ① 增量结果与一次性全量写盘**逐位一致**
    b_inc, b_full = open(p_inc, 'rb').read(), open(p_full, 'rb').read()
    assert b_inc == b_full, ('增量与全量结果不一致', len(b_inc), len(b_full))
    # ② 每次落盘只追加增量 (最后一行的字节 ~= 新增 50 行的字节)
    last_row_bytes = full_bytes / N
    assert max(deltas) < 2.0 * STEP * last_row_bytes + 64, \
        ('某次落盘写了远超"新增行"的量 ⇒ 仍在重写整表', max(deltas), STEP * last_row_bytes)
    # ③ 整轮总写盘量 O(N) 而不是 O(N²/STEP)
    naive = sum(full_bytes * k / N for k in range(STEP, N + 1, STEP))   # 旧方案: 每次全量重写
    assert sum(deltas) <= 1.2 * full_bytes, ('总写盘量应 ≈ 最终大小', sum(deltas), full_bytes)
    print('  ① 增量 ≡ 全量: %d 字节逐位一致 (%d 行)' % (len(b_inc), N))
    print('  ② 单次落盘写盘量: max %d 字节 (新增 50 行 ≈ %d 字节)' % (max(deltas), STEP * last_row_bytes))
    print('  ③ 整轮总写盘量: 增量 %d 字节 (= 最终大小 ×%.2f) vs 旧全量方案 %d 字节 (×%.1f)'
          % (sum(deltas), sum(deltas) / full_bytes, naive, naive / full_bytes))
    # ④ 幂等: 没有新行时不写
    sz0 = os.path.getsize(p_inc)
    assert flush_csv_append(p_inc, big, written) == written
    assert os.path.getsize(p_inc) == sz0, '无新行时文件被改写'
    # ⑤ 首段覆盖旧文件 (不留 stale 行)
    p_old = os.path.join(tmp, 'old.csv')
    pd.DataFrame({'step': list(range(999, 1005)), 'v': 1.0, 'txt': 'stale'}).to_csv(p_old)
    assert flush_csv_append(p_old, big.iloc[:7].reset_index(drop=True), 0) == 7
    got = pd.read_csv(p_old)
    assert len(got) == 7 and 'stale' not in set(got['txt']), '首段未覆盖旧文件'
    print('  ④ 幂等 (无新行不写) ✓   ⑤ 首段覆盖旧 run 的同名表 (无 stale 行) ✓')
    print('  ⑥ 中途落盘的文件始终可读 ⇒ 崩溃最多丢一个落盘间隔 (<50 行), 与旧语义一致')
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print('=' * 84)
print('TEST OK')
