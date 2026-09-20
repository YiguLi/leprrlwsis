#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""日志表**增量落盘**的回归测试 —— 不需要 MATLAB。

背景（实测）：主循环里"只增不减"的整表 `df` 若每 50 步做一次全量 `df.to_csv(path)`，
单次写盘量 = O(累积行数) ⇒ **迭代数越大越卡**，表现为"每 50 整数倍 step 卡一下"，
且卡顿随训练步数加剧（run9 实测该表到 67k 步已 63 MB；`td3prle` 的注释里记着同一条诊断）。
改为"首段覆盖写 + 其后只追加新行"后，每次写盘量 = O(新增行) = 常数，整轮总写盘量 O(N)。
同规模下全量覆写的整轮总写盘量是增量的约 20.5 倍，且该倍数随 N **线性增长**。

四节：
  A. **静态**（6 个 TD3 main）：主表 / `update_stats_H.csv` / episode / test9 的落盘点必须存在
     "追加"路径；裸的全量 `df*.to_csv(` 只允许出现在文件末尾的收尾安全写（或首段 `==0` 守卫）。
  A2. **单一实现**：全仓库（git 跟踪的 .py）只允许 `util.flush_csv_append` 一个增量落盘实现 ——
     禁止任何 main 再自带 `_flush_csv` 本地副本、手写 `mode='a', header=False`、或
     `iloc[...flushed_rows:].to_csv(...)` 内联变体（历史上曾有三套等价实现，易各自腐化）。
     并按各 main 的日志表形态核对入口点：**累积整表**（td3prrbb / td3prle）用
     `flush_csv_append`，**新鲜批次**（其余 4 个，rows 写完即清空）用 `flush_csv_append_batch`
     —— 两者混用会静默丢行（批次表走累积语义时 `len(rows) <= written` ⇒ 一行都不写）。
  B. **功能**（`util.flush_csv_append`）：增量写盘与一次性全量写盘**逐位一致**；
     每次落盘只追加增量（写盘量不随累积行数增长）；幂等；首段覆盖旧文件；
     中途落盘的文件始终可读（崩溃最多丢一个落盘间隔的行，与旧语义一致）。
  C. **入参形态 / index 开关 / 两种入口点**：`DataFrame` 入参、row-dict 列表入参、
     `index=True/False`、累积入口与批次入口，都必须与各自的一次性全量写**逐位一致**
     （这是把两派调用方收敛到同一实现的前提）。
"""
import os
import re
import shutil
import subprocess
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

from util import flush_csv_append, flush_csv_append_batch               # noqa: E402

TD3_MAINS = ['td3per/td3per_main.py', 'td3prrbb/td3prrbb_main.py',
             'td3prle/td3prle_main.py', 'td3prrlrb/td3prrlrb_main.py',
             'td3perrb/td3perrb_main.py', 'td3leprrlrb/td3leprrlrb_main.py']
INC = re.compile(r"flush_csv_append(?:_batch)?\(|mode='a'")   # 唯一的增量落盘写法


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
    # ③ 裸全量 df*.to_csv 只允许: 首段覆盖写(带 ==0 守卫) 或 (ii) 循环外的收尾安全写。
    #    循环内的写缩进 ≥12 空格; 收尾 `if use_csv:` 块里的写缩进 ≤8 ⇒ 用缩进判定。
    bare = [k for k, l in enumerate(L) if re.match(r'\s*df\w*\.to_csv\(', l)]
    for k in bare:
        guarded = any(re.search(r'==\s*0\b', x) for x in L[max(0, k - 3):k])
        indent = len(L[k]) - len(L[k].lstrip())
        assert guarded or indent <= 8, \
            (p, '第 %d 行有循环内裸全量 df.to_csv (缩进 %d) ⇒ O(N^2)' % (k + 1, indent))
    print('  %-32s 50 步落盘=增量 ✓  update_stats=%d 处(增量) ✓  收尾全量写=%d 处 ✓'
          % (p, n_upd, len(bare)))

print('=' * 84)
print('A2. 单一实现: 全仓库只有一个增量落盘实现 (util.flush_csv_append)')
print('=' * 84)
_tracked = [x for x in subprocess.run(
    ['git', '-c', 'core.quotepath=false', 'ls-files', '-z', '*.py'],
    capture_output=True, text=True, encoding='utf-8').stdout.split('\0') if x]
assert _tracked, 'git ls-files 无输出 (需在仓库内运行)'
# ① 不得存在任何本地副本
_locals = [q for q in _tracked if re.search(r'^def _flush_csv|^def _flush_inc|^def flush_csv_append\b',
                                            src(q), re.M) and q.replace('\\', '/') != 'util.py']
assert not _locals, ('仍有本地增量落盘副本 (应统一到 util.flush_csv_append)', _locals)
# ② 唯一实现必须在 util.py, 且只定义一次
_helper = [q.replace('\\', '/') for q in _tracked
           if re.search(r'^def flush_csv_append\b', src(q), re.M)]
assert _helper == ['util.py'], ('增量落盘实现应唯一且位于 util.py', _helper)
assert src('util.py').count('def flush_csv_append(') == 1, 'util.py 中定义了多个 flush_csv_append'
# ③ 6 个 main 必须都调用共享 helper, 且不再手写追加/内联切片
for p in TD3_MAINS:
    t = src(p)
    assert 'flush_csv_append(' in t or 'flush_csv_append_batch(' in t, (p, '未调用共享 helper')
    assert re.search(r'^from util import \*|^import util\b', t, re.M), (p, 'helper 不可见')
    assert not re.search(r"mode='a'", t), (p, '仍有手写 mode=\'a\' 追加')
    assert not re.search(r'\.iloc\[[^\]]*\]\s*\.to_csv\(', t), (p, '仍有内联 iloc 切片追加')
# ④ 入口点必须与各 main 的日志表形态匹配 —— 形态误用会**静默丢行**(不报错):
#    新鲜批次表(rows 写完即清空)若走累积语义, 当 len(rows) <= written 时一行都不写。
BATCH_MAINS = {'td3per/td3per_main.py', 'td3perrb/td3perrb_main.py',
               'td3prrlrb/td3prrlrb_main.py', 'td3leprrlrb/td3leprrlrb_main.py'}
for p in TD3_MAINS:
    t, L = src(p), src(p).split('\n')
    n_b = t.count('flush_csv_append_batch(')
    n_a = t.count('flush_csv_append(')          # `_batch(` 不含该子串(后随下划线)
    if p in BATCH_MAINS:
        assert n_b > 0 and n_a == 0, (p, '新鲜批次表必须只用 flush_csv_append_batch', n_b, n_a)
        for k, l in enumerate(L):
            if 'flush_csv_append_batch(' in l:
                if len(l) - len(l.lstrip()) <= 8:
                    continue                    # 收尾安全写 (无需清空)
                assert re.search(r'=\s*\[\]', '\n'.join(L[k:k + 5])), \
                    (p, '第 %d 行批次落盘后未清空 rows ⇒ 下一批会重复写' % (k + 1))
    else:
        assert n_a > 0 and n_b == 0, (p, '累积表必须只用 flush_csv_append', n_a, n_b)
    print('  %-32s 入口点=%s (batch %d / 累积 %d) ✓'
          % (p, 'batch' if p in BATCH_MAINS else '累积', n_b, n_a))
print('  ① 无本地副本 ✓  ② 唯一实现 = util.py (唯一 def) ✓  ③ 6 个 main 全走共享 helper ✓')
print('     git 跟踪 .py 共 %d 个, 扫描无第二套增量落盘实现' % len(_tracked))

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
    # ⑥ rows 列表入参 (另外 4 个 main 的历史形态) 与 DataFrame 入参等价
    p_rows = os.path.join(tmp, 'rows.csv')
    rows = big.to_dict('records')            # 同一份数据, 只是换成 row-dict 列表形态
    # check_dtype=False: Windows 上 np.arange 给 int32, 往返成 int64, 值/渲染文本一致
    pd.testing.assert_frame_equal(pd.DataFrame(rows), big, check_dtype=False)
    w2 = 0
    for k in range(1, N + 1):
        if k % STEP == 0:
            w2 = flush_csv_append(p_rows, rows[:k], w2)
    assert w2 == N
    assert open(p_rows, 'rb').read() == b_full, 'rows 列表入参与 DataFrame 入参结果不一致'
    print('  ④ 幂等 (无新行不写) ✓   ⑤ 首段覆盖旧 run 的同名表 (无 stale 行) ✓')
    print('  ⑥ rows 列表入参 ≡ DataFrame 入参 (%d 行) ✓' % N)
    print('  ⑦ 中途落盘的文件始终可读 ⇒ 崩溃最多丢一个落盘间隔 (<50 行), 与旧语义一致')
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print('=' * 84)
print('C. 功能: index 开关 —— 两种口径各自 ≡ 一次性全量写')
print('=' * 84)
tmp = tempfile.mkdtemp(prefix='csvinc2_')
try:
    N, STEP = 600, 25
    rows = [{'step': i, 'v': float(i) * 0.5} for i in range(N)]
    df_all = pd.DataFrame(rows)
    for idx in (True, False):
        pf = os.path.join(tmp, 'full_%s.csv' % idx)
        df_all.to_csv(pf, index=idx)
        # 两派调用方: rows 列表 (迁移前 4 个 main) / DataFrame 切片 (td3prrbb 与 prle)
        pa, pb = os.path.join(tmp, 'a_%s.csv' % idx), os.path.join(tmp, 'b_%s.csv' % idx)
        wa = wb = 0
        for k in range(1, N + 1):
            if k % STEP == 0:
                wa = flush_csv_append(pa, rows[:k], wa, index=idx)                  # 列表
                wb = flush_csv_append(pb, df_all.iloc[:k].reset_index(drop=True),
                                      wb, index=idx)                                # DataFrame
        assert wa == wb == N, (wa, wb, N)
        ba, bb, bf = (open(x, 'rb').read() for x in (pa, pb, pf))
        assert ba == bb == bf, ('index=%s 时两派入参与全量写不一致' % idx, len(ba), len(bb), len(bf))
        print('  index=%-5s  rows-列表 ≡ df-切片 ≡ 全量写: %d 字节逐位一致'
              % (idx, len(ba)))
    # 非 0 起始计数器 (中断续写) 也要一致
    p_mid = os.path.join(tmp, 'mid.csv')
    w = flush_csv_append(p_mid, df_all.iloc[:40].reset_index(drop=True), 0)
    w = flush_csv_append(p_mid, df_all.iloc[:100].reset_index(drop=True), w)
    assert w == 100
    assert open(p_mid, 'rb').read() == df_all.iloc[:100].reset_index(drop=True).to_csv(index=True).encode()
    print('  非 0 起始计数器续写 (0→40→100) ≡ 全量写 ✓')
    # ⑤ 批次形态 (4 个 main 的真实调用式): 每批只给新增行, 计数器续接全局行号。
    #    这里刻意用"每批 1 行 + written 已很大" ⇒ 若误用累积语义会**静默一行不写**。
    p_batch = os.path.join(tmp, 'batch.csv')
    nb = 0
    for i in range(N):
        nb = flush_csv_append_batch(p_batch, [rows[i]], nb)
        assert nb == i + 1, ('批次形态丢行 (累积语义误用)', i, nb)
    ref = df_all.to_csv(index=True).encode()
    assert open(p_batch, 'rb').read() == ref, '批次形态 ≠ 全量写'
    # 批次形态与累积形态对同一逻辑内容必须逐位一致 (index=False 亦然)
    p_batch2 = os.path.join(tmp, 'batch2.csv')
    nb2 = 0
    for k in range(1, N + 1):
        if k % STEP == 0:
            nb2 = flush_csv_append_batch(p_batch2, rows[nb2:k], nb2)
    assert open(p_batch2, 'rb').read() == ref, '批次形态 (按 STEP 分批) ≠ 全量写'
    print('  批次形态: 每批 1 行 × %d 批 ≡ 全量写 ✓ (累积语义误用会在此静默丢行)' % N)
    print('  批次形态 (每 %d 行一批) ≡ 全量写 ✓ —— 两种入口点对同一内容逐位一致' % STEP)
    # ⑥ 反面样本: 批次行误用累积入口 ⇒ 静默不写 (这正是两个入口点必须分开的原因)
    p_neg = os.path.join(tmp, 'neg.csv')
    wn = flush_csv_append(p_neg, [rows[0]], 500)      # 累积语义: len(rows)=1 <= 500
    assert wn == 500, (wn, '累积语义竟然递增了计数器')
    assert not os.path.exists(p_neg), '累积语义竟然写出了批次行 (预期静默丢行)'
    print('  反面样本: 批次行误用累积入口 ⇒ 返回 written=%d 且文件未创建 (静默丢行) ✓' % wn)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print('=' * 84)
print('TEST OK')
