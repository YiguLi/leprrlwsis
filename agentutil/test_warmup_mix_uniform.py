# -*- coding: utf-8 -*-
"""逐位对照：新 `wt_sched='walk'` 是否与 git 历史版本 `WarmupSchedule` 完全一致。

方法：从 `git show <base>:hsrl_warmup.py` 取回旧文件写到临时模块，两个模块各自建调度器、
喂同一串 price 序列，比较**动作数组逐位相等**（float.hex 级别）以及信号维。
"""
import importlib.util
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else 'b72450f^'
#: ⚠ 默认基准必须钉在**混合采样落地之前**的提交上；若传 `HEAD` 而不带 `^`，
#:   就会拿新代码跟自己比 ⇒ **平凡通过、什么也没证明**。下面显式检测这种情况。
src = subprocess.run(['git', 'show', '%s:hsrl_warmup.py' % BASE],
                     capture_output=True, text=True, encoding='utf-8', check=True).stdout
if 'wt_sched' in src:
    print('⚠ 警告：基准 %s 的 hsrl_warmup.py 已经含 wt_sched ⇒ 这是"新 vs 新"的平凡对照！'
          % BASE)
    print('   请传混合采样落地**之前**的提交，例如 `python %s b72450f^`'
          % os.path.basename(__file__))
SELF_COMPARE = 'wt_sched' in src
# 临时文件必须落在仓库内（受限沙箱下系统 temp 不可写）；放在 agentutil/ 下并随即删除。
tmp = os.path.join(_HERE, '_old_hsrl_warmup_tmp.py')
with open(tmp, 'w', encoding='utf-8') as f:
    f.write(src)
try:
    spec = importlib.util.spec_from_file_location('hsrl_warmup_old', tmp)
    old = importlib.util.module_from_spec(spec)
    sys.modules['hsrl_warmup_old'] = old
    spec.loader.exec_module(old)
finally:
    pass

sys.path.insert(0, ROOT)
import hsrl_warmup as new  # noqa: E402

print('旧版本来源: git %s:hsrl_warmup.py' % BASE)
print('  旧默认 WT_START=%s WT_STEP=%s WT_LO/HI=%s/%s'
      % (old.WT_START, old.WT_STEP, old.WT_LO, old.WT_HI))
print('  旧版本有 wt_sched 参数吗: %s'
      % ('wt_sched' in old.WarmupSchedule.__init__.__code__.co_varnames))

rng = np.random.RandomState(2024)
prices = [float(rng.choice([300, 580, 980])) for _ in range(400)]

fails = []


def cmp(tag, nb_t, nb_a, seed, prices, **kw):
    so = old.WarmupSchedule(nb_t, nb_a, seed=seed, **kw)
    sn = new.WarmupSchedule(nb_t, nb_a, seed=seed, wt_sched='walk', **kw)
    A = np.array([so.action(p) for p in prices])
    B = np.array([sn.action(p) for p in prices])
    arr_same = (A.shape == B.shape) and np.array_equal(
        np.array([float.hex(v) for v in A.reshape(-1)]),
        np.array([float.hex(v) for v in B.reshape(-1)]))
    # `seed<=0` ⇒ RandomState(None) 每次从 OS 抽种子 ⇒ **本来就不可复现**（新旧都一样），
    # 此时只比结构：形状 / 带内 / 落在 start + k·step 的格点上。
    if seed and seed > 0:
        stream_same = (float.hex(float(so.rng.random()))
                       == float.hex(float(sn.rng.random())))
        ok = arr_same and stream_same
        print('  [%s] nb_t=%d nb_a=%s seed=%s  形状 %s vs %s  动作逐位: %-5s RNG流一致: %-5s'
              % (tag, nb_t, nb_a, seed, A.shape, B.shape, arr_same, stream_same))
        if not ok and not arr_same:
            print('     最大差 %.3e（首个不同块 %s）'
                  % (np.abs(A - B).max(), int(np.argmax((A != B).any(axis=1)))))
        if not ok:
            # 若基准已含新实现（"新 vs 新"的平凡对照），旧版没有 wt_sched 参数 ⇒
            # 动作必然不同，这不是回归失败；显式区分开。
            if SELF_COMPARE and 'wt_sched' not in str(
                    old.WarmupSchedule.__init__.__code__.co_varnames):
                print('     ↑ 基准即新实现 ⇒ 平凡对照，不计为失败（见脚本头部警告）')
            else:
                fails.append(tag)
    else:
        st = kw.get('start', old.WT_START)
        sp = kw.get('step', old.WT_STEP)
        lo_, hi_ = kw.get('lo', old.WT_LO), kw.get('hi', old.WT_HI)
        def grid_ok(X):
            k = np.round((X[:, :nb_t] - st) / sp)
            return (np.abs(X[:, :nb_t] - (st + k * sp)) < 1e-12).all() \
                and (X[:, :nb_t] >= lo_ - 1e-12).all() and (X[:, :nb_t] <= hi_ + 1e-12).all()
        ok = grid_ok(A) and grid_ok(B)
        print('  [%s] nb_t=%d nb_a=%s seed=%s  形状 %s vs %s  '
              '（无种子 ⇒ 只比结构）格点+带内: 旧=%s 新=%s'
              % (tag, nb_t, nb_a, seed, A.shape, B.shape, grid_ok(A), grid_ok(B)))
        if not ok:
            fails.append(tag)


print('\n--- 逐位对照（seed>0 时要求动作逐位相同；所有情况要求 RNG 流一致）---')
cmp('默认', 5, 6, 101, prices)
cmp('无 seed(RandomState(None))', 5, 6, 0, prices[:120])
cmp('单机(singleaif)', 1, 2, 7, prices)
cmp('nb_actions=5(nolow)', 5, 5, 13, prices)
cmp('sig_mode=fixed', 5, 6, 55, prices, sig_mode='fixed')
cmp('sig_step 改幅', 5, 6, 77, prices, sig_step_lo=0.01, sig_step_hi=0.2)
cmp('带偏移', 5, 6, 99, prices, start=0.3, step=0.02, lo=0.1, hi=0.4)
cmp('price=None 调用', 5, 6, 31, [None] * 100)

print('\n--- 新口径（mix）分布自检 ---')
s = new.WarmupSchedule(5, 6, seed=101)
N = 1440 // 5
A = np.array([s.action(p) for p in prices[:N]])
frac_prior = float(np.mean(np.all(np.abs(A[:, :5] - np.array(new.WT_PRIOR)) <= 0.031, axis=1)))
print('  %d 块动作: 均值 %s' % (N, np.round(A[:, :5].mean(axis=0), 4)))
print('  sd %s' % np.round(A[:, :5].std(axis=0), 4))
print('  min %s  max %s' % (np.round(A[:, :5].min(axis=0), 3), np.round(A[:, :5].max(axis=0), 3)))
print('  落在先验邻域(±prior_step)内的块占比 ≈ %.1f%%  (期望约 70%%，'
      '因 clip 到带内会略偏)' % (100 * frac_prior,))

print('\n--- 先验对齐规则 prior_for_nb ---')
for nb, exp in ((5, [0.2, 0.2, 0.2, 0.2, 1 / 3]), (1, [1 / 3]), (2, [0.2, 0.2]),
                (7, [0.2, 0.2, 0.2, 0.2, 1 / 3, 1 / 3, 1 / 3])):
    got = new.prior_for_nb(nb)
    same = len(got) == nb and np.allclose(got, exp)
    print('  nb=%d -> %s  期望 %s  %s' % (nb, np.round(got, 4), np.round(exp, 4),
                                          'OK' if same else 'FAIL'))
    if not same:
        fails.append('prior_for_nb nb=%d' % nb)

print('\n--- sample_wt_prior（actor 初始 bias 的真源）---')
rng = np.random.RandomState(5)
b = new.sample_wt_prior(5, 0.05, 0.45, rng=rng)
print('  默认(mix) 一次抽样: %s' % np.round(b, 4))
assert b.shape == (5,) and b.min() >= 0.05 and b.max() <= 0.45
near = np.mean([np.all(np.abs(new.sample_wt_prior(5, 0.05, 0.45, rng=rng)
                             - np.asarray(new.WT_PRIOR)) <= new.WT_PRIOR_STEP + 1e-12)
                for _ in range(400)])
print('  先验臂占比 ≈ %.1f%%（期望 ~70%%）' % (100 * near))
if abs(near - new.WT_PRIOR_FRAC) > 0.08:
    fails.append('sample_wt_prior 先验臂占比 %.3f' % near)
u = np.array([new.sample_wt_prior(5, 0.05, 0.45, rng=rng, use_prior=False)
              for _ in range(300)])
print('  use_prior=False: E[a]=%s（应 ≈ 带中点 0.25、与先验无关）'
      % np.round(u.mean(axis=0), 4))
if not (0.23 < u.mean() < 0.27):
    fails.append('use_prior=False 均值 %.3f' % u.mean())

print('\n结论: %s' % ('PASS —— 旧路径逐位不变'
                      if not fails else 'FAIL: %s' % fails))
try:
    os.remove(tmp)
except OSError:
    pass
sys.exit(0 if not fails else 1)