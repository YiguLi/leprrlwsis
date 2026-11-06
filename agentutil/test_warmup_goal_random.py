# -*- coding: utf-8 -*-
"""**warmup 期 goal（第 6 维）纯随机**的契约测试（纯静态/纯算术，不跑仿真、不需 MATLAB）。

口径（2026-10-30 起）
--------------------
`--warmup_sig` 的默认值由 `hsrl_warmup.SIG_MODE_DEFAULT = 'random'` 决定 ⇒ **所有入口**
（td3per / td3perrb / td3perseg / td3prle / td3prrbb / td3prrlrb / td3leprrlrb /
singleaif / aifflor / aifflorpave / prle / prrbb / yawflor / testpg）的 warmup goal
都是**纯随机**：

    a6[b] ~ i.i.d. U(sig_lo, sig_hi)      默认 [-0.5, +0.5]，块内 5 步固定

不看电价、不看上一块（区别于旧口径 `price_walk` 的累积游走 + 端点裁剪）。

判据
----
  [A] 常量：`SIG_MODES` 三档、`SIG_MODE_DEFAULT='random'`。
  [B] CLI：`add_warmup_args()` 的 `--warmup_sig` 默认 = `SIG_MODE_DEFAULT`；
      **所有**调用它的 main 都没有覆盖 `sig_default`（⇒ 一起切换）。
  [C] 纯随机语义（1440 块）：都在带内、铺满两端、10 层 ≈144、均值≈0、
      std≈U 的理论值 0.2887、lag-1 自相关≈0、**第 0 块也随机**（无锚点）。
  [D] **与电价无关**：同 seed、全 300 / 全 980 / 交替 三种价格序列 ⇒ 逐位相同的序列。
  [E] 可复现 + RNG 隔离：同 seed 逐位相同；不消耗全局 numpy RNG。
  [F] 旧口径仍可用：`price_walk`（300 上调 / 580 不动 / 980 下调、首块锚点 0、步幅 ≤0.10）、
      `fixed`（恒 0）。
  [G] 采样带可调：`--warmup_sig_lo/hi` 经 `build_from_args` 生效（收窄 ⇒ 目标更可达）。
  [H] testpg 的**独立实现** `PGWarmup` 同口径（默认 random、带内均匀、`describe()` 说明）。

跑法::

    python agentutil/test_warmup_goal_random.py
"""
import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import hsrl_warmup as hw                                # noqa: E402

N_BLK = int(hw.WARMUP_STEPS) // 5                       # 7200 步 = 1440 块
N_OK = 0


def ok(cond, msg):
    global N_OK
    if not cond:
        raise AssertionError('FAIL: ' + msg)
    N_OK += 1


def sched(sig_mode=None, seed=7, **kw):
    """按需构造；`sig_mode=None` ⇒ 用**生产默认**（SIG_MODE_DEFAULT）。"""
    mode = hw.SIG_MODE_DEFAULT if sig_mode is None else sig_mode
    return hw.WarmupSchedule(5, 6, seed=seed, wt_sched='lhs', sig_mode=mode, **kw)


def seq(s, prices, n=N_BLK):
    if prices is None:
        return np.array([s.action(None)[5] for _ in range(n)])
    if np.isscalar(prices):
        return np.array([s.action(float(prices))[5] for _ in range(n)])
    return np.array([s.action(float(prices[i % len(prices)]))[5] for i in range(n)])


# =============================================================================
# [A] 常量
# =============================================================================
def sec_a_constants():
    print('\n[A] 常量（单一真源）')
    ok(hw.SIG_MODE_DEFAULT == 'random', "SIG_MODE_DEFAULT = 'random'")
    ok(set(hw.SIG_MODES) == {'random', 'price_walk', 'fixed'}, 'SIG_MODES 三档')
    ok(hw.SIG_MODES[0] == 'random', 'choices 第一项 = 默认（--help 排序友好）')
    ok(float(hw.SIG_LO) == -0.5 and float(hw.SIG_HI) == 0.5, '默认采样带 [−0.5, 0.5]')
    ok(int(hw.WARMUP_STEPS) == 7200 and N_BLK == 1440, 'warmup7200 ⇒ 1440 个块')


# =============================================================================
# [B] CLI 默认 + 所有入口都不覆盖
# =============================================================================
def sec_b_cli():
    print('\n[B] CLI 默认 + 入口扫描')
    p = argparse.ArgumentParser(add_help=False)
    hw.add_warmup_args(p)
    d = p.parse_args([])
    ok(d.warmup_sig == hw.SIG_MODE_DEFAULT, 'add_warmup_args 的 --warmup_sig 默认 = random')
    ok(d.warmup_sig_lo == hw.SIG_LO and d.warmup_sig_hi == hw.SIG_HI,
       '--warmup_sig_lo/hi 默认 = SIG_LO/SIG_HI')
    # 显式传旧口径仍可用
    ok(p.parse_args(['--warmup_sig', 'price_walk']).warmup_sig == 'price_walk',
       '显式 price_walk 可覆盖默认')
    ok(p.parse_args(['--warmup_sig', 'fixed']).warmup_sig == 'fixed', '显式 fixed 可覆盖默认')
    # --help 必须能渲染（argparse 会对 help 再做一次 % 展开 ⇒ 字面 % 必须转义）
    htxt = p.format_help()
    ok('--warmup_sig {random,price_walk,fixed}' in htxt, '--help 列出三档 mode')
    ok('纯随机' in htxt and '91%' in htxt,
       '--help 正常渲染（含"纯随机"与转义后的 91%）')
    # 扫描所有 main：调用 add_warmup_args 且**没有**覆盖 sig_default
    mains = []
    for path in glob.glob(os.path.join(ROOT, '**', '*.py'), recursive=True):
        rel = os.path.relpath(path, ROOT).replace('\\', '/')
        if rel.startswith(('agentutil/', '.', 'libraries/', 'output/', 'runs/',
                           '未命名/', 'documentation/')):
            continue
        if rel == 'hsrl_warmup.py':      # 真源自己定义 sig_default，不算"入口"
            continue
        try:
            txt = open(path, encoding='utf-8', errors='replace').read()
        except OSError:
            continue
        if 'add_warmup_args(' in txt:
            mains.append((rel, txt))
    ok(len(mains) >= 12, '调用 add_warmup_args 的入口 ≥12（实测 %d）' % len(mains))
    bad = [rel for rel, txt in mains if 'sig_default' in txt]
    ok(not bad, '没有入口覆盖 sig_default ⇒ 全部随 SIG_MODE_DEFAULT 切换（%r）' % bad)
    # 允许覆盖 yaw / wt_sched_default（aifflor 的 unified 是既有口径），但不影响 goal
    ok(sum(1 for _rel, txt in mains if "wt_sched_default='unified'" in txt) == 1,
       '只有 aifflor 覆盖 wt_sched_default（与 test_warmup_unified 一致）')
    ok(sum(1 for _rel, txt in mains if 'add_warmup_args(parser)' in txt) >= 10,
       '≥10 个入口用无参形式（其余传 wt_sched_default，同样继承 sig 默认）')


# =============================================================================
# [C] 纯随机语义
# =============================================================================
def sec_c_random():
    print('\n[C] 纯随机语义（1440 块）')
    a = seq(sched(), 300.0)        # 价格给什么都行（[D] 单独证无关）
    ok(a.shape == (N_BLK,), '1440 个 goal 样本')
    ok(a.min() >= hw.SIG_LO - 1e-12 and a.max() <= hw.SIG_HI + 1e-12,
       '全部落在 [%.1f, %.1f]（min %+.4f, max %+.4f）' % (hw.SIG_LO, hw.SIG_HI,
                                                         a.min(), a.max()))
    ok(a.min() < -0.45 and a.max() > 0.45, '铺满带的两端')
    bins = np.histogram(a, bins=np.linspace(hw.SIG_LO, hw.SIG_HI, 11))[0]
    ok(bins.min() >= 120 and bins.max() <= 175,
       '均匀：10 层每层 ≈144（实测 [%d, %d]）' % (bins.min(), bins.max()))
    ok(abs(a.mean()) < 0.05, '均值 ≈0（实测 %+.5f）' % a.mean())
    ok(0.26 < a.std() < 0.31,
       'std ≈ U(−0.5,0.5) 的理论值 0.2887（实测 %.4f）' % a.std())
    r1 = float(np.corrcoef(a[:-1], a[1:])[0, 1])
    ok(abs(r1) < 0.10, 'lag-1 自相关 ≈0 ⇒ 每块独立（实测 %+.4f）' % r1)
    # 第 0 块也随机（旧价 walk 的首块是"锚点 0"）
    ok(a[0] != 0.0, '第 0 块也是随机抽的（实测 %+.4f，不是锚点 0）' % a[0])
    a0 = [seq(sched(seed=s), None, n=1)[0] for s in (1, 2, 3, 4, 5)]
    ok(any(abs(v) > 1e-9 for v in a0) and len(set(np.round(a0, 12))) > 1,
       '不同 seed 首块不同 ⇒ 无固定锚点')


# =============================================================================
# [D] 与电价无关
# =============================================================================
def sec_d_price_free():
    print('\n[D] 纯随机与电价无关（逐位相同）')
    a_low = seq(sched(seed=11), 300.0, 300)
    a_high = seq(sched(seed=11), 980.0, 300)
    a_mid = seq(sched(seed=11), 580.0, 300)
    a_alt = seq(sched(seed=11), [300.0, 580.0, 980.0], 300)
    a_none = seq(sched(seed=11), None, 300)
    for tag, b in (('全 980', a_high), ('全 580', a_mid), ('交替', a_alt), ('None', a_none)):
        ok(np.array_equal(a_low, b), '全 300 与 %s 的价格序列 ⇒ 逐位相同' % tag)


# =============================================================================
# [E] 可复现 + RNG 隔离
# =============================================================================
def sec_e_repro():
    print('\n[E] 可复现性与 RNG 隔离')
    ok(np.array_equal(seq(sched(seed=7), 300.0), seq(sched(seed=7), 300.0)),
       '同 seed 逐位可复现')
    st0 = np.random.get_state()
    _ = seq(sched(seed=13), 300.0, 50)
    st1 = np.random.get_state()
    ok(st0[0] == st1[0] and np.array_equal(st0[1], st1[1]) and st0[2:] == st1[2:],
       'WarmupSchedule 不消耗全局 numpy RNG')
    # 训练流的独立随机流也不受影响（用 RandomState 副本对比）
    np.random.seed(2024)
    _ = np.random.random(3)
    ref = np.random.random(5)
    np.random.seed(2024)
    _ = np.random.random(3)
    _ = seq(sched(seed=99), 300.0, 20)
    got = np.random.random(5)
    ok(np.array_equal(ref, got), '抽样不扰动调用方的全局随机序列')


# =============================================================================
# [F] 旧口径仍可用
# =============================================================================
def sec_f_legacy():
    print('\n[F] 旧口径 price_walk / fixed 仍可复现')
    s = sched('price_walk', seed=101)
    ok(abs(s.action(300.0)[5]) < 1e-12, 'price_walk 首块仍是锚点 0')
    ups = seq(s, 300.0, 20)
    ok(np.all(np.diff(ups) >= -1e-12) and ups[-1] > 0, '300 ⇒ 单调上调（末值 %+.3f）' % ups[-1])
    ok(np.all(np.diff(np.r_[0.0, ups]) <= 0.10 + 1e-12), '单步 ≤0.10')
    s2 = sched('price_walk', seed=101)
    _ = seq(s2, 300.0, 20)
    mid = seq(s2, 580.0, 1)[0]
    _ = seq(s2, 580.0, 1)
    ok(abs(seq(s2, 580.0, 1)[0] - mid) < 1e-12, '580 ⇒ 不动')
    dn = seq(s2, 980.0, 6)
    ok(np.all(np.diff(np.r_[mid, dn]) <= 1e-12) and dn[-1] < mid, '980 ⇒ 下调（末值 %+.3f）' % dn[-1])
    ok(np.all(seq(sched('fixed', seed=1), 300.0, 40) == 0.0), 'fixed ⇒ 恒 0')
    # 步幅默认值仍是 0.06/0.10（只影响 price_walk）
    ok(abs(hw.SIG_STEP_LO - 0.06) < 1e-12 and abs(hw.SIG_STEP_HI - 0.10) < 1e-12,
       'SIG_STEP_LO/HI = 0.06/0.10（只对 price_walk 生效）')


# =============================================================================
# [G] 采样带可调
# =============================================================================
def sec_g_band():
    print('\n[G] --warmup_sig_lo/hi 生效')
    p = argparse.ArgumentParser(add_help=False)
    hw.add_warmup_args(p)
    d = p.parse_args(['--warmup_sig_lo', '-0.1', '--warmup_sig_hi', '0.1'])

    class _HStub(object):
        nb_turbine, nb_actions = 5, 6

    s, _rule, _stats = hw.build_from_args(
        argparse.Namespace(**vars(d), seed=7, rbb_rated_power=0.5), _HStub(), None)
    ok(s.sig_mode == 'random' and s.sig_lo == -0.1 and s.sig_hi == 0.1,
       'build_from_args 把 sig_mode/sig_lo/sig_hi 传进 schedule')
    a = seq(s, 300.0, 500)
    ok(a.min() >= -0.1 - 1e-12 and a.max() <= 0.1 + 1e-12,
       '收窄后全部落在 [−0.1, 0.1]（min %+.4f, max %+.4f）' % (a.min(), a.max()))
    # 目标量纲：span 0.4167 ⇒ ±0.1 信号 ≈ ±0.0417 MWh = 电池单块可移动量
    ok(abs(hw.SOC_TARGET - 0.25) < 1e-12, 'warmup 规则下层的默认目标是 0.25（旧注释 3.0 已过期）')


# =============================================================================
# [H] testpg 的独立实现同口径
# =============================================================================
def sec_h_testpg():
    print('\n[H] testpg 的 PGWarmup 同口径')
    from testpg.testpg_warmup import PGWarmup
    import inspect
    sig_default = inspect.signature(PGWarmup.__init__).parameters['sig_mode'].default
    ok(sig_default == hw.SIG_MODE_DEFAULT, 'PGWarmup 的 sig_mode 默认 = SIG_MODE_DEFAULT')
    w = PGWarmup(enabled=True, seed=7)
    a = np.array([w.action(300.0)[1] for _ in range(1440)])
    ok(a.min() >= hw.SIG_LO - 1e-12 and a.max() <= hw.SIG_HI + 1e-12, 'PGWarmup 的 goal 在带内')
    ok(a.min() < -0.45 and a.max() > 0.45, 'PGWarmup 的 goal 铺满两端')
    ok(abs(a.mean()) < 0.05 and abs(float(np.corrcoef(a[:-1], a[1:])[0, 1])) < 0.10,
       'PGWarmup 的 goal 均值≈0、无自相关')
    ok('纯随机' in w.describe(), 'describe() 说明是纯随机')
    # 与电价无关
    w2 = PGWarmup(enabled=True, seed=7)
    b = np.array([w2.action(980.0)[1] for _ in range(100)])
    ok(np.array_equal(a[:100], b), 'PGWarmup 的 goal 也与电价无关')
    # 静态：testpg_main 的 --warmup_sig 默认取 SIG_MODE_DEFAULT
    src = open(os.path.join(ROOT, 'testpg', 'testpg_main.py'), encoding='utf-8').read()
    ok("parser.add_argument('--warmup_sig', default=hwarm.SIG_MODE_DEFAULT" in src,
       'testpg_main 的 --warmup_sig 默认 = hwarm.SIG_MODE_DEFAULT')
    ok('--warmup_sig_lo' in src and '--warmup_sig_hi' in src,
       'testpg_main 也暴露了采样带开关')


def main():
    print('=' * 78)
    print('warmup 期 goal = 纯随机（默认）契约测试')
    print('=' * 78)
    sec_a_constants()
    sec_b_cli()
    sec_c_random()
    sec_d_price_free()
    sec_e_repro()
    sec_f_legacy()
    sec_g_band()
    sec_h_testpg()
    print('\n' + '=' * 78)
    print('全部通过：%d 条断言' % N_OK)
    print('=' * 78)


if __name__ == '__main__':
    main()
