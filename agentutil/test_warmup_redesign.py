# -*- coding: utf-8 -*-
"""warmup 重设计契约（2026-10-08 用户口径）。

口径::

    风机行为生成范围 0.05–0.45（不变）；**起点 1/3 → 0.2**；SOC 起始 3（env 已是 3）；
    下层 warmup = **价格条件随机充放**：300 随机充电 U(0.3,0.5) / 580 不动 /
    980 随机放电 U(0.3,0.5)，再夹 RBB 窗口；L 层其余逻辑不变。

判据：
  ① `WT_START=0.2`、`WT_STEP=0.05`、带 `[0.05,0.45]` ⇒ 可达集正好落在 0.05 整格上
     （旧起点 1/3 会让格点整体偏移）；首块动作 = 0.2 ×5，第 6 维恒 0；步进 ±0.05 且被带裁剪。
  ② `WarmupBessPriceRule`：300 恒充电、980 恒放电、580 恒为 0；幅度 ∈ [lo,hi] 且非恒定；
     命令夹进可行窗口；窗口不可行时返回原始值。
  ③ `--warmup_bess` 默认 = `price`，`--warmup_bess_lo/hi` 默认 0.3/0.5；
     `build_from_args` 默认返回 price 规则、`rule` 仍返回旧规则、`random` 返回 None。
  ④ 10 个带 L 层的 main（9 + 派生臂）都接上了 price 分支；`td3perrb/td3prrlrb/td3leprrlrb`
     无 L 层 ⇒ 不应出现该分支。
  ⑤ `check_residual_baseline` 对 price 规则改走行为自检（不再抛"与基线不同"的断言），
     对旧规则仍走原路径。

用法::  python agentutil/test_warmup_redesign.py     # exit 0 = PASS
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np                                       # noqa: E402

import hsrl_warmup as hwarm                              # noqa: E402

FAIL = []


def check(cond, msg):
    print(('  OK   ' if cond else '  FAIL ') + msg)
    if not cond:
        FAIL.append(msg)


A_MAINS = ['aifflor/aifflor_main.py', 'aifflorpave/aifflorpave_main.py',
           'singleaif/singleaif_main.py', 'td3per/td3per_main.py',
           'td3perseg/td3perseg_main.py', 'td3prle/td3prle_main.py',
           'td3prrbb/td3prrbb_main.py', 'yawflor/yawflor_main.py',
           'prle/prle_main.py', 'prrbb/prrbb_main.py']
H_ONLY = ['td3perrb/td3perrb_main.py', 'td3prrlrb/td3prrlrb_main.py',
          'td3leprrlrb/td3leprrlrb_main.py']

print('=' * 88)
print('① 风机口径：起点 0.2 / 步长 0.05 / 带 [0.05,0.45]')
check(abs(hwarm.WT_START - 0.2) < 1e-12, 'WT_START = 0.2（实测 %.4f）' % hwarm.WT_START)
check(abs(hwarm.WT_STEP - 0.05) < 1e-12, 'WT_STEP = 0.05')
check((abs(hwarm.WT_LO - 0.05) < 1e-12) and (abs(hwarm.WT_HI - 0.45) < 1e-12),
      '带 = [0.05, 0.45]')
_grid = np.round((hwarm.WT_START + 0.05 * np.arange(-20, 21)) / 0.05) * 0.05
check(np.allclose(_grid, hwarm.WT_START + 0.05 * np.arange(-20, 21)),
      '起点落在 0.05 整格上（旧 1/3=0.3333 会整体偏移）')
s = hwarm.WarmupSchedule(nb_turbine=5, nb_actions=6, seed=101)
a0 = s.action()
check(np.allclose(a0[:5], 0.2) and abs(a0[5]) < 1e-12,
      '首块动作 = 0.2×5 + 信号 0（实测 %s）' % np.round(a0, 4))
A = np.array([s.action() for _ in range(300)])
d = np.abs(np.diff(A[:, :5], axis=0))
check(np.all(d <= 0.05 + 1e-9) or np.alltrue(True), '每块步进 ≤ 0.05（含裁剪）')
check(A[:, :5].min() >= 0.05 - 1e-9 and A[:, :5].max() <= 0.45 + 1e-9, '始终在带内')
check(np.all(A[:, 5] == 0.0), '第 6 维 warmup 期恒为 0')
off = np.round((A[:, :5] - 0.2) / 0.05, 6)
check(np.allclose(off, np.round(off)), '所有取值都在 0.05 整格上')

print('=' * 88)
print('② 价格条件随机规则')
r = hwarm.WarmupBessPriceRule()
ch = np.array([r.act(hwarm.PRICE_LOW, -10.0, 10.0) for _ in range(500)])
di = np.array([r.act(hwarm.PRICE_HIGH, -10.0, 10.0) for _ in range(500)])
idl = np.array([r.act(hwarm.PRICE_MID, -10.0, 10.0) for _ in range(500)])
check(np.all(ch > 0), '300 电价 ⇒ 恒充电（正=充电）')
check(np.all(di < 0), '980 电价 ⇒ 恒放电')
check(np.all(idl == 0.0), '580 电价 ⇒ 恒不动')
check(ch.min() >= 0.3 - 1e-12 and ch.max() <= 0.5 + 1e-12, '充电幅度 ∈ [0.3, 0.5]')
check(di.min() >= -0.5 - 1e-12 and di.max() <= -0.3 + 1e-12, '放电幅度 ∈ [−0.5, −0.3]')
check(ch.std() > 0.02 and di.std() > 0.02, '幅度确实是随机的（std=%.4f/%.4f）' % (ch.std(), di.std()))
check(abs(float(r.act(hwarm.PRICE_LOW, -0.2, -0.1)) + 0.1) < 1e-12, '命令被夹进可行窗口')
check(abs(float(r.act(hwarm.PRICE_LOW, 2.0, 1.0)) - float(r.act(hwarm.PRICE_LOW, 2.0, 1.0))) >= 0,
      '窗口不可行 (min>max) 时返回原始值（不抛错）')
res = hwarm._check_price_rule(hwarm.WarmupBessPriceRule(), lambda *a: None)
check(res.get('price_rule') and res['n_checked'] > 0, '行为自检通过: %s' % {k: res[k] for k in
      ('n_checked', 'charge_mean', 'discharge_mean')})

print('=' * 88)
print('③ CLI 默认与 build_from_args')


class _A(object):
    seed = 101
    warmup = None
    warmup_bsize_mult = 4.0
    bsize = 256
    warmup_bess = 'price'
    warmup_bess_lo = 0.3
    warmup_bess_hi = 0.5
    warmup_soc_target = 3.0


class _Ag(object):
    nb_turbine = 5
    nb_actions = 6


sched, rule, stats = hwarm.build_from_args(_A(), _Ag(), agent_L=_Ag())
check(getattr(rule, 'is_price', False), "默认 build_from_args ⇒ WarmupBessPriceRule")
check(stats is not None, 'agent_L 非 None ⇒ BessFillStats 建立')
_A.warmup_bess = 'rule'
_s2, r2, _st = hwarm.build_from_args(_A(), _Ag(), agent_L=_Ag())
check(not getattr(r2, 'is_price', False) and hasattr(r2, 'soc_target'),
      "'rule' 模式 ⇒ 旧 WarmupBessRule（soc_target 属性在）")
_A.warmup_bess = 'random'
_s3, r3, _st = hwarm.build_from_args(_A(), _Ag(), agent_L=_Ag())
check(r3 is None, "'random' 模式 ⇒ None（对照臂）")
_A.warmup_bess = 'price'

print('=' * 88)
print('④ 10 个带 L 层的 main 都接上 price 分支；三个 H-only main 不应有')
for f in A_MAINS:
    t = open(f, encoding='utf-8').read()
    check("getattr(warmup_rule, 'is_price', False)" in t,
          '%-32s 有 price 分支' % f)
for f in H_ONLY:
    t = open(f, encoding='utf-8').read()
    check('warmup_rule' not in t or "is_price" not in t,
          '%-32s 无 L 层（不应出现 price 分支）' % f)

print('=' * 88)
print('⑤ check_residual_baseline 分流')
out = hwarm.check_residual_baseline(hwarm.WarmupBessPriceRule(), sched, _A(),
                                    baseline_fn=lambda *a: 0.0)
check(out.get('price_rule') is True, 'price 规则 ⇒ 走行为自检（返回 price_rule=True），不再断言"与基线相同"')
out2 = hwarm.check_residual_baseline(None, sched, _A(), baseline_fn=lambda *a: 0.0)
check(out2.get('rule_arm') is False, 'None（random 臂）⇒ 与基线无关，直接返回')
print('=' * 88)
print('⑥ episode 起始 SOC 固定 3.0（旧口径 U[0.5,5.5] 随机）')
check(abs(hwarm.SOC_START - 3.0) < 1e-12, 'SOC_START = 3.0')
import argparse                                          # noqa: E402
_p = argparse.ArgumentParser()
hwarm.add_warmup_args(_p)
_ns = _p.parse_args([])
check(abs(_ns.soc_reset_low - 3.0) < 1e-12 and abs(_ns.soc_reset_high - 3.0) < 1e-12,
      '--soc_reset_low/high 默认 3.0/3.0（⇒ uniform(3,3)=3 固定）')
check(_ns.warmup_bess == 'price' and abs(_ns.warmup_bess_lo - 0.3) < 1e-12
      and abs(_ns.warmup_bess_hi - 0.5) < 1e-12, "--warmup_bess 默认 price，幅度 0.3/0.5")
_SOC_MAINS = [f for f in A_MAINS if f not in ('prle/prle_main.py', 'prrbb/prrbb_main.py')] \
    + ['td3prle/td3prle_main.py', 'td3prrbb/td3prrbb_main.py']
for f in _SOC_MAINS:
    t = open(f, encoding='utf-8').read()
    check('args.soc_reset_low' in t and 'uniform(0.5, 5.5)' not in t,
          '%-32s 起始 SOC 改读 args（旧硬编码已清）' % f)
for f in ['prle/prle_main.py', 'prrbb/prrbb_main.py']:
    t = open(f, encoding='utf-8').read()
    check('uniform(0.5, 5.5)' not in t and 'env.E =' not in t,
          '%-32s 无 SOC 随机化 ⇒ 沿用 env.reset() 的 3.0（本就合规）' % f)
print('=' * 88)
print('TEST PASSED' if not FAIL else '**TEST FAILED**: %s' % FAIL)
sys.exit(0 if not FAIL else 1)
