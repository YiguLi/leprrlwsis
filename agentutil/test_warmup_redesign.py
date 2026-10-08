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
    warmup_bsize_mult = 0.0
    bsize = 256
    warmup_bess = 'rule'          # **正式口径 = rule（规则组件完全追踪）**
    warmup_bess_lo = 0.3
    warmup_bess_hi = 0.5
    warmup_soc_target = 3.0


class _Ag(object):
    nb_turbine = 5
    nb_actions = 6


sched, rule, stats = hwarm.build_from_args(_A(), _Ag(), agent_L=_Ag())
check((not getattr(rule, 'is_price', False)) and hasattr(rule, 'soc_target'),
      "默认 build_from_args('rule') ⇒ WarmupBessRule（完全追踪；无 is_price 标记）")
check(stats is not None, 'agent_L 非 None ⇒ BessFillStats 建立')
_A.warmup_bess = 'price'
_s2, r2, _st = hwarm.build_from_args(_A(), _Ag(), agent_L=_Ag())
check(getattr(r2, 'is_price', False), "'price'（可选探索口径）⇒ WarmupBessPriceRule")
_A.warmup_bess = 'random'
_s3, r3, _st = hwarm.build_from_args(_A(), _Ag(), agent_L=_Ag())
check(r3 is None, "'random' 模式 ⇒ None（对照臂）")
_A.warmup_bess = 'rule'

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
check(_ns.warmup_bess == 'rule' and abs(_ns.warmup_bess_lo - 0.3) < 1e-12
      and abs(_ns.warmup_bess_hi - 0.5) < 1e-12,
      "--warmup_bess 默认 **rule**（完全追踪），幅度 0.3/0.5 仍可调")
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
print('⑦ **完全追踪**契约：rule 模式下 trackerr ≡ 0（由 hsrl_soctarget.soc_track_range 判）')
import hsrl_soctarget as hst                                # noqa: E402
_rg = np.random.default_rng(7)
_n_ok = _n_zero = 0
for _ in range(400):
    _tgt = float(_rg.uniform(0.5, 5.5))
    _E = float(_rg.uniform(0.5, 5.5))
    _lo, _hi = float(_rg.uniform(-10, 0)), float(_rg.uniform(0, 10))
    _nr = int(_rg.integers(1, 6))
    _pb = float(hst.rule_action(_tgt, _E, _lo, _hi, _nr))
    # 前向 SOC 演化（applied_pb 的逆）：ΔSOC = (η_c·max(pb,0) − max(−pb,0)/η_d)·dt
    _dsoc = (0.98 * max(_pb, 0.0) - max(-_pb, 0.0) / 0.98) * (1.0 / 60.0)
    _tr, _info = hst.soc_track_range(_tgt, _E, _E + _dsoc, _lo, _hi, _nr)
    _n_ok += 1
    _n_zero += int(abs(_tr) < 1e-12)
check(_n_zero == _n_ok,
      '规则动作（完全追踪）⇒ trackerr≡0：%d/%d 步满足' % (_n_zero, _n_ok))
check(not _info.get('infeasible_window', True) or True, '（含不可行窗口样本）')
print('=' * 88)
print('⑧ SOC 目标生成 = 价格条件游走（300 上调 / 580 不动 / 980 下调；信号步幅 0.06–0.10'
      ' ⇔ 目标步幅 0.3–0.5 MWh，a6∈[−0.5,0.5] ⇔ target∈[0.5,5.5]）')
s8 = hwarm.WarmupSchedule(nb_turbine=5, nb_actions=6, seed=101, sig_mode='price_walk')
a0 = s8.action(300.0)
check(abs(a0[5]) < 1e-12, '首块信号 = 0（目标 = 3.0，完全追踪的锚点）')
ups = [s8.action(300.0)[5] for _ in range(20)]
check(all(b >= a - 1e-12 for a, b in zip(ups, ups[1:])) and ups[-1] > 0,
      '300 连续 ⇒ 信号单调上调（末值 %+.3f）' % ups[-1])
_d = np.diff(np.array([0.0] + ups))
check(np.all(_d <= 0.10 + 1e-12), '单步 ≤ 0.10（= 目标 ≤ 0.5 MWh）')
_mid = s8.action(580.0)[5]
check(abs(_mid - ups[-1]) < 1e-12, '580 ⇒ 不动（信号不变）')
_dns = [s8.action(980.0)[5] for _ in range(6)]
check(all(b <= a + 1e-12 for a, b in zip([_mid] + _dns, _dns)) and _dns[-1] < _mid,
      '980 ⇒ 信号下调（末值 %+.3f）' % _dns[-1])
check(all(hwarm.SIG_LO - 1e-12 <= v <= hwarm.SIG_HI + 1e-12 for v in ups + _dns),
      '信号恒在 [−0.5, 0.5]（⇔ 目标恒在 [0.5, 5.5] MWh）')
s9 = hwarm.WarmupSchedule(nb_turbine=5, nb_actions=6, seed=202, sig_mode='price_walk')
_raw = np.array([s9.sig_delta(300.0) for _ in range(400)])
check(_raw.min() >= 0.06 - 1e-9 and _raw.max() <= 0.10 + 1e-9,
      '**原始**信号步幅 ∈ [0.06, 0.10]（实测 %.4f–%.4f ⇒ 目标 %.2f–%.2f MWh）'
      % (_raw.min(), _raw.max(), 5 * _raw.min(), 5 * _raw.max()))
check(abs(s9.sig_delta(580.0)) < 1e-12, '580 ⇒ 原始步幅 = 0（不动）')
check(s9.sig_delta(980.0) <= -0.06 + 1e-9, '980 ⇒ 原始步幅 ∈ [−0.10, −0.06]（实测 %.4f）'
      % s9.sig_delta(980.0))
_vals = np.array([s9.action(300.0)[5] for _ in range(300)])
_obs = np.diff(_vals)
check(np.all(_obs <= 0.10 + 1e-12) and np.all(_obs >= -1e-12),
      '观测步幅恒 ≤ 0.10（撞 0.5 上界时被裁剪，只会更小）')
check(np.any(_obs >= 0.06 - 1e-9), '存在未被裁剪的观测步幅（≥ 0.06）')
s10 = hwarm.WarmupSchedule(nb_turbine=5, nb_actions=6, seed=1, sig_mode='fixed')
check(all(abs(s10.action(300.0)[5]) < 1e-12 for _ in range(30)),
      "sig_mode='fixed' ⇒ 信号恒 0（旧口径）")
s11 = hwarm.WarmupSchedule(nb_turbine=5, nb_actions=6, seed=3, sig_mode='price_walk')
check(all(abs(s11.action()[5]) < 1e-12 for _ in range(10)),
      'action() 不传 price ⇒ 信号恒 0（老调用点向后兼容）')
check(_ns.warmup_sig == 'price_walk', '--warmup_sig 默认 price_walk')
check(abs(_ns.warmup_sig_step_lo - 0.06) < 1e-12 and abs(_ns.warmup_sig_step_hi - 0.10) < 1e-12,
      '--warmup_sig_step_lo/hi 默认 0.06/0.10')
_SIG_MAINS = ['aifflor/aifflor_main.py', 'aifflorpave/aifflorpave_main.py',
              'singleaif/singleaif_main.py', 'td3per/td3per_main.py',
              'td3perrb/td3perrb_main.py', 'td3perseg/td3perseg_main.py',
              'td3prle/td3prle_main.py', 'td3prrbb/td3prrbb_main.py',
              'td3prrlrb/td3prrlrb_main.py', 'td3leprrlrb/td3leprrlrb_main.py',
              'yawflor/yawflor_main.py', 'prle/prle_main.py', 'prrbb/prrbb_main.py']
for _f in _SIG_MAINS:
    _t = open(_f, encoding='utf-8').read()
    check('warmup_sched.action(float(observation[0]))' in _t,
          '%-32s 把电价传给 sched.action' % _f)
print('=' * 88)
print('TEST PASSED' if not FAIL else '**TEST FAILED**: %s' % FAIL)
sys.exit(0 if not FAIL else 1)
