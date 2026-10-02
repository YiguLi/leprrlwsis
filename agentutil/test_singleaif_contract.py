# -*- coding: utf-8 -*-
"""singleaif 契约冒烟测试 (不需 MATLAB) —— 上层 H 只输出 1 个风机动作 + 1 信号, 广播到 5 台。

singleaif = TD3PER 的改造: H 动作维数 6 → 2 ([a_wt, soc_signal]), 单一风机动作广播
到全部 5 台; 其余 (L 层 RL BESS / 非分段奖励 / SOC 目标信号 / n 步回报 / EMA 对偶 /
PER 等) 不变。本测试验证**只有**这些差异、且 penalty_NN_H 只计一次越界 (不 ×5)。

覆盖:
  [A] `CheckAction(len==2)` 裁剪与越界罚 (penalty_NN_H 对应修改)
  [B] `violation_amount/derate(n_dims=1)`: 越界乘性系数只对单一风机维计
  [C] `CheckAction` 的 len==6/5/1 旧分支未被破坏 (回归)
  [D] 源码级契约: singleaif_main / singleaif_ddpg / singleaif_myfunc 的关键改动点都在

不 import `singleaif_main` (它在模块顶层 `matlab.engine.start_matlab()`);
只 import `singleaif_myfunc` (其 import 链 hsrl_cost/hsrl_soctarget/util/test_scenario
均无 matlab/myfunc/env ⇒ 无需引擎)。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np

import singleaif.singleaif_myfunc as mf


def main():
    fails = []

    def check(name, cond, detail=''):
        if cond:
            print('  [OK  ] %s %s' % (name, detail))
        else:
            print('  [FAIL] %s %s' % (name, detail))
            fails.append(name)

    # ---- [A] CheckAction(len==2) ----
    print('[A] CheckAction(len==2) = [a_wt, soc_signal], 罚只计一次:')
    a, p = mf.CheckAction(np.array([0.3, 0.0]))
    check('合法(0.3,0) → 原样 + 罚 0',
          np.allclose(a, [0.3, 0.0]) and p == 0, 'got %s p=%s' % (a, p))
    a, p = mf.CheckAction(np.array([0.6, 0.0]))
    check('WT +0.1 → clip 0.5 + 罚 40 (=400×0.1)',
          np.allclose(a, [0.5, 0.0]) and abs(p - 40.0) < 1e-9, 'got %s p=%s' % (a, p))
    a, p = mf.CheckAction(np.array([-0.1, 0.0]))
    check('WT −0.1 → clip 0 + 罚 40',
          np.allclose(a, [0.0, 0.0]) and abs(p - 40.0) < 1e-9, 'got %s p=%s' % (a, p))
    a, p = mf.CheckAction(np.array([0.3, 0.7]))
    check('信号 +0.2 → clip 0.5 + 罚 80',
          np.allclose(a, [0.3, 0.5]) and abs(p - 80.0) < 1e-9, 'got %s p=%s' % (a, p))
    a, p = mf.CheckAction(np.array([0.6, -0.6]))
    check('WT+0.1 且 信号−0.1 → 罚 40+40=80',
          np.allclose(a, [0.5, -0.5]) and abs(p - 80.0) < 1e-9, 'got %s p=%s' % (a, p))
    check('返回 2 维 (不是 6 维)',
          len(mf.CheckAction(np.array([0.3, 0.0]))[0]) == 2)

    # ---- [B] violation_derate n_dims=1 ----
    print('[B] violation_amount/derate(n_dims=1):')
    check('只对 dim0(单一风机维) 计越界',
          abs(mf.violation_amount(np.array([0.6, 0.9]), n_dims=1) - 0.1) < 1e-9)
    check('derate = exp(-c·0.1)',
          abs(mf.violation_derate(np.array([0.6, 0.0]), coef=1.0, n_dims=1) - np.exp(-0.1)) < 1e-12)

    # ---- [C] 旧分支回归 ----
    print('[C] CheckAction 旧分支回归:')
    check('len==6 仍 6 维', len(mf.CheckAction(np.array([0.2] * 6))[0]) == 6)
    check('len==5 仍 5 维', len(mf.CheckAction(np.array([0.2] * 5))[0]) == 5)
    a, p = mf.CheckAction(np.array([5.0]))
    check('len==1 仍 clip 到 ±10', len(a) == 1 and abs(a[0] - 5.0) < 1e-9)

    # ---- [D] 源码级契约 ----
    print('[D] 源码级契约 (关键改动点都在):')
    rd = lambda *p: open(os.path.join(ROOT, *p), encoding='utf-8').read()
    s_main = rd('singleaif', 'singleaif_main.py')
    s_ddpg = rd('singleaif', 'singleaif_ddpg.py')
    s_myf = rd('singleaif', 'singleaif_myfunc.py')
    checks = [
        ("RUN_TAG='singleaif'", "RUN_TAG = 'singleaif'" in s_main),
        ("--env 默认 SINGLEAIF", "default='SINGLEAIF'" in s_main),
        ("H nb_actions=2 + nb_wt=1", "nb_actions=2, args=args, nb_wt=1" in s_main),
        ("main 广播 np.full(args.nb_turbine", "np.full(args.nb_turbine" in s_main),
        ("main violation_derate n_dims=1",
         "violation_derate(original_action_H, args.viol_derate, n_dims=1)" in s_main),
        ("myfunc 广播 np.full(env_test.num_turbine", "np.full(env_test.num_turbine" in s_myf),
        ("myfunc violation_derate n_dims=1",
         "violation_derate(action_H_raw, viol_derate_coef, n_dims=1)" in s_myf),
        ("CheckAction len==2 分支", "elif len(action) == 2:" in s_myf),
        ("ddpg 签名 nb_wt=None", "nb_wt=None" in s_ddpg),
        ("ddpg nb_turbine 跟随 nb_wt",
         "int(nb_wt) if nb_wt is not None else int(args.nb_turbine)" in s_ddpg),
        ("init_wt_action_bias n_wt_dims=1", "n_wt_dims=1" in s_main),
        ("import 改为 singleaif.*", "from singleaif.singleaif_ddpg import SP_TD3" in s_main),
    ]
    for name, ok in checks:
        check(name, ok)

    print()
    if fails:
        print('FAILED (%d): %s' % (len(fails), ', '.join(fails)))
        return 1
    print('ALL PASSED')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
