# -*- coding: utf-8 -*-
"""契约：`td3per/td3per_hsrl.py` 与 `aifflor/aifflor_hsrl.py` **数值逐位一致**。

背景
----
`--hsrl` 是"块总放电量"方法分支。td3per 侧的实现是 aifflor 侧的移植
（见 `documentation/td3per_vs_aifflor.md` §5）。两者**只允许**在
"引用哪个 myfunc 包"与 docstring 上不同 ⇒ 任何数值漂移都是 bug。

本测试（纯数值 / 不推进仿真 / 不 import env）：
  1. 常量逐个相等（HSRL_* / OBS_*）；
  2. 纯函数在随机输入网格上逐位相等（feasible / check_action / reward / cost /
     obs / amortized / lam bounds / dual update）；
  3. `cal_rew_rb_hsrl` 用合成 obs/obs2/rbb_info 对比 r1/c1/details 关键项；
  4. 源码层面：td3per 侧不 import aifflor。

运行：`python agentutil/test_td3per_hsrl_parity.py`
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

from aifflor import aifflor_hsrl as A          # noqa: E402
from td3per import td3per_hsrl as T            # noqa: E402

FAIL = []
N = 0


def check(name, cond, detail=''):
    global N
    N += 1
    if not cond:
        FAIL.append('%s %s' % (name, detail))


def eq(name, x, y, tol=0.0):
    if isinstance(x, str) or isinstance(y, str):
        ok = (x == y)
        check(name, ok, '' if ok else '(a=%r b=%r)' % (x, y))
        return
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    ok = a.shape == b.shape and np.allclose(a, b, rtol=0.0, atol=tol, equal_nan=True)
    check(name, ok, '' if ok else '(a=%r b=%r)' % (x, y))


CONSTS = ['HSRL_BLOCK', 'DT', 'ETA_C', 'ETA_D', 'P_RATED', 'E_MIN', 'E_MAX',
          'HSRL_PEN6_COEF', 'HSRL_G_FULL', 'HSRL_VIOL_COEF', 'HSRL_DEG_COEF',
          'HSRL_L_LAM_MIN', 'HSRL_L_LAM_MAX', 'HSRL_L_LAM_INIT',
          'HSRL_L_EMA_LAM_LR', 'HSRL_D_BLOCK_L', 'HSRL_D_BLOCK_L_MIN',
          'HSRL_D_BLOCK_H_MIN', 'HSRL_L_COST_LIMIT', 'HSRL_L_CURRICULUM_BLOCKS',
          'HSRL_DUAL_Q_AGG', 'HSRL_DUAL_SRC', 'HSRL_DUAL_Q_SRC',
          'HSRL_DUAL_RECENT_N', 'HSRL_COST_DELTA', 'HSRL_TEST_STEPS',
          'HSRL_TEST_PRICE', 'HSRL_TEST_WIND_COL', 'HSRL_TEST_WIND_START',
          'OBS_U_SCALE', 'OBS_PR_SCALE', 'OBS_E_SCALE', 'OBS_PG_SCALE',
          'OBS_G_SCALE', 'OBS_NEED_SCALE']


def test_constants():
    for c in CONSTS:
        eq('const.' + c, getattr(A, c), getattr(T, c))


def test_pure_functions():
    rng = np.random.RandomState(20261106)
    for _ in range(200):
        E = float(rng.uniform(-0.05, 0.55))
        eq('feasible_total(E=%.3f) ctx=%s' % (E, ''),
           A.feasible_total(E), T.feasible_total(E))
        eq('feasible_power', A.feasible_power(E), T.feasible_power(E))

        a = np.concatenate([rng.uniform(-0.6, 1.0, 5),
                            [rng.uniform(-4.0, 4.0)]])
        ra = A.check_action_H_hsrl(a, E)
        rb = T.check_action_H_hsrl(a, E)
        eq('check_action_H_hsrl.applied', ra[0], rb[0])
        eq('check_action_H_hsrl.pen', [ra[1], ra[2]], [rb[1], rb[2]])

        g = float(rng.uniform(-2.5, 2.5))
        sd = float(rng.uniform(0.0, 2.5))
        nrem = int(rng.randint(1, 6))
        aL = float(rng.uniform(-1.0, 1.0))
        eq('plan_need', A.plan_need(g, sd, nrem), T.plan_need(g, sd, nrem))
        eq('lower_cost_hsrl', A.lower_cost_hsrl(g, sd, nrem, aL),
           T.lower_cost_hsrl(g, sd, nrem, aL))
        eq('lower_reward_hsrl', A.lower_reward_hsrl(0.3, aL),
           T.lower_reward_hsrl(0.3, aL))
        obs = [rng.uniform(0, 25), rng.uniform(0, 1000), rng.uniform(0, 0.5),
               rng.uniform(-3, 3), g, A.plan_need(g, sd, nrem)]
        eq('build_L_obs_hsrl', A.build_L_obs_hsrl(*obs), T.build_L_obs_hsrl(*obs))
        eq('amortized_rule_action', A.amortized_rule_action(g, sd, nrem, E),
           T.amortized_rule_action(g, sd, nrem, E))
        eq('discharge_of', A.discharge_of(0.4), T.discharge_of(0.4))
        la = A.check_action_L_hsrl([aL], E)
        lb = T.check_action_L_hsrl([aL], E)
        eq('check_action_L_hsrl.applied', la[0], lb[0])
        eq('check_action_L_hsrl.pen', [la[1], la[2]], [lb[1], lb[2]])

    for lo, hi, init in ((-1.0, 1.0, 0.0), (2.0, 50.0, 10.0), (0.5, 3.0, 1.5)):
        eq('lam1_phys_bounds', A.lam1_phys_bounds(lo, hi),
           T.lam1_phys_bounds(lo, hi))


class _Agent(object):
    def __init__(self):
        self.lam = 1.0
        self.lam_min = 0.1
        self.lam_max = 9.0
        self.discount = 0.95
        self.nb_states = 6
        self.device = None
        self.ema_cost = None

    def update_ema_cost(self, c):
        self.ema_cost = float(c) if self.ema_cost is None else \
            0.98 * self.ema_cost + 0.02 * float(c)


def test_dual():
    for src, rate, ema in (('q', 3.0, 1.0), ('ema', None, 2.0), ('q', 0.0, 0.5)):
        a1, a2 = _Agent(), _Agent()
        a1.ema_cost = a2.ema_cost = ema
        A.dual_update_H_hsrl(a1, 1.0, q_rate=rate, src=src, ema=ema)
        T.dual_update_H_hsrl(a2, 1.0, q_rate=rate, src=src, ema=ema)
        eq('dual_update_H_hsrl(%s)' % src, a1.lam, a2.lam)
        b1, b2 = _Agent(), _Agent()
        b1.ema_cost = b2.ema_cost = ema
        A.dual_update_L_hsrl(b1, 0.1, q_rate=rate, src=src, ema=ema)
        T.dual_update_L_hsrl(b2, 0.1, q_rate=rate, src=src, ema=ema)
        eq('dual_update_L_hsrl(%s)' % src, b1.lam, b2.lam)


def test_reward():
    rng = np.random.RandomState(7)
    for _ in range(100):
        obs = np.array([rng.uniform(250, 1000), rng.uniform(3, 20),
                        rng.uniform(0, 0.5), rng.uniform(-1, 1)])
        obs2 = np.array([obs[0], obs[1], rng.uniform(0, 0.5), rng.uniform(-1, 1)])
        action_rbb = np.concatenate([rng.uniform(0, 0.5, 5),
                                     [rng.uniform(-2.5, 2.5)],
                                     [rng.uniform(-0.5, 0.5)]])
        rbb = dict(pbrule=float(action_rbb[-1]), unfeasible_gap=0.0,
                   clip_amount=0.0, is_infeasible=False, inevitable_violation=0.0)
        kw = dict(penalty_NN=1.0, time=A.DT, ptur=float(rng.uniform(0, 5)),
                  pb_cmd=0.0, turbine_derate=1.0)
        r1a, c1a, da = A.cal_rew_rb_hsrl(obs, obs2, action_rbb, rbb, **kw)
        r1b, c1b, db = T.cal_rew_rb_hsrl(obs, obs2, action_rbb, rbb, **kw)
        eq('cal_rew_rb_hsrl.r1', r1a, r1b)
        eq('cal_rew_rb_hsrl.c1', c1a, c1b)
        for k in ('turbine_profit', 'bess_profit', 'revenue', 'degradation',
                  'degradation_ref490', 'P_F', 'SOC', 'c1_fluct'):
            eq('cal_rew_rb_hsrl.' + k, da[k], db[k])


def test_source():
    p = os.path.join(ROOT, 'td3per', 'td3per_hsrl.py')
    src = open(p, encoding='utf-8').read()
    check('td3per_hsrl 不 import aifflor', 'import aifflor' not in src)
    check('td3per_hsrl 引用 td3per_myfunc',
          'td3per.td3per_myfunc' in src or 'td3per_myfunc' in src)


def main():
    test_constants()
    test_pure_functions()
    test_dual()
    test_reward()
    test_source()
    if FAIL:
        print('FAIL (%d/%d)' % (len(FAIL), N))
        for f in FAIL[:30]:
            print('  ✗', f)
        return 1
    print('PASS: %d 项契约全部通过（td3per_hsrl ≡ aifflor_hsrl）' % N)
    return 0


if __name__ == '__main__':
    sys.exit(main())
