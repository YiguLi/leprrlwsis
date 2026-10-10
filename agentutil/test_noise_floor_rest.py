# -*- coding: utf-8 -*-
"""验证剩余 3 个 TD3 系列的探索噪声（**已统一为 td3per 格式**）。

统一格式的幅度表（2026-10 起：**风机维恒定 `wt_noise`，不再随 ε 衰减**）
--------------------------------------------------------------------
  H 层 (nb_actions = nt+1 = 6):
      ε=1 ⇒ [wt_noise]×5 + [sig_noise_max]        = [0.05]*5 + [0.2]
      ε=0 ⇒ [wt_noise]×5 + [sig_noise_min]        = [0.05]*5 + [0.01]
      （WT 维两式相同 ⇒ 风机噪声恒定 0.05；只有第 6 维 SOC 信号从 0.2 衰减到 0.01）
  L 层 (nb_actions = 1，非风机维):
      仍为 [wt_noise] → [wt_noise_floor] = [0.05] → [0.01]（main 随后会覆写）
  （注：真实调用里 H 的 nb_turbine 决定维度；本测试用 nb_turbine=5 ⇒ 走 nt+1 分支）
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402


class Args:
    pass


def mkargs():
    a = Args()
    a.cuda = False
    a.seed = 0
    a.hidden1 = 32
    a.hidden2 = 32
    a.init_w = 3e-3
    a.rate = 1e-3
    a.prate = 1e-3
    a.lr_lam = 1e-3
    a.discount = 0.98
    a.tau = 0.001
    a.bsize = 64
    a.rmsize = 200
    a.PER_alpha = 0.6
    a.PER_beta = 0.4
    a.PER_epsilon = 0.01
    a.beta_increment_per_sampling = 0.001
    a.epsilon = 50000
    a.lam = 0.1
    a.max_episode_length = 1440
    a.nb_turbine = 5
    a.cost_single_limit = 1000.0
    a.ou_mu = 0.0
    a.ou_sigma = 0.2
    a.ou_theta = 0.15
    a.policy_delay = 2
    a.policy_noise = 0.1
    a.noise_clip = 0.3
    a.cost_critic_mode = 'max'
    a.nstep = 1
    a.target_agg = 'min'
    return a


CASES = [
    ('td3prle/td3prle_ddpg.py', 'td3prle.td3prle_ddpg', 'SP_TD3_PRLE', 6),
    ('td3prrbb/td3prrbb_ddpg.py', 'td3prrbb.td3prrbb_ddpg', 'SP_TD3', 6),
    ('td3leprrlrb/td3leprrlrb_ddpg.py', 'td3leprrlrb.td3leprrlrb_ddpg',
     'SP_TD3_LEPRRLRB', 6),
]


def measure(K, eps, nb_actions=6, n=8000):
    ag = K(nb_states=6, nb_actions=nb_actions, args=mkargs(),
           wt_noise=0.05, sig_noise_max=0.2, sig_noise_min=0.01,
           wt_noise_floor=0.01)
    ag.is_training = True
    ag.epsilon = eps
    ag.depsilon = 0.0
    s = np.zeros(6, dtype=np.float32)
    acts = np.array([ag.select_action(s, decay_epsilon=False) for _ in range(n)])
    return (acts.max(axis=0) - acts.min(axis=0)) / 2.0


def main():
    ok = True
    for path, mod, cls, na in CASES:
        m = __import__(mod, fromlist=[cls])
        K = getattr(m, cls)
        print('=' * 96)
        print('%s  (%s)' % (path, cls))
        print('=' * 96)
        h1 = measure(K, 1.0, na)
        h0 = measure(K, 0.0, na)
        exp1 = np.array([0.05] * 5 + [0.2]) if na == 6 else np.full(na, 0.05)
        # 2026-10: 风机维恒定 wt_noise ⇒ ε=0 的 WT 维仍是 0.05（只有信号维落到 0.01）
        exp0 = (np.array([0.05] * 5 + [0.01]) if na == 6
                else np.full(na, 0.05 if na > 1 else 0.01))
        print('  ε=1 实测: %s' % np.array2string(h1, precision=4))
        print('       期望: %s' % np.array2string(exp1, precision=4))
        print('  ε=0 实测: %s' % np.array2string(h0, precision=4))
        print('       期望: %s' % np.array2string(exp0, precision=4))
        g1 = bool(np.all(np.abs(h1 - exp1) < 0.006))
        g0 = bool(np.all(np.abs(h0 - exp0) < 0.004))
        print('  ε=1 匹配(td3per 格式): %s ; ε=0 = WT 恒定/信号地板: %s'
              % ('OK' if g1 else '**偏差**', 'OK' if g0 else '**偏差**'))
        ok = ok and g1 and g0
        print()
    # 维度分支核对（nb_actions = nb_turbine + 1 是 H；否则是 L）
    print('=' * 96)
    print('维度分支核对（nb_turbine=5）')
    print('=' * 96)
    m = __import__('td3prle.td3prle_ddpg', fromlist=['SP_TD3_PRLE'])
    for na in (6, 1):
        ag = m.SP_TD3_PRLE(nb_states=6, nb_actions=na, args=mkargs())
        print('  nb_actions=%d (nb_turbine=5) ⇒ amp_e=%s floor_e=%s'
              % (na, np.array2string(ag.explore_amp, precision=3),
                 np.array2string(ag.explore_floor, precision=3)))
    print()
    print('%s' % ('全部通过 ✅' if ok else '**有失败** ❌'))
    return 0 if ok else 1


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
