# -*- coding: utf-8 -*-
"""验证探索噪声幅度表：ε=1 → ε=0 时幅度从 explore_amp 收敛到 explore_floor（不再为 0）。

**2026-10 口径修正**：风机（WT）维的幅度**恒定 = `wt_noise`（默认 0.05）**，不再随 ε 衰减
（用户口径"全部风机、风机行为噪声恒为 0.05"）；只有第 6 维 SOC 信号仍按
`sig_noise_max(0.2) → sig_noise_min(0.01)` 衰减。实现方式是让 WT 维的
`explore_amp == explore_floor == wt_noise`，凸组合公式本身不变。

自建最小 args（`SP_TD3.__init__` 需要 24 个字段），构造 H 层（6 维）与 L 层（1 维），
打印并核对：
  ε=1.0 幅度 = explore_amp
  ε=0.5 幅度 = 中点
  ε=0.0 幅度 = explore_floor（**关键：必须 > 0**）
并用 `select_action` 实测 ε=0/ε=1 时的实际抖动范围，断言 WT 维**两者相等且 ≈ 0.05**。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402


class A:
    pass


def mkargs():
    a = A()
    a.cuda = False
    a.seed = 0
    a.hidden1 = 64
    a.hidden2 = 64
    a.init_w = 3e-3
    a.rate = 1e-3
    a.prate = 1e-3
    a.lr_lam = 1e-3
    a.discount = 0.98
    a.tau = 0.001
    a.bsize = 256
    a.rmsize = 40000
    a.PER_alpha = 0.6
    a.PER_beta = 0.4
    a.PER_epsilon = 0.01
    a.beta_increment_per_sampling = 0.001
    a.epsilon = 4000
    a.lam = 0.1
    a.max_episode_length = 1440
    a.nb_turbine = 5
    a.cost_single_limit = 1000.0
    a.ou_mu = 0.0
    a.ou_sigma = 0.2
    a.ou_theta = 0.15
    return a


def main():
    import td3per.td3per_ddpg as D
    args = mkargs()

    print('=' * 96)
    print('探索噪声幅度表  amp(ε) = explore_amp·ε + explore_floor·(1−ε)')
    print('=' * 96)
    for tag, na, nt in (('H 层 (WT 5 维 + 信号维)', 6, 5), ('L 层 (1 维)', 1, 5)):
        ag = D.SP_TD3(nb_states=6, nb_actions=na, args=args,
                      wt_noise=0.05, sig_noise_max=0.2, sig_noise_min=0.01,
                      wt_noise_floor=0.01)
        print('\n  [%s]' % tag)
        print('    explore_amp   = %s' % np.array2string(ag.explore_amp, precision=4))
        print('    explore_floor = %s' % np.array2string(ag.explore_floor, precision=4))
        print('    %-8s %s' % ('ε', '  '.join('%9s' % ('dim%d' % i)
                                              for i in range(na))))
        for eps in (1.0, 0.75, 0.5, 0.25, 0.0):
            amp = ag.explore_amp * eps + ag.explore_floor * (1.0 - eps)
            print('    %-8.2f %s' % (eps, '  '.join('%9.5f' % v for v in amp)))
        # 断言
        a1 = ag.explore_amp * 1.0 + ag.explore_floor * 0.0
        a0 = ag.explore_amp * 0.0 + ag.explore_floor * 1.0
        ok1 = np.allclose(a1, ag.explore_amp)
        ok0 = np.all(a0 > 0.0)
        print('    ε=1 ⇒ 等于 explore_amp : %s' % ('OK' if ok1 else '**FAIL**'))
        print('    ε=0 ⇒ 全维 > 0（不再停止探索）: %s (最小 %.5f)'
              % ('OK' if ok0 else '**FAIL**', float(a0.min())))

    # ---- 实测 select_action 在 ε=0 时的抖动 ----
    print('\n' + '=' * 96)
    print('实测 `select_action`：把 ε 强制为 0 后，动作仍带 ±floor 的抖动')
    print('=' * 96)
    ag = D.SP_TD3(nb_states=6, nb_actions=6, args=args,
                  wt_noise=0.05, sig_noise_max=0.2, sig_noise_min=0.01,
                  wt_noise_floor=0.01)
    ag.is_training = True
    s = np.zeros(6, dtype=np.float32)
    for eps_val, lab in ((0.0, 'ε=0'), (1.0, 'ε=1')):
        ag.epsilon = eps_val
        acts = np.array([ag.select_action(s, decay_epsilon=False) for _ in range(4000)])
        lo = acts.min(axis=0)
        hi = acts.max(axis=0)
        print('  %-6s 各维实测范围半宽: %s' % (lab, np.array2string((hi - lo) / 2.0,
                                                                  precision=4)))
    ag.epsilon = 0.0
    acts = np.array([ag.select_action(s, decay_epsilon=False) for _ in range(4000)])
    hw0 = (acts[:, :5].max(0) - acts[:, :5].min(0)) / 2.0
    print('  ε=0 时 5 个 WT 维的抖动半宽 = %s （期望 ≈ 0.05 = WT 恒定幅度）'
          % np.array2string(hw0, precision=4))
    ag.epsilon = 1.0
    acts1 = np.array([ag.select_action(s, decay_epsilon=False) for _ in range(4000)])
    hw1 = (acts1[:, :5].max(0) - acts1[:, :5].min(0)) / 2.0
    print('  ε=1 时 5 个 WT 维的抖动半宽 = %s （应与 ε=0 相同）'
          % np.array2string(hw1, precision=4))
    ok_wt = bool(np.all(np.abs(hw0 - 0.05) < 0.004) and np.all(np.abs(hw1 - 0.05) < 0.004))
    print('  ⇒ WT 维恒为 0.05（ε=0 与 ε=1 一致）: %s' % ('OK' if ok_wt else '**FAIL**'))
    return 0 if ok_wt else 1


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
