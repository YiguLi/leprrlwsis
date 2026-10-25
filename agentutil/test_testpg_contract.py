# -*- coding: utf-8 -*-
"""`testPg` 契约测试（**只读、不需要 MATLAB**）。

把"testPg 与 aifflor 的单因子差异"变成可执行判据：

A. 环境：`ptur = rate·Pmax(u)`、Pmax 由固定动作 `[0.2,0.2,0.2,0.2,0.33]` 现算、
   reset 满发、rate=1 与 aifflor 固定动作逐位一致、预取不改数值；
B. 动作/罚：2 维 checkAction（rate [0,1] + goal [-0.5,0.5]）、rate 越界罚、
   goal 越界罚开关、越界乘性罚只算 rate；
C. 奖励：`cal_rew_rb_pg` 与 aifflor 的 `Cal_Rew_RB` 在相同输入下**逐位相等**
   （只把 signal 的位置换成第 2 维）；L 观测与 `hst.build_L_obs` 同式；
D. agent：2 维随机动作/探索噪声/target smoothing 幅度/cost 裁剪/输出层 bias；
E. 9-case：45 步测试可跑通，指标有限且 `ptur ≈ rate·Pmax`；
F. 不变量：固定动作 = 用户口径；testPg 不 import matlab。

用法
----
    <python> agentutil\\test_testpg_contract.py
"""
import os
import sys
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

from floris_import_guard import check_interpreter as _chk, sanitize as _sanitize  # noqa: E402
_chk()
_sanitize()

FAIL = []


def ok(cond, msg):
    print('  %s %s' % ('PASS' if cond else 'FAIL', msg))
    if not cond:
        FAIL.append(msg)


def _args(**kw):
    """SP_TD3/SP_TD3_PG 构造所需的最小 args 集合。"""
    base = dict(
        cuda=False, seed=1, hidden1=16, hidden2=100, init_w=0.003,
        prate=1e-4, rate=3e-3, rmsize=1000, bsize=8, tau=1e-3,
        discount=0.95, epsilon=100, lam=2.0, lr_lam=0.1,
        cost_single_limit=1.0, PER_alpha=0.6, PER_beta=0.4,
        PER_epsilon=0.01, beta_increment_per_sampling=3e-5,
        nb_turbine=5, critic_arch='v1', critic_act='relu',
        critic_out_init='3e3', critic_feat_power=0,
        critic_feat_power_k=1.357e-3, critic_action_first=0,
        cost_clip=1, farm_level=0, aux_model=0, na_box=5,
        pg_rate_noise=0.05, pg_rate_noise_floor=None,
        sig_noise_max=0.2, sig_noise_min=0.0,
        ou_theta=0.15, ou_sigma=0.2, ou_mu=0.0,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def main():
    import aifflor.aifflor_turbine as AT
    from aifflor.aifflor_env import AifflorPhysics, WindBatteryAIF
    import aifflor.aifflor_myfunc as AMF
    import hsrl_soctarget as hst
    from rbb.rbb_rule_bess import RuleBasedBESS
    from testpg.testpg_env import WindBatteryPG, PG_FIXED_ACTION
    from testpg import testpg_myfunc as PMF
    from testpg.testpg_ddpg import SP_TD3_PG
    from testpg.testpg_warmup import PGWarmup

    print('=' * 90)
    print('A. 环境：固定行为 Pmax + rate 缩放 + 预取')
    print('=' * 90)
    ph = AifflorPhysics(nt=5, calibrated=False, power_scale=AT.LEVEL_COEF)
    env = WindBatteryPG(0.5, 5, False, None, physics=ph)
    obs = env.reset()
    u, pm = float(env.u), float(env.Pmax(float(env.u)))
    ok(np.shape(obs) == (4,), 'reset() 观测 4 维 [pr, u, E, P_G_last]')
    ok(abs(float(obs[3]) - pm) < 1e-12,
       'reset 的 P_G_prev = Pmax（rate=1，满发）: obs[3]=%.9f Pmax=%.9f'
       % (float(obs[3]), pm))
    ok(abs(pm - float(ph.power(u, PG_FIXED_ACTION))) < 1e-12,
       'Pmax == physics.power(u, A_FIXED)（固定动作 = %s）'
       % np.round(PG_FIXED_ACTION, 4).tolist())
    for r in (0.0, 0.25, 0.5, 1.0):
        v = float(env.PowerOutput(u, [r, 0.3]))
        ok(abs(v - r * pm) < 1e-12, 'rate=%.2f ⇒ ptur=%.9f == rate·Pmax' % (r, v))
    ok(float(env.PowerOutput(u, [1.0, -0.4])) == pm,
       'rate=1 ⇒ ptur 恰为 Pmax（与 goal 无关）')
    ok(float(env.PowerOutput(u, env.nominal())) == pm,
       '标称 5 维动作路径（reset）⇒ rate=1，ptur=Pmax')
    ok(env.PowerOutput(3.0, [1.0, 0.0]) == 0.0
       and env.PowerOutput(26.0, [1.0, 0.0]) == 0.0,
       'u<3.5 / u>25 ⇒ 0（围栏，未调引擎）')
    nd0 = ph.n_direct + ph.n_batch
    env.PowerOutput(3.0, [0.9, 0.0])
    ok(ph.n_direct + ph.n_batch == nd0, '围栏命中时**不调用**物理引擎')

    # rate=1 与 aifflor（固定动作）逐位同值：单因子对照的核心不变量
    ph_a = AifflorPhysics(nt=5, calibrated=False, power_scale=AT.LEVEL_COEF)
    env_a = WindBatteryAIF(0.5, 5, False, None, physics=ph_a)
    env_a.reset()
    for uu in (4.0, 6.5, 9.0, 12.0, 18.0):
        a = float(env.PowerOutput(uu, [1.0, 0.0]))
        b = float(env_a.PowerOutput(uu, PG_FIXED_ACTION))
        ok(abs(a - b) < 1e-12,
           'rate=1 在 u=%.1f 与 aifflor 固定动作同值（%.9f vs %.9f）' % (uu, a, b))

    # 预取：对同一批 u 的 Pmax 与直算一致，且缓存无坏值
    us = np.array([6.0, 7.5, 9.0, 10.5, 12.0])
    direct = np.array([float(env.Pmax(float(x))) for x in us])
    ph2 = AifflorPhysics(nt=5, calibrated=False, power_scale=AT.LEVEL_COEF)
    env2 = WindBatteryPG(0.5, 5, False, None, physics=ph2)
    env2.reset()
    env2.physics._cache = {}
    env2.physics.prefetch(us, PG_FIXED_ACTION)
    cached = np.array([float(env2.Pmax(float(x))) for x in us])
    ok(np.max(np.abs(direct - cached)) < 1e-12, '预取 5 个 u 的 Pmax 与直算逐位一致')
    ok(all(np.isfinite(v) for v in env2.physics._cache.values()),
       '预取缓存里没有非有限值（%d 条）' % len(env2.physics._cache))

    # step：2 维动作可驱动 env；pw_tru2 = rate·Pmax(u_next)
    pw = float(env.PowerOutput(float(env.u), [0.5, 0.0]))
    out = env.step(np.array([0.5, 0.0]), np.array([0.0]), pw)
    pw2, obs2 = float(out[0]), out[1]
    ok(len(out) == 7, 'step() 返回 7 元组（与 env.py 同形）')
    ok(abs(pw2 - 0.5 * float(env.Pmax(float(env.u)))) < 1e-12,
       '下一步 ptur = rate·Pmax(u_next)')
    ok(abs(float(obs2[3]) - pw) < 1e-12,
       'P_G(t) = pw_tru(t) − pbrule（电池命令=0 ⇒ P_G=ptur）')

    print('\n' + '=' * 90)
    print('B. 动作检查 / 罚')
    print('=' * 90)
    for a in ([0.5, 0.0], [1.0, 0.5], [0.0, -0.5]):
        ap, pen = PMF.CheckAction(a)
        ok(pen == 0.0 and np.allclose(ap, a), '界内 %s ⇒ 无罚、动作不变' % a)
    ap, pen = PMF.CheckAction([1.2, -0.8])
    ok(np.allclose(ap, [1.0, -0.5]), '越界动作 clip 到 [1.0, -0.5]')
    ok(abs(pen - 5.0) < 1e-12,
       'rate 越界 0.2 ⇒ 2.0；goal 越界 0.3 ⇒ 3.0；总罚 5.0（两者都罚）')
    PMF.PG_GOAL_PEN = False
    _, pen_rate = PMF.CheckAction([1.2, -0.8])
    PMF.PG_GOAL_PEN = True
    ok(abs(pen_rate - 2.0) < 1e-12, '--pg_goal_pen 0 ⇒ 只罚 rate（2.0）')
    ap, pen = PMF.CheckAction([0.7])          # 下层 1 维，沿用 aifflor 的 200/单位
    ok(np.allclose(ap, [0.5]) and abs(pen - 40.0) < 1e-9,
       '1 维动作委托 aifflor 口径：clip 0.5、罚 200×0.2=40')

    ok(abs(PMF.violation_amount_pg([1.2, 0.9]) - 0.2) < 1e-12,
       'violation_amount_pg 只算 rate 越界（goal 不计）')
    ok(abs(PMF.violation_amount_pg([-0.3, -0.9]) - 0.3) < 1e-12,
       'rate<0 的越界量 = 0.3')
    ok(PMF.violation_derate_pg([2.0, 0.0], coef=0.0) == 1.0,
       'coef<=0 ⇒ f 恰为 1.0（旧路径逐位不变）')
    ok(abs(PMF.violation_derate_pg([1.5, 0.0], coef=2.0)
           - float(np.exp(-2.0 * 0.5))) < 1e-12,
       'f = exp(-c·rate 越界量)')

    print('\n' + '=' * 90)
    print('C. 奖励 / L 观测：与 aifflor 同式（只换 signal 位置）')
    print('=' * 90)
    obs1 = np.array([980.0, 10.0, 0.25, 3.0])
    obs2 = np.array([980.0, 10.5, 0.27, 3.3])
    rbb = {'pbrule': -0.2, 'unfeasible_gap': 0.0, 'clip_amount': 0.0,
           'is_infeasible': False, 'inevitable_violation': 0.0}
    rate, goal, ptur = 0.8, 0.1, 2.4
    r1, c1, det = PMF.cal_rew_rb_pg(obs1, obs2, np.array([rate, goal]), rbb,
                                    penalty_NN_H=1.0, ptur=ptur,
                                    pb_cmd=-0.1, pmax=3.0)
    r1a, c1a, deta = AMF.Cal_Rew_RB(
        obs1, obs2, np.array([0.2] * 5 + [goal, -0.2]), rbb,
        penalty_NN_H=1.0, ptur=ptur, pb_cmd=-0.1)
    ok(abs(r1 - r1a) < 1e-12, 'r1 逐位相同（%.12f vs %.12f）' % (r1, r1a))
    ok(abs(c1 - c1a) < 1e-12, 'c1 逐位相同（%.12f vs %.12f）' % (c1, c1a))
    for k in ('soc_signal', 'soc_target', 'P_F', 'SOC', 'turbine_profit',
              'bess_profit', 'degradation', 'revenue', 'penalty_NN_H'):
        ok(abs(float(det[k]) - float(deta[k])) < 1e-12,
           'details[%s] 与 aifflor 相同' % k)
    ok(abs(det['rate'] - rate) < 1e-12 and abs(det['pmax'] - 3.0) < 1e-12,
       'details 新增 rate / pmax')
    ok(abs(det['ptur_max'] - ptur / rate) < 1e-12,
       'details.ptur_max = ptur/rate')

    n_rem = 3
    L_pg = PMF.build_L_obs_pg(obs1, ptur, np.array([rate, goal]), rbb, n_rem)
    L_ref = hst.build_L_obs(float(obs1[-1]), ptur, float(obs1[2]),
                            hst.soc_target_from_signal(goal), n_rem)
    ok(np.max(np.abs(L_pg - L_ref)) < 1e-15,
       'L 观测 == hsrl_soctarget.build_L_obs（5 维）')

    print('\n' + '=' * 90)
    print('D. agent：2 维动作的 TD3 接口')
    print('=' * 90)
    a = _args()
    ag = SP_TD3_PG(nb_states=4, args=a, lam_max=100.0, lam_min=2.0,
                   ema_lam=True, policy_delay=2, policy_noise=0.1,
                   noise_clip=0.3, cost_critic_mode='max',
                   nstep=5, target_agg='min')
    ok(ag.nb_actions == 2, 'H nb_actions = 2')
    ok(np.allclose(ag.action_amp, [0.5, 0.5]), 'target smoothing amp = [0.5, 0.5]')
    ok(np.allclose(ag.explore_amp, [0.05, 0.2])
       and np.allclose(ag.explore_floor, [0.05, 0.0]),
       'rate 恒定 ±0.05；goal 0.2→0.0')
    ra = np.array([ag.random_action() for _ in range(200)])
    ok(ra.shape == (200, 2) and (ra[:, 0] >= 0).all() and (ra[:, 0] <= 1).all()
       and (ra[:, 1] >= -0.5).all() and (ra[:, 1] <= 0.5).all(),
       'random_action: rate∈[0,1], goal∈[-0.5,0.5]')
    ag.is_training = False
    d = ag.select_action(np.zeros(4))
    ok(np.shape(d) == (2,), '评测（is_training=False）输出 2 维确定性动作')
    ag.is_training = True
    ok(PMF.init_pg_action_bias(ag, rate_value=0.75, goal_value=0.1, verbose=False),
       'init_pg_action_bias 返回 True')
    ok(abs(float(ag.actor.fc4.bias[0]) - 0.75) < 1e-6
       and abs(float(ag.actor.fc4.bias[1]) - 0.1) < 1e-6
       and abs(float(ag.actor_target.fc4.bias[0]) - 0.75) < 1e-6,
       'init_pg_action_bias 写入 actor 与 actor_target')
    import torch
    cb = ag._clip_box(torch.tensor([[1.3, -0.9], [0.4, 0.7]]))
    ok(np.allclose(cb.numpy(), [[1.0, -0.9], [0.4, 0.7]]),
       '_clip_box: 只把 rate 夹到 [0,1]，goal 原样')

    print('\n' + '=' * 90)
    print('E. warmup：[rate~U(0,1), goal 价格游走]')
    print('=' * 90)
    w = PGWarmup(enabled=True, sig_mode='price_walk', seed=7)
    acts = np.array([w.action([300.0, 580.0, 980.0][i % 3])
                     for i in range(300)])
    ok(acts.shape == (300, 2) and acts[:, 0].min() >= 0 and acts[:, 0].max() <= 1,
       'warmup 动作 300 个：rate 覆盖 [0,1]（min=%.3f max=%.3f）'
       % (acts[:, 0].min(), acts[:, 0].max()))
    ok(acts[:, 1].min() >= -0.5 and acts[:, 1].max() <= 0.5,
       'warmup goal 始终在 [-0.5,0.5]')
    ok(PGWarmup(enabled=False).action(300.0) is None,
       '--warmup_sched random ⇒ action() 返回 None')

    print('\n' + '=' * 90)
    print('F. 9-case：45 步可跑通、指标有限、ptur=rate·Pmax')
    print('=' * 90)
    ag_H = SP_TD3_PG(nb_states=4, args=_args(), lam_max=100.0, lam_min=2.0,
                     ema_lam=True, policy_delay=2, policy_noise=0.1,
                     noise_clip=0.3, cost_critic_mode='max',
                     nstep=1, target_agg='min')
    PMF.init_pg_action_bias(ag_H, rate_value=0.8, goal_value=0.0, verbose=False)
    from aifflor.aifflor_ddpg import SP_TD3
    ag_L = SP_TD3(nb_states=5, nb_actions=1, args=_args(),
                  lam_max=2.0, lam_min=0.5, ema_lam=True,
                  policy_delay=2, policy_noise=0.1, noise_clip=0.3,
                  cost_critic_mode='max', nstep=1, target_agg='min')
    ag_L.lam = 1.0
    ag_L.cost_limit = 0.6
    ag_L.is_training = True
    env_t = WindBatteryPG(0.5, 5, False, None,
                          physics=AifflorPhysics(nt=5, calibrated=False,
                                                 power_scale=AT.LEVEL_COEF))
    rbb = RuleBasedBESS(max_E=0.5, min_E_factor=0.0, rated_power=0.5,
                        eta_c=0.98, eta_d=0.98, time=1.0 / 60,
                        fluctuation_limit=3.0)
    cases, overall = PMF.nine_case_test_pg(ag_H, ag_L, env_t, rbb, seed=42)
    ok(len(cases) == 9, '9 个 case')
    ok(all(np.isfinite([m['r1'], m['c1'], m['r2'], m['c2'],
                        m['profit'], m['rate'], m['pmax'], m['ptur']]).all()
           for m in cases), '所有 case 的指标有限')
    ok(all(0.0 <= m['rate'] <= 1.0 and m['pmax'] >= 0.0
           and m['ptur'] <= m['pmax'] + 1e-9 for m in cases),
       'rate∈[0,1]、pmax≥0、ptur≤pmax')
    ok(set(('r1', 'c1', 'rate', 'pmax', 'ptur', 'profit_econ')) <= set(overall),
       'overall 含 testPg 诊断键')

    print('\n' + '=' * 90)
    print('G. 不变量：固定动作 = 用户口径；不引入 MATLAB')
    print('=' * 90)
    ok(np.allclose(PG_FIXED_ACTION, [0.2, 0.2, 0.2, 0.2, 0.33]),
       '固定风机行为 = [0.2, 0.2, 0.2, 0.2, 0.33]')
    ok('matlab' not in sys.modules, '导入 testPg 全链路**未**引入 matlab')

    print('\n' + '=' * 90)
    if FAIL:
        print('结果: **%d 项失败** ❌' % len(FAIL))
        for f in FAIL:
            print('   - %s' % f)
        return 1
    print('结果: 全部通过 ✅')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
