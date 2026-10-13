# -*- coding: utf-8 -*-
"""`aifflor --hsrl` 契约测试（**纯算术 + 桩环境；不需要 MATLAB / 不跑 FLORIS 训练**）。

把 `--hsrl` 分支的每一条口径变成可执行判据：

  A. 常量/量纲：退化 = 越界罚的 1/10、λ2 区间对数对称、观测刻度 = 满量程
  B. 可行域：`feasible_power` ⇔ RBB 物理边界**逐位一致**；`feasible_total` 的两条约束
     （额定功率 + SOC 能量）+ 物理一致性（把 D_max 花完恰好不越 SOC 下界）
  C. c_L 指数式 ⇔ `未命名/PER_main0.py:116-118` 的原式**逐位一致**；摊还计划 ⇒ c_L ≡ 0
  D. 符号约定（g>0 ⇔ 放电 ⇔ pb<0）与 H/L 动作裁剪 + 越界罚（含"不封顶"）
  E. L reward 分量与系数比（**不含** rulebase 波动裁剪；退化只对放电计费）
  F. H reward 去掉 bessdeg（与旧 `Cal_Rew_RB` 之差恰为 `490·max(−pb,0)·dt`）
  G. L 观测 5 维与刻度（并把"Σ/n_rem 不可观测"这一已知局限写成可执行记录）
  H. λ2 对偶区间/地板/乘性步长；warmup 摊还执行器自检（c_L≡0 + 恒在物理域内）
  I. 旧路径未被触碰（`train()` 里不含任何 hsrl 调用）
  J. 20 步测试协议端到端（桩 env + 桩 agent）：4 块 / 电价恒 300 / 开局 SOC = 0.25 /
     c_L 与 r2 的接线与手算逐位一致；摊还执行臂 ⇒ Σ放电 == g 且 c_L ≡ 0

用法
----
    "<本项目环境的 python>" agentutil\\test_aifflor_hsrl_contract.py

⚠ 本测试**不导入 floris**（只用到 `aifflor_hsrl` 与 `aifflor_myfunc`；第 I 节才 import
   `aifflor_main`，并把"没有 floris"降级为 SKIP）。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

FAIL = []


def check(cond, msg):
    print('  %s %s' % ('PASS' if cond else 'FAIL', msg))
    if not cond:
        FAIL.append(msg)
    return bool(cond)


# =============================================================================
# 桩：最小 env / agent（只实现 20 步测试用到的接口；SOC 动力学与 env.WindBattery 同式）
# =============================================================================
class _StubEnv:
    def __init__(self, E0=0.25, ptur=3.0, max_E=0.5, min_E=0.0,
                 eta_c=0.98, eta_d=0.98, dt=1.0 / 60.0):
        self.E = E0
        self.ptur = float(ptur)
        self.max_E, self.min_E = max_E, min_E
        self.eta_c, self.eta_d, self.time = eta_c, eta_d, dt
        self.price = [300.0]
        self.wind = [10.0]
        self.n_env_clip = 0

    # ---- env 接口（与 env.WindBattery 同形）----
    def ScenarioReset(self, wind_seq, price_seq):
        self.wind = [float(x) for x in np.asarray(wind_seq).reshape(-1)]
        self.price = [float(x) for x in np.asarray(price_seq).reshape(-1)]
        self.idx = 0
        self.E = 5.0                      # **复刻 env.py 的硬编码**（测试函数随后覆写）
        self.pr, self.u = self.price[0], self.wind[0]
        self.P_G_prev = self.ptur         # 标称动作下的初始 P_G
        return np.array([self.pr, self.u, self.E, self.P_G_prev])

    def PowerOutput(self, u0, act_wind):
        return self.ptur

    def ScenarioStep(self, applied_action_H, act_bat, pw_tru):
        self.idx += 1
        i = min(self.idx, len(self.price) - 1)
        self.pr, self.u = self.price[i], self.wind[i]
        pw2 = self.ptur
        pb = float(np.asarray(act_bat, dtype=float).reshape(-1)[0])
        P_C, P_D = max(0.0, pb), abs(min(pb, 0.0))
        E_temp = self.E + (P_C * self.eta_c - P_D / self.eta_d) * self.time
        if E_temp > self.max_E:
            P_C = (self.max_E - self.E) / self.eta_c / self.time
            self.E = self.max_E
            self.n_env_clip += 1
        elif E_temp < self.min_E:
            P_D = (self.E - self.min_E) * self.eta_d / self.time
            self.E = self.min_E
            self.n_env_clip += 1
        else:
            self.E = E_temp
        applied = np.array([P_C - P_D])
        P_G = float(pw_tru) - P_C + P_D
        self.P_G_prev = P_G
        return (pw2, np.array([self.pr, self.u, self.E, P_G]), applied, 0.0,
                False, False, {'P_G': P_G})


class _StubH:
    """固定动作的 H 桩：前 5 维 0.2、第 6 维（块总放电量）= g。"""

    def __init__(self, g=1.5):
        self.g = float(g)
        self.is_training = True

    def select_action(self, obs):
        return np.array([0.2] * 5 + [self.g], dtype=float)

    def reset(self, obs):
        return None


class _StubL:
    """固定 pb 的 L 桩（正=充电）。默认 −0.1 ⇒ 每步放电 0.1 MW·min。"""

    def __init__(self, pb=-0.1):
        self.pb = float(pb)
        self.is_training = True

    def select_action(self, obs):
        return np.array([self.pb], dtype=float)

    def reset(self, obs):
        return None


class _StubAmortizedL:
    """按**摊还计划**执行的 L 桩：pb = −(g−Σ)/n_rem（内部计数器复刻块内进度）。

    本桩假设 env 不裁剪自身命令（本测试的 SOC 远离上下界 ⇒ 成立），
    因此它给出的命令 = 实际施加值 ⇒ c_L 应恒为 0（"零成本动作 = 摊还计划"）。
    """

    def __init__(self, g=1.5):
        self.g = float(g)
        self.is_training = True
        self.cmds = []

    def reset(self, obs):
        self.i = 0
        self.s = 0.0

    def select_action(self, obs):
        n_rem = 5 - self.i
        need = (self.g - self.s) / float(n_rem)
        self.i += 1
        self.s += need
        self.cmds.append(-need)
        return np.array([-need], dtype=float)


def main():
    import aifflor.aifflor_hsrl as H
    import aifflor.aifflor_myfunc as AMF
    from rbb.rbb_rule_bess import RuleBasedBESS

    P_RATED, ETA_C, ETA_D, DT, BLOCK = H.P_RATED, H.ETA_C, H.ETA_D, H.DT, H.HSRL_BLOCK
    E_MAX, E_MIN = H.E_MAX, H.E_MIN
    E_MID = 0.25

    # =====================================================================
    print('\n' + '=' * 90)
    print('A. 常量 / 量纲（用户口径）')
    print('=' * 90)
    # =====================================================================
    check(abs(H.HSRL_DEG_COEF * 10.0 - H.HSRL_VIOL_COEF) < 1e-12,
          '退化系数 = 越界罚的 1/10（%.1f vs %.1f）' % (H.HSRL_DEG_COEF, H.HSRL_VIOL_COEF))
    check(H.HSRL_VIOL_COEF == AMF.PENALTY_NN_L_COEF and H.HSRL_PEN6_COEF == AMF.PENALTY_NN_H_COEF,
          '越界罚沿用既有系数（L=%.0f, H6=%.0f）' % (H.HSRL_VIOL_COEF, H.HSRL_PEN6_COEF))
    check(abs(H.HSRL_L_LAM_INIT - np.sqrt(H.HSRL_L_LAM_MIN * H.HSRL_L_LAM_MAX)) < 1e-12,
          'λ2 区间对数对称：初值 %.0f = sqrt(%.0f·%.0f)'
          % (H.HSRL_L_LAM_INIT, H.HSRL_L_LAM_MIN, H.HSRL_L_LAM_MAX))
    check((H.HSRL_L_LAM_MIN, H.HSRL_L_LAM_MAX) == (2.0, 50.0),
          'λ2 ∈ [2, 50]（随 c_L 无量纲口径重标）')
    check(abs(H.HSRL_G_FULL - BLOCK * P_RATED) < 1e-12 and abs(H.HSRL_G_FULL - 2.5) < 1e-12,
          '满量程 g = δ·P_rated = %.2f MW·min' % H.HSRL_G_FULL)
    check(H.OBS_G_SCALE == H.HSRL_G_FULL and H.OBS_E_SCALE == E_MAX,
          '观测刻度：g/%.1f、SOC/%.1f' % (H.OBS_G_SCALE, H.OBS_E_SCALE))
    check(abs(H.HSRL_WARMUP_G_SCALE - 2.0 * H.HSRL_G_FULL) < 1e-12,
          'warmup 信号定标 = 2·满量程（sig ±0.5 → g ∓2.5）')

    # =====================================================================
    print('\n' + '=' * 90)
    print('B. 可行域：每步物理域 ⇔ RBB 物理边界；块级总放电域的两条约束')
    print('=' * 90)
    # =====================================================================
    rbb = RuleBasedBESS(max_E=E_MAX, min_E_factor=0.0, rated_power=P_RATED,
                        eta_c=ETA_C, eta_d=ETA_D, time=DT, fluctuation_limit=float('inf'))
    rg = np.random.default_rng(20261014)
    n_bit, bad_bit = 0, 0
    bad_cons, bad_sim = 0, 0
    for _ in range(500):
        E = float(rg.uniform(E_MIN, E_MAX))
        lo, hi = H.feasible_power(E)
        _pb, info = rbb.compute_pbrule(pb_original=0.0, ptur=0.0, pg_prev=0.0, E=E)
        if lo == info['pbphysicsmin'] and hi == info['pbphysicsmax']:
            n_bit += 1
        else:
            bad_bit += 1
        c_max, d_max = H.feasible_total(E)
        # 约束①: 额定功率;  约束②: SOC 能量
        if not (d_max <= BLOCK * P_RATED + 1e-12
                and c_max <= BLOCK * P_RATED + 1e-12
                and d_max <= (E - E_MIN) * (1.0 / DT) * ETA_D + 1e-12
                and c_max <= (E_MAX - E) * (1.0 / DT) / ETA_C + 1e-12):
            bad_cons += 1
        # 物理一致性: 以恒定功率把 D_max 花完, SOC 恰好不低于下界
        pb = -d_max / BLOCK
        e = E
        for _k in range(BLOCK):
            e = e + (max(pb, 0.0) * ETA_C - max(-pb, 0.0) / ETA_D) * DT
        if e < E_MIN - 1e-9:
            bad_sim += 1
    check(bad_bit == 0, 'feasible_power ⇔ RBB pbphysicsmin/max 逐位一致（%d/500）' % n_bit)
    check(bad_cons == 0, 'feasible_total 满足额定功率与 SOC 能量两条约束（500 组）')
    check(bad_sim == 0, 'D_max 恰好在 SOC 下界收口（把 D_max 花完不越界）')
    check(H.feasible_total(E_MID) == (BLOCK * P_RATED, BLOCK * P_RATED),
          'E=0.25（中点）⇒ 可行域 = ±%.1f MW·min（额定功率受限）' % (BLOCK * P_RATED))
    e_near = E_MAX - 0.01
    c_near, d_near = H.feasible_total(e_near)
    check(abs(c_near - (E_MAX - e_near) * (1.0 / DT) / ETA_C) < 1e-12 and d_near == BLOCK * P_RATED,
          'E 贴近上界(%.2f) ⇒ 充电上限 %.3f = 剩余容量口径（≠ 乘 δ）' % (e_near, c_near))
    check(H.feasible_total(E_MIN) == (BLOCK * P_RATED, 0.0),
          'E=0 ⇒ 放电上限 0（无电可放）')

    # =====================================================================
    print('\n' + '=' * 90)
    print('C. c_L 指数式：与 未命名/PER_main0.py 原式逐位一致；摊还计划 ⇒ c_L ≡ 0')
    print('=' * 90)
    # =====================================================================
    bad_c, n_c = 0, 0
    rg2 = np.random.default_rng(7)
    for _ in range(2000):
        tt = int(rg2.integers(0, 5))
        n_rem = 5 - tt                                   # = δ − t_L
        g = float(rg2.uniform(-2.5, 2.5))
        prev = list(rg2.uniform(-0.5, 0.5, tt))          # 本块**已执行**的实际放电量
        a_L = float(rg2.uniform(-0.5, 0.5))              # 本步实际放电量
        # 原式（未命名/PER_main0.py:118）:
        #     c2 = -np.exp(-abs((signal - sum(pb_l[:-1]))/(5 - tt) - pb)) + 1
        # 这里把 (signal, pb) 换成同一代数量 (g, a_L) —— 断言的是**代数形式逐位相同**
        ref = -np.exp(-abs((g - sum(prev)) / (5 - tt) - a_L)) + 1.0
        got = H.lower_cost_hsrl(g, sum(prev), n_rem, a_L)
        n_c += 1
        if abs(got - ref) > 1e-15:
            bad_c += 1
    check(bad_c == 0, 'c_L = 1 − exp(−|(g−Σ)/(δ−t_L) − a_L|) 与原式逐位一致（%d 组）' % n_c)
    check(H.lower_cost_hsrl(2.0, 0.5, 3, H.plan_need(2.0, 0.5, 3)) == 0.0,
          '按摊还计划执行 ⇒ c_L ≡ 0（精确 0）')
    check(abs(H.lower_cost_hsrl(0.0, 0.0, 5, 0.5) - (1.0 - np.exp(-0.5))) < 1e-15,
          '偏离 0.5 ⇒ c_L = 1 − e^{−0.5} = %.6f（饱和型，∈[0,1)）'
          % (1.0 - np.exp(-0.5)))
    check(H.plan_need(2.0, 0.0, 1) == 2.0,
          'n_rem = 1（最后一步）⇒ 参照 = 本块剩余总量（必须补齐）')
    check(H.lower_cost_hsrl(0.0, 0.0, 5, 5.0) < 1.0,
          'c_L 有界（|dev|→∞ 也只趋近 1，不是线性无界罚）')

    # =====================================================================
    print('\n' + '=' * 90)
    print('D. 符号约定 + 动作裁剪/越界罚（H 第 6 维 & L）')
    print('=' * 90)
    # =====================================================================
    c_max, d_max = H.feasible_total(E_MID)
    # 第 6 维: 放电为正, 越界线性罚 400/MW·min, 不封顶
    a_in = np.array([0.2, 0.2, 0.2, 0.2, 0.2, d_max - 0.1])
    ap, p5, p6 = H.check_action_H_hsrl(a_in, E_MID)
    check(p5 == 0.0 and p6 == 0.0 and np.allclose(ap, a_in) and ap[5] == a_in[5],
          '域内动作 ⇒ 零罚且逐位不变（含 g 不夹）')
    ap, p5, p6 = H.check_action_H_hsrl(np.array([0.6, -0.1, 0.2, 0.2, 0.2, d_max + 0.4]), E_MID)
    check(abs(p5 - AMF.PENALTY_NN_H_COEF * (0.1 + 0.1)) < 1e-12 and ap[0] == 0.5 and ap[1] == 0.0,
          '前 5 维沿用旧口径（clip [0,0.5] + 400/单位）')
    check(abs(p6 - H.HSRL_PEN6_COEF * 0.4) < 1e-12 and ap[5] == d_max,
          'g 越上界 0.4 ⇒ 罚 %.1f（400/MW·min）且夹到 D_max' % (H.HSRL_PEN6_COEF * 0.4))
    ap, p5, p6 = H.check_action_H_hsrl(np.array([0.2] * 5 + [-c_max - 0.3]), E_MID)
    check(abs(p6 - H.HSRL_PEN6_COEF * 0.3) < 1e-12 and ap[5] == -c_max,
          'g 越下界 0.3 ⇒ 罚 %.1f 且夹到 −C_max' % (H.HSRL_PEN6_COEF * 0.3))
    _ap, _p5, p6_far = H.check_action_H_hsrl(np.array([0.2] * 5 + [d_max + 100.0]), E_MID)
    check(abs(p6_far - H.HSRL_PEN6_COEF * 100.0) < 1e-9,
          'g 越界 100 ⇒ 罚 %.0f（**不封顶**，保持界外恒定恢复力）' % p6_far)
    # 符号: g>0 ⇒ 命令放电 ⇒ pb<0
    check(H.discharge_of(-0.3) == 0.3 and H.discharge_of(0.3) == -0.3,
          'a_L = −pb（放电为正；pb 正=充电，与 env 同号）')
    check(H.amortized_rule_action(1.0, 0.0, 5, E_MID) < 0.0,
          'g>0（要放电）⇒ 摊还命令 pb<0（放电）')
    # L: 只按物理域裁剪
    pb_lo, pb_hi = H.feasible_power(E_MID)
    ap, pen, viol = H.check_action_L_hsrl(np.array([0.9]), E_MID)
    check(ap[0] == pb_hi and abs(viol - 0.4) < 1e-15
          and abs(pen - H.HSRL_VIOL_COEF * 0.4) < 1e-12,
          'L 动作越上界 0.4 ⇒ viol=0.4, 罚 %.1f，夹到 +%.2f' % (pen, pb_hi))
    ap, pen, viol = H.check_action_L_hsrl(np.array([-0.7]), E_MID)
    check(ap[0] == pb_lo and abs(viol - 0.2) < 1e-15,
          'L 动作越下界 0.2 ⇒ viol=0.2，夹到 %.2f' % pb_lo)
    ap, pen, viol = H.check_action_L_hsrl(np.array([-0.25]), E_MID)
    check(pen == 0.0 and ap[0] == -0.25, 'L 动作在物理域内 ⇒ 零罚且不夹')

    # =====================================================================
    print('\n' + '=' * 90)
    print('E. L reward：r2 = −200·越界 − 20·放电（无 rulebase 波动裁剪）')
    print('=' * 90)
    # =====================================================================
    r2 = H.lower_reward_hsrl(0.4, 0.5)
    check(abs(r2 - (-(H.HSRL_VIOL_COEF * 0.4 + H.HSRL_DEG_COEF * 0.5))) < 1e-12,
          'r2 = −(200·0.4 + 20·0.5) = %.2f' % r2)
    r2_charge = H.lower_reward_hsrl(0.0, -0.5)
    check(r2_charge == 0.0,
          '退化只对**放电**计费（充电 a_L<0 ⇒ 退化项 = 0，r2 = 0）')
    check(abs(H.lower_reward_hsrl(1.0, 0.0) + H.HSRL_VIOL_COEF) < 1e-12,
          '越界罚 200 元/MW（线性，随越界量）')
    # 与"含 rulebase 波动裁剪"的旧 r2 对照: 旧 r2 = −1·min(clip_amount, 20)
    _clip_incl_fluct = abs(-0.9 - pb_hi)          # 命令 vs 施加(含窗口段) 才是旧口径的被罚量
    check(abs(AMF.lower_reward(_clip_incl_fluct) - (-_clip_incl_fluct)) < 1e-12
          and abs(H.lower_reward_hsrl(0.4, 0.0)) > 0,
          'HSRL 的 r2 **不再**等于 −clip_amount（旧 rulebase 波动裁剪罚已移除）')

    # =====================================================================
    print('\n' + '=' * 90)
    print('F. H reward：去掉 bessdeg（与旧 Cal_Rew_RB 的差恰为 490·max(−pb,0)·dt）')
    print('=' * 90)
    # =====================================================================
    obs = np.array([980.0, 10.0, E_MID, 2.0])
    obs2 = np.array([980.0, 10.0, E_MID - 0.002, 2.9])
    act6 = np.array([0.2] * 5 + [0.0])       # 旧 Cal_Rew_RB 会读 action_rbb[5]（HSRL 版不读）
    for pb in (-0.4, 0.0, 0.3):
        info = {'pbrule': pb, 'unfeasible_gap': 0.0, 'clip_amount': 0.0,
                'is_infeasible': False, 'inevitable_violation': 0.0}
        r_new, c_new, d_new = H.cal_rew_rb_hsrl(obs, obs2, act6, info,
                                                penalty_NN=0.0, time=DT, ptur=2.9)
        r_old, c_old, d_old = AMF.Cal_Rew_RB(obs, obs2, act6, info,
                                             penalty_NN_H=0.0, time=DT, ptur=2.9)
        deg = 490.0 * max(-pb, 0.0) * DT
        check(abs((r_new - r_old) - deg) < 1e-12,
              'pb=%+.2f: r1(HSRL) − r1(旧) = 490·max(−pb,0)·dt = %.6f' % (pb, deg))
        check(d_new['degradation'] == 0.0 and abs(d_new['degradation_ref490'] - deg) < 1e-12,
              'pb=%+.2f: 奖励内 degradation ≡ 0，诊断列 degradation_ref490 = %.6f' % (pb, deg))
    check(abs(H.cal_rew_rb_hsrl(obs, obs2, act6, {'pbrule': 0.3, 'unfeasible_gap': 0.0,
                                                  'clip_amount': 0.0, 'is_infeasible': False,
                                                  'inevitable_violation': 0.0},
                                penalty_NN=100.0, time=DT, ptur=2.9)[0]
              - (980.0 * 2.9 * DT - 980.0 * 0.3 * DT - 100.0)) < 1e-12,
          'r1 = pr·P_G·dt − penalty_NN（P_G = ptur − pbrule）')

    # =====================================================================
    print('\n' + '=' * 90)
    print('G. L 观测：5 维 / 刻度 / （已知局限）Σ 与 n_rem 不可观测')
    print('=' * 90)
    # =====================================================================
    o = H.build_L_obs_hsrl(10.0, 300.0, 0.25, 1.5, 2.5)
    check(o.shape == (5,), '观测维数 = 5')
    check(np.allclose(o, [1.0, 0.3, 0.5, 0.3, 1.0]),
          '观测 = [u/10, pr/1000, SOC/0.5, P_G/5, g/2.5] = %s' % np.round(o, 4).tolist())
    o2 = H.build_L_obs_hsrl(10.0, 300.0, 0.25, 1.5, -2.5)
    check(o2[4] == -1.0, 'g 满量程负向 ⇒ 第 5 维 = −1')
    o_a = H.build_L_obs_hsrl(10.0, 300.0, 0.25, 1.5, 1.0)
    o_b = H.build_L_obs_hsrl(10.0, 300.0, 0.25, 1.5, 1.0)
    check(np.array_equal(o_a, o_b),
          '（局限记录）观测不含 Σ/n_rem ⇒ 不同块内进度给出**同一个**观测'
          '（c_L 的参照量对策略不可观测；刻意保留的原方法形态）')

    # =====================================================================
    print('\n' + '=' * 90)
    print('H. λ2 对偶 + warmup 摊还执行器自检')
    print('=' * 90)
    # =====================================================================

    class _A:
        def __init__(self, ema, lam):
            self.ema_cost = ema
            self.lam = lam
    a = _A(1.0, 10.0)
    H.dual_update_L_hsrl(a, 0.5)                       # ratio = 2 ⇒ λ 上升
    check(a.lam > 10.0, 'EMA/d_block = 2 ⇒ λ2 上升（乘性对数步长）')
    a = _A(0.1, 10.0)
    H.dual_update_L_hsrl(a, 0.5)                       # ratio = 0.2 ⇒ λ 下降
    check(a.lam < 10.0, 'EMA/d_block = 0.2 ⇒ λ2 回落')
    a = _A(1e9, 10.0)
    H.dual_update_L_hsrl(a, 0.5)
    check(a.lam == H.HSRL_L_LAM_MAX, 'λ2 封顶 = %.0f' % H.HSRL_L_LAM_MAX)
    a = _A(0.0, H.HSRL_L_LAM_MIN)
    H.dual_update_L_hsrl(a, 0.5)
    check(a.lam == H.HSRL_L_LAM_MIN,
          'λ2 触底 = %.0f（EMA=0 时再压也不破下界）' % H.HSRL_L_LAM_MIN)
    a = _A(0.5, 10.0)
    H.dual_update_L_hsrl(a, 0.5)                       # ratio = 1 ⇒ 不动点
    check(abs(a.lam - 10.0) < 1e-12, 'EMA == d_block ⇒ λ2 不动（不动点）')
    chk = H.check_amortized_baseline()
    check(chk['fail'] == 0 and chk['max_c_unclipped'] == 0.0,
          '摊还执行器自检: %d 例全过（未被裁剪时 c_L ≡ 0；被裁剪 %d 例）'
          % (chk['n_checked'], chk['n_clipped']))

    # =====================================================================
    print('\n' + '=' * 90)
    print('I. 旧路径未被触碰（train() 里没有任何 hsrl 调用）')
    print('=' * 90)
    # =====================================================================
    _M = None
    try:
        import aifflor.aifflor_main as _M
    except Exception as e:                                   # pragma: no cover
        print('  SKIP I 节（无法 import aifflor_main: %s）' % e)
    if _M is not None:
        import td3per.td3per_myfunc as TMF
        names = tuple(_M.train.__code__.co_names) + tuple(_M.train.__code__.co_varnames)
        check(not any('hsrl' in str(n).lower() for n in names),
              'train()（旧口径）里不含 hsrl* 名字 ⇒ 旧路径逐位不变')
        check(_M.CheckAction is TMF.CheckAction,
              'train() 仍用共享的 CheckAction（未被 HSRL 版遮蔽）')
        check(callable(getattr(_M, 'train_hsrl', None)),
              'train_hsrl() 存在（HSRL 分支并列实现）')
    # 已有契约（旧口径）也要在: 旧口径的核心函数仍是共享真源
    check(AMF.lower_reward is not H.lower_reward_hsrl,
          'HSRL 的 r2 是新函数（旧 lower_reward 未被改动）')

    # =====================================================================
    print('\n' + '=' * 90)
    print('J. 20 步测试协议端到端（桩 env + 桩 agent）')
    print('=' * 90)
    # =====================================================================
    pool = np.zeros((100, 2))
    pool[:, 0] = 9.0
    pool[:, 1] = 11.0
    G_TGT = 1.5

    def _run(stub_L):
        env = _StubEnv(E0=E_MID)
        aH, aL = _StubH(G_TGT), stub_L
        per_blk, overall, meta = H.twenty_step_test_hsrl(
            aH, aL, env, rbb, pool, seed=1234, phase_mode=0)
        return env, per_blk, overall, meta

    env, per_blk, overall, meta = _run(_StubL(pb=-0.1))
    check(len(per_blk) == 4 and meta['n_steps'] == 20,
          '20 步 = 4 个块（返回 %d 个块）' % len(per_blk))
    check(meta['price'] == 300.0 and all(abs(p - 300.0) < 1e-12 for p in env.price[:20]),
          '电价恒 300（20 步都是同一个价）')
    check(abs(meta['soc_start'] - 0.25) < 1e-12 and abs(per_blk[0]['soc_start'] - 0.25) < 1e-12,
          '开局 SOC = 0.25 MWh（覆写了 env.ScenarioReset 的硬编码 5.0）')
    check(0 <= meta['wind_col'] < pool.shape[1] and 0 <= meta['wind_start'] <= 80
          and len(meta['wind_seq']) == 20,
          '风速段: 随机列 %d + 起点 %d 取 20 步' % (meta['wind_col'], meta['wind_start']))
    # 手算 c_L（每块 5 步: Σ_prev = 0.1k, n_rem = 5−k, a_L = 0.1）
    c_expected = sum(1.0 - np.exp(-abs((G_TGT - 0.1 * k) / (5 - k) - 0.1)) for k in range(5))
    check(abs(per_blk[0]['c2'] - c_expected) < 1e-12,
          '块 c_L 之和 = 手算 %.9f（逐块 Σ 递推与公式接线一致）' % c_expected)
    check(abs(overall['c2'] - 4 * c_expected) < 1e-12,
          '总 c_L = 4 × %.9f = %.9f' % (c_expected, 4 * c_expected))
    check(abs(per_blk[0]['r2'] - (-H.HSRL_DEG_COEF * 0.1 * 5)) < 1e-12,
          '块 r2 = −20·0.1·5 = %.2f（无越界 ⇒ 只有退化项）' % per_blk[0]['r2'])
    check(abs(per_blk[0]['sum_disch'] - 0.5) < 1e-12 and abs(per_blk[0]['g'] - G_TGT) < 1e-12,
          '块内 Σ 放电 = 0.5（5 步 × 0.1），g = %.1f（未完成计划 ⇒ c_L>0）' % G_TGT)
    check(overall['pf_viol'] == 0 and overall['cvm'] == 1,
          '恒定风机出力 ⇒ P_F 只在**首步**出现 0.1 MW（复位基线的 pb=0）: '
          'pf_viol = 0, cvm = 1（实得 %d/%d）' % (overall['pf_viol'], overall['cvm']))
    check(abs(overall['dsoc'] - (-20 * 0.1 / 60.0 / ETA_D)) < 1e-9,
          'ΔSOC = −20 步 × 0.1/(60·η_d) = %.6f MWh' % overall['dsoc'])
    check(abs(overall['profit_econ'] - (overall['profit'] + overall['dsoc'] * 580.0)) < 1e-9,
          '存货结算 profit_econ = profit + ΔSOC·580（与 9-case 同口径）')

    env2, per_blk2, overall2, meta2 = _run(_StubAmortizedL(G_TGT))
    check(abs(overall2['c2']) < 1e-15,
          '摊还执行臂 ⇒ c_L ≡ 0（"零成本动作 = 摊还计划"端到端成立；实得 %.3e）'
          % overall2['c2'])
    check(all(abs(m['sum_disch'] - G_TGT) < 1e-12 for m in per_blk2),
          '摊还执行臂 ⇒ 每块 Σ放电 == g（计划被精确兑现）')
    check(abs(per_blk2[0]['r2'] - (-H.HSRL_DEG_COEF * G_TGT)) < 1e-12,
          '摊还执行臂的块 r2 = −20·g = %.2f（只有退化成本）' % per_blk2[0]['r2'])
    check(env2.n_env_clip == 0 and env.n_env_clip == 0,
          '两条臂都未触发 env 的 SOC 边界修正（在物理域内 ⇒ 施加值 == 命令值）')

    print('\n' + '=' * 90)
    if FAIL:
        print('结果: **%d 项失败**' % len(FAIL))
        for f in FAIL:
            print('   - %s' % f)
        return 1
    print('结果: 全部通过 ✓')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
