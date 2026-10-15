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
    """固定动作的 H 桩：前 5 维 `a_wt`、第 6 维（块总放电量）= g。"""

    def __init__(self, g=1.5, a_wt=0.2):
        self.g = float(g)
        self.a_wt = float(a_wt)
        self.is_training = True

    def select_action(self, obs):
        return np.array([self.a_wt] * 5 + [self.g], dtype=float)

    def reset(self, obs):
        return None


class _StubL:
    """固定 pb 的 L 桩（正=充电）。默认 −0.1 ⇒ 每步放电 0.1 MW·min。"""

    def __init__(self, pb=-0.1):
        self.pb = float(pb)
        self.is_training = True
        self.obs = []

    def select_action(self, obs):
        self.obs.append(np.asarray(obs, dtype=float).copy())
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
        self.obs = []

    def reset(self, obs):
        self.i = 0
        self.s = 0.0

    def select_action(self, obs):
        self.obs.append(np.asarray(obs, dtype=float).copy())
        n_rem = 5 - self.i
        need = (self.g - self.s) / float(n_rem)
        self.i += 1
        self.s += need
        self.cmds.append(-need)
        return np.array([-need], dtype=float)


def main():
    import aifflor.aifflor_hsrl as H
    import aifflor.aifflor_myfunc as AMF
    import hsrl_cost as hc
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
    print('G. L 观测：6 维 / 刻度 / 第 6 维 = c_L 的参照点 / （残余局限）n_rem 不可观测')
    print('=' * 90)
    # =====================================================================
    o = H.build_L_obs_hsrl(10.0, 300.0, 0.25, 1.5, 2.5, 0.5)
    check(o.shape == (6,), '观测维数 = 6')
    check(np.allclose(o, [1.0, 0.3, 0.5, 0.3, 1.0, 0.2]),
          '观测 = [u/10, pr/1000, SOC/0.5, P_G/5, g/2.5, need/2.5] = %s'
          % np.round(o, 4).tolist())
    check(abs(o[5] - 0.5 / H.OBS_NEED_SCALE) < 1e-15 and H.OBS_NEED_SCALE == H.OBS_G_SCALE,
          '第 6 维 = need/%.1f（与 g 同刻度）' % H.OBS_NEED_SCALE)
    o2 = H.build_L_obs_hsrl(10.0, 300.0, 0.25, 1.5, -2.5, -2.5)
    check(o2[4] == -1.0 and o2[5] == -1.0, 'g/need 满量程负向 ⇒ 第 5/6 维 = −1')
    o3 = H.build_L_obs_hsrl(10.0, 300.0, 0.25, 1.5, 2.5, -0.5)
    check(o3[4] == 1.0 and abs(o3[5] + 0.2) < 1e-15,
          'need 可正可负（这里 need=−0.5 ⇒ 第 6 维 = −0.2）⇒ 该维不是 g 的副本')
    # 参照点必须与 c_L 用的是**同一个量**（零成本动作 = 令 a_L = obs[5]·2.5）
    _g, _s, _nr = 1.7, 0.4, 3
    _need = H.plan_need(_g, _s, _nr)
    _o = H.build_L_obs_hsrl(10.0, 300.0, 0.25, 1.5, _g, _need)
    check(abs(_o[5] * H.OBS_NEED_SCALE - _need) < 1e-15
          and H.lower_cost_hsrl(_g, _s, _nr, _need) == 0.0,
          '观测第 6 维 × 刻度 == plan_need，且 a_L 取该值时 c_L ≡ 0（参照点接线正确）')
    # 残余局限: 由 (g, need) 无法反解 (Σ, n_rem) ⇒ 下一状态的参照点仍不可预测
    _g, _need = 1.5, 0.3
    _pairs = [(_g - _need * _nr, _nr) for _nr in (5, 3)]   # (Σ, n_rem): need 完全相同
    _obs_a = H.build_L_obs_hsrl(10.0, 300.0, 0.25, 1.5, _g, _need)
    _obs_b = H.build_L_obs_hsrl(10.0, 300.0, 0.25, 1.5, _g, _need)
    _next_a = H.plan_need(_g, _pairs[0][0] + 0.5, _pairs[0][1] - 1)
    _next_b = H.plan_need(_g, _pairs[1][0] + 0.5, _pairs[1][1] - 1)
    check(np.array_equal(_obs_a, _obs_b) and abs(_next_a - _next_b) > 1e-12,
          '（残余局限记录）n_rem 仍不可观测: (Σ,n_rem) = (%.2f,5)/(%.2f,3) 给出**同一观测**，'
          '但下一参照点 = %.3f vs %.3f（本步代价只看本步 need/a_L ⇒ 本步已给全）'
          % (_pairs[0][0], _pairs[1][0], _next_a, _next_b))

    # =====================================================================
    print('\n' + '=' * 90)
    print('H. λ2/λ1 对偶：Q 版与 EMA 版、区间/地板/乘性步长、warmup 摊还执行器自检')
    print('=' * 90)
    # =====================================================================

    class _A:
        def __init__(self, ema, lam):
            self.ema_cost = ema
            self.lam = lam
    # ---- Q 分子（默认）：乘性、单调、封顶/触底 ----
    a = _A(1.0, 10.0)
    H.dual_update_L_hsrl(a, 0.5, q_rate=1.0)            # ratio = 2 ⇒ λ 上升
    check(a.lam > 10.0, 'Q 分子: rate/d_block = 2 ⇒ λ2 上升（乘性对数步长）')
    a = _A(1.0, 10.0)
    H.dual_update_L_hsrl(a, 0.5, q_rate=0.1)            # ratio = 0.2 ⇒ λ 回落
    check(a.lam < 10.0, 'Q 分子: rate/d_block = 0.2 ⇒ λ2 回落')
    a = _A(1.0, 10.0)
    H.dual_update_L_hsrl(a, 0.5, q_rate=0.5)            # ratio = 1 ⇒ 不动点
    check(abs(a.lam - 10.0) < 1e-12, 'Q 分子: rate == d_block ⇒ λ2 不动（不动点）')
    a = _A(1.0, 10.0)
    H.dual_update_L_hsrl(a, 0.5, q_rate=1e9)
    check(a.lam == H.HSRL_L_LAM_MAX, 'λ2 封顶 = %.0f' % H.HSRL_L_LAM_MAX)
    a = _A(1.0, H.HSRL_L_LAM_MIN)
    H.dual_update_L_hsrl(a, 0.5, q_rate=0.0)
    check(a.lam == H.HSRL_L_LAM_MIN,
          'λ2 触底 = %.0f（rate=0 时再压也不破下界）' % H.HSRL_L_LAM_MIN)
    # ---- EMA 分子（A/B 对照臂）与回退 ----
    a = _A(1.0, 10.0)
    H.dual_update_L_hsrl(a, 0.5, src='ema', ema=1.0)
    check(a.lam > 10.0, 'EMA 分子（--hsrl_dual_src ema）仍可用（A/B 对照）')
    a = _A(1.0, 10.0)
    H.dual_update_L_hsrl(a, 0.5, q_rate=None, ema=1.0)
    check(a.lam > 10.0, 'q_rate 缺失（critic 未建立）⇒ **回退 EMA**，对偶不停摆')
    a = _A(None, 10.0)
    check(H.dual_update_L_hsrl(a, 0.5, q_rate=None, ema=None) is None and a.lam == 10.0,
          'Q 与 EMA 都缺 ⇒ 不动 λ（返回 None）')

    # ---- Q → "每块成本速率" 的折算（duck-typed agent，纯算术）----
    import torch as _torch

    class _StubQ:
        def __init__(self, q1, q2, gamma=0.95, sc=100.0, vs=1000.0):
            self.device = 'cpu'
            self.discount = float(gamma)
            self.cost_scale = float(sc)
            self.value_scale = float(vs)
            self.seen = []

            class _Act:
                def __call__(self, s):
                    return _torch.zeros(s.shape[0], 1)
            self.actor = _Act()

            def _q(v):
                def f(sa):
                    self.seen.append(np.asarray(sa[1].detach().cpu()).reshape(-1).copy())
                    return _torch.full((1, 1), float(v))
                return f
            self.critic_c = _q(q1)
            self.critic_c2 = _q(q2)

    _obs5 = H.build_L_obs_hsrl(10.0, 300.0, 0.25, 1.5, 0.1, 0.02)
    _stubq = _StubQ(q1=2.0, q2=4.0, gamma=0.95, sc=100.0)
    _rate, _info = H.q_cost_rate(_stubq, _obs5, horizon='infinite', agg='mean',
                                return_info=True)
    check(abs(_rate - (1.0 - 0.95) * 3.0 * 100.0) < 1e-9,
          'H 视界: rate = (1−γ)·mean(q1,q2)·s_c = %.3f（γ=0.95, q̄=3, s_c=100）' % _rate)
    check(abs(_info['q_raw'] - 300.0) < 1e-9,
          '口径还原: 原始 Q = q̄·s_c = %.1f（critic 吃的是 c/s_c）' % _info['q_raw'])
    _rmin, _ = H.q_cost_rate(_stubq, _obs5, horizon='infinite', agg='min', return_info=True)
    _rmax, _ = H.q_cost_rate(_stubq, _obs5, horizon='infinite', agg='max', return_info=True)
    check(_rmin < _rate < _rmax,
          'twin 聚合三档有序: min %.1f < mean %.1f < max %.1f' % (_rmin, _rate, _rmax))
    _K = (1.0 - 0.98 ** 5) / (5 * (1.0 - 0.98))
    _stubq98 = _StubQ(q1=0.5, q2=0.5, gamma=0.98, sc=1.0)
    _rateL, _ = H.q_cost_rate(_stubq98, _obs5, horizon='finite', agg='mean',
                              return_info=True)
    check(abs(_rateL - 0.5 / _K) < 1e-9,
          'L 视界（5 步有限）: rate = Q/K = %.6f, K = %.4f' % (_rateL, _K))
    check(abs(H.cost_rate_from_q(300.0, 0.95, horizon='infinite') - 15.0) < 1e-9,
          '折算固定点: Q = d/(1−γ) ⇒ rate = d（Q=300, γ=0.95 ⇒ 15 = d_block_H）')
    # 传了 a_t ⇒ 必须用**实际执行的动作**评估（否则与 critic 训练分布不一致，系统性低估）
    _r_given, _i_given = H.q_cost_rate(_stubq, _obs5, a_t=np.array([0.77]),
                                       horizon='infinite', agg='mean', return_info=True)
    check(_i_given['action_given'] and abs(_stubq.seen[-1][0] - 0.77) < 1e-6,
          'a_t 给了 ⇒ critic 用实际执行的动作评估（实测收到 %.3f）' % _stubq.seen[-1][0])
    _stubq.seen = []
    H.q_cost_rate(_stubq, _obs5, horizon='infinite', agg='mean')
    check(abs(_stubq.seen[-1][0]) < 1e-9,
          '未给 a_t ⇒ 退回 actor 的确定性动作（本桩输出 0）')

    # ---- S1: 近端 N 条批均值 × **当前策略**动作（`--hsrl_dual_q_src recent_batch`）----
    from PER_buffer import PrioritizedReplayBuffer
    _np = np

    class _StubRecent:
        """duck-typed agent：真回放缓冲（含环绕）+ 与存储动作**不同**的 actor 输出。"""

        def __init__(self, n_app=12, cap=8, a_out=0.9, gamma=0.95, sc=100.0):
            self.device = 'cpu'
            self.discount = float(gamma)
            self.cost_scale = float(sc)
            self.value_scale = 1000.0
            self.memory = PrioritizedReplayBuffer(cap, 0.6)
            for i in range(n_app):
                self.memory.append(_np.array([float(i), float(i) + 100.0], dtype=_np.float32),
                                   _np.array([-3.0], dtype=_np.float32),   # 存储的是**旧行为动作**
                                   float(i), float(i) * 2.0,
                                   _np.array([float(i) + 1, float(i) + 101.0], dtype=_np.float32),
                                   0.0)

            class _Act:
                def __call__(self, s):
                    return _torch.full((s.shape[0], 1), float(a_out))
            self.actor = _Act()
            # critic 只依赖动作（10·a）⇒ 用它反查"到底喂了哪个动作"
            self.critic_c = lambda sa: 10.0 * sa[1]
            self.critic_c2 = lambda sa: 10.0 * sa[1]

    _sr = _StubRecent(n_app=12, cap=8, a_out=0.9, gamma=0.95, sc=100.0)
    _s_states, _s_acts, _s_costs = H.recent_states(_sr, 4)
    check(_s_states[:, 0].tolist() == [8.0, 9.0, 10.0, 11.0]
          and _s_costs.tolist() == [16.0, 18.0, 20.0, 22.0],
          '近端读取（环形已环绕）: 最近 4 条 = 第 8~11 次写入，按写入顺序')
    _s_all, _, _ = H.recent_states(_sr, 100)
    check(_s_all.shape[0] == 8 and _s_all[:, 0].tolist() == [4.0, 5, 6, 7, 8, 9, 10, 11],
          '请求条数 > 容量时只给容量内、仍按写入顺序（n=%d）' % _s_all.shape[0])
    _rate_b, _info_b = H.policy_batch_cost_rate(_sr, n=4, horizon='infinite', agg='mean',
                                                return_info=True)
    check(abs(_info_b['q_agg'] - 9.0) < 1e-6,
          'critic 收到的是 **actor 的动作 0.9**（q = 10·0.9 = 9），不是缓冲里存的 −3.0')
    check(abs(_rate_b - (1.0 - 0.95) * 9.0 * 100.0) < 1e-9,
          'S1 折算（H 视界）: rate = (1−γ)·mean_i Q(s_i,π(s_i))·s_c = %.3f' % _rate_b)
    _srl = _StubRecent(gamma=0.98, sc=1.0)
    _rate_bl, _info_bl = H.policy_batch_cost_rate(_srl, n=256, horizon='finite', agg='mean',
                                                  return_info=True)
    _K98 = (1.0 - 0.98 ** 5) / (5 * (1.0 - 0.98))
    check(abs(_rate_bl - 9.0 / _K98) < 1e-9 and _info_bl['n_batch'] == 8,
          'S1 折算（L 视界）+ 缓冲不足时用满 8 条: rate = Q/K = %.4f' % _rate_bl)
    check(H.HSRL_DUAL_Q_SRC == 'update_batch' and H.HSRL_DUAL_RECENT_N == 256,
          '默认口径: λ1 用 update_batch（本块 update 的 Qc 批均值）/ λ2 用 recent_batch N=256')

    # ---- update_batch: 直接用 update_stats 里的 q_c_mean / q_c2_mean（**已是原始口径**）----
    class _AUpd:
        def __init__(self, stats, gamma=0.95, sc=100.0):
            self.device = 'cpu'
            self.discount = float(gamma)
            self.cost_scale = float(sc)
            self.value_scale = 1000.0
            self.update_stats = stats

    _st4 = [{'q_c_mean': 300.0, 'q_c2_mean': 320.0}, {'q_c_mean': 340.0, 'q_c2_mean': 360.0}]
    _au = _AUpd(_st4, gamma=0.95, sc=100.0)
    _ru, _iu = H.update_batch_cost_rate(_au, horizon='infinite', agg='mean', return_info=True)
    _q_exp = 0.5 * (0.5 * (300 + 320) + 0.5 * (340 + 360))          # = 330（先 twin 均值再批均值）
    check(abs(_iu['q_raw'] - _q_exp) < 1e-9 and abs(_ru - (1 - 0.95) * _q_exp) < 1e-9,
          'update_batch（H 视界）: q_raw = %.1f（**不再乘 s_c**，因为它已是原始口径）⇒ rate = %.3f'
          % (_iu['q_raw'], _ru))
    check(abs(_iu['q_agg'] - _q_exp / 100.0) < 1e-9,
          '诊断列 q_agg 报网络口径（÷s_c = %.4f），便于与点估计/近端批的列比较' % _iu['q_agg'])
    _rul, _iul = H.update_batch_cost_rate(_AUpd(_st4, gamma=0.98, sc=1.0), horizon='finite',
                                          agg='mean', return_info=True)
    check(abs(_rul - _q_exp / _K98) < 1e-9,
          'update_batch（L 视界，若将来给 L 开 record_update_stats）: rate = Q/K = %.3f' % _rul)
    _rmax_u, _ = H.update_batch_cost_rate(_au, horizon='infinite', agg='max', return_info=True)
    check(abs(_rmax_u - 0.05 * 340.0) < 1e-9, 'twin 聚合 max 作用在批均值上（%.1f）' % _rmax_u)
    _re, _ie = H.update_batch_cost_rate(_AUpd([]), horizon='infinite', return_info=True)
    check(_re is None and _ie['empty'] and _ie['n_update'] == 0,
          '统计为空（warmup / 未开 record_update_stats）⇒ 返回 None，由调用方回退 EMA')
    _a = _A(0.0, 0.967)
    H.dual_update_H_hsrl(_a, 15.0, q_rate=None, src='q', ema=30.0)
    check(_a.lam > 0.967, '分子为 None 时 H 对偶**自动回退 EMA**（λ1 仍会动）')

    # ---- λ1：物理价区间 ↔ actor 空间、单调与封顶 ----
    _kl = float(hc.lam_scale())
    _lo1, _hi1, _init1 = H.lam1_phys_bounds()
    check(abs(_init1 - H.HSRL_LAM1_PHYS_INIT * _kl) < 1e-15
          and abs(_hi1 - H.HSRL_LAM1_PHYS_MAX * _kl) < 1e-15,
          'λ1 区间（actor）= 物理价 × lam_scale(%.3g) = [%.4f, %.4f]' % (_kl, _lo1, _hi1))
    a = _A(0.0, _init1)
    H.dual_update_H_hsrl(a, 15.0, q_rate=30.0)          # ratio = 2
    check(a.lam > _init1 and a.lam <= _hi1, 'H 对偶（Q 分子）: ratio=2 ⇒ λ1 上升且不破上界')
    a = _A(0.0, _init1)
    H.dual_update_H_hsrl(a, 15.0, q_rate=15.0)
    check(abs(a.lam - _init1) < 1e-15, 'H 对偶: ratio == 1 ⇒ λ1 不动（不动点）')
    a = _A(0.0, _init1)
    H.dual_update_H_hsrl(a, 15.0, q_rate=1e9)
    check(a.lam == _hi1, 'λ1 封顶 = %.4f（actor 空间）' % _hi1)
    a = _A(0.0, _init1)
    H.dual_update_H_hsrl(a, 15.0, q_rate=None, src='ema', ema=30.0)
    check(a.lam > _init1, 'H 对偶的 EMA 臂（A/B）仍可用')

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
    print('J. 20 步测试协议端到端（桩 env + 桩 agent；固定风速段）')
    print('=' * 90)
    # =====================================================================
    # 风速池做够宽，让默认固定段 (col=HSRL_TEST_WIND_COL, start=HSRL_TEST_WIND_START)
    # 落在池内、不被 clip；并把该段填成 README 记的那 20 个真实风速值 ⇒ 可逐位核对选段。
    _SEG = np.array([10.37, 10.35, 12.35, 10.53, 11.23, 9.95, 11.76, 10.31, 12.86,
                     10.29, 12.03, 8.71, 9.11, 9.68, 9.88, 10.74, 8.74, 7.70, 10.90,
                     9.43])
    pool = np.full((1440, 130), 10.0)
    pool[H.HSRL_TEST_WIND_START:H.HSRL_TEST_WIND_START + 20,
         H.HSRL_TEST_WIND_COL] = _SEG
    G_TGT = 1.5

    def _run(stub_L, stub_H=None, **kw):
        env = _StubEnv(E0=E_MID)
        aH, aL = (stub_H or _StubH(G_TGT)), stub_L
        per_blk, overall, meta = H.twenty_step_test_hsrl(
            aH, aL, env, rbb, pool, phase_mode=0, **kw)
        return env, per_blk, overall, meta

    stub_const = _StubL(pb=-0.1)
    env, per_blk, overall, meta = _run(stub_const)
    check(len(per_blk) == 4 and meta['n_steps'] == 20,
          '20 步 = 4 个块（返回 %d 个块）' % len(per_blk))
    check(meta['price'] == 300.0 and all(abs(p - 300.0) < 1e-12 for p in env.price[:20]),
          '电价恒 300（20 步都是同一个价）')
    check(abs(meta['soc_start'] - 0.25) < 1e-12 and abs(per_blk[0]['soc_start'] - 0.25) < 1e-12,
          '开局 SOC = 0.25 MWh（覆写了 env.ScenarioReset 的硬编码 5.0）')
    # ---- 固定风速段（不再随机起步）----
    check((meta['wind_col'], meta['wind_start'])
          == (H.HSRL_TEST_WIND_COL, H.HSRL_TEST_WIND_START),
          '默认用**固定**段 col=%d / start=%d' % (H.HSRL_TEST_WIND_COL, H.HSRL_TEST_WIND_START))
    check(np.allclose(meta['wind_seq'], _SEG),
          '固定段的风速序列与 README 记录的 20 个值逐位一致')
    _m2 = _run(_StubL(pb=-0.1), seed=777)[3]
    _m3 = _run(_StubL(pb=-0.1), random_wind=True, seed=1)[3]
    _m4 = _run(_StubL(pb=-0.1), random_wind=True, seed=2)[3]
    _m5 = _run(_StubL(pb=-0.1), random_wind=True, seed=1)[3]
    check((_m2['wind_col'], _m2['wind_start'])
          == (H.HSRL_TEST_WIND_COL, H.HSRL_TEST_WIND_START),
          '固定段不受 --hsrl_test_seed 影响（随机起步已关闭）')
    check((_m3['wind_col'], _m3['wind_start']) != (_m4['wind_col'], _m4['wind_start'])
          and (_m3['wind_col'], _m3['wind_start']) == (_m5['wind_col'], _m5['wind_start']),
          '--hsrl_test_wind random: 不同 seed 抽不同段、同 seed 可复现（对照臂仍可用）')
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

    # ---- 观测第 6 维（c_L 参照点）在该协议下的接线 ----
    _obs_need = [float(o[5]) * H.OBS_NEED_SCALE for o in stub_const.obs]
    _need_exp = [(G_TGT - 0.1 * k) / (5 - k) for k in range(5)]
    check(all(abs(a - b) < 1e-12 for a, b in zip(_obs_need[:5], _need_exp)),
          '每块第 6 维 = 决策时刻的 need = %s（Σ 只用**本步之前**的累计）'
          % np.round(_need_exp, 4).tolist())
    check(len(stub_const.obs[0]) == 6 and len(_run(_StubAmortizedL(G_TGT))[1]) == 4,
          '测试协议里的 L 观测也是 6 维（与训练同口径）')

    # ---- r1 的记账必须与训练同口径: r1 = pr·P_G·dt − (pen5 + pen6) ----
    _env_bad, _pb_bad, _ov_bad, _ = _run(_StubL(pb=-0.1), _StubH(G_TGT, a_wt=0.6))
    _pen5_step = AMF.PENALTY_NN_H_COEF * 0.1 * 5     # 5 维 × 每维越界 0.1
    _pen5_blk = _pen5_step * 5                       # 块级动作的罚在块内**按步**摊入（5 次）
    _dr1 = _pb_bad[0]['r1'] - per_blk[0]['r1']
    check(abs(_pb_bad[0]['pen5'] - _pen5_blk) < 1e-9,
          '前 5 维越界(0.6>0.5) 的罚计入并记录: 块内 pen5 = %.1f (= %.0f/步 × 5 步)'
          % (_pb_bad[0]['pen5'], _pen5_step))
    check(abs(_dr1 + _pen5_blk) < 1e-9,
          'r1 确实扣掉 pen5（实测 Δr1 = %.1f = −%.0f）⇒ 测试与训练记账同口径'
          % (_dr1, _pen5_blk))

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
