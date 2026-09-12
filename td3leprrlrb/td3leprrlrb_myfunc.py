"""
TD3LEPRRLRB — Piecewise Reward + Rule-Based BESS + LLM 分段微调
==============================================================
= td3prrlrb (Piecewise Reward RL + 规则 BESS 下层) + PRLE 的 LLM 元控制器层。

与 td3prrlrb 的【唯一】差别: 风速分段阈值 (LOW_MAX / HIGH_MIN) 不再是写死的
常量, 而是可变模块状态 —— 启动时由 --seg 定起点, 训练中每 N 步由 LLM 依据
窗口统计在线微调 (硬约束: 原设定 ±1 m/s), 阈值一变就用当前阈值重算回放池里
历史转移的 (r1, c1) (见 td3leprrlrb_buffer.LEPRRLRBBuffer.relabel_all)。

Upper layer (H, RL):
  r1 = wT*(pr*ptur*time) + wB*(-pr*pbrule*time)
       - 490*max(-pbrule,0)*time            (电池退化, 由命令方 H 付费)
       - penalty_NN_H
       wT = wT_u(u) × wT_p(pr),  wB = wB_p(pr) × wB_soc(SOC, wind)

  c1 = C1_RISK[wind] * [ 10*(P_F-3)+ + 0.1*P_F ]      (波动越限 + 平滑)
       + W_GAP * unfeasible_gap                       (平抑可行性缺口)
       + [块末结算] gap_penalty                        (SOC 目标可行性罚)

  a6 = SOC 目标信号 (-0.5 ~ 0.5)  →  soc_target = 3 + 5*a6 ∈ [0.5, 5.5]
       块级(每 5 步)固定; 上层为"不可行的目标"付费。

Lower layer (L, 规则): SocTracker (见 td3leprrlrb_rule_bess.py)
  在 RBB 可行窗口内以最大能力跟踪 soc_target:
    * 物理可行 + 平抑波动 = 窗口硬保证 → clip_amount ≡ 0
    * 死区 SOC_TOL (=0.1 MWh = 6 MW·min) 内不动作, 保留平抑作业余量
  因此下层无需 RL; 块末残差 |soc_target - SOC| 近似等于"不可行量"。

【阈值依赖面】风速档 wind_level(u) 同时进入
  (a) 收益权重 wT = WT_WIND[wind]×WT_PRICE[price] 与 wB (经 SOC_SAFE_RANGES[wind])
  (b) 风险系数 C1_RISK[wind]
所以阈值一动, 历史的 r1/c1 口径就变了 → 必须 relabel (本模块提供两个
relabel_*_substep 辅助函数, 只重算阈值相关的乘法因子)。

Constants
---------
PRICE 分档: ≥980 HIGH, ≤300 LOW, 其余 MEDIUM
WIND  分档: 动态 —— 初值 7.0/11.0, 由 --seg 覆盖, 之后 LLM 在 ±1 m/s 内微调
"""

import numpy as np
import pandas as pd

from util import *


# =============================================================================
# 0. Dynamic wind thresholds (mutable module-level state)
# =============================================================================
# 模块默认值 7.0/11.0 (与 td3prrlrb 写死的常量一致); main 启动时按 --seg
# (percentile / kmeans / expert) 用 set_wind_thresholds() 设定起点, 之后由
# LLM 元控制器每 N 步在【原设定 ±1 m/s】内在线微调。

WIND_LOW_MAX  = 7.0   # u < 7.0   → LOW wind
WIND_HIGH_MIN = 11.0  # u >= 11.0 → HIGH wind, else MEDIUM

_wind_low_max  = WIND_LOW_MAX
_wind_high_min = WIND_HIGH_MIN


def get_wind_thresholds():
    """Return the current (wind_low_max, wind_high_min) in m/s."""
    return _wind_low_max, _wind_high_min


def set_wind_thresholds(low_max, high_min):
    """Set the wind thresholds with physical validity clamping.

    Constraints enforced (与 td3prrbb/td3prle 一致):
      - low_max  ∈ [3.0, high_min - 1.0]
      - high_min ∈ [low_max + 1.0, 22.0]
      - always low_max < high_min (至少 1.0 m/s 间隔)

    Returns
    -------
    (actual_low_max, actual_high_min) : (float, float)
        Values after clamping (may differ from the request).
    """
    global _wind_low_max, _wind_high_min
    low_max = float(np.clip(low_max, 3.0, high_min - 1.0))
    high_min = float(np.clip(high_min, low_max + 1.0, 22.0))
    _wind_low_max = low_max
    _wind_high_min = high_min
    return _wind_low_max, _wind_high_min


# =============================================================================
# 1. Price / Wind classification
# =============================================================================

PRICE_HIGH   = 980
PRICE_MEDIUM = 580
PRICE_LOW    = 300


def get_price_level(pr):
    """Classify electricity price into 3 levels."""
    if pr >= PRICE_HIGH:
        return 'HIGH'
    elif pr <= PRICE_LOW:
        return 'LOW'
    else:
        return 'MEDIUM'


def get_wind_level(u):
    """Classify wind speed into LOW / MEDIUM / HIGH using CURRENT thresholds."""
    if u < _wind_low_max:
        return 'LOW'
    elif u >= _wind_high_min:
        return 'HIGH'
    else:
        return 'MEDIUM'


# =============================================================================
# 2. Segmented weight lookup (multiplicative decomposition)
# =============================================================================
# 机会侧（利润）与风险侧（c1）分离: wT = wT_u(u)×wT_p(pr), wB = wB_p(pr)×wB_soc(SOC,wind)

WT_WIND = {'LOW': 1.2, 'MEDIUM': 1.0, 'HIGH': 0.9}
WT_PRICE = {'LOW': 1.0, 'MEDIUM': 1.0, 'HIGH': 1.05}
WB_PRICE = {'LOW': 0.8, 'MEDIUM': 1.0, 'HIGH': 1.2}
WB_SOC_OUTSIDE = 0.7

# c1 风险系数：三档
C1_RISK = {'LOW': 0.8, 'MEDIUM': 1.0, 'HIGH': 1.2}
# unfeasible_gap（平抑可行性缺口, MW）进入 c1 的系数
W_GAP = 10.0

# SOC 目标跟踪死区 (MWh): 0.1 MWh = 6 MW × 1 min
#   —— 下层规则控制器的死区与该裕度【必须同源】: 否则规则停在死区边缘时,
#      上层会被收一笔永远消不掉的"幽灵税"。
SOC_TOL = 0.1
# SOC 目标映射: a6 ∈ [-0.5, 0.5] → soc_target ∈ [0.5, 5.5]
SOC_TARGET_CENTER = 3.0
SOC_TARGET_SPAN = 5.0
SOC_TARGET_MIN, SOC_TARGET_MAX = 0.5, 5.5

# 退化系数 (元/MWh): 放电侧; 充电侧可以取更小 (k_c = 0 表示只在放电计退化)
DEG_COST_DISCHARGE = 490.0
DEG_COST_CHARGE = 0.0


def get_wT(wind_level, price_level):
    """Turbine revenue weight: multiplicative wind × price decomposition."""
    return WT_WIND[wind_level] * WT_PRICE[price_level]


def get_wB(price_level, wind_level, SOC):
    """BESS revenue weight: price × SOC-margin decomposition."""
    soc_min, soc_max = SOC_SAFE_RANGES[wind_level]
    w_soc = 1.0 if soc_min <= SOC <= soc_max else WB_SOC_OUTSIDE
    return WB_PRICE[price_level] * w_soc


# =============================================================================
# 3. SOC 安全带 (仅用于日志指标; 新设计下不参与任何成本)
# =============================================================================
SOC_SAFE_RANGES = {
    'LOW':    (0.5, 5.5),
    'MEDIUM': (1.0, 5.0),
    'HIGH':   (1.5, 4.5),
}


def Cal_soc_deviation(SOC, wind_level='MEDIUM'):
    """Wind-segmented SOC deviation outside the safe band (log metric only)."""
    soc_min, soc_max = SOC_SAFE_RANGES[wind_level]
    dev = 0.0
    if SOC < soc_min:
        dev = soc_min - SOC
    elif SOC > soc_max:
        dev = SOC - soc_max
    return dev


# =============================================================================
# 4. Upper-level r1
# =============================================================================

def Cal_r1_PRRBB(pr, ptur, pbrule, penalty_NN_H, wind_level, price_level,
                 SOC=3.0, time=1.0 / 60):
    """
    r1 = wT * (pr * ptur * time)             ← 风机收益
       + wB * (-pr * pbrule * time)          ← BESS 收益 (正=pbrule<0 放电)
       - 490 * max(-pbrule, 0) * time        ← 退化
       - penalty_NN_H
    """
    w_wind = get_wT(wind_level, price_level)
    w_bess = get_wB(price_level, wind_level, SOC)

    turbine_profit = w_wind * pr * ptur * time
    bess_profit    = w_bess * (-pr * pbrule * time)
    degradation    = (DEG_COST_DISCHARGE * max(-pbrule, 0.0)
                      + DEG_COST_CHARGE * max(pbrule, 0.0)) * time

    return turbine_profit + bess_profit - degradation - penalty_NN_H


# =============================================================================
# 4b. Reward / cost relabeling helpers (LLM 阈值变动后重算历史转移)
# =============================================================================
# 阈值改动只影响两处乘法因子:
#   (a) 收益权重 wT = WT_WIND[wind(u)] × WT_PRICE[price]
#                 wB = WB_PRICE[price] × wB_soc(SOC, SOC_SAFE_RANGES[wind(u)])
#   (b) 风险系数 C1_RISK[wind(u)]
# 其余项 (pr·ptur·dt, −pr·pbrule·dt, 退化, penalty_NN_H, P_F 项, W_GAP·gap)
# 都与阈值无关, 因此缓存"与阈值无关的原料"(raw) + 少数上下文标量即可精确重算。

def relabel_r1_substep(u, pr, turbine_raw, bess_raw, degradation, penalty_NN_H,
                       SOC=3.0):
    """用【当前】阈值重算单个子步的 r1.

    Parameters
    ----------
    u : float                     该子步风速 (m/s)
    pr : float                    该子步电价 (RMB/MWh)
    turbine_raw : float           pr * ptur * dt        (与阈值无关)
    bess_raw : float              -pr * pbrule * dt     (与阈值无关)
    degradation : float           退化成本             (与阈值无关)
    penalty_NN_H : float          上层动作越界罚        (与阈值无关)
    SOC : float                   决策时刻 SOC (MWh) —— 因为本项目保留了
                                  "BESS 收益按 SOC 贴边折减"(WB_SOC_OUTSIDE=0.7),
                                  wB 仍依赖 SOC 落在哪一档安全带上, 而安全带
                                  又由 wind_level 决定, 故 SOC 必须随原料一起
                                  缓存, 否则 relabel 不精确。

    Returns
    -------
    r1 : float
    """
    wind_level = get_wind_level(u)
    price_level = get_price_level(pr)
    w_wind = get_wT(wind_level, price_level)
    w_bess = get_wB(price_level, wind_level, SOC)
    return w_wind * turbine_raw + w_bess * bess_raw - degradation - penalty_NN_H


def relabel_c1_substep(u, P_F, unfeasible_gap=0.0, gap_pen=0.0):
    """用【当前】阈值重算单个子步的 c1.

    c1 = C1_RISK[wind(u)] · [10·(P_F−3)⁺ + 0.1·P_F]
         + W_GAP·unfeasible_gap     (与阈值无关, 原样带回)
         + gap_pen                  (块末 SOC 目标罚, 与阈值无关; 仅块内最后
                                     一个子步非零 —— 见 main 的 context 记账)

    parameters 中 unfeasible_gap / gap_pen 都从存档 context 原样读回, 保证
    relabel 前后成本口径逐项一致 (td3prle 曾因漏存块末跟踪项而在 relabel 时
    静默丢掉该项, 本项目已修正)。
    """
    coeff = C1_RISK[get_wind_level(u)]
    if P_F > 3.0 + 1e-4:
        base = coeff * (10.0 * (P_F - 3.0) + 0.1 * P_F)
    else:
        base = coeff * (0.1 * P_F)
    return base + W_GAP * unfeasible_gap + float(gap_pen)


# =============================================================================
# 5. SOC 目标映射 与 上层可行性(gap)惩罚
# =============================================================================

def soc_target_from_signal(a6):
    """上层信号 → SOC 目标: 3 + 5*a6, 并夹在 [0.5, 5.5] (合法目标带)."""
    return float(np.clip(SOC_TARGET_CENTER + SOC_TARGET_SPAN * float(a6),
                         SOC_TARGET_MIN, SOC_TARGET_MAX))


def gap_penalty(soc_target, SOC_end, tol=SOC_TOL):
    """上层可行性罚 (块末结算): 10 * max(0, |soc_target - SOC_end| - tol).

    由于下层规则是"最大能力跟踪", 块末残差 ≈ 不可行量;
    tol 与下层死区同源 → 只罚真正的不可行部分。
    """
    return W_GAP * max(0.0, abs(float(soc_target) - float(SOC_end)) - tol)


# =============================================================================
# 6. Combined reward (H 层)
# =============================================================================

def Cal_Rew_RLRB(observation, observation2, action_rbb, rbb_info,
                 penalty_NN_H=0.0, time=1.0 / 60, ptur=None,
                 pb_cmd=None):
    """
    TD3PRRLRB 上层奖励/成本.

    Parameters
    ----------
    observation  : [pr, u, E, P_G_last]
    observation2 : [pr', u', E', P_G]
    action_rbb   : [a1..a5, soc_signal, pbrule]
    rbb_info     : dict from RuleBasedBESS.compute_pbrule()
    penalty_NN_H : 上层动作越界罚
    ptur         : 风机功率 (MW)
    pb_cmd       : 下层规则输出的功率指令 (MW) —— 用于记录额定裁剪 (规则下应为 0)

    Returns
    -------
    r1, c1_step, details      # 注意: 不含块末 gap 罚 (由 main 在块末累加)
    """
    if ptur is None:
        raise ValueError("Cal_Rew_RLRB: ptur must be provided")

    pr_old = float(observation[0])
    u = float(observation[1])
    P_G_new = float(observation2[-1])
    P_G_old = float(observation[-1])
    SOC_new = float(observation2[2])

    pbrule = float(rbb_info['pbrule'])
    unfeasible_gap = float(rbb_info['unfeasible_gap'])
    clip_amount = float(rbb_info['clip_amount'])
    P_F = abs(P_G_new - P_G_old)

    soc_signal = float(action_rbb[5])
    soc_target = soc_target_from_signal(soc_signal)

    wind_level = get_wind_level(u)
    price_level = get_price_level(pr_old)
    SOC_decision = float(observation[2])

    # ==== r1 (H 机会侧) ====
    r1 = Cal_r1_PRRBB(pr_old, ptur, pbrule, penalty_NN_H,
                      wind_level, price_level, SOC=SOC_decision, time=time)

    # ==== c1 (H 风险侧) ====
    c1_risk_coeff = C1_RISK[wind_level]
    if P_F > 3.0 + 1e-4:
        c1 = c1_risk_coeff * (10.0 * (P_F - 3.0) + 0.1 * P_F)
    else:
        c1 = c1_risk_coeff * (0.1 * P_F)
    c1 = c1 + W_GAP * unfeasible_gap

    # ==== details ====
    w_wind = get_wT(wind_level, price_level)
    w_bess = get_wB(price_level, wind_level, SOC_decision)
    turbine_profit = w_wind * pr_old * ptur * time
    bess_profit = w_bess * (-pr_old * pbrule * time)
    soc_dev = Cal_soc_deviation(SOC_new, wind_level)

    details = {
        'wind_level': wind_level,
        'price_level': price_level,
        'w_wind': w_wind,
        'w_bess': w_bess,
        'pr': pr_old,
        'u': u,
        'turbine_profit': turbine_profit,
        'bess_profit': bess_profit,
        'revenue': turbine_profit + bess_profit,
        'degradation': (DEG_COST_DISCHARGE * max(-pbrule, 0.0)
                        + DEG_COST_CHARGE * max(pbrule, 0.0)) * time,
        'soc_signal': soc_signal,
        'soc_target': soc_target,
        'soc_delta': SOC_new - soc_target,
        'soc_tol': SOC_TOL,
        # 决策时刻 SOC: wB 的 SOC 贴边折减以此为准 → relabel 需要缓存它
        'SOC_decision': SOC_decision,
        'soc_deviation': soc_dev,
        'ptur': ptur,
        'pbrule': pbrule,
        'pb_cmd': float(pb_cmd) if pb_cmd is not None else np.nan,
        # 规则下层自身已落在窗口内 ⇒ 该差值 = RBB 兜底接管量 (仅不可行步 >0)
        'pb_rbb_clip': (abs(float(pb_cmd) - pbrule) if pb_cmd is not None
                        else 0.0),
        'P_F': P_F,
        'SOC': SOC_new,
        'unfeasible_gap': unfeasible_gap,
        'clip_amount': clip_amount,
        'is_infeasible': rbb_info.get('is_infeasible', False),
        'inevitable_violation': rbb_info.get('inevitable_violation', 0.0),
        'penalty_NN_H': penalty_NN_H,
        'c1_fluct': c1,
    }

    return r1, c1, details


# =============================================================================
# 7. Action checking
# =============================================================================

def CheckAction(action, charge=10.):
    """Clip actions to feasible ranges. WT: [0,0.5], Signal: [-0.5,0.5], Bat: [-charge,charge].

    上层 (6 维): WT 越界与信号越界按 100/MW 计罚, 并封顶 50 (penalty_NN_H)。
    下层 (1 维): 规则控制器已保证落在 RBB 窗口内, 这里只做额定功率安全限幅。
    """
    penalty = 0
    SIGNAL_MIN, SIGNAL_MAX = -0.5, 0.5

    if len(action) == 6:
        for j in action[:5]:
            ma, mi = max(0.5, j), min(0, j)
            penalty += 100 * ((ma - 0.5) + (0. - mi))
        sig = action[5]
        if sig > SIGNAL_MAX:
            penalty += 100 * (sig - SIGNAL_MAX)
        elif sig < SIGNAL_MIN:
            penalty += 100 * (SIGNAL_MIN - sig)
        applied = np.append(np.clip(action[:5], 0., 0.5),
                            np.clip(action[5], SIGNAL_MIN, SIGNAL_MAX))
        penalty = min(penalty, 50.0)
        return applied, penalty

    elif len(action) == 5:
        for j in action[:5]:
            ma, mi = max(0.5, j), min(0, j)
            penalty += 200 * ((ma - 0.5) + (0. - mi))
        return np.clip(action[:5], 0., 0.5), penalty

    elif len(action) == 1:
        for j in action:
            ma, mi = max(charge, j), min(-charge, j)
            penalty += 200 * ((ma - charge) + (-charge - mi))
        return np.clip(action, -charge, charge), penalty

    return action, penalty


# =============================================================================
# 8. Normalization
# =============================================================================

def Normalize(obs, max_E=6.):
    return np.array([obs[0] / 1000, obs[1] / 10, obs[2] / 10, obs[3] / 10])


def Original(obs, max_E=6.):
    return np.array([obs[0] * 1000, obs[1] * 10, obs[2] * 10, obs[3] * 10])


# =============================================================================
# 9. 9-case 测试 (下层为规则控制器)
# =============================================================================

try:
    from test_scenario import build_9case_scenario
except ImportError:
    import sys as _sys, os as _os
    _HSRL_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _HSRL_ROOT not in _sys.path:
        _sys.path.insert(0, _HSRL_ROOT)
    from test_scenario import build_9case_scenario


def nine_case_test(agent_H, tracker, env_test, rbb_system,
                   scenario=None, test_csv=False, output=None, seed=None):
    """
    TD3LEPRRLRB 45-step 9-case in-training test (下层 = 规则控制器).

    Layout: wind LOW->MEDIUM->HIGH; within each wind segment price
    LOW->MEDIUM->HIGH; each case = 5 steps = one upper-level period.

    【动态阈值】场景本身按【固定设计档】(LOW<7.0 / MEDIUM[7.0,11.0) / HIGH≥11.0)
    生成, 但奖励用【当前】动态阈值计算 —— 于是每个 case 额外记录
    'project_wind_levels'(用当前阈值对场景风速重新分档的结果), 阈值漂移一目了然。

    Returns
    -------
    case_metrics : list of dict (length 9)  —— 每 case 的 5 步均值:
        r1 / c1(步级+块末gap) / gap_pen / cvm(c1_step>0 的步数) /
        pf_viol(P_F > 3 的步数, 规则下层下应≈0, 仅物理不可行时>0) /
        profit(5 步利润之和) / soc_track_err(块末 |target-SOC|) /
        project_wind_levels(该 case 内出现过的当前风速档)
    overall : dict —— 9 个 case 的均值 (cvm 为总计; profit 为 45 步总和),
        并附带当前 wind_low_max / wind_high_min
    """
    if scenario is None:
        wind_seq, price_seq, case_ids, blocks = build_9case_scenario(seed=seed)
    else:
        wind_seq, price_seq, case_ids, blocks = scenario

    obs = env_test.ScenarioReset(wind_seq, price_seq)
    norm_obs = Normalize(obs)
    time = 1.0 / 60
    agent_H.is_training = False
    step = 0
    num_iterations = len(wind_seq)
    n_per_case = num_iterations // len(blocks)

    per_case = {i: {'r1': 0.0, 'c1': 0.0, 'cvm': 0, 'pf_viol': 0, 'profit': 0.0,
                    'gap_pen': 0.0, 'soc_track_err': 0.0, 'project_winds': set()}
                for i in range(len(blocks))}
    df = pd.DataFrame()
    soc_target = SOC_TARGET_CENTER

    while step < num_iterations:
        if step % 5 == 0:
            action_H = agent_H.select_action(norm_obs)
            action_H, penalty_NN_H = CheckAction(action_H)
            pw_tru = env_test.PowerOutput(obs[1], action_H)
            soc_target = soc_target_from_signal(action_H[-1])

        # ---- 下层: 规则控制器 (RBB 窗口内最大能力跟踪) ----
        _, rbb_info_obs = rbb_system.compute_pbrule(
            pb_original=0.0, ptur=float(pw_tru),
            pg_prev=float(obs[-1]), E=float(obs[2]))
        pb_cmd = tracker.act(float(obs[2]), soc_target,
                             rbb_info_obs['pbrulemin'], rbb_info_obs['pbrulemax'])
        action_L = np.array([pb_cmd])

        action = np.concatenate((action_H, action_L))

        pb_original = float(action[-1])
        ptur = float(pw_tru)
        pg_prev = float(obs[-1])
        E_current = float(obs[2])

        pbrule, rbb_info = rbb_system.compute_pbrule(
            pb_original=pb_original, ptur=ptur,
            pg_prev=pg_prev, E=E_current,
        )

        action_rbb = action.copy()
        action_rbb[-1] = pbrule

        pw_tru2, obs2, action_L_env, penalty_L, done, done_L, info = \
            env_test.ScenarioStep(action_rbb[:6], action_rbb[6:], pw_tru)

        action_rbb[6:] = action_L_env
        norm_obs2 = Normalize(obs2)

        r1, c1_step, details = Cal_Rew_RLRB(
            obs, obs2, action_rbb, rbb_info,
            penalty_NN_H=penalty_NN_H, time=time,
            ptur=ptur, pb_cmd=pb_cmd,
        )

        # 块末: SOC 目标可行性罚 (tol 与下层规则死区同源)
        gap_pen = 0.0
        if step % 5 == 4:
            gap_pen = gap_penalty(soc_target, float(obs2[2]), tol=tracker.soc_tol)
        c1 = c1_step + gap_pen

        bi = step // 5
        pc = per_case[bi]
        pc['r1'] += r1
        pc['c1'] += c1
        pc['gap_pen'] += gap_pen
        if c1_step > 0:
            pc['cvm'] += 1
        if details['P_F'] > 3.0 + 1e-4:
            pc['pf_viol'] += 1
        if step % 5 == 4:
            pc['soc_track_err'] = abs(soc_target - float(obs2[2]))
        # 当前（动态）阈值下该步实际落在哪一档 → 用于观察阈值漂移
        pc['project_winds'].add(details['wind_level'])

        # profit: P_G * pr * dt - 490*max(-pb,0)*dt (与训练 episode_profit 同口径)
        profit_step = obs2[-1] * obs[0] * time - 490.0 * max(-float(pbrule), 0.0) * time
        pc['profit'] += profit_step

        if test_csv:
            df = pd.concat([df, pd.DataFrame([{
                'step': step,
                'case_wind': case_ids[step][0],
                'case_price': case_ids[step][1],
                'wind_level': details['wind_level'],
                'price_level': details['price_level'],
                'u': details['u'], 'pr': details['pr'],
                'r1': r1, 'c1_step': c1_step, 'gap_pen': gap_pen, 'c1': c1,
                'profit': profit_step,
                'soc_signal': details['soc_signal'],
                'soc_target': details['soc_target'],
                'soc_delta': details['soc_delta'],
                'SOC': details['SOC'],
                'pb_cmd': pb_cmd, 'pbrule': rbb_info['pbrule'],
                'clip_amount': rbb_info['clip_amount'],
                'P_F': details['P_F'],
                'action': action_rbb,
            }])], ignore_index=True)

        norm_obs = norm_obs2
        obs = obs2
        pw_tru = pw_tru2
        step += 1

    agent_H.is_training = True
    if test_csv and output:
        df.to_csv(f'{output}/test_s_rlrb_9case.csv', index=False)

    wind_low_max, wind_high_min = get_wind_thresholds()
    case_metrics = []
    for i, b in enumerate(blocks):
        pc = per_case[i]
        case_metrics.append({
            'wind_level': b['wind_level'],
            'price_level': b['price_level'],
            'r1': pc['r1'] / n_per_case,
            'c1': pc['c1'] / n_per_case,
            'gap_pen': pc['gap_pen'] / n_per_case,
            'cvm': pc['cvm'],
            'pf_viol': pc['pf_viol'],
            'profit': pc['profit'],
            'soc_track_err': pc['soc_track_err'],
            'project_wind_levels': sorted(pc['project_winds']),
        })

    overall = {
        'r1': float(np.mean([m['r1'] for m in case_metrics])),
        'c1': float(np.mean([m['c1'] for m in case_metrics])),
        'gap_pen': float(np.mean([m['gap_pen'] for m in case_metrics])),
        'cvm': int(sum(m['cvm'] for m in case_metrics)),
        'pf_viol': int(sum(m['pf_viol'] for m in case_metrics)),
        'profit': float(sum(m['profit'] for m in case_metrics)),
        'soc_track_err': float(np.mean([m['soc_track_err'] for m in case_metrics])),
        # 当前动态阈值 (LLM 微调结果) 隨测试一并记录
        'wind_low_max': wind_low_max,
        'wind_high_min': wind_high_min,
    }
    return case_metrics, overall
