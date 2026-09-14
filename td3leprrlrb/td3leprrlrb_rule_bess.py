"""
Rule-Based BESS (PRRBB) System + SocTracker
==========================================
Copied verbatim from td3prrlrb (identical RBB + rule-layer logic) for
TD3LEPRRLRB. 本文件与被复制版本逻辑逐字一致 —— 本项目的 LLM 层只在线微调
【风速分段阈值】(奖励权重与风险分档的界面), 不触及可行性窗口与下层规则控制器。

`RuleBasedBESS`  — 计算 RBB 可行功率窗口 (波动 ±limit ∩ 物理 SOC/额定), 硬裁剪 pb。
`SocTracker`     — 下层 BESS 的【规则控制器】, 取代下层 RL:
                   在 RBB 可行窗口内, 以最大能力跟踪上层给出的 SOC 目标。
"""

import numpy as np


class RuleBasedBESS:
    """
    Rule-Based BESS system that clips the BESS agent's power output (pb)
    to satisfy both grid fluctuation constraints and physical SOC/power limits.

    Parameters
    ----------
    max_E : float
        Maximum battery energy capacity (MWh).
    min_E_factor : float
        Fraction of max_E for minimum SOC (default 0.2).
    rated_power : float
        Rated charge/discharge power limit (MW).
    eta_c : float
        Charging efficiency.
    eta_d : float
        Discharging efficiency.
    time : float
        Time step duration in hours (default 1/60 = 1 minute).
    fluctuation_limit : float
        Maximum allowed grid power fluctuation between steps (MW).
    """

    def __init__(self, max_E=6.0, min_E_factor=0.2, rated_power=10.0,
                 eta_c=0.98, eta_d=0.98, time=1.0 / 60, fluctuation_limit=3.0):
        self.max_E = max_E
        self.min_E = min_E_factor * max_E
        self.rated_power = rated_power
        self.eta_c = eta_c
        self.eta_d = eta_d
        self.time = time
        self.fluctuation_limit = fluctuation_limit

    def compute_pbrule(self, pb_original, ptur, pg_prev, E):
        """
        Compute the rule-based BESS power pbrule.

        Parameters
        ----------
        pb_original : float or np.ndarray
            Original BESS power output from the lower-level agent (MW).
            Positive = charging, negative = discharging.
        ptur : float
            Turbine output power for the current step (MW).
        pg_prev : float
            Grid power from the end of the previous step, pg(t-1) (MW).
        E : float
            Current battery state of charge / energy (MWh).

        Returns
        -------
        pbrule : float or np.ndarray
            Clipped BESS power to be passed to the environment (MW).
        info : dict
            Dictionary containing all intermediate values for logging:
            - pb_original: original pb before clipping
            - pbrule: final clipped power
            - pbfluctmin, pbfluctmax: fluctuation smoothing range
            - pbphysicsmin, pbphysicsmax: physical feasible range
            - pbrulemin, pbrulemax: final feasible range (intersection)
            - clip_amount: |pb - pbrule|
            - unfeasible_gap: max(0, pbrulemin - pbrulemax)
            - soc: current SOC (= E)
        """
        # ---- 1. Fluctuation Smoothing Range ----
        # Goal: |pg(t) - pg(t-1)| <= fluctuation_limit
        # pg(t) = ptur - pb  (grid power = turbine - battery)
        # |ptur - pb - pg_prev| <= fluctuation_limit
        # -fluctuation_limit <= ptur - pb - pg_prev <= fluctuation_limit
        # ptur - pg_prev - fluctuation_limit <= pb <= ptur - pg_prev + fluctuation_limit
        pbfluctmin = ptur - pg_prev - self.fluctuation_limit
        pbfluctmax = ptur - pg_prev + self.fluctuation_limit

        # ---- 2. Physical Feasible Range ----
        # SOC constraint: E must stay within [min_E, max_E]
        #   Max charge power: (max_E - E) / (eta_c * time)
        #   Max discharge power: -(E - min_E) * eta_d / time
        # Rated power constraint: |pb| <= rated_power
        max_charge_from_soc = (self.max_E - E) / (self.eta_c * self.time)
        max_discharge_from_soc = (E - self.min_E) * self.eta_d / self.time

        pbphysicsmax = min(self.rated_power, max_charge_from_soc)
        pbphysicsmin = max(-self.rated_power, -max_discharge_from_soc)

        # ---- 3. Combined Feasible Range ----
        pbrulemin = max(pbfluctmin, pbphysicsmin)
        pbrulemax = min(pbfluctmax, pbphysicsmax)

        # ---- 4. Compute pbrule ----
        pb = float(pb_original) if isinstance(pb_original, np.ndarray) else pb_original

        if pbrulemin > pbrulemax:
            # ============================================================
            # INFEASIBLE CASE: fluctuation range ∩ physical range = ∅
            #
            # This happens when |ptur - pg_prev| > rated_power + fluctuation_limit
            # (e.g. wind gust causes >13 MW turbine output change in one step).
            # Even max charge/discharge at rated power cannot fully absorb the
            # fluctuation — a P_F violation is PHYSICALLY UNAVOIDABLE.
            #
            # Strategy: pick the edge of the physical range that is CLOSEST
            # to the fluctuation range, thereby MINIMIZING the inevitable
            # fluctuation violation.
            # ============================================================
            if pbfluctmin > pbphysicsmax:
                # Fluctuation range entirely ABOVE physical range.
                # Need more charge power than physically possible.
                # → Charge at max feasible rate to minimize violation.
                pbrule = float(pbphysicsmax)
            elif pbfluctmax < pbphysicsmin:
                # Fluctuation range entirely BELOW physical range.
                # Need more discharge power than physically possible.
                # → Discharge at max feasible rate to minimize violation.
                pbrule = float(pbphysicsmin)
            else:
                # Should never reach here (pbrulemin > pbrulemax implies one
                # of the above two conditions must hold), but guard anyway.
                pbrule = float(np.clip(pb, pbphysicsmin, pbphysicsmax))

            # Compute the INEVITABLE fluctuation violation (MW beyond limit)
            # P_F = |ptur - pbrule - pg_prev|, limit = fluctuation_limit
            inevitable_violation = max(0.0,
                abs(ptur - pbrule - pg_prev) - self.fluctuation_limit)
        else:
            # Feasible case: intersection exists, clip pb into it.
            # This guarantees P_F ≤ fluctuation_limit.
            pbrule = float(np.clip(pb, pbrulemin, pbrulemax))
            inevitable_violation = 0.0

        # ---- 5. Collect Info ----
        clip_amount = abs(pb - pbrule)
        unfeasible_gap = max(0.0, pbrulemin - pbrulemax)

        info = {
            'pb_original': pb,
            'pbrule': pbrule,
            'pbfluctmin': pbfluctmin,
            'pbfluctmax': pbfluctmax,
            'pbphysicsmin': pbphysicsmin,
            'pbphysicsmax': pbphysicsmax,
            'pbrulemin': pbrulemin,
            'pbrulemax': pbrulemax,
            'clip_amount': clip_amount,
            'unfeasible_gap': unfeasible_gap,
            'inevitable_violation': inevitable_violation,
            'is_infeasible': pbrulemin > pbrulemax,
            'soc': E,
        }

        return pbrule, info


# =============================================================================
# 下层规则控制器 (rule-based lower layer, 取代下层 RL)
# =============================================================================

def block_cap_charge(pbrulemax, eta_c, time, n_steps=5):
    """本块最大可充电量 (MWh): 整块贴着可行上界充电."""
    return max(0.0, float(pbrulemax)) * eta_c * time * n_steps


def block_cap_discharge(pbrulemin, eta_d, time, n_steps=5):
    """本块最大可放电量 (MWh): 整块贴着可行下界放电."""
    return max(0.0, -float(pbrulemin)) * (time / eta_d) * n_steps


class SocTracker:
    """
    确定性 SOC 目标跟踪器 —— TD3LEPRRLRB 的下层 BESS 控制器.

    设计依据 (见项目文档):
      * 下层任务优先级: 物理可行 > 平抑波动 > 追踪上层目标;
      * 前两项由 RBB 可行窗口 [pbrulemin, pbrulemax] 硬保证;
      * 追踪目标 = 一条规则: `pb = clip(ΔSOC/gain, pbrulemin, pbrulemax)`
        (即"一步填平所需功率"投影到可行窗口上), 且**不做死区豁免** ——
        残差 0.01 MWh 也照追 (见 soc_tol 说明)。

    【死区归属 (本轮修正)】0.1 MWh 的豁免【只属于上层罚项】, 不属于下层:
      * 上层 `gap_penalty = 10·max(0, |target−SOC_末| − SOC_TOL)` —— H 只对超过
        0.1 MWh 的残差付费 (这是"H 不必为零头付费"的**记账约定**);
      * 下层 `soc_tol` 默认 **0.0** —— 只要 δ≠0 就用满窗口能力去追, δ=0 时只做
        窗口允许的最小被迫吞吐 (不主动制造吞吐, 但也不放弃任何可达残差)。
      这样不对称的方向是**安全**的: 下层能消掉的残差总是多于上层豁免的
      (0 < 0.1), 不会出现"规则停在某处不动、上层却照罚"的幽灵税。
      ⚠ 若把 `soc_tol` 调回 >0 (复现旧的"最小吞吐"行为), 必须保证
        `tracker.soc_tol ≤ --soc_tol`, 否则幽灵税回归。

    由于 SOC 是纯积分器、代价对 |目标-实际| 单调, 该规则即 5 步块内的
    最优努力策略 (best effort); 因此上层在块末观察到的残差 |target - SOC|
    近似等于"不可行量"—— 上层只需为不可行付费 (见 main 的 gap 惩罚)。

    注意: 块末残差未必归零。风阶跃迫使的吞吐 (窗口不含 0 时 `clip(0, 窗口)`)
    可以**过冲**目标, 也可能把残差推过 0.1 —— 那部分不是"目标不可行", 而是
    "停不下来", 属于 gap 罚口径的已知不精确处。

    Parameters
    ----------
    soc_tol : float  下层跟踪死区 (MWh)。默认 0.0 = 尽力追随 (推荐);
                     设为 >0 复现旧的"死区内不动作"行为 (仅用于 A/B 对照)。
    eta_c, eta_d : float  充/放效率。
    time : float  步长 (小时)。
    rated_power : float  额定功率 (MW)。非空窗口下规则不必用它限幅 —— RBB 物理
                     边界已含 ±rated, "夹进窗口"天然满足额定约束; 只在窗口为空
                     (RBB 将忽略指令) 时用它把请求量压回执行器范围。
    """

    def __init__(self, soc_tol=0.0, eta_c=0.98, eta_d=0.98, time=1.0 / 60,
                 rated_power=10.0):
        self.soc_tol = soc_tol
        self.eta_c = eta_c
        self.eta_d = eta_d
        self.time = time
        self.rated_power = rated_power

    # ---- 每 MW 电力在一步内改变多少 SOC (MWh) ----
    def gain_charge(self):
        return self.eta_c * self.time          # 充电: pb>0

    def gain_discharge(self):
        return self.time / self.eta_d          # 放电: pb<0

    def act(self, E, soc_target, pbrulemin, pbrulemax):
        """返回本步 BESS 功率指令 pb (MW, 正=充电).

        规则只有一条 (不含任何限幅/死区附加机制):

            need = ΔSOC / gain        # "一步填平 SOC 残差"所需功率 (ΔSOC×60/η)
            pb   = clip(need, pbrulemin, pbrulemax)

        即: **在可行窗口 pbrule 内取最接近目标的动作** —— need 落在窗口外时贴窗口
        边界 (用满可用权限), 落在窗口内时就是 need 本身 (一步到位)。
        非空窗口下 `rated_power` 不必单独限幅: RBB 的物理边界 pbphysicsmin/max 已含
        ±rated, 所以"夹进窗口"天然满足额定约束。

        【窗口为空时】(pbrulemin > pbrulemax = 物理与波动约束无交集) RBB 会忽略指令,
        直接取物理边界中"最靠近波动窗口"的一端出力; 这里只把请求量压回 ±rated_power
        (执行器上限), 免得日志里留下物理上给不出的指令值。此时 RBB 的 clip_amount
        可能非零 —— 那是"指令被忽略"的标记, 不是平抑约束被破坏 (|ΔP_G| ≤ 3 在这类
        风阶跃下物理上不可达, 已由 inevitable_violation 单列)。

        【已知行为: 窗口可以把动作顶离 need】
        窗口位置 = ptur − pg_prev ± 3, 而本步 pb 会成为下一步 pg_prev 的一部分。
        因此 |pb| > 波动限值时, 动作会把下一步窗口推离 0, 下一步的 clip 又会把它
        顶回窗口边界 —— 表现为**被迫同向吞吐**, 可能过冲目标 (实测 δ=0.30 的块:
        第 2 步 need=9.37 → 充 9 MW; 第 3 步窗口变 [6,10], need 仅 0.37 却被顶到
        6 MW → 块末残差 0.006 → 0.141)。风阶跃使 |ptur − pg_prev| > 13 时窗口被
        压成单点 {10}, 同理强制 10 MW 吞吐 (SOC 变 0.163 MWh)。
        这是"波动约束优先于 SOC 目标"的直接后果, 不是本规则的缺陷; 要改只能重排
        优先级 (会破坏 |ΔP_G| ≤ 3)。
        """
        delta = float(soc_target) - float(E)

        # 已在目标上: need = 0, 于是 clip(0, 窗口) —— 不主动制造吞吐, 但仍须落在
        # 可行窗口内 (风速阶跃使窗口不含 0 时, 做最小幅度的被迫吞吐; 规则自己完成
        # 该裁剪, 非空窗口下使 RBB 的 clip_amount 恒为 0, 只做真正的安全兜底)。
        # soc_tol > 0 时在 |δ| ≤ soc_tol 内同样按 need=0 处理 (复现旧的"最小吞吐"行为)。
        if abs(delta) <= self.soc_tol:
            if pbrulemin > pbrulemax:
                return 0.0
            return float(np.clip(0.0, pbrulemin, pbrulemax))

        # 一步填平 SOC 残差所需的功率
        g = self.gain_charge() if delta > 0 else self.gain_discharge()
        need = delta / g

        # 不可行窗口: 交给 RBB 兜底 (该分支 RBB 会忽略指令, 直接贴物理边界出力)。
        # 指令本身压回 ±额定: 否则日志里会出现 17.98 MW 这种执行器给不出的数字,
        # 且 RBB 的 clip_amount 诊断会被无意义的原始量污染。
        if pbrulemin > pbrulemax:
            return float(np.clip(need, -self.rated_power, self.rated_power))

        # 夹进可行窗口 = 在 pbrule 内取最接近目标的动作
        return float(np.clip(need, pbrulemin, pbrulemax))

