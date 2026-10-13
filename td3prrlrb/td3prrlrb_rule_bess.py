"""
Rule-Based BESS (PRRBB) System + SocTracker
==========================================
Copied from td3prrbb (identical RBB logic) and EXTENDED with the rule-based
lower-layer controller used by TD3PRRLRB.

`RuleBasedBESS`  — 计算 RBB 可行功率窗口 (波动 ±limit ∩ 物理 SOC/额定), 硬裁剪 pb。
`SocTracker`     — 下层 BESS 的【规则控制器】, 取代下层 RL:
                   机制只有一句话 —— 把"一步填平 SOC 所需功率"夹进可行窗口:
                       pb = clip( (soc_target − E) / g , pbrulemin, pbrulemax )
                   两层豁免分离 (参考 td3leprrlrb 实现):
                     * 0.1 MWh 死区**只属于上层** gap 罚 (记账约定);
                     * 下层 `soc_tol = 0.0` —— 残差 0.01 MWh 也照追。
                   不做"取窗口边界最大功率": 那会把下一步窗口推离 0, 逼出过冲。
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

    def __init__(self, max_E=0.50, min_E_factor=0.2, rated_power=0.5,
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
    确定性 SOC 目标跟踪器 —— TD3PRRLRB 的下层 BESS 控制器.

    设计依据 (见项目文档):
      * 下层任务优先级: 物理可行 > 平抑波动 > 追踪上层目标;
      * 前两项由 RBB 可行窗口 [pbrulemin, pbrulemax] 硬保证;
      * 追踪目标 = "一步填平": 取 `ΔSOC / g` (一步填平所需功率) 再夹进窗口,
        **不做死区豁免** —— 残差 0.01 MWh 也照追;
      * **不取窗口边界的最大功率**: 顶格会把下一步的可行窗口推离 0 (见 act 推导),
        逼出"自致过冲"; 只取所需功率则残差小时功率自然小, 问题不存在。

    【死区归属 (参考 td3leprrlrb 实现)】0.1 MWh 的豁免【只属于上层罚项】,
    不属于下层:
      * 上层 `gap_penalty = 10·max(0, |target−SOC_末| − SOC_TOL)` —— H 只对超过
        0.1 MWh 的残差付费 (这是"H 不必为零头付费"的**记账约定**);
      * 下层 `soc_tol` 默认 **0.0** —— 只要 δ≠0 就按所需功率追, δ=0 时只做
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
    soc_tol : float  下层跟踪死区 (MWh)。**默认 0.0 = 尽力追随**;
                     设为 >0 复现旧的"死区内不动作"行为 (仅用于 A/B 对照)。
    eta_c, eta_d : float  充/放效率。
    time : float  步长 (小时)。
    rated_power : float  额定功率 (MW), 用于把指令先限幅到 [-rated, rated]。
    pb_cap : float or None
        **可选**的自愿指令硬上限 (MW), 默认 None = 不限制。仅为实验保留:
        更紧的上限会砍掉"目标远时"的合法能力 (实测: 中性风下每块涌现能力
        ≈0.5 MWh, 而 |pb|≤3 会把能力锁到 ~0.21 MWh)。正常工况不需要它 ——
        "不顶格"这件事由 `ΔSOC/g` 本身保证, 不需要额外的天花板。
    """

    def __init__(self, soc_tol=0.0, eta_c=0.98, eta_d=0.98, time=1.0 / 60,
                 rated_power=0.5, pb_cap=None):
        self.soc_tol = soc_tol
        self.eta_c = eta_c
        self.eta_d = eta_d
        self.time = time
        self.rated_power = rated_power
        # 可选的自愿指令硬上限 (MW)。默认 None = 不限制 —— "不顶格"由 ΔSOC/g 保证;
        # 设成小值会砍掉"目标远时"的合法能力 (见类文档)。仅为实验保留。
        self.pb_cap = pb_cap
        # 诊断侧信道: act() 每次更新, 供主循环记录 (不影响返回值)
        self.last_pb_voluntary = 0.0   # 一步填平所需功率 (MW, 未夹窗口前)
        self.last_pb_forced = 0.0      # pb_applied − pb_voluntary:
                                       #   >0 = 被窗口/兜底**被迫多做**的部分
                                       #   <0 = 被窗口**限制**掉的部分

    # ---- 每 MW 电力在一步内改变多少 SOC (MWh) ----
    def gain_charge(self):
        return self.eta_c * self.time          # 充电: pb>0

    def gain_discharge(self):
        return self.time / self.eta_d          # 放电: pb<0

    def effective_pb_cap(self):
        """自愿指令的有效上限 (MW)。默认无上限 (= 额定功率)。"""
        if self.pb_cap is None or self.pb_cap <= 0:
            return float(self.rated_power)
        return float(min(self.rated_power, self.pb_cap))

    def act(self, E, soc_target, pbrulemin, pbrulemax):
        """返回本步 BESS 功率指令 pb (MW, 正=充电).

        E          : 当前 SOC (MWh)
        soc_target : 上层给出的目标 SOC (MWh)
        pbrulemin/max : 本步 RBB 可行窗口 (MW)

        【机制 (定稿)】取"一步填平 SOC 所需功率"再夹进可行窗口:
            delta = soc_target − E
            pb    = clip( delta / g , pbrulemin , pbrulemax )     # g = η·dt
        这就是"在 pbrule 内对着目标最大化"的最小充分形式。

        【为什么不是"取窗口边界的最大功率"】
        窗口位置 = ptur − pg_prev ± 3, 而本步 pb 会成为下一步 pg_prev 的组成部分
        (pg_prev' = ptur − pb) ⇒ 下一步窗口中心 = ΔP_tur + pb。若"顶格"到一个
        远大于所需值的功率, 下一步窗口就会整体偏离 0, 规则被**逼着继续同向吞吐**
        → 越过目标 (自致过冲)。只取 ΔSOC/g 时, 残差小则功率小, 窗口自然停在
        含 0 的位置, 下一步就能停在目标上 —— 不需要任何额外的功率上限。

        【残留局限 (属于外部几何, 不是规则的错)】
        若 |delta| > 3g (≈0.05 MWh), 一步填平本身就要求 >3 MW, 下一步窗口仍可能
        不含 0 而被逼补一点, 过冲量 ≈ max(0, |delta| − 3g); 上层 0.1 MWh 豁免
        可吸收其中一部分。风阶跃 (|ΔP_tur| 大) 造成的窗口偏移同理, 规则只能取
        窗口内离 0 最近的点。另注: 约束只加在**并网功率**上 (|ΔP_G|≤3), 所以
        电池功率的步变化允许量实际是 `3 + ΔP_tur` (实测 p50≈3.9 / p90≈6.1 MW),
        不是 3 —— 能力随风况涌现, 不该被常数锁死。
        """
        delta = float(soc_target) - float(E)

        # (1) 已在目标上 (δ == 0; soc_tol 默认 0 ⇒ 该分支仅在此触发):
        #     不主动制造吞吐, 但仍须落在可行窗口内 —— 风速阶跃 >3 MW/min 时
        #     窗口可能不含 0, 此时做最小幅度的**被迫**吞吐。规则自己完成该裁剪,
        #     使 RBB 的 clip_amount 恒为 0 (只做真正的安全兜底)。
        #     soc_tol > 0 时复现旧行为 (δ 在死区内一律不追)。
        if abs(delta) <= self.soc_tol:
            if pbrulemin > pbrulemax:
                self.last_pb_voluntary = 0.0
                self.last_pb_forced = 0.0
                return 0.0
            pb = float(np.clip(0.0, pbrulemin, pbrulemax))
            self.last_pb_voluntary = 0.0
            self.last_pb_forced = pb
            return pb

        # (2) 一步填平所需功率 (残差 0.01 MWh 也照追)
        g = self.gain_charge() if delta > 0 else self.gain_discharge()
        pb = delta / g

        # (3) 额定功率限幅 (+ 可选的实验性硬上限; 默认无上限)
        cap = self.effective_pb_cap()
        pb = float(np.clip(pb, -cap, cap))
        pb_vol = pb

        # (4) 不可行窗口 (pbrulemin > pbrulemax): 交由 RBB 的不可行处理逻辑
        if pbrulemin > pbrulemax:
            self.last_pb_voluntary = pb_vol
            self.last_pb_forced = 0.0
            return pb

        # (5) 夹进可行窗口 (窗口才是唯一的行动边界)
        pb = float(np.clip(pb, pbrulemin, pbrulemax))
        self.last_pb_voluntary = pb_vol
        self.last_pb_forced = pb - pb_vol        # >0 = 被迫多做的量
        return pb

