"""
Rule-Based BESS (RBB) System
=============================
Belongs to the BESS agent, NOT the environment.

After WT agent and BESS agent output actions, and wfsim returns ptur and pb,
this module computes the rule-based feasible power pbrule by combining:
  1. Fluctuation smoothing range: keeps |pg(t) - pg(t-1)| <= 3 MW
  2. Physical feasible range: SOC limits + rated power limits

The final pbrule replaces pb as the actual BESS power fed to the environment.
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

    def compute_pbrule_min_discharge(self, ptur, pg_prev, E):
        """**nolow 分支专用**: BESS 只负责波动平抑, 功率取平抑可行窗口内的"放电最小"值。

        窗口 = 平抑范围 ∩ 物理可行范围 = `[pbrulemin, pbrulemax]` (见 `compute_pbrule`)。
        符号约定与 `compute_pbrule` 一致: **pb > 0 = 充电** ⇒ "放电最小" = 窗口最靠充电端
        = `pbrulemax`:
          * 窗口跨 0        ⇒ 取正端 (充电, 不放电);
          * 窗口全在放电侧  ⇒ 取最靠 0 的负值 (放电**最少**);
          * 窗口全在充电侧  ⇒ 取最大充电值 (物理/平抑都允许的上界)。
        窗口不可行 (`pbrulemin > pbrulemax`) 时沿用 `compute_pbrule` 的既有处理
        (取物理边界、最小化不可避免的越限), 不在这里另造口径。

        与 `compute_pbrule` 的唯一差别: 不再由上层/L 层命令 `pb_original` 决定窗口内的落点,
        而是固定落在"最不放电"的一端 ⇒ `clip_amount ≡ 0` (没有命令被裁掉这回事)。
        """
        _, info = self.compute_pbrule(pb_original=0.0, ptur=ptur, pg_prev=pg_prev, E=E)
        if not info['is_infeasible']:
            info['pbrule'] = float(info['pbrulemax'])
        # nolow 没有"上层/下层的命令"这回事 ⇒ 这两个诊断量恒定为 0/自身
        info['pb_original'] = float(info['pbrule'])
        info['clip_amount'] = 0.0
        return float(info['pbrule']), info
