"""
九例分段测试场景生成器（训练中 45 步测试用）
==============================================

Wind segmentation (user-specified, aligned with PRRBB/PRLE initial thresholds):
    LOW    < 7.0 m/s
    MEDIUM [7.0, 11.0) m/s
    HIGH   >= 11.0 m/s

Scenario layout (fixed order, 45 steps = 3 wind x 3 price x 5 steps):
    wind LOW     (steps 0-14)  : price LOW -> MEDIUM -> HIGH (5 steps each)
    wind MEDIUM  (steps 15-29) : price LOW -> MEDIUM -> HIGH
    wind HIGH    (steps 30-44) : price LOW -> MEDIUM -> HIGH

Each (wind_level, price_level) case covers exactly 5 steps = one upper-level
period, so the upper agent makes exactly one decision per case.

Wind profile design (每步在档内抖动、有增有减、衔接顺畅):
    Within each 15-step segment the profile is a paired sawtooth:
        - steps are processed in pairs (2j, 2j+1):
            du(2j)   = b + a            (up step)
            du(2j+1) = b - a            (down step, same magnitude a)
        - the last step of the segment uses du = b only.
    a ~ U(b+0.03, min(b+0.20, 0.45)) is re-sampled per pair, so every pair
    jitters one step up and one step down (有增有减), with per-call
    randomness (每档内部随机), while pairs cancel exactly:
        sum(du) over the segment = 15*b = end_u - start_u
    so the segment endpoint is exact. b = (end_u - start_u)/15.
    The walk continues across segment boundaries (no reset), and the fixed
    start/end values are chosen so the boundary jumps are small
    (LOW->MEDIUM ~0.54-0.70, MEDIUM->HIGH ~0.63-0.75 m/s), keeping the whole
    45-step sequence smooth (衔接顺畅).
    Value bounds (verified, see module docstring math):
        LOW    in [4.80, 6.79]  ( < 7.0, >= 3.5)
        MEDIUM in [7.04, 10.79] (in [7.0, 11.0))
        HIGH   in [11.33, 15.35] (>= 11.0, < 25)
    All steps stay strictly inside their designed band.
"""

import numpy as np

WIND_LOW_MAX = 7.0    # u < 7.0      -> LOW wind
WIND_HIGH_MIN = 11.0  # u >= 11.0    -> HIGH wind, else MEDIUM

PRICES = {'LOW': 300, 'MEDIUM': 580, 'HIGH': 980}

STEPS_PER_CASE = 5
PRICE_ORDER = ('LOW', 'MEDIUM', 'HIGH')
SEGMENT_LENGTH = 15

# (wind_level, start_u, end_u)
_SEGMENTS = (
    ('LOW',    5.0,  6.7),
    ('MEDIUM', 7.05, 10.85),
    ('HIGH',   11.0, 15.5),
)


def build_9case_scenario(seed=None):
    """
    Build the 45-step 9-case test scenario.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility. None = fully random per call.

    Returns
    -------
    wind_seq : list of float (length 45)
        Wind speed at each step (m/s), LOW->MEDIUM->HIGH segments.
    price_seq : list of int (length 45)
        Electricity price at each step (RMB/MWh), LOW->MEDIUM->HIGH
        within each wind segment.
    case_ids : list of (wind_level, price_level) (length 45)
        Designed case label of each step.
    blocks : list of dict (length 9)
        One entry per case: {'wind_level', 'price_level', 'start'}.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random

    wind_seq = []
    price_seq = []
    case_ids = []
    blocks = []
    u = _SEGMENTS[0][1]  # start of the LOW segment
    t = 0

    for wind_level, start_u, end_u in _SEGMENTS:
        b = (end_u - start_u) / SEGMENT_LENGTH
        a_max = min(b + 0.20, 0.45)
        for price_level in PRICE_ORDER:
            blocks.append({
                'wind_level': wind_level,
                'price_level': price_level,
                'start': t,
            })
            for k in range(STEPS_PER_CASE):
                pos = t % SEGMENT_LENGTH
                if pos == SEGMENT_LENGTH - 1:
                    du = b                       # segment final step: drift only
                elif pos % 2 == 0:
                    a = float(rng.uniform(b + 0.03, a_max))
                    du = b + a                   # up step
                else:
                    du = b - a                   # down step (same a as the pair)
                u = round(u + du, 2)
                wind_seq.append(u)
                price_seq.append(PRICES[price_level])
                case_ids.append((wind_level, price_level))
                t += 1

    return wind_seq, price_seq, case_ids, blocks


def get_scenario_wind_level(u):
    """Classify a scenario wind speed into LOW/MEDIUM/HIGH (7/11 bands)."""
    if u < WIND_LOW_MAX:
        return 'LOW'
    elif u >= WIND_HIGH_MIN:
        return 'HIGH'
    return 'MEDIUM'
