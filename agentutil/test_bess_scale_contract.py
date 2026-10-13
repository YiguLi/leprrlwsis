# -*- coding: utf-8 -*-
"""新 BESS 口径的契约烟测（只读，不需 MATLAB）。"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np          # noqa: E402
import hsrl_soctarget as hst   # noqa: E402
import hsrl_cost as hc        # noqa: E402
import hsrl_warmup as hw     # noqa: E402

FAIL = []


def ck(cond, msg):
    print(('  OK   ' if cond else '  FAIL ') + msg)
    if not cond:
        FAIL.append(msg)


print('=' * 92)
print('新 BESS 口径契约（0.5 MW / 0.5 MWh / SOC_TOL 0.01）')
print('=' * 92)

print('\n[1] 常量')
ck(abs(hst.SOC_TARGET_CENTER - 0.25) < 1e-12, 'SOC_TARGET_CENTER = %.4f' % hst.SOC_TARGET_CENTER)
ck(abs(hst.SOC_TARGET_SPAN - 5 / 12) < 1e-12, 'SOC_TARGET_SPAN = %.6f' % hst.SOC_TARGET_SPAN)
ck(abs(hst.SOC_TARGET_MIN - 0.25 / 6) < 1e-12 and abs(hst.SOC_TARGET_MAX - 0.25 * 11 / 6) < 1e-12,
   'SOC_TARGET_MIN/MAX = %.6f / %.6f' % (hst.SOC_TARGET_MIN, hst.SOC_TARGET_MAX))
ck(abs(hc.SOC_TOL - 0.01) < 1e-15, 'hc.SOC_TOL = %.4f' % hc.SOC_TOL)
ck(abs(hw.SOC_START - 0.25) < 1e-12, 'hw.SOC_START = %.4f' % hw.SOC_START)
ck(abs(hw.RATED_POWER - 0.5) < 1e-12, 'hw.RATED_POWER = %.4f MW' % hw.RATED_POWER)
ck(abs(hst.OBS_SOC_SCALE - 0.5) < 1e-12, 'OBS_SOC_SCALE = %.4f' % hst.OBS_SOC_SCALE)
ck(abs(hst.OBS_POWER_SCALE - 5.0) < 1e-12, 'OBS_POWER_SCALE = %.4f' % hst.OBS_POWER_SCALE)

print('\n[2] SOC 目标映射（信号 -> 目标，必须落在容量 [0, 0.5] 内）')
for a6 in (-0.5, -0.25, 0.0, 0.25, 0.5):
    t = hst.soc_target_from_signal(a6)
    ok = (hst.SOC_TARGET_MIN - 1e-12) <= t <= (hst.SOC_TARGET_MAX + 1e-12)
    print('    a6=%+5.2f -> target %.6f  %s' % (a6, t, 'OK' if ok else 'FAIL(超容量)'))
    if not ok:
        FAIL.append('soc_target_from_signal(%s)=%s' % (a6, t))
ck(abs(hst.soc_target_from_signal(0.0) - 0.25) < 1e-12, 'a6=0 -> 中心 0.25')
ck(hst.SOC_TARGET_MAX <= 0.5 + 1e-12, 'SOC_TARGET_MAX(%.4f) <= 容量 0.5' % hst.SOC_TARGET_MAX)

print('\n[3] 物理动力学（满充满放时长）')
eta, dt = 0.98, 1.0 / 60
rated, cap = 0.5, 0.5
print('    rated=%.2f MW, cap=%.2f MWh, eta=%.2f' % (rated, cap, eta))
print('    满充时长 = cap/(rated*eta) = %.1f min  （旧口径 6/(10*0.98)=36.7 min）'
      % (cap / (rated * eta) * 60))
print('    单步能量 = rated*eta*dt = %.5f MWh  => 满充需 %.0f 步'
      % (rated * eta * dt, cap / (rated * eta * dt)))

print('\n[4] rule_action 在新口径下不作妖（delta=0 应返回窗口内最小被迫吞吐）')
lo, hi = -0.5, 0.5
r0 = hst.rule_action(0.25, 0.25, lo, hi, 5, eta_c=eta, eta_d=eta, time=dt, rated_power=rated)
print('    rule_action(target=0.25, E=0.25, W=[-0.5,0.5]) = %.6f' % r0)
ck(abs(r0) <= 0.5 + 1e-12, '指令在物理限幅内')
r1 = hst.rule_action(0.5, 0.0, lo, hi, 5, eta_c=eta, eta_d=eta, time=dt, rated_power=rated)
print('    rule_action(target=0.5, E=0.0,  ... ) = %.6f  (delta/g=%.4f 步)' % (r1, 0.5 / (eta * dt)))
ck(abs(r1) <= rated + 1e-12, '已被 rated=0.5 限幅')
r2 = hst.rule_action(0.25, 0.25, lo, hi, 5, eta_c=eta, eta_d=eta, time=dt, rated_power=0.5)
ck(abs(r2 - r0) < 1e-12, 'rated_power 参数生效（显式传入 ≡ 默认）')

print('\n[5] L 层观测刻度（SOC 维必须落在可分辨区间）')
for E in (hst.SOC_TARGET_MIN, 0.25, hst.SOC_TARGET_MAX):
    v = E / hst.OBS_SOC_SCALE
    print('    SOC=%.4f -> obs %.4f' % (E, v))
ck(hst.SOC_TARGET_MAX / hst.OBS_SOC_SCALE <= 1.0 + 1e-9,
   'SOC 维归一化后 <= 1.0（满量程）')

print('\n[6] CheckAction 的新功率上限')
from td3per.td3per_myfunc import CheckAction as CA6
ap, pen = CA6(np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.0]))
ck(pen == 0, 'H 层合法动作无罚')
al, penl = CA6(np.array([5.0]))
print('    CheckAction([5.0]) -> %.4f (penalty %.1f)' % (al[0], penl))
ck(abs(al[0] - 0.5) < 1e-12, '下层命令被夹到 ±0.5 MW')

print('\n' + '=' * 92)
print('结论: %s' % ('全部通过' if not FAIL else '%d 项失败: %s' % (len(FAIL), FAIL)))
print('=' * 92)
sys.exit(0 if not FAIL else 1)
