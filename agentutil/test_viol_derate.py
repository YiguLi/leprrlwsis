# -*- coding: utf-8 -*-
"""越界乘性惩罚 (--viol_derate) 回归测试。

契约 (见 td3per/td3per_myfunc.py 的 "越界乘性惩罚" 小节):
  r1 = turbine_profit · exp(-c·x) + bess_profit - degradation - penalty_NN_H
  x  = Σ_{j=0..4} 越界量        (只算机组限功率维 [0,0.5]; **dim5=signal 不计**)
  * c<=0 或 x==0 ⇒ f 恰为 1.0 ⇒ 与历史路径**逐位一致**
  * **记录列不变**: turbine_profit / revenue / degradation / bess_profit / profit
    一律是未打折的原始口径 (用户硬性要求); 被打折的只有送进 critic 的 r1
  * r1 单调不增: f<=1 且 turbine_profit>=0 ⇒ 越界在任何情况下都不会让 r1 变大
    (乘 turbine_profit 而非净利润, 正是为了排除"亏损步上 f<1 反而奖励越界"的符号陷阱)

运行: python agentutil/test_viol_derate.py   (不需要 MATLAB)
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np

from td3per.td3per_myfunc import (
    violation_amount, violation_derate, Cal_r1_continuous, Cal_Rew_RB,
    VIOL_DERATE_COEF, VIOL_DERATE_DIMS,
)
from td3per import td3per_myfunc as M

FAIL = []


def check(cond, msg):
    print(('  [OK]   ' if cond else '  [FAIL] ') + msg)
    if not cond:
        FAIL.append(msg)


def section(t):
    print('\n' + t)
    print('-' * len(t))


# ---------------------------------------------------------------------------
section('1. violation_amount: 只算前 5 维, dim5 不计')
# ---------------------------------------------------------------------------
check(violation_amount([0.2, 0.3, 0.0, 0.5, 0.1, 0.0]) == 0.0, '界内动作 -> x == 0')
a = np.array([0.7, -0.1, 0.5, 0.0, 0.2, 0.0])
check(abs(violation_amount(a) - 0.3) < 1e-12,
      'dim0 越界 0.2 + dim1 越界 0.1 -> x == 0.3 (实得 %.6f)' % violation_amount(a))
a5 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])       # dim5 = 1.0 严重越界
check(violation_amount(a5) == 0.0,
      'dim5(signal) 越界 1.5 **不计入 x** (按用户决定; 实得 %.6f)' % violation_amount(a5))
check(VIOL_DERATE_DIMS == 5, 'VIOL_DERATE_DIMS == 5')
# 恰好落在边界上不算越界
check(violation_amount([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) == 0.0, '恰好在上界 (0.5) 不算越界')
check(violation_amount([0.0, 0.0, 0.0, 0.0, 0.0, -0.5]) == 0.0, '恰好在下界 (0.0) 不算越界')

# ---------------------------------------------------------------------------
section('2. violation_derate: 关闭/无越界时恰为 1.0')
# ---------------------------------------------------------------------------
check(violation_derate(a, 0.0) == 1.0, 'c=0 (关闭) -> f 恰为 1.0 (逐位)')
check(violation_derate(a, -3.0) == 1.0, 'c<0 -> f 恰为 1.0')
check(violation_derate([0.1] * 6, VIOL_DERATE_COEF) == 1.0, 'x=0 -> f 恰为 1.0')
f = violation_derate(a, 5.0)
check(abs(f - np.exp(-5.0 * 0.3)) < 1e-15, 'f == exp(-c·x) (实得 %.12f)' % f)
check(violation_derate([0.6] * 5, 5.0) < violation_derate([0.51] * 5, 5.0),
      'x 越大 f 越小 (单调)')
check(violation_derate([0.6] * 5, 10.0) < violation_derate([0.6] * 5, 5.0),
      'c 越大 f 越小 (单调)')

# ---------------------------------------------------------------------------
section('3. Cal_r1_continuous: 默认口径逐位不变 + 只缩放风机项')
# ---------------------------------------------------------------------------
rng = np.random.RandomState(7)
ok_bit, ok_only_tp, ok_mono = True, True, True
for _ in range(200):
    pr = float(rng.choice([300.0, 580.0, 980.0]))
    ptur = float(rng.uniform(0.0, 60.0))
    pbrule = float(rng.uniform(-10.0, 10.0))
    pen = float(rng.uniform(0.0, 120.0))
    wl = str(rng.choice(['LOW', 'MEDIUM', 'HIGH']))
    pl = str(rng.choice(['LOW', 'MEDIUM', 'HIGH']))
    soc = float(rng.uniform(0.5, 5.5))
    f = float(rng.uniform(0.1, 1.0))

    r_new = Cal_r1_continuous(pr, ptur, pbrule, pen, wl, pl, SOC=soc, turbine_derate=f)
    r_off = Cal_r1_continuous(pr, ptur, pbrule, pen, wl, pl, SOC=soc, turbine_derate=1.0)
    r_def = Cal_r1_continuous(pr, ptur, pbrule, pen, wl, pl, SOC=soc)   # 不传 = 1.0

    # 3a. 默认 (不传 turbine_derate) 与显式 1.0 逐位相同
    if not (r_off == r_def):
        ok_bit = False
    # 3b. 只有风机项被缩放: r(f) - r(1) == turbine_profit*(f-1)
    tp = M.get_wT(wl, pl) * pr * ptur * (1.0 / 60)
    if abs((r_new - r_off) - tp * (f - 1.0)) > 1e-9 * max(1.0, abs(r_off)):
        ok_only_tp = False
    # 3c. r1 单调不增 (f<=1 且 tp>=0)
    if r_new > r_off + 1e-12:
        ok_mono = False
check(ok_bit, '不传 turbine_derate == 显式 1.0, 200 组随机输入逐位相同')
check(ok_only_tp, 'r1(f)-r1(1) == turbine_profit·(f-1): 只有风机收益项被缩放')
check(ok_mono, 'f<=1 ⇒ r1 单调不增 (越界永不改善 r1), 200 组随机输入')

# 一个手算锚点
rp = Cal_r1_continuous(980.0, 40.0, -5.0, 30.0, 'HIGH', 'HIGH', SOC=3.0, turbine_derate=1.0)
rn = Cal_r1_continuous(980.0, 40.0, -5.0, 30.0, 'HIGH', 'HIGH', SOC=3.0, turbine_derate=0.5)
print('      手算锚点: r1(f=1)=%.10f, r1(f=0.5)=%.10f, 差=%.10f' % (rp, rn, rn - rp))
check(rn < rp, 'f=0.5 时 r1 变小')

# ---------------------------------------------------------------------------
section('4. Cal_Rew_RB: 记录列 (turbine_profit/revenue/degradation) 必须不变')
# ---------------------------------------------------------------------------
obs = np.array([580.0, 9.0, 3.0, 5.0])
obs2 = np.array([980.0, 9.0, 3.1, 6.0])
rbb_info = {'pbrule': -3.0, 'unfeasible_gap': 0.0, 'clip_amount': 0.0}
action_rbb = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.2, -3.0])
kw = dict(penalty_NN_H=12.0, time=1.0 / 60, ptur=40.0, pb_cmd=-3.0)
r1_off, _, d_off = Cal_Rew_RB(obs, obs2, action_rbb, rbb_info, turbine_derate=1.0, **kw)
r1_on, _, d_on = Cal_Rew_RB(obs, obs2, action_rbb, rbb_info, turbine_derate=0.25, **kw)
for k in ('turbine_profit', 'revenue', 'bess_profit', 'degradation'):
    same = (d_off[k] == d_on[k])
    check(same, 'details[%r] 不受 turbine_derate 影响 (%.12f == %.12f)' % (k, d_off[k], d_on[k]))
check(r1_on < r1_off, 'r1 受影响且变小 (%.6f < %.6f)' % (r1_on, r1_off))
check(abs((r1_off - r1_on) - d_off['turbine_profit'] * (1.0 - 0.25)) < 1e-9,
      'r1 的减少量恰为 turbine_profit·(1-f)')
# 关闭时必须逐位相同
r1_off2, _, d_off2 = Cal_Rew_RB(obs, obs2, action_rbb, rbb_info,
                                turbine_derate=violation_derate(action_rbb, 0.0), **kw)
check(r1_off2 == r1_off and d_off2['revenue'] == d_off['revenue'],
      'turbine_derate=1.0 路径与"关闭系数"路径逐位一致')

# ---------------------------------------------------------------------------
section('5. 静态接线: main 与 nine_case_test')
# ---------------------------------------------------------------------------
src_main = open('td3per/td3per_main.py', encoding='utf-8').read()
src_my = open('td3per/td3per_myfunc.py', encoding='utf-8').read()
check("'--viol_derate'" in src_main, 'main 有 CLI 开关 --viol_derate')
check('violation_derate(original_action_H, args.viol_derate)' in src_main,
      'main 在块首用**原始**动作算系数 (原动作, 不是 clip 后的)')
check('turbine_derate=viol_derate' in src_main, 'main 把系数传进 Cal_Rew_RB')
check('viol_derate_coef=args.viol_derate' in src_main,
      'main 把系数传给 nine_case_test (test9 与训练同口径)')
check('action_H_raw = agent_H.select_action(norm_obs)' in src_my,
      'nine_case_test 先留住原始动作再 CheckAction')
check('turbine_derate=_derate' in src_my, 'nine_case_test 把系数传进 Cal_Rew_RB')
check('def violation_derate(' in src_my and 'def violation_amount(' in src_my,
      'helper (violation_amount/violation_derate) 定义在 td3per_myfunc')
# 其它五个方法不应被牵连
import glob
others = [p for p in glob.glob('*/[a-z]*_myfunc.py') if 'td3per/' not in p.replace('\\', '/')]
bad = [p for p in others if 'violation_derate' in open(p, encoding='utf-8').read()]
check(not bad, '其它方法的 *_myfunc.py 未被牵连 (%d 个文件)' % len(others))

print('\n' + '=' * 60)
if FAIL:
    print('失败 %d 项:' % len(FAIL))
    for f in FAIL:
        print('  - ' + f)
    sys.exit(1)
print('全部通过 ✓')
