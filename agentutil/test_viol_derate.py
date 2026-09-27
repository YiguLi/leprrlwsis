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
section('6. nine_case_test 端到端: 分解三列 + 记录列不变 (桩 env, 不需 MATLAB)')
# ---------------------------------------------------------------------------
from td3per.td3per_myfunc import nine_case_test
from rbb.rbb_rule_bess import RuleBasedBESS


class _StubAgent:
    def __init__(self, a, nb_states=6):
        self.a = np.asarray(a, dtype=float).ravel().copy()
        self.is_training = True
        self.nb_states = nb_states
        self.nb_actions = self.a.size

    def select_action(self, obs, *a, **k):
        return self.a.copy()

    def reset(self, obs=None, *a, **k):
        self.s_t = obs
        return None


class _StubEnv:
    """只实现 nine_case_test 需要的三个方法 (PowerOutput 用确定性公式代替 MATLAB)。"""

    def __init__(self, n_turbine=5):
        self.num_turbine = n_turbine
        self.max_E, self.eta, self.time, self.E = 6.0, 0.98, 1.0 / 60, 5.0
        self._i, self._wind, self._price = 0, [], []

    def PowerOutput(self, u, act_wind):
        a = np.clip(np.asarray(act_wind, dtype=float).ravel()[:self.num_turbine], 0.0, 0.5)
        return float(a.sum() * 10.0 * min(max(float(u), 0.0), 12.0) / 12.0)

    def ScenarioReset(self, wind_seq, price_seq):
        self._wind, self._price, self._i = list(wind_seq), list(price_seq), 0
        self.E = 5.0
        pw = self.PowerOutput(self._wind[0], np.ones(self.num_turbine) * 0.3333)
        return np.array([float(self._price[0]), float(self._wind[0]), self.E, pw])

    def ScenarioStep(self, applied_action_H, act_bat, pw_tru):
        self._i = min(self._i + 1, len(self._wind) - 1)
        u, pr = float(self._wind[self._i]), float(self._price[self._i])
        pw2 = self.PowerOutput(u, applied_action_H)
        cmd = float(np.asarray(act_bat, dtype=float).ravel()[0])
        P_C, P_D = max(0.0, cmd), abs(min(cmd, 0.0))
        E_t = self.E + (P_C * self.eta - P_D / self.eta) * self.time
        if E_t > self.max_E:
            P_C = (self.max_E - self.E) / self.eta / self.time
            self.E = self.max_E
        elif E_t < 0.0:
            P_D = self.E * self.eta / self.time
            self.E = 0.0
        else:
            self.E = E_t
        P_G = float(pw_tru) - P_C + P_D
        return (pw2, np.array([pr, u, self.E, P_G]), np.array([P_C - P_D]),
                0.0, False, False, {'P_F': 0.0, 'pbrule': P_C - P_D, 'P_G': P_G})


def _run_h(coef, a_H):
    """用越界的上层动作真跑一次 45 步 9-case 测试。"""
    return nine_case_test(_StubAgent(a_H), _StubAgent([0.0]), _StubEnv(),
                          RuleBasedBESS(), seed=42, viol_derate_coef=coef)


A_H_VIOL = [0.70, 0.55, -0.10, 0.30, 0.30, 0.05]      # x = 0.20+0.05+0.10 = 0.35
cm0, ov0 = _run_h(0.0, A_H_VIOL)
cm5, ov5 = _run_h(5.0, A_H_VIOL)
# None ⇒ 用模块默认 VIOL_DERATE_COEF (现为 0.0 = 关闭)。动态跟随常量, 改默认值不会再破测试。
cmD, ovD = _run_h(M.VIOL_DERATE_COEF, A_H_VIOL)
cmN, ovN = _run_h(None, A_H_VIOL)

for k in ('turbine_profit', 'derate_loss', 'penalty_NN_H'):
    check(k in ov0 and k in cm0[0], 'overall/case 都新增了 %r' % k)
check(ov0['derate_loss'] == 0.0, 'c=0 ⇒ derate_loss 恰为 0 (实得 %.12f)' % ov0['derate_loss'])
check(ov5['derate_loss'] > 0.0, 'c=5 且动作越界 ⇒ derate_loss > 0 (实得 %.6f)' % ov5['derate_loss'])
check(any(m['derate_loss'] > 0 for m in cm5), 'per-case 也记录了 derate_loss')

# 记录列不变 (用户硬性要求): 轨迹不依赖奖励 ⇒ profit/dsoc/profit_econ 逐位相同
same = all(cm0[i]['profit'] == cm5[i]['profit'] and cm0[i]['dsoc'] == cm5[i]['dsoc']
           and cm0[i]['profit_econ'] == cm5[i]['profit_econ'] for i in range(9))
check(same, '**per-case profit/dsoc/profit_econ 在 c=0 与 c=5 之间逐位相同** (记录列不变)')
check(ov0['profit'] == ov5['profit'] and ov0['profit_econ'] == ov5['profit_econ'],
      '**overall profit/profit_econ 逐位相同**')
check(ov5['r1'] < ov0['r1'], 'c=5 的 r1 更低 (%.4f < %.4f)' % (ov5['r1'], ov0['r1']))

# 恒等式: profit/45 - r1 == (penalty_NN_H + derate_loss)/45   (test9_compare [6] 依赖此式)
for tag, ov in (('c=0', ov0), ('c=5', ov5)):
    lhs = ov['profit'] / 45.0 - ov['r1']
    rhs = (ov['penalty_NN_H'] + ov['derate_loss']) / 45.0
    check(abs(lhs - rhs) < 1e-9 * max(1.0, abs(rhs)),
          'overall 恒等式 %s: profit/45-r1 == (罚+削减)/45 (%.10f vs %.10f)' % (tag, lhs, rhs))
for tag, cms in (('c=0', cm0), ('c=5', cm5)):
    err = max(abs(m['profit'] / 5.0 - m['r1'] - (m['penalty_NN_H'] + m['derate_loss']) / 5.0)
              for m in cms)
    check(err < 1e-9, 'per-case 恒等式 %s: 最大偏差 %.2e' % (tag, err))

check(ovN['r1'] == ovD['r1'] and cmN[0]['derate_loss'] == cmD[0]['derate_loss'],
      'viol_derate_coef=None 等价于模块默认 VIOL_DERATE_COEF (=%.1f, 现为关闭)' % M.VIOL_DERATE_COEF)
check(M.VIOL_DERATE_COEF == 0.0,
      '默认值 = 0 (关闭): c=5.0 经 run14 实测有害, 已回退 (见 README)')

# ---------------------------------------------------------------------------
section('7. 静态接线: main 与 nine_case_test')
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
# 分解三列必须落盘 (test9_compare [6] 消费)
check(src_my.count("pc['turbine_profit'] += details['turbine_profit']") == 1
      and src_my.count("pc['derate_loss'] += details['turbine_profit'] * (1.0 - _derate)") == 1,
      'nine_case_test 逐步累加 turbine_profit / derate_loss')
check("'turbine_profit': m['turbine_profit']" in src_main
      and "'turbine_profit': overall['turbine_profit']" in src_main,
      'main 把 turbine_profit/derate_loss/penalty_NN_H 写进 df_test9 (per-case + OVERALL)')
src_tool = open('agentutil/test9_compare.py', encoding='utf-8').read()
check('derate_loss' in src_tool and 'turbine_profit' in src_tool
      and 'penalty_NN_H' in src_tool and '未分离' in src_tool,
      'test9_compare [6] 读三列并分开报 (缺列时退回"未分离")')
src_model = open('model.py', encoding='utf-8').read()
check('故意不做有界压缩' in src_model and 'tanh' in src_model,
      'model.py 在 Actor 输出处注明 tanh 有害 (含实测依据)')
# 六个 TD3 方法**应有** helper (批次 1 已同步); **非 TD3** 方法不得被牵连
import glob
SIX_TD3 = {'td3per', 'td3perrb', 'td3prrlrb', 'td3prrbb', 'td3prle', 'td3leprrlrb'}
others = [p for p in glob.glob('*/[a-z]*_myfunc.py')
          if os.path.basename(os.path.dirname(p)) not in SIX_TD3]
bad = [p for p in others if 'violation_derate' in open(p, encoding='utf-8').read()]
check(not bad, '非 TD3 方法的 *_myfunc.py 未被牵连 (%d 个: %s)'
      % (len(others), ', '.join(sorted(os.path.basename(os.path.dirname(p)) for p in others))))
missing = [k for k in SIX_TD3
           if 'violation_derate' not in open('%s/%s_myfunc.py' % (k, k), encoding='utf-8').read()]
check(not missing, '六个 TD3 方法的 myfunc 都已同步 helper (缺: %s)' % (missing or '无'))

print('\n' + '=' * 60)
if FAIL:
    print('失败 %d 项:' % len(FAIL))
    for f in FAIL:
        print('  - ' + f)
    sys.exit(1)
print('全部通过 ✓')
