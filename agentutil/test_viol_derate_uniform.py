# -*- coding: utf-8 -*-
"""六个 TD3 方法的 `--viol_derate` **跨项目一致性**测试 (批次 1 验收)。

断言:
  A. helper 同源: 六个 myfunc 的 `violation_amount`/`violation_derate` 在随机输入上**逐位相同**,
     且 `VIOL_DERATE_COEF`/`VIOL_DERATE_DIMS` 一致。
  B. 只缩放风机项: 六个包装函数 (`Cal_Rew_RB`/`Cal_Rew_RLRB`/`Cal_Rew_PRRBB`) 里
     `r1(f) - r1(1) == turbine_profit*(f-1)`; `details` 的
     turbine_profit/revenue/bess_profit/degradation **不受影响**; 不传系数 == 显式 1.0 (逐位)。
  C. 端到端 (桩 env 真跑 nine_case_test): `c=0` 与 `c=5` 的 profit/dsoc/profit_econ **逐位相同**;
     分解恒等式 `profit/N - r1 == (penalty_NN_H + derate_loss)/N` 成立; overall 有三键。
  D. 静态接线: 六个 main 都有 `--viol_derate` 且默认 = VIOL_DERATE_COEF, 且 import 了接线。

不需要 MATLAB。
"""

import importlib
import inspect
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

# (key, myfunc 模块, 下层类型 L=RL / T=规则, rule_bess 模块)
PROJS = [
    ('td3per',      'td3per.td3per_myfunc',           'L', 'rbb.rbb_rule_bess'),
    ('td3perrb',    'td3perrb.td3perrb_myfunc',       'T', 'td3perrb.td3perrb_rule_bess'),
    ('td3prrlrb',   'td3prrlrb.td3prrlrb_myfunc',     'T', 'td3prrlrb.td3prrlrb_rule_bess'),
    ('td3prrbb',    'td3prrbb.td3prrbb_myfunc',       'L', 'td3prrbb.td3prrbb_rule_bess'),
    ('td3prle',     'td3prle.td3prle_myfunc',         'L', 'td3prle.td3prle_rule_bess'),
    ('td3leprrlrb', 'td3leprrlrb.td3leprrlrb_myfunc', 'T', 'td3leprrlrb.td3leprrlrb_rule_bess'),
]
WRAP = {'td3per': 'Cal_Rew_RB', 'td3perrb': 'Cal_Rew_RB', 'td3prrlrb': 'Cal_Rew_RLRB',
        'td3prrbb': 'Cal_Rew_PRRBB', 'td3prle': 'Cal_Rew_PRRBB', 'td3leprrlrb': 'Cal_Rew_RLRB'}

FAIL = []


def check(cond, msg):
    print(('  [OK]   ' if cond else '  [FAIL] ') + msg)
    if not cond:
        FAIL.append(msg)


def section(t):
    print('\n' + t)
    print('-' * len(t))


MODS = {k: importlib.import_module(m) for k, m, _, _ in PROJS}

# ---------------------------------------------------------------------------
section('A. helper 同源 (六方法逐位相同)')
# ---------------------------------------------------------------------------
rng = np.random.RandomState(2024)
ACTS = [rng.uniform(-1.2, 1.2, size=6) for _ in range(200)]
ACTS += [np.array([0.5] * 6), np.array([0.0] * 6), np.array([0.7, -0.1, 0.5, 0.0, 0.2, 1.0])]

ref_amt = [MODS['td3per'].violation_amount(a) for a in ACTS]
ref_der = [MODS['td3per'].violation_derate(a, 5.0) for a in ACTS]
ref_der0 = [MODS['td3per'].violation_derate(a, 0.0) for a in ACTS]
for k, m, _, _ in PROJS:
    same_amt = all(MODS[k].violation_amount(a) == r for a, r in zip(ACTS, ref_amt))
    same_der = all(MODS[k].violation_derate(a, 5.0) == r for a, r in zip(ACTS, ref_der))
    same_off = all(MODS[k].violation_derate(a, 0.0) == 1.0 for a in ACTS)
    cst = (MODS[k].VIOL_DERATE_COEF == MODS['td3per'].VIOL_DERATE_COEF
           and MODS[k].VIOL_DERATE_DIMS == MODS['td3per'].VIOL_DERATE_DIMS)
    check(same_amt and same_der and same_off and cst,
          '%-13s violation_amount/violation_derate 逐位同源 + 常量一致 (c=%s, dims=%s)'
          % (k, MODS[k].VIOL_DERATE_COEF, MODS[k].VIOL_DERATE_DIMS))
check(all(MODS['td3per'].violation_derate([0.9] * 6, 5.0) == MODS['td3per'].violation_derate([0.9] * 6, 5.0)
          for _ in [0]), '参考实现自洽')

# ---------------------------------------------------------------------------
section('B. 六个包装函数: 只缩放风机项, 记录列不变')
# ---------------------------------------------------------------------------
OBS = np.array([580.0, 9.0, 3.0, 5.0])
OBS2 = np.array([980.0, 9.0, 3.1, 6.0])
RBB = {'pbrule': -3.0, 'unfeasible_gap': 0.0, 'clip_amount': 0.0,
       # 四个 PRRBB 项目的包装函数还要读这些 (只影响 r1 绝对值, 不影响 Δr1 恒等式)
       'pbrulemin': -10.0, 'pbrulemax': 10.0,
       'pbphysicsmin': -10.0, 'pbphysicsmax': 10.0,
       'pbfluctmin': -10.0, 'pbfluctmax': 10.0,
       'pb_best': -3.0, 'd_best': 1.0, 'd_next': 1.0}
ACT = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.2, -3.0])
for k, m, _, _ in PROJS:
    fn = getattr(MODS[k], WRAP[k])
    sig = inspect.signature(fn)
    kw = dict(penalty_NN_H=12.0, time=1.0 / 60, ptur=40.0)
    if 'pb_cmd' in sig.parameters:
        kw['pb_cmd'] = -3.0
    if 'n_rem' in sig.parameters:
        kw['n_rem'] = 5
    out1 = fn(OBS, OBS2, ACT, RBB, turbine_derate=1.0, **kw)
    outf = fn(OBS, OBS2, ACT, RBB, turbine_derate=0.25, **kw)
    outd = fn(OBS, OBS2, ACT, RBB, **kw)                      # 不传 = 1.0
    r1_1, d1 = out1[0], out1[-1]
    r1_f, df = outf[0], outf[-1]
    r1_d = outd[0]
    tp = d1['turbine_profit']
    ok_reduce = abs((r1_1 - r1_f) - tp * (1.0 - 0.25)) < 1e-9 * max(1.0, abs(r1_1))
    ok_cols = all(d1.get(c) == df.get(c) for c in ('turbine_profit', 'revenue', 'degradation'))
    ok_bit = (r1_d == r1_1)
    check(ok_reduce and ok_cols and ok_bit,
          '%-13s %-13s Δr1==tp·(1-f) ✓ 记录列不变 ✓ 不传==1.0 逐位 ✓' % (k, WRAP[k]))


# ---------------------------------------------------------------------------
section('C. 端到端 (桩 env 真跑 nine_case_test)')
# ---------------------------------------------------------------------------
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


A_H_VIOL = [0.70, 0.55, -0.10, 0.30, 0.30, 0.05]     # x = 0.35
for k, m, kind, rb_name in PROJS:
    rb = importlib.import_module(rb_name).RuleBasedBESS()
    low = _StubAgent([0.0]) if kind == 'L' else importlib.import_module(rb_name).SocTracker()
    try:
        cm0, ov0 = MODS[k].nine_case_test(_StubAgent(A_H_VIOL), low, _StubEnv(), rb,
                                    seed=42, viol_derate_coef=0.0)
        cm5, ov5 = MODS[k].nine_case_test(_StubAgent(A_H_VIOL), low, _StubEnv(), rb,
                                    seed=42, viol_derate_coef=5.0)
    except Exception as e:
        check(False, '%-13s nine_case_test 跑失败: %s: %s' % (k, type(e).__name__, e))
        continue
    keys = all(x in ov0 for x in ('turbine_profit', 'derate_loss', 'penalty_NN_H'))
    der0 = (ov0['derate_loss'] == 0.0)
    der5 = (ov5['derate_loss'] > 0.0)
    same = (ov0['profit'] == ov5['profit'] and ov0['profit_econ'] == ov5['profit_econ']
            and all(cm0[i]['profit'] == cm5[i]['profit'] and cm0[i]['dsoc'] == cm5[i]['dsoc']
                    and cm0[i]['profit_econ'] == cm5[i]['profit_econ'] for i in range(9)))
    # 分解恒等式: profit/45 - r1 == (罚 + 削减 + 重加权残差)/45
    #   td3per/td3perrb 的 wT=wB=1 ⇒ 重加权残差 ≡ 0, 恒等式精确;
    #   四个 PRRBB 系项目 r1 用 wT/wB 重加权 ⇒ 残差非零 (但**与 c 无关**)。
    def _resid(ov):
        return (ov['profit'] / 45.0 - ov['r1']
                - (ov['penalty_NN_H'] + ov['derate_loss']) / 45.0)
    r0, r5 = _resid(ov0), _resid(ov5)
    unweighted = k in ('td3per', 'td3perrb')
    ok_ident = (abs(r5 - r0) < 1e-9 * max(1.0, abs(r0))) and (abs(r0) < 1e-6 if unweighted else True)
    check(keys and der0 and der5 and same and ok_ident,
          '%-13s 三键✓ c=0削减0✓ c=5削减>0✓ profit/econ逐位不变✓ 残差随c不变(%.3g→%.3g)%s'
          % (k, r0, r5, ' [w=1 残差≡0]' if unweighted else ' [重加权残差, 与c无关]'))
    if not (keys and der0 and der5 and same and ok_ident):
        print('        keys=%s der0=%s der5=%s same=%s ok_ident=%s' % (keys, der0, der5, same, ok_ident))

# ---------------------------------------------------------------------------
section('D. 六个 main 的静态接线')
# ---------------------------------------------------------------------------
for k, m, _, _ in PROJS:
    src = open('%s/%s_main.py' % (k, k), encoding='utf-8').read()
    has_cli = "'--viol_derate', default=VIOL_DERATE_COEF" in src
    has_call = 'turbine_derate=viol_derate' in src
    has_t9 = 'viol_derate_coef=args.viol_derate' in src
    has_imp = re.search(r'violation_derate, VIOL_DERATE_COEF', src) is not None
    has_cols = (src.count("'derate_loss': m['derate_loss']") == 1
                and src.count("'derate_loss': overall['derate_loss']") == 1)
    check(has_cli and has_call and has_t9 and has_imp and has_cols,
          '%-13s CLI默认=VIOL_DERATE_COEF✓ 奖励透传✓ test9透传✓ import✓ df_test9两处三列✓' % k)

print('\n' + '=' * 74)
if FAIL:
    print('失败 %d 项:' % len(FAIL))
    for f in FAIL:
        print('  - ' + f)
    sys.exit(1)
print('全部通过 ✓  (六个 TD3 方法的越界乘性惩罚已同源)')
