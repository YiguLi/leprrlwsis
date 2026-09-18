#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""六个 RL-BESS 下层数值口径的不变量测试 —— 不需要 MATLAB。

分两支：
  * **SOC 目标支**（td3per / td3prrbb / prle / td3prle）：口径集中在单一真源 `hsrl_soctarget.py`
    （`r2 = −1·min(clip_amount, 20)`、`c2 = +1·trackerr`、`d_block_L = 1.2`、地板 0.15、
    `cost_limit = 0.6`、λ2 ∈ [0.5, 2] 无量纲兑换率 (η2 = 0.02 乘性对数步长)）。本文件只查"接线是否都指向真源 + 是否把
    **命令值**交给 RBB（两段 clip 都计费）"；
    **四项目逐位等价**与目标映射/零成本动作性质见 `test_soctarget_uniform.py`。
  * **其它支**（per / prrbb 影子电价）：系数仍在各自项目里，本文件按项目核对取值，
    并断言"÷100 后物理量守恒"：d_block_L / c2系数 = 0.02 MWh、η2 × c2系数 = 10 / 7。
  * 三处的 `Cal_r2_PER_baseline`（LEGACY 基线）系数**刻意保持 ÷100 之前的原值**。
"""
import importlib
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors='replace')
    except Exception:
        pass

OLD_ETA = 0.01

# TD3 全系列已改用**残余动作** (命令 = 规则基线 + Δ); prle 是 DDPG 兄弟, 仍是绝对动作
TD3_RESIDUAL = ('td3per', 'td3prrbb', 'td3prle')

SOC_FAMILY = {                      # 项目 -> (main, myfunc)
    'td3per': ('td3per/td3per_main.py', 'td3per/td3per_myfunc.py'),
    'td3prrbb': ('td3prrbb/td3prrbb_main.py', 'td3prrbb/td3prrbb_myfunc.py'),
    'prle': ('prle/prle_main.py', 'prle/prle_myfunc.py'),
    'td3prle': ('td3prle/td3prle_main.py', 'td3prle/td3prle_myfunc.py'),
}
# 其它支: 项目 -> (main, c2系数, d_block_L, 地板, ÷100 前的 c2 系数, ÷100 前的 d_block_L)
OTHER = {
    'per': ('PER_main0.py', 10.0, 0.2, 0.025, 1000.0, 20.0),
    'prrbb': ('prrbb/prrbb_main.py', 7.0, 0.14, 0.0175, 700.0, 14.0),
}
LEGACY = {'prrbb': 'prrbb/prrbb_myfunc.py', 'prle': 'prle/prle_myfunc.py',
          'td3prrbb': 'td3prrbb/td3prrbb_myfunc.py'}


def src(p):
    return open(p, encoding='utf-8').read()


def one(pat, txt, tag):
    m = re.findall(pat, txt)
    assert len(m) == 1, (tag, m)
    return m[0]


def num_or_ref(txt, ref_pat, lit_pat, tag):
    """取一个量: 既接受 `hst.L_XXX` 形式, 也接受字面量（真源取值由另一个测试校验）。"""
    m = re.findall(ref_pat, txt)
    if m:
        assert len(m) == 1, (tag, m)
        return None
    return float(one(lit_pat, txt, tag))


print('=' * 84)
print('A. SOC 目标支: 四个项目都接在单一真源 hsrl_soctarget 上')
print('=' * 84)
import hsrl_soctarget as hst                                     # noqa: E402
assert (hst.L_CLIP_COEF, hst.L_TRACK_COEF, hst.L_D_BLOCK,
        hst.L_D_BLOCK_MIN, hst.L_COST_LIMIT) == \
       (1.0, 1.0, 1.2, 0.15, 0.6)
assert hst.L_CLIP_CAP == 20.0               # r2 封顶 (实测窗内容量 ≤20 ⇒ 不截断真实值)
_g = 0.98 * (1.0 / 60.0)
assert abs(hst.L_D_BLOCK * _g - 0.02) < 2e-3          # 物理等价: 允许搁置 0.02 MWh/块
# λ2 = 无量纲兑换率: 每 MW c2 至多值 2 MW r2 / 至少值 0.5 MW r2, 关于中性 1.0 对数对称
assert (hst.L_LAM_MIN, hst.L_LAM_MAX, hst.L_LAM_INIT) == (0.5, 2.0, 1.0)
assert abs(hst.L_LAM_MAX * hst.L_LAM_MIN - 1.0) < 1e-12
assert abs(hst.L_LAM_INIT ** 2 - hst.L_LAM_MAX * hst.L_LAM_MIN) < 1e-12
# 乘性对数步长 (看无量纲比值 EMA/d_block) + curriculum 以"块"计量
assert hst.L_EMA_LAM_LR == 0.02 and hst.L_CURRICULUM_BLOCKS == 50
for proj, (main_py, mf_py) in SOC_FAMILY.items():
    mt, ft = src(main_py), src(mf_py)
    if proj == 'td3per':
        # td3per 保留**本地包装**（转发真源，由 test_soctarget_uniform 第 4 节钉住逐位一致）
        assert 'def soc_track_slack(' in ft and 'L_CLIP_COEF = hst.L_CLIP_COEF' in ft \
            and 'L_SOC_COEF = hst.L_TRACK_COEF' in ft, proj
    else:
        assert 'import hsrl_soctarget as hst' in ft, (proj, 'myfunc 未接真源')
        for call in ('hst.soc_target_from_signal(', 'hst.lower_reward(',
                     'hst.lower_cost_soc_track(', 'hst.build_L_obs('):
            assert call in ft, (proj, call)
    assert 'import hsrl_soctarget as hst' in mt or proj == 'td3per', (proj, 'main 未接真源')
    assert re.search(r'(?:hst\.)?gap_penalty\(', mt), (proj, '块末 gap 罚未接真源')
    assert re.search(r'sum_r1 -= (?:gap_pen|c1_soc_track)\b', mt), \
        (proj, 'uniform 下块末追踪罚未进 r1')
    if proj in TD3_RESIDUAL:
        # 残余动作 (TD3 全系列): 命令 = 规则基线 + Δ ⇒ clip_amount 仍含两段 clip
        assert re.search(r'pb_cmd_L = float\(pb_rule\) \+ float\(delta_L\[0\]\)', mt), \
            (proj, '残余动作: 命令必须是 规则基线 + Δ')
        assert re.search(r'pb_original = float\(pb_cmd_L\)', mt), \
            (proj, '未把**命令值**交给 compute_pbrule ⇒ clip_amount 只含窗口那段 (|cmd|>10 死区)')
        assert re.search(r'original_action_L = delta_L', mt), (proj, '入池动作必须是 Δ')
    else:
        # prle (DDPG, 不在 td3 系列内): 仍是绝对动作, a_L 本身就是命令值
        assert re.search(r'pb_original = float\(np\.ravel\(np\.asarray\(original_action_L', mt), \
            (proj, '未把**命令值**交给 compute_pbrule ⇒ clip_amount 只含窗口那段 (|cmd|>10 死区)')
    # L 层对偶走单一真源 + λ2 初值/curriculum 计量显式 + 计数每块 1 次
    assert 'hst.dual_update_L(' in mt, (proj, 'L 对偶未走 hsrl_soctarget.dual_update_L')
    assert re.search(r'def build_L_obs_distilled\([^)]*n_rem=None', ft), \
    (proj, '观测构造器缺少 n_rem (块内剩余步数)')
    assert re.search(r'agent_L\.lam = hst\.L_LAM_INIT', mt), (proj, 'λ2 初值未显式设为真源')
    assert re.search(r'consecutive_satisfied_L >= hst\.L_CURRICULUM_BLOCKS', mt), \
        (proj, 'L curriculum 未按"块数"计量')
    assert mt.count('consecutive_satisfied_L += 1') == 1, \
        (proj, 'L curriculum 计数出现多次 ⇒ 仍嵌在按 UTD 计数的循环里')
    d = num_or_ref(mt, r'(?m)^\s*d_block_L = hst\.L_D_BLOCK\b',
                   r'(?m)^\s*d_block_L = ([0-9.]+)', proj + ' d_block_L')
    f = num_or_ref(mt, r'(?m)^\s*d_block_L_min = hst\.L_D_BLOCK_MIN\b',
                   r'(?m)^\s*d_block_L_min = ([0-9.]+)', proj + ' floor')
    c = num_or_ref(mt, r'agent_L\.cost_limit = hst\.L_COST_LIMIT',
                   r'agent_L\.cost_limit = ([0-9.]+)', proj + ' cost_limit')
    n_lo = len(re.findall(r'ema_lam=True, ema_alpha=0\.02, ema_lam_lr=hst\.L_EMA_LAM_LR', mt))
    n_hi = len(re.findall(r'ema_lam=True, ema_alpha=0\.02, ema_lam_lr=0\.01', mt))
    assert n_lo == 1 and n_hi == 1, (proj, n_lo, n_hi)
    for v, exp in ((d, 1.2), (f, 0.15), (c, 0.6)):
        assert v is None or abs(v - exp) < 1e-12, (proj, v, exp)
    body = re.sub(r'(?s)def Cal_r2_PER_baseline.*?(?=^def )', '', ft, flags=re.M)
    for bad in (r'shadow_profit\s*=',                       # 旧套利项赋值
                r'shadow_price = float\(action_rbb\[5\]\) \* 2\.0',   # 旧信号映射
                r'r2 = 0\.1 \* shadow_profit'):             # 旧 r2 公式
        assert not re.search(bad, body), (proj, '训练路径仍含 ' + bad)
    assert "reward_details['shadow_profit']" not in mt, proj
    print('  {:<9s} myfunc 走 hst.*; main: d_block_L={} 地板={} cost_limit={} η2 指向真源; 无影子项'
          .format(proj, d if d else 'hst.L_D_BLOCK', f if f else 'hst.L_D_BLOCK_MIN',
                  c if c else 'hst.L_COST_LIMIT'))

print('=' * 84)
print('B. 其它支 (影子电价口径, 未迁移): 系数与物理量守恒')
print('=' * 84)
for proj, (main_py, c2coef, dbl, floor, old_c2, old_dbl) in OTHER.items():
    mt = src(main_py)
    d = float(one(r'(?m)^\s*d_block_L = ([0-9.]+)', mt, proj))
    f = float(one(r'(?m)^\s*d_block_L_min = ([0-9.]+)', mt, proj))
    cl = float(one(r'agent_L\.cost_limit = ([0-9.]+)', mt, proj))
    n_lo = len(re.findall(r'ema_lam=True, ema_alpha=0\.02, ema_lam_lr=1\.0', mt))
    n_hi = len(re.findall(r'ema_lam=True, ema_alpha=0\.02, ema_lam_lr=0\.01', mt))
    assert (d, f, cl) == (dbl, floor, 0.1), (proj, d, f, cl)
    assert n_lo == 1 and n_hi == 1, (proj, n_lo, n_hi)
    assert abs(d / c2coef - old_dbl / old_c2) < 1e-12, (proj, d, c2coef)
    assert abs(1.0 * c2coef - OLD_ETA * old_c2) < 1e-12, (proj, c2coef)
    print('  {:<9s} c2系数={:<5g} d_block_L={:<6g} 地板={:<7g} cost_limit={} η2=1.0 | '
          'd/c2={:g} (原 {:g}) η2·c2={:g} (原 {:g})'
          .format(proj, c2coef, d, f, cl, d / c2coef, old_dbl / old_c2,
                  1.0 * c2coef, OLD_ETA * old_c2))

print('=' * 84)
print('C. LEGACY 基线 (Cal_r2_PER_baseline) 刻意保持 ÷100 之前的原值')
print('=' * 84)
for proj, p in LEGACY.items():
    t = src(p)
    m = re.search(r'(?s)def Cal_r2_PER_baseline.*?(?=^def )', t, flags=re.M)
    assert m, proj
    assert re.search(r'10\.0 \* shadow_profit - 700\.0 \* soc_dev', m.group(0)), \
        (proj, '基线被改动')
    print('  {:<9s} 仍为 10·shadow_profit − 700·soc_deviation (对照旧 PER 设计)'.format(proj))

print('=' * 84)
print('D. r2 计**两段 clip**: 传命令值不改 pbrule、clip_amount = 两段合计、病态饱和有封顶')
print('=' * 84)
assert hst.lower_reward(1e6) == -hst.L_CLIP_CAP                 # 封顶生效
assert hst.lower_reward(hst.L_CLIP_CAP + 0.5, cap=float('inf')) == -(hst.L_CLIP_CAP + 0.5)
# 四个项目各自的 RuleBasedBESS: rbb 版 (td3per) 与 prrbb 家族版 (其余三个, 三份逐字节相同)
RBB = {
    'td3per': 'rbb.rbb_rule_bess',
    'td3prrbb': 'td3prrbb.td3prrbb_rule_bess',
    'prle': 'prrbb.prrbb_rule_bess',
    'td3prle': 'td3prle.td3prle_rule_bess',
}
rng = np.random.RandomState(11)
mx_inv = mx_two = mx_gain = mx_lin = 0.0
n_inv = n_two = n_over = n_deg = n_cap = 0
for proj, mod in RBB.items():
    RB = importlib.import_module(mod).RuleBasedBESS()
    for _ in range(1000):
        E = float(rng.uniform(0.0, 6.0))
        ptur = float(rng.uniform(0.0, 60.0))
        pg = float(rng.uniform(0.0, 60.0))
        cmd = float(rng.uniform(-60.0, 60.0))
        s = float(np.clip(cmd, -10.0, 10.0))
        u_cmd, i_cmd = RB.compute_pbrule(pb_original=cmd, ptur=ptur, pg_prev=pg, E=E)
        u_s, i_s = RB.compute_pbrule(pb_original=s, ptur=ptur, pg_prev=pg, E=E)
        mx_inv = max(mx_inv, abs(u_cmd - u_s))
        n_inv += 1
        if abs(u_cmd) > 10.0 + 1e-9:            # SOC 越带的退化窗口 (实测真实 run 中 |pbrule|≤10)
            n_deg += 1
            continue
        over = max(abs(cmd) - 10.0, 0.0)
        mx_two = max(mx_two, abs(i_cmd['clip_amount'] - (i_s['clip_amount'] + over)))
        n_two += 1
        if over > 1e-9:
            n_over += 1
            # 死区消失: 越界后 r2 必须仍随 |cmd| 变负; 未触顶时恰好多扣 over (斜率 1)
            r2_cmd = hst.lower_reward(i_cmd['clip_amount'])
            r2_s = hst.lower_reward(i_s['clip_amount'])
            assert r2_cmd <= r2_s + 1e-12, (proj, cmd, r2_cmd, r2_s)
            exp = -(min(i_cmd['clip_amount'], hst.L_CLIP_CAP)
                    - min(i_s['clip_amount'], hst.L_CLIP_CAP))
            mx_gain = max(mx_gain, abs((r2_cmd - r2_s) - exp))
            if i_cmd['clip_amount'] <= hst.L_CLIP_CAP + 1e-12:
                mx_lin = max(mx_lin, abs((r2_cmd - r2_s) + over))
            else:
                n_cap += 1
    print('  {:<9s} 1000 组: 传命令值后 pbrule 不变 ✓  两段归因 ✓  越界样本 {} 组'
          .format(proj, n_over))
assert mx_inv < 1e-12, mx_inv
assert mx_two < 1e-12, mx_two
assert mx_gain < 1e-12, mx_gain
assert mx_lin < 1e-12, mx_lin
print('  {} 组: max|pbrule(cmd) − pbrule(clip(cmd,±10))| = {:.3e}（pbrule 与环境零变化）'
      .format(n_inv, mx_inv))
print('  {} 组 (|pbrule|≤10): max|clip_amount − (窗口段 + 越界段)| = {:.3e}；'
      '越界 {} 组 r2 恰好多扣 |cmd|−10（斜率 1，无死区，偏差 {:.1e}），触顶 {} 组'
      .format(n_two, mx_two, n_over, mx_lin, n_cap))
print('  SOC 越带的退化窗口样本 {} 组已跳过 (|pbrule|>10, 真实 run 未见)'.format(n_deg))

print('=' * 84)
print('E. λ2 对偶律 (乘性, 只吃无量纲比值): 不动点 / 饱和 / 回落 / 刻度免疫 / 限幅')
print('=' * 84)


class _Ag:
    """最小假体: 只需 .ema_cost 与 .lam。"""
    def __init__(self, ema, lam=hst.L_LAM_INIT):
        self.ema_cost, self.lam = ema, lam


# 不动点: EMA_c2 = d_block_L (预算刚好满足) ⇒ λ2 不动
_a = _Ag(hst.L_D_BLOCK)
assert abs(hst.dual_update_L(_a, hst.L_D_BLOCK) - hst.L_LAM_INIT) < 1e-15
# 违约 (实测当前工况 EMA/d ≈ 13/1.2 ≈ 11) ⇒ 上升, 4 块内顶到上限
_a = _Ag(11.0 * hst.L_D_BLOCK)
_traj = [hst.dual_update_L(_a, hst.L_D_BLOCK) for _ in range(4)]
assert _traj[0] > hst.L_LAM_INIT * 1.15 and _traj[-1] == hst.L_LAM_MAX, _traj
# 超额完成 (比值 0.5) ⇒ 回落, 且不越下界
_a = _Ag(0.5 * hst.L_D_BLOCK)
_traj2 = [hst.dual_update_L(_a, hst.L_D_BLOCK) for _ in range(80)]
assert _traj2[-1] == hst.L_LAM_MIN, _traj2[-1]
# 尺度免疫: 同一比值、不同绝对刻度 ⇒ 同一乘子 (加性式 λ+=η(EMA−d) 做不到)
_x1, _x2 = _Ag(3.0 * 1.0), _Ag(3.0 * 1000.0)
assert abs(hst.dual_update_L(_x1, 1.0) - hst.dual_update_L(_x2, 1000.0)) < 1e-15
# ema 未建立 ⇒ 完全不动
assert hst.dual_update_L(_Ag(None), hst.L_D_BLOCK) is None
assert hst.dual_update_L(None, hst.L_D_BLOCK) is None
# 限幅: 任何比值迭代任意步都不越界
for _r in (1e-6, 1.0, 1e6):
    _a = _Ag(_r * hst.L_D_BLOCK, lam=hst.L_LAM_INIT)
    for _ in range(200):
        _v = hst.dual_update_L(_a, hst.L_D_BLOCK)
    assert hst.L_LAM_MIN - 1e-12 <= _v <= hst.L_LAM_MAX + 1e-12, (_r, _v)
print('  EMA=d_block ⇒ 不动; 比值 11 ⇒ {} 块到上限 {:.1f}; 比值 0.5 ⇒ 回落到下限 {:.1f}'
      .format(len(_traj), hst.L_LAM_MAX, hst.L_LAM_MIN))
print('  同比值不同刻度 ⇒ 同一乘子 (无量纲); ema 未建立 ⇒ 不动; 任意比值恒在 [0.5, 2] 内')

print('=' * 84)
print('TEST OK')
