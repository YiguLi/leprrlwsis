#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""六个 RL-BESS 下层数值口径的不变量测试 —— 不需要 MATLAB。

分两支：
  * **SOC 目标支**（td3per / td3prrbb / prle）：口径集中在单一真源 `hsrl_soctarget.py`
    （`r2 = −1·clip`、`c2 = +1·trackerr`、`d_block_L = 1.2`、地板 0.15、`cost_limit = 0.6`、
    L 对偶 η2 = 1.0）。本文件只查"接线是否都指向真源"；
    **三项目逐位等价**与目标映射/零成本动作性质见 `test_soctarget_uniform.py`。
  * **其它支**（per / prrbb 影子电价、td3prle 目标中心二次势）：系数仍在各自项目里，
    本文件按项目核对取值，并断言"÷100 后物理量守恒"：
        d_block_L / c2系数 = 0.02 MWh（td3prle 为 4 MWh²）、η2 × c2系数 = 10 / 7 / 0.1。
  * 三处的 `Cal_r2_PER_baseline`（LEGACY 基线）系数**刻意保持 ÷100 之前的原值**。
"""
import os
import re
import sys

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
        hst.L_D_BLOCK_MIN, hst.L_COST_LIMIT, hst.L_EMA_LAM_LR) == \
       (1.0, 1.0, 1.2, 0.15, 0.6, 0.16)
_g = 0.98 * (1.0 / 60.0)
assert abs(hst.L_D_BLOCK * _g - 0.02) < 2e-3          # 物理等价: 允许搁置 0.02 MWh/块
assert abs(hst.L_EMA_LAM_LR / _g - 10.0) < 0.3        # 对偶物理步长与旧口径一致
for proj, (main_py, mf_py) in SOC_FAMILY.items():
    mt, ft = src(main_py), src(mf_py)
    if proj == 'td3per':
        # td3per 保留**本地包装**（转发真源，由 test_soctarget_uniform 第 4 节钉住逐位一致）
        assert 'def soc_track_slack(' in ft and 'L_CLIP_COEF = hst.L_CLIP_COEF' in ft \
            and 'L_SOC_COEF = hst.L_TRACK_COEF' in ft, proj
    else:
        assert 'import hsrl_soctarget as hst' in ft, (proj, 'myfunc 未接真源')
        for call in ('hst.soc_target_from_signal(', 'hst.lower_reward(',
                     'hst.lower_cost_soc_track(', 'hst.obs_target_dim('):
            assert call in ft, (proj, call)
    assert 'import hsrl_soctarget as hst' in mt or proj == 'td3per', (proj, 'main 未接真源')
    assert re.search(r'(?:hst\.)?gap_penalty\(', mt), (proj, '块末 gap 罚未接真源')
    assert re.search(r'sum_r1 -= (?:gap_pen|c1_soc_track)\b', mt), \
        (proj, 'uniform 下块末追踪罚未进 r1')
    d = num_or_ref(mt, r'(?m)^\s*d_block_L = hst\.L_D_BLOCK\b',
                   r'(?m)^\s*d_block_L = ([0-9.]+)', proj + ' d_block_L')
    f = num_or_ref(mt, r'(?m)^\s*d_block_L_min = hst\.L_D_BLOCK_MIN\b',
                   r'(?m)^\s*d_block_L_min = ([0-9.]+)', proj + ' floor')
    c = num_or_ref(mt, r'agent_L\.cost_limit = hst\.L_COST_LIMIT',
                   r'agent_L\.cost_limit = ([0-9.]+)', proj + ' cost_limit')
    n_lo = len(re.findall(r'ema_lam=True, ema_alpha=0\.02, ema_lam_lr=(?:hst\.L_EMA_LAM_LR|1\.0)', mt))
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
print('TEST OK')
