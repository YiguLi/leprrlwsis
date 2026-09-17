#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RL-BESS 下层数值口径 ÷100 的不变量测试 —— 不需要 MATLAB。

口径(用户决定):
  下层值函数的目标应贴近 0(r2 理想为 0、c2 理想为 0), 因此把 L 层的**数值单位**
  统一降两个数量级: r2 的 clip 系数 100→1、shadow 系数 10→0.1、c2 系数 ÷100;
  `d_block_L`/地板/`cost_limit` ÷100(同一物理容忍度), L 的加性对偶步长 η2 ×100
  (λ2 轨迹不变, 因为 ema_c2 与 d 同步缩小)。H 层 `--value_scale` 与本文件无关。

断言分三层:
  A. 六个 RL-BESS 项目的系数/预算/步长 → 目标值;
  B. **物理量守恒**: d_block_L / c2系数 与 η2 × c2系数 与 ÷100 之前一致;
  C. 动态: 真的以新系数算出 r2/c2(导入各 myfunc);
  外加: `Cal_r2_PER_baseline`(LEGACY 基线) 系数**刻意保持原值**。
"""
import importlib
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)                                         # 相对数据/输出路径以仓库根为基准
sys.path.insert(0, ROOT)                               # 项目模块 (hsrl_*, myfunc, util, td3*/...)
sys.path.insert(0, _HERE)                              # agentutil 内互相 import
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors='replace')
    except Exception:
        pass

OLD_ETA = 0.01          # ÷100 之前的 L 对偶步长

# 项目: (main, myfunc 或 None, clip系数, shadow系数, c2系数, d_block_L, 地板,
#        ÷100 之前的 c2 系数, ÷100 之前的 d_block_L)
CASES = {
    'per':      ('PER_main0.py',                None,                        1.0,  0.1, 10.0, 0.2,  0.025,  1000.0, 20.0),
    'prrbb':    ('prrbb/prrbb_main.py',         'prrbb.prrbb_myfunc',        1.0,  0.1, 7.0,  0.14, 0.0175, 700.0,  14.0),
    'prle':     ('prle/prle_main.py',           'prle.prle_myfunc',          1.0,  0.1, 7.0,  0.14, 0.0175, 700.0,  14.0),
    'td3per':   ('td3per/td3per_main.py',       'td3per.td3per_myfunc',      1.0,  None, 10.0, 0.2,  0.025,  1000.0, 20.0),
    'td3prrbb': ('td3prrbb/td3prrbb_main.py',   'td3prrbb.td3prrbb_myfunc',  1.0,  0.1, 7.0,  0.14, 0.0175, 700.0,  14.0),
    'td3prle':  ('td3prle/td3prle_main.py',     'td3prle.td3prle_myfunc',    1.0,  None, 0.1,  0.4,  0.4,    10.0,   40.0),
}


def src(p):
    return open(p, encoding='utf-8').read()


def one(pat, txt, tag):
    m = re.findall(pat, txt)
    assert len(m) == 1, (tag, m)
    return m[0]


print('=' * 78)
print('A/B. 系数·预算·步长 与 物理量守恒')
print('=' * 78)
for proj, (main_py, myfunc, clip, shadow, c2coef, dbl, floor, old_c2, old_dbl) in CASES.items():
    mt, ft = src(main_py), (src(myfunc.replace('.', '/') + '.py') if myfunc else src(main_py))

    # --- 系数 (代码行, 非文档) ---
    if proj == 'td3per':
        assert one(r'(?m)^L_CLIP_COEF = ([0-9.]+)$', ft, proj + ' L_CLIP_COEF') == repr(clip)
        assert one(r'(?m)^L_SOC_COEF = ([0-9.]+)$', ft, proj + ' L_SOC_COEF') == repr(c2coef)
    elif proj == 'td3prle':
        assert one(r'(?m)^L_CLIP_PEN_COEF = ([0-9.]+)$', ft, proj) == repr(clip)
        assert one(r'(?m)^C2_SOC_COEFF = ([0-9.]+)$', ft, proj) == repr(c2coef)
        # 封顶 = 10 MW × clip 系数 (原本 1000/100 = 10 MW, ÷100 后 10/1 = 10 MW)
        assert one(r'(?m)L_CLIP_PEN_TOTAL_CAP = ([0-9.]+)$', ft, proj) == repr(10.0 * clip)
    else:
        # 代码行缩进: per 在 main 的嵌套循环里 = 8 空格; 三个 myfunc = 4 空格。
        # (6 空格的 Cal_r2_PER_baseline(LEGACY) 段刻意不改, 2 空格的模块文档单独查)
        ind = 8 if proj == 'per' else 4
        pre = r'(?m)^\s{%d}' % ind
        r2 = one(pre + r'r2 = ([0-9.]+) \* shadow_profit - ([0-9.]+) \* clip_amount$',
                 ft, proj + ' code r2')
        assert float(r2[0]) == shadow and float(r2[1]) == clip, (proj, r2)
        c2 = one(pre + r'c2 = ([0-9.]+) \* soc_dev(?:iation)?$', ft, proj + ' code c2')
        assert float(c2) == c2coef, (proj, c2)
        if proj != 'per':      # 三个 myfunc 的模块文档必须同步
            d2 = one(r'(?m)^  r2 = ([0-9.]+) \* shadow_profit - ([0-9.]+) \* clip_amount$',
                     ft, proj + ' doc r2')
            assert float(d2[0]) == shadow and float(d2[1]) == clip, (proj, d2)

    # --- 预算 / 地板 / cost_limit / L 对偶步长 ---
    d = float(one(r'(?m)^\s*d_block_L = ([0-9.]+)', mt, proj + ' d_block_L'))
    f = float(one(r'(?m)^\s*d_block_L_min = ([0-9.]+)', mt, proj + ' floor'))
    cl = float(one(r'agent_L\.cost_limit = ([0-9.]+)', mt, proj + ' cost_limit'))
    # L 的 η2 只出现一次 'ema_lam_lr=1.0', 且必须落在 agent_L 构造函数之后;
    # H 的 'ema_lam_lr=0.01' 保持原样 (uniform 模式下还会被 hc.h_settings() 覆写)
    n_lo = len(re.findall(r'ema_lam=True, ema_alpha=0\.02, ema_lam_lr=1\.0', mt))
    n_hi = len(re.findall(r'ema_lam=True, ema_alpha=0\.02, ema_lam_lr=0\.01', mt))
    assert n_lo == 1 and n_hi == 1, (proj, n_lo, n_hi)
    assert mt.index('agent_L = ') < mt.index('ema_lam=True, ema_alpha=0.02, ema_lam_lr=1.0'), proj
    eta = 1.0
    assert d == dbl and f == floor, (proj, d, f)
    assert cl == 0.1, (proj, cl)
    assert eta == 1.0, (proj, eta)

    # --- B. 物理量守恒 ---
    assert abs(d / c2coef - old_dbl / old_c2) < 1e-12, (proj, d, c2coef, old_dbl, old_c2)
    assert abs(eta * c2coef - OLD_ETA * old_c2) < 1e-12, (proj, eta, c2coef)
    if shadow is not None:
        # shadow:clip 的相对权重不变 (÷100 前 10:100 = 1:10)
        assert abs((shadow / clip) - (10.0 / 100.0)) < 1e-12, (proj, shadow, clip)
    print('  {:<9s} clip={:.0f} shadow={} c2系数={:<5g} d_block_L={:<6g} 地板={:<7g} '
          'cost_limit={} η2={} | d/c2={:g} (原 {:g}) η2·c2={:g} (原 {:g})'
          .format(proj, clip, '—' if shadow is None else shadow, c2coef, d, f, cl, eta,
                  d / c2coef, old_dbl / old_c2, eta * c2coef, OLD_ETA * old_c2))

print('\n  ⇒ 六项 d_block_L/c2系数 (= 物理容忍度) 与 η2×c2系数 (= 对偶速度) 均与 ÷100 前一致')

print('=' * 78)
print('C. 动态: 用新系数实算 r2 / c2')
print('=' * 78)
tm = importlib.import_module('td3per.td3per_myfunc')
assert tm.L_CLIP_COEF == 1.0 and tm.L_SOC_COEF == 10.0
assert tm.lower_reward(0.0) == 0.0 and tm.lower_reward(1.0) == -1.0
assert abs(tm.lower_reward(3.0) - (-3.0)) < 1e-12
# 零成本动作 = 规则动作 ⇒ slack = 0 ⇒ c2 = 0 (设计目的: c2 也应为 0)
# δ = 0.1 MWh 时规则只需 pb = 0.1/g ≈ 6.12 MW (≤ 额定 10) ⇒ 一步到位, slack = 0
c2_zero, info = tm.lower_cost_soc_track(3.1, 3.0, 3.1, 0.0, 10.0, return_info=True)
assert abs(c2_zero) < 1e-12, (c2_zero, info)
# 完全不动 ⇒ slack = 可达量 0.1 MWh ⇒ c2 = 10 × 0.1 = 1.0
c2_bad, info_bad = tm.lower_cost_soc_track(3.1, 3.0, 3.0, 0.0, 10.0, return_info=True)
assert abs(c2_bad - 1.0) < 1e-9, (c2_bad, info_bad)
print('  td3per: r2(0)=0, r2(1 MW)=-1 元; 规则动作 c2=0, 差 0.1 MWh 可达量 c2={:.2f}'.format(c2_bad))

tle = importlib.import_module('td3prle.td3prle_myfunc')
assert tle.C2_SOC_COEFF == 0.1 and tle.L_CLIP_PEN_COEF == 1.0
assert tle.L_CLIP_PEN_TOTAL_CAP == 10.0
assert abs(tle.C2_SOC_COEFF * (2.0 ** 2) - 0.4) < 1e-12   # = d_block_L (预算以 ±2 MWh 标定)
assert min(tle.L_CLIP_PEN_COEF * 10.5, tle.L_CLIP_PEN_TOTAL_CAP) == 10.0  # 封顶仍对应 10 MW
print('  td3prle: C2(±2 MWh)=0.4=d_block_L; 裁剪惩罚封顶 10 元 ⇔ 合计 10 MW (与 ÷100 前同)')

for proj, mf in (('prrbb', 'prrbb.prrbb_myfunc'), ('prle', 'prle.prle_myfunc'),
                 ('td3prrbb', 'td3prrbb.td3prrbb_myfunc')):
    m = importlib.import_module(mf)
    base = m.Cal_r2_PER_baseline(1.0, 1.0, 3.0)
    print('  {:<9s} Cal_r2_PER_baseline(1MW,sp=1,SOC=3)={:+.3f}  (LEGACY 基线, 系数保持 10/700)'
          .format(proj, base))
    assert abs(base - (10.0 * (-1.0 * 1.0) - 700.0 * 0.0)) < 1e-9, base

print('=' * 78)
print('D. H 层未受影响')
print('=' * 78)
import hsrl_cost as hc
assert hc.D_BLOCK == 15.0, hc.D_BLOCK          # d_block_H 仍是"3MW×5步"的物理容忍度
assert hc.value_scale() == 1000.0              # H 层归一化照旧
tt = src('td3per/td3per_main.py')
assert 'd_block_H = _h[\'d_block\'] if _h else 50.0' in tt or "d_block_H = _h['d_block']" in tt
print('  d_block_H = {:.0f} (物理"3MW 波动容忍度"), H 层 s={:g} 未变'.format(
    hc.D_BLOCK, hc.value_scale()))
print('=' * 78)
print('TEST OK')
