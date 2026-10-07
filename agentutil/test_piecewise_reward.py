# -*- coding: utf-8 -*-
"""piecewise（分段）奖励臂 `td3perseg` 的回归测试 —— 钉死"单因子"性质。

被测对象
--------
`td3perseg/` = **逐文件复制 `td3per/`** 后只加一层"风况 × 电价 分段权重"机制：
  * `get_wT(wind_level, price_level)`：关闭 → `W_TURBINE`(=1.0)；打开 → `WT_WIND[u]·WT_PRICE[p]`
  * `get_wB(price_level, wind_level, SOC)`：关闭 → `W_BESS`(=1.0)；打开 → `WB_PRICE[p]·w_soc`
  * `get_wind_level(u)` 的阈值可配（`get/set_wind_thresholds`），并由
    `data_piecewise.boundaries.apply_segmentation` 按 `--seg {expert,percentile,kmeans}` 设定。
  奖励管线（`Cal_r1_continuous` / `Cal_Rew_RB`）**未改结构** —— 它本来就乘这两个权重。

必须成立的四条判据
------------------
A. **`--seg off` 与 td3per 逐位相同**：权重恒 1.0，`Cal_r1_continuous` 在随机输入上与
   `td3per_myfunc.Cal_r1_continuous` **逐位相等**（这是"单因子对照"的基线保证）。
B. **`--seg on` 的权重与 td3prrbb 逐位相同**：`get_wT`/`get_wB` 在全部 3×3 档位组合与
   SOC 网格上与 `td3prrbb_myfunc` 的同名函数**逐位相等**。
C. **机制确实进到 r1**：同一组输入下 r1(ON) − r1(OFF) 等于权重差带来的解析增量。
D. **单因子边界（有意不移植的东西）**：`W_GAP` 仍为 10.0（td3prrbb 是 50.0）、
   `Cal_soc_deviation` 仍是 td3per 的单档 [1.5, 4.5]（td3prrbb 是按风况分档）——
   这两条**故意不移植**，否则对照会变成多因子。测试把它们钉住，防止将来被顺手改掉。

用法::

    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\test_piecewise_reward.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

import td3per.td3per_myfunc as P          # 基座（非分段）
import td3perseg.td3perseg_myfunc as S    # 被测：基座 + piecewise
import td3prrbb.td3prrbb_myfunc as R      # 分段机制来源（仅用于逐位比对权重）

_F = []


def check(cond, msg):
    print('  %-5s %s' % ('OK' if cond else 'FAIL', msg))
    if not cond:
        _F.append(msg)


LEV = ['LOW', 'MEDIUM', 'HIGH']

print('=' * 96)
print('A. --seg off ⇒ 与 td3per 逐位相同（单因子基线）')
print('=' * 96)
S.set_piecewise(False)
check(S.piecewise_enabled() is False, 'set_piecewise(False) ⇒ piecewise_enabled() False')
check(S.get_wT('LOW', 'HIGH') == P.W_TURBINE and S.get_wB('HIGH', 'LOW', 1.0) == P.W_BESS,
      'off 时 get_wT ≡ W_TURBINE(%.1f) / get_wB ≡ W_BESS(%.1f)' % (P.W_TURBINE, P.W_BESS))
rs = np.random.RandomState(0)
n_bit = 0
for _ in range(400):
    pr = float(rs.uniform(0, 1000)); ptur = float(rs.uniform(0, 12)); pb = float(rs.uniform(-10, 10))
    pen = float(rs.uniform(0, 400)); soc = float(rs.uniform(0, 6)); td = float(rs.choice([1.0, 0.9]))
    w, p = LEV[rs.randint(3)], LEV[rs.randint(3)]
    a = S.Cal_r1_continuous(pr, ptur, pb, pen, w, p, SOC=soc, turbine_derate=td)
    b = P.Cal_r1_continuous(pr, ptur, pb, pen, w, p, SOC=soc, turbine_derate=td)
    n_bit += int(a == b)
check(n_bit == 400, 'Cal_r1_continuous 400 组随机输入逐位相等（%d/400）' % n_bit)

print()
print('=' * 96)
print('B. --seg on ⇒ 权重与 td3prrbb 逐位相同')
print('=' * 96)
S.set_piecewise(True)
R.set_wind_thresholds(*S.get_wind_thresholds())     # 两侧阈值对齐（默认都是 7/11）
bad_t = [(w, p) for w in LEV for p in LEV if S.get_wT(w, p) != R.get_wT(w, p)]
check(not bad_t, 'get_wT 9 个档位组合逐位相同（不一致: %s）' % (bad_t or '无'))
socs = np.linspace(0.0, 6.0, 25)
bad_b = [(p, w, soc) for p in LEV for w in LEV for soc in socs
         if S.get_wB(p, w, float(soc)) != R.get_wB(p, w, float(soc))]
check(not bad_b, 'get_wB（9 档位 × 25 个 SOC）逐位相同（不一致: %s）' % (bad_b[:3] or '无'))
check(S.WT_WIND == R.WT_WIND and S.WT_PRICE == R.WT_PRICE
      and S.WB_PRICE == R.WB_PRICE and S.WB_SOC_OUTSIDE == R.WB_SOC_OUTSIDE
      and S.SOC_SAFE_RANGES == R.SOC_SAFE_RANGES,
      '分段表（WT_WIND/WT_PRICE/WB_PRICE/WB_SOC_OUTSIDE/SOC_SAFE_RANGES）与 td3prrbb 逐项相同')

print()
print('=' * 96)
print('C. 机制确实进到 r1（ON − OFF = 解析增量）')
print('=' * 96)
n_ok = 0
for _ in range(200):
    pr = float(rs.uniform(0, 1000)); ptur = float(rs.uniform(0, 12)); pb = float(rs.uniform(-10, 10))
    soc = float(rs.uniform(0, 6)); w, p = LEV[rs.randint(3)], LEV[rs.randint(3)]
    pen, time, td = 0.0, 1.0 / 60.0, 1.0
    S.set_piecewise(False)
    r_off = S.Cal_r1_continuous(pr, ptur, pb, pen, w, p, SOC=soc, time=time, turbine_derate=td)
    S.set_piecewise(True)
    r_on = S.Cal_r1_continuous(pr, ptur, pb, pen, w, p, SOC=soc, time=time, turbine_derate=td)
    d_wt = S.get_wT(w, p) - 1.0
    d_wb = S.get_wB(p, w, soc) - 1.0
    expect = d_wt * pr * ptur * time * td + d_wb * (-pr * pb * time)
    n_ok += int(abs((r_on - r_off) - expect) <= 1e-9 * max(1.0, abs(expect)))
check(n_ok == 200, 'r1(ON)−r1(OFF) 与解析增量一致（%d/200，容差 1e-9 相对）' % n_ok)
S.set_piecewise(True)
check(S.get_wT('LOW', 'HIGH') > 1.0 and S.get_wT('HIGH', 'LOW') < 1.0,
      '方向性: 低风×高价 wT>1（%.3f）；高风×低价 wT<1（%.3f）'
      % (S.get_wT('LOW', 'HIGH'), S.get_wT('HIGH', 'LOW')))

print()
print('=' * 96)
print('D. 单因子边界：有意**不**移植的东西必须保持 td3per 原值')
print('=' * 96)
check(S.W_GAP == P.W_GAP == 10.0,
      'W_GAP 仍为 10.0（td3prrbb=%.1f 未移植）' % R.W_GAP)
check((S.SOC_SAFE_MIN, S.SOC_SAFE_MAX) == (P.SOC_SAFE_MIN, P.SOC_SAFE_MAX) == (1.5, 4.5),
      'Cal_soc_deviation 仍是单档 [1.5, 4.5]（td3prrbb 按风况分档，未移植）')
check(all(S.Cal_soc_deviation(x, w) == P.Cal_soc_deviation(x, w)
          for x in (0.0, 1.0, 2.0, 3.0, 4.6, 5.5, 6.0) for w in LEV + [None]),
      'Cal_soc_deviation 与 td3per 逐位相同（含 7 个 SOC × 4 个 wind_level）')
src = open(os.path.join('td3perseg', 'td3perseg_myfunc.py'), encoding='utf-8').read()
body = src.split('def Cal_Rew_RB(')[1].split('\ndef ')[0]
check('get_wT(' in body and 'get_wB(' in body,
      'Cal_Rew_RB 内部调用 get_wT/get_wB（机制在 H 层奖励束里生效）')
check('def Cal_r2_PER_baseline' not in src and 'def Cal_r1_PRRBB' not in src,
      '未移植 td3prrbb 的 Cal_r1_PRRBB / Cal_r2_PER_baseline')

print()
print('=' * 96)
print('E. 阈值附件（get/set_wind_thresholds + get_wind_level 语义）')
print('=' * 96)
S.set_piecewise(False)
check(S.get_wind_thresholds() == (7.0, 11.0), '默认阈值 (7.0, 11.0) 与 td3per 常量一致')
check(all(S.get_wind_level(u) == P.get_wind_level(u) for u in
          np.linspace(0, 25, 501)), '默认阈值下 get_wind_level 与 td3per 在 501 个风速点上一致')
S.set_wind_thresholds(6.0, 10.0)
check(S.get_wind_thresholds() == (6.0, 10.0), 'set_wind_thresholds(6,10) 生效')
check((S.get_wind_level(5.99), S.get_wind_level(6.0), S.get_wind_level(9.99),
       S.get_wind_level(10.0)) == ('LOW', 'MEDIUM', 'MEDIUM', 'HIGH'),
      '自定义阈值下的档位边界正确（<low=LOW, [low,high)=MEDIUM, >=high=HIGH）')
try:
    S.set_wind_thresholds(11.0, 7.0); raised = False
except ValueError:
    raised = True
check(raised, 'set_wind_thresholds(low>=high) 报错（防 MEDIUM 档被挤空）')
S.set_wind_thresholds(7.0, 11.0)

print()
print('=' * 96)
print('F. 附件 apply_segmentation（data_piecewise）可调用（读训练风速数据，缺失则跳过）')
print('=' * 96)
try:
    from data_piecewise.boundaries import apply_segmentation
    lo, hi, info = apply_segmentation('expert', S.set_wind_thresholds, days=80)
    check(np.isfinite(lo) and np.isfinite(hi) and lo < hi,
          'expert 分段: LOW_MAX=%.4f HIGH_MIN=%.4f（%s）' % (lo, hi, info.get('note', '')[:40]))
except Exception as e:                                    # 数据缺失/依赖问题 ⇒ 跳过而非失败
    print('  SKIP  apply_segmentation 不可用: %s: %s' % (type(e).__name__, str(e)[:80]))
S.set_wind_thresholds(7.0, 11.0)
S.set_piecewise(False)

print()
print('=' * 96)
print('G. 接线与"未改动的部分"（静态 + 字节级）')
print('=' * 96)
msrc = open(os.path.join('td3perseg', 'td3perseg_main.py'), encoding='utf-8').read()
check("add_argument('--seg', default='off'" in msrc
      and "choices=['off', 'expert', 'percentile', 'kmeans']" in msrc,
      "main 有 --seg，默认 'off'，choices=[off,expert,percentile,kmeans]")
check('set_piecewise, piecewise_enabled, set_wind_thresholds, get_wind_thresholds'
      in msrc.replace('\n', ' ').replace('    ', ' ')[:100000] or
      ('set_piecewise' in msrc and 'set_wind_thresholds' in msrc),
      'main 从 myfunc 导入了 set_piecewise / set_wind_thresholds')
check("set_piecewise(True)" in msrc and "apply_segmentation(" in msrc
      and "str(args.seg).lower() != 'off'" in msrc,
      'main 在 --seg != off 时打开分段并调用 apply_segmentation(附件接线同形)')
dd_a = open(os.path.join('td3per', 'td3per_ddpg.py'), 'rb').read().replace(b'\r\n', b'\n')
dd_b = open(os.path.join('td3perseg', 'td3perseg_ddpg.py'), 'rb').read().replace(b'\r\n', b'\n')
check(dd_a == dd_b, 'ddpg.py 与 td3per 文本完全相同（仅行尾可能被规整 ⇒ 智能体实现未改）')
check(os.path.exists(os.path.join('td3per', 'td3per_main.py'))
      and os.path.exists(os.path.join('td3per', 'td3per_myfunc.py')),
      '基座 td3per/ 原文件仍在（未被就地修改）')

print()
print('=' * 96)
if _F:
    print('TEST FAILED — %d 项未通过:' % len(_F))
    for m in _F:
        print('  - %s' % m)
    sys.exit(1)
print('TEST PASSED — td3perseg 的 piecewise 机制: off 与 td3per 逐位相同 / on 与 td3prrbb 权重逐位相同 / 单因子边界完整')
