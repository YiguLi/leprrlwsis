#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""24h 相位口径统一 回归测试（六方法）
=====================================

背景
----
六方法中**五个**采用统一口径（`td3per` / `td3perrb` / `td3prrlrb` / `td3prrbb` / `td3prle`）：

    Normalize_H(obs, minute=0, mode=1, table=None)      # 4 维基础 + 相位特征
    PHASE_DIM = {0: 0, 1: 2, 2: 5}
    day_price_schedule() / phase_daily_table() / phase_features() / phase_minute_of_day()
    REP_MINUTE = {'LOW': 210, 'MEDIUM': 840, 'HIGH': 540}
    rep_minute_for_price_level(level)                   # 名字与数值电价两种调用都对

`td3leprrlrb` 是**旧口径**：同一语义、不同 API
（`Normalize_H(obs, phase_minute=0, phase_feat=1, max_E=6.)`、`REP_MINUTE=1440` 是周期、
`representative_phase_minute` 只吃数值电价、`phase_features(m, mode)` 无 table 参数）。
本测试对它做**语义等价比对**，而不是要求签名一致。

本测试守住的四件事
------------------
A. 五个统一模块的 helper 齐全，且 `Normalize_H` / `phase_features` / `phase_daily_table`
   在这些模块之间**逐位相同**（跨项目可比的根基）。
B. `rep_minute_for_price_level` 五模块一致；档位名('LOW')与数值电价(300)两种调用
   给出同一分钟；未知输入回退 210（历史 bug: 数值电价查不到档位名 ⇒ 全回退 LOW）。
C. `td3leprrlrb` 与统一口径**语义等价**（代表分钟映射相同；逐位向量除签名外相同）。
D. 接线契约（静态扫描）：
   * 六个 main 都有 `--phase_feat default=1 choices=[0,1,2]`；
     五个统一 main 的 `nb_states_H = 4 + PHASE_DIM[...]`（旧口径为 `4 + _phase_dim`）；
   * 六个 main 的 `--bsize 256` / `--rmsize 40000`（六个 TD3 方法已统一）；
   * 两个接收方（td3prrbb/td3prle）的 `nine_case_test` **接受** phase_mode/phase_table、
     **调用** `Normalize_H`、且函数体内**不再有** `Normalize(obs)` / `Normalize(obs2)`；
     main 向 `nine_case_test` 传 `phase_mode=phase_mode, phase_table=phase_table`；
   * `test_five_days.py` 的 **六个** TD3 项目 `h_obs='normalize_H'`，三个旧项目仍为 `'normalize'`
     （`h_obs='normalize'` 时该文件**强制**要求 4 维权重，故这是 6 维 checkpoint 能加载的前提）。

注意: 纯静态 + 纯 Python 数值比对, **不依赖 MATLAB**。
"""
import io
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np                                                  # noqa: E402
import importlib                                                    # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors='replace')
    except Exception:
        pass

UNIFORM = ['td3per', 'td3perrb', 'td3prrlrb', 'td3prrbb', 'td3prle', 'yawflor']
LEGACY = 'td3leprrlrb'
ALL6 = UNIFORM + [LEGACY]

_fails = []


def check(cond, msg):
    print('  {} {}'.format('OK  ' if cond else 'FAIL', msg))
    if not cond:
        _fails.append(msg)


def read_text(path):
    with io.open(path, encoding='utf-8', errors='replace') as fh:
        return fh.read()


def func_body(src, name):
    """取顶层函数 name 的函数体（到下一个顶层 def / 文件尾）。"""
    m = re.search(r'^def %s\(' % re.escape(name), src, re.M)
    if not m:
        return None
    nxt = re.search(r'^def ', src[m.end():], re.M)
    return src[m.start():m.end() + (nxt.start() if nxt else len(src))]


print('=' * 100)
print('A. 五个统一模块: helper 齐全 + 跨模块逐位相同')
print('=' * 100)
mods = {}
for key in UNIFORM:
    mods[key] = importlib.import_module('%s.%s_myfunc' % (key, key))
    miss = [n for n in ('Normalize', 'Normalize_H', 'PHASE_DIM', 'day_price_schedule',
                        'phase_daily_table', 'phase_features', 'phase_minute_of_day',
                        'REP_MINUTE', 'rep_minute_for_price_level')
            if not hasattr(mods[key], n)]
    check(not miss, '%s: helper 齐全%s' % (key, '' if not miss else ' (缺 %s)' % miss))

check(all(mods[k].PHASE_DIM == {0: 0, 1: 2, 2: 5} for k in UNIFORM),
      'PHASE_DIM 五模块一致 = {0:0, 1:2, 2:5}')
check(all(mods[k].REP_MINUTE == {'LOW': 210, 'MEDIUM': 840, 'HIGH': 540} for k in UNIFORM),
      "REP_MINUTE 五模块一致 = {'LOW':210,'MEDIUM':840,'HIGH':540}")

OBS = np.array([580.0, 8.0, 3.0, 1.5])
MINUTES = [0, 1, 210, 419, 420, 540, 660, 839, 840, 1019, 1020, 1260, 1380, 1439]
ref = {}
for mode in (0, 1, 2):
    table = mods[UNIFORM[0]].phase_daily_table() if mode >= 2 else None
    ref[mode] = {m: mods[UNIFORM[0]].Normalize_H(OBS, m, mode, table) for m in MINUTES}
for key in UNIFORM[1:]:
    same = True
    for mode in (0, 1, 2):
        table = mods[key].phase_daily_table() if mode >= 2 else None
        for m in MINUTES:
            v = mods[key].Normalize_H(OBS, m, mode, table)
            if v.shape != ref[mode][m].shape or not np.array_equal(v, ref[mode][m]):
                same = False
    check(same, '%s: Normalize_H(mode 0/1/2 × %d 分钟) 与 td3per 逐位相同'
          % (key, len(MINUTES)))

tabs = [mods[k].phase_daily_table() for k in UNIFORM]
check(all(t.shape == tabs[0].shape and np.array_equal(t, tabs[0]) for t in tabs),
      'phase_daily_table() 五模块逐位相同 (shape=%s)' % (tabs[0].shape,))
check(all(np.array_equal(mods[k].day_price_schedule(), mods[UNIFORM[0]].day_price_schedule())
          for k in UNIFORM), 'day_price_schedule() 五模块逐位相同')

print()
print('=' * 100)
print('B. rep_minute_for_price_level: 两种调用约定 + 回退')
print('=' * 100)
EXPECT = {'LOW': 210, 'MEDIUM': 840, 'HIGH': 540}
for key in UNIFORM:
    f = mods[key].rep_minute_for_price_level
    by_name = {nm: f(nm) for nm in EXPECT}
    by_num = {nm: f(pr) for nm, pr in (('LOW', 300), ('MEDIUM', 580), ('HIGH', 980))}
    check(by_name == EXPECT and by_num == EXPECT,
          '%s: 档位名 %s / 数值电价 %s 都正确' % (key, by_name, by_num))
    # 回退只针对**无法解析**的输入；合法数值（哪怕极端）应走数值分档.
    check(f('nonsense') == 210,
          "%s: 无法解析的输入回退 210（历史 bug 的兜底）" % key)
    _lvl = mods[key].get_price_level(12345.0)
    check(f(12345) == mods[key].REP_MINUTE[_lvl],
          '%s: 极端数值电价 12345 走数值分档 → %s=%d（非回退）'
          % (key, _lvl, f(12345)))

print()
print('=' * 100)
print('C. td3leprrlrb（旧口径）与统一口径语义等价')
print('=' * 100)
leg = importlib.import_module('%s.%s_myfunc' % (LEGACY, LEGACY))
check(hasattr(leg, 'Normalize_H') and hasattr(leg, 'phase_features'),
      'td3leprrlrb: 有相位支持 (Normalize_H / phase_features)')
check(getattr(leg, 'PHASE_REP_MINUTE', None) == {300: 210, 580: 840, 980: 540},
      'td3leprrlrb: PHASE_REP_MINUTE = {300:210, 580:840, 980:540}（与统一口径同映射）')
check(all(leg.representative_phase_minute(pr) == EXPECT[nm]
          for nm, pr in (('LOW', 300), ('MEDIUM', 580), ('HIGH', 980))),
      'td3leprrlrb: representative_phase_minute 与统一口径一致')
leg_same, bad = True, None
for mode in (0, 1, 2):
    for m in MINUTES:
        a = leg.Normalize_H(OBS, m, phase_feat=mode)
        if a.shape != ref[mode][m].shape or not np.array_equal(a, ref[mode][m]):
            leg_same, bad = False, (mode, m, a, ref[mode][m])
            break
check(leg_same, 'td3leprrlrb: Normalize_H 逐位等于统一口径%s'
      % ('' if leg_same else '（首个不符 mode=%d minute=%d: %s vs %s）' % bad[:2] + str(bad[2:])))

print()
print('=' * 100)
print('D. 接线契约（静态扫描）')
print('=' * 100)
P_MAIN = {k: os.path.join(k, '%s_main.py' % k) for k in ALL6}
# --bsize / --rmsize 默认 = 256 / 40000 (**六个 TD3 方法已统一**; 均启用 H 层 n 步滑窗,
# 1 条 = n 块 = 5n 环境步 ⇒ 40000 条 ≈ 2M 环境步。更早的 "其余五方法 80000" 例外表
# 已在批次 2 取消)。
BSIZE_DEFAULT = 256
RMSIZE_DEFAULT = 40000
for k in ALL6:
    src = read_text(P_MAIN[k])
    check("'--phase_feat', default=1, type=int, choices=[0, 1, 2]" in src,
          '%s: --phase_feat 默认 1 / choices [0,1,2]' % k)
    check("'--bsize', default=%d" % BSIZE_DEFAULT in src
          and 'default=%d,' % RMSIZE_DEFAULT in src,
          '%s: --bsize %d / --rmsize %d' % (k, BSIZE_DEFAULT, RMSIZE_DEFAULT))
    check("'--nstep', default=10" in src and "'--target_agg', default='min'" in src,
          '%s: --nstep 默认 10 / --target_agg 默认 min' % k)
    check('nstep=args.nstep, target_agg=args.target_agg' in src,
          '%s: H 层构造接线 nstep/target_agg' % k)
for k in UNIFORM:
    src = read_text(P_MAIN[k])
    check('nb_states_H = 4 + PHASE_DIM[int(getattr(args, \'phase_feat\', 0))]' in src,
          '%s: nb_states_H = 4 + PHASE_DIM[...]' % k)
check('nb_states=4 + _phase_dim' in read_text(P_MAIN[LEGACY]),
      '%s: nb_states=4 + _phase_dim（旧口径, 语义相同）' % LEGACY)

for k in ('td3prrbb', 'td3prle'):
    msrc = read_text(os.path.join(k, '%s_myfunc.py' % k))
    body = func_body(msrc, 'nine_case_test') or ''
    check('phase_mode=0, phase_table=None' in body,
          '%s_myfunc.nine_case_test: 接受 phase_mode/phase_table' % k)
    check('Normalize(obs)' not in body and 'Normalize(obs2)' not in body,
          '%s_myfunc.nine_case_test: 函数体内无 4 维 Normalize(obs)/Normalize(obs2)' % k)
    check(body.count('Normalize_H(') >= 2,
          '%s_myfunc.nine_case_test: 两处观测都走 Normalize_H (%d 次)' % (k, body.count('Normalize_H(')))
    check('_pminute' in body and 'rep_minute_for_price_level' in body,
          '%s_myfunc.nine_case_test: 用 _pminute 映射代表相位' % k)
    dead = func_body(msrc, 'prrbb_S_test') or ''
    check(dead.count('Normalize(obs') == 2 and '⚠ 历史遗留' in msrc,
          '%s_myfunc.prrbb_S_test: 仍是 4 维但有 ⚠ 无调用者的标注' % k)
    msrc = read_text(P_MAIN[k])
    check('Normalize(observation' not in msrc and 'Normalize(observation2' not in msrc,
          '%s_main: 两处 H 观测已是 Normalize_H' % k)
    check('phase_mode=phase_mode, phase_table=phase_table' in msrc,
          '%s_main: 向 nine_case_test 传 phase_mode/phase_table' % k)
    check('phase_mismatch' in msrc and 'phase_minute_of_day' in msrc,
          '%s_main: 相位自检 + 相位分钟接线到位' % k)

tf = read_text('test_five_days.py')
n_phase = len(re.findall(r"h_obs='normalize_H'", tf))
n_plain = len(re.findall(r"h_obs='normalize'(?!_H)", tf))
check(n_phase == 7 and n_plain == 3,
      "test_five_days.py: 7 个项目 h_obs='normalize_H' (2026-09-30 起含 yawflor) "
      "+ 3 个旧项目 'normalize' (实得 %d/%d)"
      % (n_phase, n_plain))

print()
print('=' * 100)
if _fails:
    print('TEST FAILED — %d 项未通过:' % len(_fails))
    for f in _fails:
        print('  - ' + f)
    sys.exit(1)
print('TEST PASSED — 相位口径统一（五统一 + 一旧口径语义等价）与接线契约全部成立')
