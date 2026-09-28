# -*- coding: utf-8 -*-
r"""把**动作先验**从"单机 Cp 最优"改到"风场总功率最优"（一次性迁移，默认 dry-run）。

背景
----
`agentutil/probe_a_farm_structure.py` 实测（MATLAB R2025a，u = 6/8/10/12/14）：

  * 风场总功率的**集体峰在 a = 0.20**；全体 `a = 1/3`（单机 Cp 最优 4a(1−a)² 的峰位）
    只有 3.446 MW，比 a = 0.20 的 3.718 MW **低 7.3%**（u=10）；
  * 峰功率精确 ∝ u³ ⇒ 最优动作**与风速无关**，是一个常向量；
  * ±2% 平台 = [0.20, 0.25]，±5% 平台 = [0.15, 0.30]，逐维最优 `[.2 .2 .2 .2 .25]`。

而本仓库的动作先验全部基于"**单机** Cp 保持 86%~100% 最优"：
`WT_START=0.33 / WT_LO,WT_HI=0.20,0.45`、`init_wt_action_bias(U(0.25,0.40))`、
`env.*Reset()` 的标称动作 `0.3333`。

新口径（由上面的实测直接读出）
------------------------------
  `WT_START = 0.20`（= 峰位）、`WT_LO, WT_HI = 0.15, 0.25`（= ±2% 平台两端；
  三档游走 {0.15,0.20,0.25} 的平稳分布均匀 ⇒ `E[a] ≡ 0.20` = 峰位）、
  `init_wt_action_bias(lo=0.17, hi=0.23)`（均值 0.20，且 ⊂ 新 warmup 带）、
  `env` 标称动作 `0.3333 → 0.20`。

用法
----
    python agentutil/port_wt_prior.py            # dry-run: 打印每处锚点的命中数与 diff
    python agentutil/port_wt_prior.py --apply    # 写盘（全部锚点命中数正确才写）

⚠ 训练中禁止 `--apply`（MATLAB 按时间戳重载 `*.m`；本脚本不碰 `*.m`，但训练中的进程
   仍会读到改过的 `.py`）。
"""

import argparse
import difflib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

MAINS = ['td3per/td3per_main.py', 'td3perrb/td3perrb_main.py',
         'td3prrbb/td3prrbb_main.py', 'td3prle/td3prle_main.py',
         'td3prrlrb/td3prrlrb_main.py']

# (相对路径, 锚点, 替换, 期望命中次数)
PATCHES = []


def P(path, old, new, exp=1):
    PATCHES.append((path, old, new, exp))


# ---------------- 1) hsrl_warmup.py：单一真源里的 4 个常量 + 判据文字 ----------------
P('hsrl_warmup.py', '各维从 **0.33** 起步', '各维从 **0.20** 起步', 1)
P('hsrl_warmup.py',
  '（Cp = 4a(1−a)² 的最优点，也是\n'
  '        `env.reset()` 的标称动作 a=1/3 与 actor 的 `init_wt_action_bias(0.25~0.40)` 中心）；',
  '（**风场总功率**的峰位 —— 实测 u=6/8/10/12/14 的集体峰全在 a=0.20，且峰功率精确\n'
  '        ∝ u³ ⇒ 最优动作**与风速无关**、是常向量；这也是 `env.reset()` 的标称动作与\n'
  '        actor 的 `init_wt_action_bias(0.17~0.23)` 中心）；',
  1)
P('hsrl_warmup.py',
  '        **[0.20, 0.45]**（该带内 Cp 仍为最优的 86%~100%，E[ΣCp]=2.82 ≈ 95% 最优；\n'
  '        clip 到 [0,0.5] 则 89% 的样本会出现某维 ≤0.05、ΣCp 掉到 83% 最优）；',
  '        **[0.15, 0.25]**（= 实测风场功率 ±2% 平台的两端：0.15 损 2.7%、0.20 为峰、\n'
  '        0.25 损 1.0%；三档游走 {0.15,0.20,0.25} 的平稳分布均匀 ⇒ E[a] ≡ 0.20 = 峰位）；',
  1)
P('hsrl_warmup.py',
  '    RNG 用**独立 RandomState**',
  '    ⚠ 2026-09-28 判据修正：此前 `WT_START=0.33 / WT_LO,WT_HI=0.20,0.45`、\n'
  '    `init_wt_action_bias(U(0.25,0.40))` 的判据是"**单机** C_p = 4a(1−a)² 保持\n'
  '    86%~100% 最优"，而**单机 C_p 最优 ≠ 风场总功率最优**（全体 1/3 比 a=0.20\n'
  '    低 7.3%）。现按实测的"风场功率平台"重定，并把 actor 的 init bias 一并移正。\n\n'
  '    RNG 用**独立 RandomState**',
  1)
P('hsrl_warmup.py',
  'WT_START = 0.33          # Cp 最优点 (= 1/3; env.reset 的标称动作)',
  'WT_START = 0.20          # 风场功率最优点 (实测 u∈[6,14] 全为 0.20; env.reset 标称动作)',
  1)
P('hsrl_warmup.py',
  'WT_LO, WT_HI = 0.20, 0.45   # 直接 clip 的安全带 (Cp 保持 86%~100% 最优)',
  'WT_LO, WT_HI = 0.15, 0.25   # clip 安全带 = 实测风场功率 ±2% 平台 {0.15,0.20,0.25}',
  1)
P('hsrl_warmup.py', '# 第 0 块 = 起始点 (0.33, 信号 0)', '# 第 0 块 = 起始点 (0.20, 信号 0)', 1)
P('hsrl_warmup.py',
  "help='warmup 期上层动作调度: fixed=0.33 起每块±0.05 clip[0.20,0.45]'",
  "help='warmup 期上层动作调度: fixed=0.20 起每块±0.05 clip[0.15,0.25]'",
  1)
P('hsrl_warmup.py',
  "help='warmup 风机动作起始值 (默认 0.33 = Cp 最优)')",
  "help='warmup 风机动作起始值 (默认 0.20 = 风场功率最优)')",
  1)
P('hsrl_warmup.py', "help='warmup 风机动作 clip 下界 (默认 0.20)')",
  "help='warmup 风机动作 clip 下界 (默认 0.15)')", 1)
P('hsrl_warmup.py', "help='warmup 风机动作 clip 上界 (默认 0.45)')",
  "help='warmup 风机动作 clip 上界 (默认 0.25)')", 1)

# ---------------- 2) env.*Reset() 的标称动作 (P_G_prev 基准) ----------------
P('env.py', 'a_temp = np.ones(self.num_turbine) * 0.3333',
  'a_temp = np.ones(self.num_turbine) * 0.20', 3)

# ---------------- 3) 5 个 main 的 init_wt_action_bias ----------------
for _m in MAINS:
    P(_m, 'def init_wt_action_bias(agent, lo=0.25, hi=0.40, sig_value=0.0, verbose=True):',
      'def init_wt_action_bias(agent, lo=0.17, hi=0.23, sig_value=0.0, verbose=True):', 1)
    P(_m, 'init_wt_action_bias(agent_H, lo=0.25, hi=0.40, sig_value=0.0)',
      'init_wt_action_bias(agent_H, lo=0.17, hi=0.23, sig_value=0.0)', 1)

# ---------------- 4) td3leprrlrb 的模块级常量 ----------------
P('td3leprrlrb/td3leprrlrb_main.py', 'WT_BIAS_LOW = 0.25', 'WT_BIAS_LOW = 0.17', 1)
P('td3leprrlrb/td3leprrlrb_main.py', 'WT_BIAS_HIGH = 0.40', 'WT_BIAS_HIGH = 0.23', 1)

# ---------------- 5) 评测主脚本的标称动作 (与 env.reset 保持一致) ----------------
P('test_five_days.py', 'a_temp = np.ones(env.num_turbine) * 0.3333',
  'a_temp = np.ones(env.num_turbine) * 0.20', 1)
P('test_compare_projects.py', 'a_temp = np.ones(env.num_turbine) * 0.3333',
  'a_temp = np.ones(env.num_turbine) * 0.20', 1)
P('test_simple.py', 'a_temp = np.ones(NB_TURBINE) * 0.3333',
  'a_temp = np.ones(NB_TURBINE) * 0.20', 1)

# ---------------- 6) 数值形式的先验文字 (纯 ASCII 锚点; 命中数随文件不同, 报出即可) ----------------
for _m in MAINS + ['td3leprrlrb/td3leprrlrb_main.py']:
    P(_m, '[0.25,0.40]', '[0.17,0.23]', None)
    P(_m, '[0.25, 0.40]', '[0.17, 0.23]', None)
    P(_m, 'U(0.25,0.40)', 'U(0.17,0.23)', None)
    P(_m, 'U(0.25, 0.40)', 'U(0.17, 0.23)', None)
    P(_m, 'Cp 高效区', '风场功率最优带', None)
    P(_m, '到 风场功率最优带', '到“风场功率最优带”', None)

# 单机 Cp 区间: 新带 0.17~0.23 对应 Cp 0.468~0.545, **低于**单机最优 0.593
# (锚点与顺序无关: 前面的 `[0.25,0.40]` 替换已经改过同一行, 故只锚后半句)
for _m in ['td3per/td3per_main.py', 'td3perrb/td3perrb_main.py', 'td3prrlrb/td3prrlrb_main.py']:
    P(_m,
      ' 对应 Cp≈0.56~0.59, 即从"正常工作点"出发。',
      ' 对应单机 Cp≈0.468~0.545 —— **低于**单机最优 0.593：风场最优本就落在单机 Cp 的'
      '**次优点**(上游降载减少尾流亏损, 全场反而多出 7.3%, 实测 u=10)。',
      1)

# ---------------- 7) 测试断言 ----------------
T = 'agentutil/test_warmup_schedule.py'
P(T, 'assert np.allclose(a0[:5], 0.33) and a0[5] == 0.0, a0',
  'assert np.allclose(a0[:5], 0.20) and a0[5] == 0.0, a0', 1)
P(T, 'assert A[:, :5].min() >= 0.20 - 1e-12 and A[:, :5].max() <= 0.45 + 1e-12',
  'assert A[:, :5].min() >= 0.15 - 1e-12 and A[:, :5].max() <= 0.25 + 1e-12', 1)
P(T, 'wall = np.isclose(A[:, :5], 0.20) | np.isclose(A[:, :5], 0.45)',
  'wall = np.isclose(A[:, :5], 0.15) | np.isclose(A[:, :5], 0.25)', 1)
P(T,
  "cp = 4 * A[:, :5] * (1 - A[:, :5]) ** 2\n"
  "print('E[ΣCp]={:.3f} = 最优 {:.3f} 的 {:.1f}%'.format(cp.sum(1).mean(), 5 * 16 / 27,\n"
  "                                                     100 * cp.sum(1).mean() / (5 * 16 / 27)))\n"
  "assert cp.sum(1).mean() > 2.75, cp.sum(1).mean()",
  """# 判据换成**风场功率**(而不是"单机 ΣCp"): 用 probe_a_farm_structure.py 的实测平台
#   u=10: 峰 3.717941 MW @ a=0.20; a=0.15 → 3.619155 (−2.7%); a=0.25 → 3.680407 (−1.0%)
_PEAK, _P_LO, _P_HI = 3.717941, 3.619155, 3.680407
_wall_loss = max((_PEAK - _P_LO) / _PEAK, (_PEAK - _P_HI) / _PEAK)
a_mean = A[:, :5].mean()
print('warmup wt: E[a]={:.4f} (峰位 0.20), a∈[{:.2f},{:.2f}], 两端功率代价 ≤{:.1f}%'
      .format(a_mean, A[:, :5].min(), A[:, :5].max(), 100 * _wall_loss))
assert abs(a_mean - 0.20) < 0.02, a_mean     # 三档游走 {0.15,0.20,0.25} 的平稳均值 = 峰位
assert _wall_loss < 0.03, _wall_loss         # 带两端距峰不超过 3%""",
  1)
P(T, '    (0.33, 0.05, 0.20, 0.45)', '    (0.20, 0.05, 0.15, 0.25)', 1)
P(T, '--warmup_wt_* 默认与对照臂', '--warmup_wt_* 默认(风场功率最优先验)与对照臂', 1)


def main():
    ap = argparse.ArgumentParser(description='动作先验迁移 (默认 dry-run)',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true', help='真正写盘 (默认只打印)')
    args = ap.parse_args()

    # 按文件归并，逐个锚点核对命中数；任何一处不符 ⇒ 整体不写盘
    by_file = {}
    for path, old, new, exp in PATCHES:
        by_file.setdefault(path, []).append((old, new, exp))

    problems, changed = [], {}
    for path, items in by_file.items():
        if not os.path.exists(path):
            problems.append('%s: 文件不存在' % path)
            continue
        raw = open(path, 'rb').read()
        nl = '\r\n' if raw.count(b'\r\n') * 2 >= raw.count(b'\n') else '\n'
        with open(path, encoding='utf-8', errors='replace') as f:
            src = f.read()
        out, log = src, []
        for old, new, exp in items:
            n = out.count(old)
            if exp is None:                       # 未指定 ⇒ 只报告, 不参与判定
                log.append(('?', n, old))
                if n:
                    out = out.replace(old, new)
                continue
            if n != exp:
                problems.append('%s: 锚点命中 %d 次 (要求 %d) ⇒ %r'
                                % (path, n, exp, old[:60]))
                continue
            out = out.replace(old, new)
            log.append(('OK', n, old))
        changed[path] = (out, nl, log)

    for path, (out, _nl, log) in changed.items():
        print('\n' + '=' * 100)
        print('=== %s ===' % path)
        for st, n, old in log:
            print('  [%s] x%-2d %r' % (st, n, old[:70]))
        if out != open(path, encoding='utf-8', errors='replace').read():
            old_lines = open(path, encoding='utf-8', errors='replace').read().splitlines()
            for ln in difflib.unified_diff(old_lines, out.splitlines(), lineterm='',
                                           fromfile=path + ' (现在)', tofile=path + ' (改后)'):
                print(ln)

    if problems:
        print('\n' + '!' * 100)
        print('锚点未全部命中 (%d 处) ⇒ 整体不写盘:' % len(problems))
        for p in problems:
            print('  !! %s' % p)
        return 1

    if not args.apply:
        print('\n[dry-run] 未写盘。确认后加 --apply（**先停掉正在训练的 run**）。')
        return 0

    for path, (out, nl, _log) in changed.items():
        with open(path, 'w', encoding='utf-8', newline=nl) as f:
            f.write(out)
        print('已写入 %s (换行 %s)' % (path, 'CRLF' if nl == '\r\n' else 'LF'))
    print('\n下一步验证:')
    print('  python agentutil/test_warmup_schedule.py      # 不需 MATLAB')
    print('  python agentutil/test_phase_uniform.py        # 不需 MATLAB (跨项目契约)')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
