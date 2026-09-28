# -*- coding: utf-8 -*-
r"""把动作先验**放宽为中性宽带**：只编码单机物理，不编码"风场最优在哪"（一次性，默认 dry-run）。

为什么再改一次（诚实记录）
--------------------------
2026-09-28 第一轮把先验从"单机 Cp 最优"改到"对准实测均匀动作峰位"
（`0.20 / [0.15,0.25] / U(0.17,0.23)`）——**那一轮是错的**，两个错：

1. 那个 0.20 只是**均匀动作族**上的峰，且逐维结论来自**坐标轮询**（每次只动一维、
   其余钉在 0.20），**不是联合最优**；用来支持"维间无竞争"的"拼接值一致"更是
   循环论证（拼接的那组配置本就不变）。
2. 更根本的是**方法**：**风场最优是随尾流状态变化的单调上升斜坡** ——
   最下游机组背风侧无机组，其功率 ∝ `CT(a₅)·Ue³`，而上游动作固定时 `Ue` 与 `a₅`
   无关 ⇒ 它应被推向单机最优 `≈1/3`；需要降载留风的是**上游**。
   这个**形状**取决于风速/湍流/来流/间距（即尾流效应），**是策略要学的函数**，
   不是可以用几个仿真点测定并写进先验的常数。把先验对准某个测出来的"峰位"，
   等于用常数替策略回答它该学的东西；而且第一轮的窄带 `[0.15,0.25]`
   会从**上方掐住末台**，把"偏高"的偏差换成了"偏低"。

本轮口径（中性宽带，只有一条带、一处真源）
------------------------------------------
  `WT_START = 1/3`      单机 `C_p = 4a(1−a)²` 的最优点 —— **只讲单机物理**；
  `WT_LO, WT_HI = 0.05, 0.45`
                        只排除 a<0.05 的**停机悬崖**（`C_p`<18%）与 a>0.45 的
                        低效高诱导区；**不预设**风场最优的形状；
  `init_wt_action_bias(lo, hi) = (0.05, 0.45)`
                        与上面**同一条带**（warmup 与策略起步同分布这条不变量得以
                        精确保留，且不需要再引入"中心 = 我认为的最优"这类断言）；
  `env.*Reset()` 的标称动作 `0.20 → 0.3333`（回到单机最优点；它是 `P_G_prev` 的
                        波动基准，任何取值都是约定，选单机物理点不带风场主张）。

  旧带宽度 0.25、新带宽度 0.40；旧带的病不在"窄"，而在"位置"（把 a<0.20 整段排除）。

测试口径同步
------------
`test_warmup_schedule.py` 的判据由"实测平台上的风场功率代价"改为**纯契约**：
起点 = 1/3、界内、步长 ≤0.05、截断只发生在贴墙处、且**中性宽带必须被真正走到两端**
（seed=7 的 41 块实测 min=0.050 / max=0.450 / mean=0.3288；2001 块平稳均值 = 0.2501
= 带中点）。**不再**断言任何"效率/最优"类判据。

用法
----
    python agentutil/relax_wt_prior.py            # dry-run: 打印命中数与 diff
    python agentutil/relax_wt_prior.py --apply    # 写盘（全部锚点命中数正确才写）
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
         'td3prrlrb/td3prrlrb_main.py', 'td3leprrlrb/td3leprrlrb_main.py']

PATCHES = []


def P(path, old, new, exp=1):
    PATCHES.append((path, old, new, exp))


# ---------------- 1) hsrl_warmup.py：单一真源 ----------------
P('hsrl_warmup.py',
  '      * 5 个风机轴向因子各维从 **0.20** 起步（**风场总功率**的峰位 —— 实测 u=6/8/10/12/14 的集体峰全在 a=0.20，且峰功率精确\n'
  '        ∝ u³ ⇒ 最优动作**与风速无关**、是常向量；这也是 `env.reset()` 的标称动作与\n'
  '        actor 的 `init_wt_action_bias(0.17~0.23)` 中心）；',
  '      * 5 个风机轴向因子各维从 **1/3** 起步（单机 `C_p = 4a(1−a)²` 的最优点 —— 只讲\n'
  '        **单机物理**，**不对"风场最优在哪"表态**；`env.reset()` 的标称动作与 actor 的\n'
  '        `init_wt_action_bias` 共用下面同一条带状区间）；',
  1)
P('hsrl_warmup.py',
  '        **[0.15, 0.25]**（= 实测风场功率 ±2% 平台的两端：0.15 损 2.7%、0.20 为峰、\n'
  '        0.25 损 1.0%；三档游走 {0.15,0.20,0.25} 的平稳分布均匀 ⇒ E[a] ≡ 0.20 = 峰位）；',
  '        **[0.05, 0.45]**（**中性宽带**：只排除 a<0.05 的停机悬崖〔`C_p`<18%〕与\n'
  '        a>0.45 的低效高诱导区，**不预设**风场最优的形状）；',
  1)
P('hsrl_warmup.py',
  '    ⚠ 2026-09-28 判据修正：此前 `WT_START=0.33 / WT_LO,WT_HI=0.20,0.45`、\n'
  '    `init_wt_action_bias(U(0.25,0.40))` 的判据是"**单机** C_p = 4a(1−a)² 保持\n'
  '    86%~100% 最优"，而**单机 C_p 最优 ≠ 风场总功率最优**（全体 1/3 比 a=0.20\n'
  '    低 7.3%）。现按实测的"风场功率平台"重定，并把 actor 的 init bias 一并移正。',
  '    ⚠ 2026-09-28 两轮修正（记录以免重犯）：旧值 `0.33 / [0.20,0.45] / U(0.25,0.40)` 的\n'
  '    判据是"**单机** C_p 保持 86%~100% 最优"，用错了对象。第一轮改成"对准均匀动作的\n'
  '    功率峰 `0.20 / [0.15,0.25] / U(0.17,0.23)`"——**同样是错的**：那个峰只是**均匀族**\n'
  '    上坐标轮询的结果，而联合最优是**随尾流状态变化的单调上升斜坡**（最下游机组背风侧\n'
  '    无机组，其功率 ∝ `CT(a₅)·Ue³` 且上游动作固定时 `Ue` 与 `a₅` 无关 ⇒ 应推向单机最优\n'
  '    ≈1/3；需要降载留风的是上游）。这个**形状**取决于尾流效应，**是策略要学的函数**，\n'
  '    不是可预先测定的常数；而第一轮的窄带还会从**上方掐住末台**。故本轮改为**中性宽带**。',
  1)
P('hsrl_warmup.py',
  'WT_START = 0.20          # 风场功率最优点 (实测 u∈[6,14] 全为 0.20; env.reset 标称动作)',
  'WT_START = 1.0 / 3       # 单机 Cp 最优点 (= 1/3; 只讲单机物理, 不预设风场最优形状)',
  1)
P('hsrl_warmup.py',
  'WT_LO, WT_HI = 0.15, 0.25   # clip 安全带 = 实测风场功率 ±2% 平台 {0.15,0.20,0.25}',
  'WT_LO, WT_HI = 0.05, 0.45   # 中性宽带: 只排除停机悬崖(a<0.05)与低效高区(a>0.45)',
  1)
P('hsrl_warmup.py', '# 第 0 块 = 起始点 (0.20, 信号 0)', '# 第 0 块 = 起始点 (1/3, 信号 0)', 1)
P('hsrl_warmup.py',
  "help='warmup 期上层动作调度: fixed=0.20 起每块±0.05 clip[0.15,0.25]'",
  "help='warmup 期上层动作调度: fixed=1/3 起每块±0.05 clip[0.05,0.45]'",
  1)
P('hsrl_warmup.py',
  "help='warmup 风机动作起始值 (默认 0.20 = 风场功率最优)')",
  "help='warmup 风机动作起始值 (默认 1/3 = 单机 Cp 最优)')",
  1)
P('hsrl_warmup.py', "help='warmup 风机动作 clip 下界 (默认 0.15)')",
  "help='warmup 风机动作 clip 下界 (默认 0.05)')", 1)
P('hsrl_warmup.py', "help='warmup 风机动作 clip 上界 (默认 0.25)')",
  "help='warmup 风机动作 clip 上界 (默认 0.45)')", 1)

# ---------------- 2) main：init bias 放到同一条中性带 ----------------
# 5 个用 lo=/hi= 直接传参的项目 (每个文件 2 处: 函数签名默认 + 调用点)
for _m in ['td3per/td3per_main.py', 'td3perrb/td3perrb_main.py',
           'td3prrbb/td3prrbb_main.py', 'td3prle/td3prle_main.py',
           'td3prrlrb/td3prrlrb_main.py']:
    P(_m, 'lo=0.17, hi=0.23', 'lo=0.05, hi=0.45', 2)
# td3leprrlrb 用模块级常量
P('td3leprrlrb/td3leprrlrb_main.py', 'WT_BIAS_LOW = 0.17', 'WT_BIAS_LOW = 0.05', 1)
P('td3leprrlrb/td3leprrlrb_main.py', 'WT_BIAS_HIGH = 0.23', 'WT_BIAS_HIGH = 0.45', 1)

# td3per / td3perrb / td3prrlrb 的单行说明
for _m in ['td3per/td3per_main.py', 'td3perrb/td3perrb_main.py', 'td3prrlrb/td3prrlrb_main.py']:
    P(_m,
      '    bias∈[lo,hi]=[0.17,0.23] 对应单机 Cp≈0.468~0.545 —— **低于**单机最优 0.593：'
      '风场最优本就落在单机 Cp 的**次优点**(上游降载减少尾流亏损, 全场反而多出 7.3%, 实测 u=10)。',
      '    bias∈[lo,hi]=[0.05,0.45] 覆盖"上游降载 → 末台单机最优"的整个区间, '
      '**故意不预设**风场最优的形状\n'
      '    (它随尾流状态(风速/湍流/来流)变化, 是**策略要学的函数**, 不是可预先测定的常数)。',
      None)
# td3prrbb / td3prle 的三行说明
for _m in ['td3prrbb/td3prrbb_main.py', 'td3prle/td3prle_main.py']:
    P(_m,
      '    U(0.17,0.23) 对应单机 Cp≈0.468~0.545 —— **低于**单机最优 0.593, 因为**风场**最优\n'
      '    本就落在单机 Cp 的次优点(上游降载减少尾流亏损, 全场反而多出 7.3%, 实测 u=10);\n',
      '    U(0.05,0.45) 覆盖"上游降载 → 末台单机最优"的整个区间, **故意不预设**风场最优的形状\n'
      '    (它随尾流状态变化, 是**策略要学的函数**);\n',
      None)
# td3leprrlrb 的注释块
P('td3leprrlrb/td3leprrlrb_main.py',
  '# 处方: 前 5 维 (风机桨距) bias 采在 U(0.17, 0.23) —— **风场总功率**的最优带\n'
  '#   (实测 probe_a_farm_structure.py: u=6/8/10/12/14 的集体峰全在 a=0.20, 且峰功率精确\n'
  '#    ∝ u³ ⇒ 最优动作与风速无关; 注意该带内**单机** Cp 只有 0.468~0.545, 低于单机最优\n'
  '#    0.593 —— 风场最优本就落在单机 Cp 的次优点, 上游降载减少尾流亏损);\n',
  '# 处方: 前 5 维 (风机桨距) bias 采在 U(0.05, 0.45) —— 覆盖"上游降载 → 末台单机最优"的\n'
  '#   整个区间, **故意不预设**风场最优的形状: 末台背风无机组应推向单机最优 ≈1/3, 上游\n'
  '#   降载留风, 而这个形状随尾流状态变化, 是**策略要学的函数**, 不是可预先测定的常数;\n',
  None)
# 术语：不再叫"风场功率最优带"
for _m in MAINS:
    P(_m, '风场功率最优带', '中性宽带', None)

# ---------------- 3) env.*Reset() 的标称动作回到单机最优点 ----------------
P('env.py', 'a_temp = np.ones(self.num_turbine) * 0.20',
  'a_temp = np.ones(self.num_turbine) * 0.3333', 3)
P('test_five_days.py', 'a_temp = np.ones(env.num_turbine) * 0.20',
  'a_temp = np.ones(env.num_turbine) * 0.3333', 1)
P('test_compare_projects.py', 'a_temp = np.ones(env.num_turbine) * 0.20',
  'a_temp = np.ones(env.num_turbine) * 0.3333', 1)
P('test_simple.py', 'a_temp = np.ones(NB_TURBINE) * 0.20',
  'a_temp = np.ones(NB_TURBINE) * 0.3333', 1)

# ---------------- 4) 测试：判据改成纯契约 ----------------
T = 'agentutil/test_warmup_schedule.py'
P(T, 'assert np.allclose(a0[:5], 0.20) and a0[5] == 0.0, a0',
  'assert np.allclose(a0[:5], 1.0 / 3) and a0[5] == 0.0, a0', 1)
P(T, 'assert A[:, :5].min() >= 0.15 - 1e-12 and A[:, :5].max() <= 0.25 + 1e-12',
  'assert A[:, :5].min() >= 0.05 - 1e-12 and A[:, :5].max() <= 0.45 + 1e-12', 1)
P(T, 'wall = np.isclose(A[:, :5], 0.15) | np.isclose(A[:, :5], 0.25)',
  'wall = np.isclose(A[:, :5], 0.05) | np.isclose(A[:, :5], 0.45)', 1)
P(T,
  """# 判据换成**风场功率**(而不是"单机 ΣCp"): 用 probe_a_farm_structure.py 的实测平台
#   u=10: 峰 3.717941 MW @ a=0.20; a=0.15 → 3.619155 (−2.7%); a=0.25 → 3.680407 (−1.0%)
_PEAK, _P_LO, _P_HI = 3.717941, 3.619155, 3.680407
_wall_loss = max((_PEAK - _P_LO) / _PEAK, (_PEAK - _P_HI) / _PEAK)
a_mean = A[:, :5].mean()
print('warmup wt: E[a]={:.4f} (峰位 0.20), a∈[{:.2f},{:.2f}], 两端功率代价 ≤{:.1f}%'
      .format(a_mean, A[:, :5].min(), A[:, :5].max(), 100 * _wall_loss))
assert abs(a_mean - 0.20) < 0.02, a_mean     # 三档游走 {0.15,0.20,0.25} 的平稳均值 = 峰位
assert _wall_loss < 0.03, _wall_loss         # 带两端距峰不超过 3%""",
  """# 判据 = **纯契约**, 不判"最优在哪": 风场最优是随尾流状态变化的单调上升斜坡
# (最下游机组应推向单机最优 ≈1/3, 上游降载留风), 它是**策略要学的函数**,
# 不是可预先测定的常数 —— 故这里只验中性宽带被真正走到两端 + 起点 = 单机最优点。
a_mean = A[:, :5].mean()
print('warmup wt: E[a]={:.4f} (起点 1/3), a∈[{:.2f},{:.2f}] (中性宽带 [0.05,0.45])'
      .format(a_mean, A[:, :5].min(), A[:, :5].max()))
assert A[:, :5].min() <= 0.10 and A[:, :5].max() >= 0.40, \\
    '中性宽带必须被真正走到两端 (seed=7 的 41 块实测 min=0.05 / max=0.45)'
assert 0.20 < a_mean < 0.40, a_mean          # 从 1/3 出发的有界游走, 均值应在带中部附近""",
  1)
P(T, '    (0.20, 0.05, 0.15, 0.25)',
  '    (1.0 / 3, 0.05, 0.05, 0.45)', 1)
P(T, '默认(风场功率最优先验)与对照臂', '默认(中性宽带领先验)与对照臂', 1)


def main():
    ap = argparse.ArgumentParser(description='动作先验放宽为中性宽带 (默认 dry-run)',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true', help='真正写盘 (默认只打印)')
    args = ap.parse_args()

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
            if exp is None:
                log.append(('?', n, old))
                if n:
                    out = out.replace(old, new)
                continue
            if n != exp:
                problems.append('%s: 锚点命中 %d 次 (要求 %d) ⇒ %r' % (path, n, exp, old[:60]))
                continue
            out = out.replace(old, new)
            log.append(('OK', n, old))
        changed[path] = (out, nl, log)

    for path, (out, _nl, log) in changed.items():
        print('\n' + '=' * 100)
        print('=== %s ===' % path)
        for st, n, old in log:
            print('  [%s] x%-2d %r' % (st, n, old[:70]))
        cur = open(path, encoding='utf-8', errors='replace').read()
        if out != cur:
            for ln in difflib.unified_diff(cur.splitlines(), out.splitlines(), lineterm='',
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
    print('\n下一步验证: python agentutil/test_warmup_schedule.py')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
