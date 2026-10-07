# -*- coding: utf-8 -*-
"""把 `aifflor/aifflor_main.py` 派生成 `aifflorpave/aifflorpave_main.py`（PAVE arm）。

仓库惯例（见 `agentutil/port_yawflor.py` / `port_aifflor.py`）：
  * 每条替换要求**命中次数精确匹配**，任一不符则整体不写盘；
  * `--check` 做**反向还原**：把生成文件里的新文本换回旧文本必须**逐位等于源文件**
    ⇒ 这就是"aifflorpave_main 与 aifflor_main 只差替换表列出的这几处"的机器可读证明。

替换表（7 条，全部为**单因子**：只接 PAVE arm 的 agent/CLI/日志身份，不改训练口径）:
  1. ddpg 导入  + SP_TD3_PAVE 一行
  2. RUN_TAG    'aifflor' → 'aifflorpave'（CSV 名随之变成 df_aifflorpave.csv）
  3. --env 默认 'AIFFLOR' → 'AIFFLORPAVE'（日志目录/TB run 名）
  4. agent_H 用子类 SP_TD3_PAVE（agent_L 仍是基类 ⇒ L 层逐位不变）
  5. agent_H 构造处追加 8 个 PAVE kwargs
  6. --wt_noise_floor 之后追加 9 个 PAVE CLI
  7. `if args.wt_bias_init:` 之前插入一行 PAVE 口径打印

用法::

    python agentutil/port_aifflorpave.py [--apply] [--check]
    # 默认 dry-run（只打印差异与命中次数）

自带路径 bootstrap（可被任意 CWD 调用）。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

SRC = 'aifflor/aifflor_main.py'
DST = 'aifflorpave/aifflorpave_main.py'

# ------------------------------------------------------------------ 替换表
DDPG_IMPORT_OLD = "from aifflor.aifflor_ddpg import SP_TD3"
DDPG_IMPORT_NEW = ("from aifflor.aifflor_ddpg import SP_TD3\n"
                   "from aifflorpave.aifflorpave_ddpg import SP_TD3_PAVE")

PAVE_KWARGS = (
    "                     critic_act=args.critic_act,\n"
    "                     pave_w_mpr=args.pave_w_mpr,\n"
    "                     pave_w_vfc=args.pave_w_vfc,\n"
    "                     pave_w_curv=args.pave_w_curv,\n"
    "                     pave_sigma=args.pave_sigma,\n"
    "                     pave_delta=args.pave_delta,\n"
    "                     pave_nv=args.pave_nv,\n"
    "                     pave_every=args.pave_every,\n"
    "                     pave_report_only=bool(args.pave_report_only),\n"
    "                     pave_log_every=args.pave_log_every,\n"
)

PAVE_CLI = (
    "\n"
    "    # ---- PAVE 式 critic 几何正则 (aifflorpave arm 专有; 全 0/relu 时逐位等于 aifflor) ----\n"
    "    parser.add_argument('--critic_act', default='relu', type=str,\n"
    "                        choices=['relu', 'silu'],\n"
    "                        help='H 层 profit critic 的激活: relu=基线逐位同构; '\n"
    "                             'silu=C² 平滑(PAVE 算 Hessian 的前提)')\n"
    "    parser.add_argument('--pave_w_mpr', default=0.0, type=float,\n"
    "                        help='MPR 权重: 罚 ‖∇_aQ(s+ε,a)−∇_aQ(s,a)‖² (已按 ‖∇_aQ‖² 归一化)')\n"
    "    parser.add_argument('--pave_w_vfc', default=0.0, type=float,\n"
    "                        help='VFC 权重: 罚 ‖∇_aQ(s,a)−∇_aQ(s\\',a)‖² (直接压制相邻步梯度翻转)')\n"
    "    parser.add_argument('--pave_w_curv', default=0.0, type=float,\n"
    "                        help='Curv 权重: 罚 max(0, vᵀ∇²_aaQv+δ) 保凹性; 仅 act=silu 有效')\n"
    "    parser.add_argument('--pave_sigma', default=0.05, type=float,\n"
    "                        help='MPR 的状态扰动 std (归一化观测尺度)')\n"
    "    parser.add_argument('--pave_delta', default=0.0, type=float,\n"
    "                        help='Curv 的严格凹裕度 δ (原始曲率量纲; 用 --pave_report_only 1 标定)')\n"
    "    parser.add_argument('--pave_nv', default=1, type=int,\n"
    "                        help='Curv 的 Hutchinson 采样数')\n"
    "    parser.add_argument('--pave_every', default=1, type=int,\n"
    "                        help='每几次 critic 更新附加 1 次几何步 (1=每次; k 可匹配算力)')\n"
    "    parser.add_argument('--pave_report_only', default=0, type=int, choices=[0, 1],\n"
    "                        help='1=只测量三项量级与 gscale 并写 pave_stats.csv, 不施加 (标定权重用)')\n"
    "    parser.add_argument('--pave_log_every', default=50, type=int,\n"
    "                        help='pave_stats.csv 每几步记一行 (0=关闭)')\n"
)

PAVE_PRINT = ("    print('[PAVE] ' + agent_H.pave_summary())\n")


def _replace_once(text, old, new, tag, hits_log):
    n = text.count(old)
    hits_log.append((tag, n))
    if n != 1:
        return text
    return text.replace(old, new, 1)


def build(src_text):
    hits = []
    t = src_text
    t = _replace_once(t, DDPG_IMPORT_OLD, DDPG_IMPORT_NEW, '1 ddpg 导入', hits)
    t = _replace_once(t, "RUN_TAG = 'aifflor'", "RUN_TAG = 'aifflorpave'", '2 RUN_TAG', hits)
    t = _replace_once(t, "default='AIFFLOR',", "default='AIFFLORPAVE',", '3 --env 默认', hits)
    t = _replace_once(t, 'agent_H = SP_TD3(nb_states=nb_states_H, nb_actions=6, args=args,',
                      'agent_H = SP_TD3_PAVE(nb_states=nb_states_H, nb_actions=6, args=args,',
                      '4 agent_H 用子类', hits)
    t = _replace_once(t, "wt_noise_floor=args.wt_noise_floor,\n",
                      "wt_noise_floor=args.wt_noise_floor,\n" + PAVE_KWARGS,
                      '5 追加 PAVE kwargs', hits)
    t = _replace_once(t, "折算单机功率抖动约 ±2.4%')",
                      "折算单机功率抖动约 ±2.4%')" + PAVE_CLI,
                      '6 追加 PAVE CLI', hits)
    t = _replace_once(t, "    if args.wt_bias_init:",
                      PAVE_PRINT + "    if args.wt_bias_init:",
                      '7 PAVE 口径打印', hits)
    return t, hits


def revert(dst_text):
    """反向还原（用于 --check）。"""
    t = dst_text
    t = t.replace(DDPG_IMPORT_NEW, DDPG_IMPORT_OLD, 1)
    t = t.replace("RUN_TAG = 'aifflorpave'", "RUN_TAG = 'aifflor'", 1)
    t = t.replace("default='AIFFLORPAVE',", "default='AIFFLOR',", 1)
    t = t.replace('agent_H = SP_TD3_PAVE(nb_states=nb_states_H, nb_actions=6, args=args,',
                  'agent_H = SP_TD3(nb_states=nb_states_H, nb_actions=6, args=args,', 1)
    t = t.replace("wt_noise_floor=args.wt_noise_floor,\n" + PAVE_KWARGS,
                  "wt_noise_floor=args.wt_noise_floor,\n", 1)
    t = t.replace("折算单机功率抖动约 ±2.4%')" + PAVE_CLI, "折算单机功率抖动约 ±2.4%')", 1)
    t = t.replace(PAVE_PRINT + "    if args.wt_bias_init:", "    if args.wt_bias_init:", 1)
    return t


INIT_PY = '''# -*- coding: utf-8 -*-
"""`aifflorpave` —— aifflor 的**PAVE 式 critic 几何正则**分支方法（验证臂）。

目的（本仓库实测驱动）：基线 critic 的动作梯度场几何很差 ——
  * 相邻决策 ∇_aQ 符号**翻转率 35%**（Lee et al. 2026 的 Base 为 9–35.5%）；
  * 5 台机组间 ∇_aQ **同号率 45.7%**（≈随机）；
  * 全 6×6 ∇²_aaQ **负定性率 4.5%**（该文 Base 14–47%）⇒ μ≈0。
该文的理论界是策略敏感度 L ≤ M/μ，并给出**只作用于 critic** 的三项辅助损失
MPR / VFC / Curv。本方法的可检验预言：**施加几何正则后翻转率应显著下降且成绩改善**；
若只加大网络或只调噪声，翻转率不变（已实测）。

与 `aifflor` 的关系：本包**不复制** ddpg 主体，而是
  * `aifflorpave_pave.py` —— 新增（CriticPAVE + MPR/VFC/Curv 实现）；
  * `aifflorpave_ddpg.py` —— `SP_TD3` 的**子类**（只加可切换激活 + 一个附加几何优化步）；
  * `aifflorpave_main.py` —— 由 `agentutil/port_aifflorpave.py` 从 aifflor_main **机械派生**
    （7 处替换，`--check` 可逆证明）。
`--critic_act relu --pave_w_* 0` 时本 arm **逐位等于 aifflor 基线**。

L 层（BESS 残差动作）仍用基类 `SP_TD3` ⇒ 残差/零初始化设计不受影响。
"""

from aifflorpave.aifflorpave_ddpg import SP_TD3_PAVE
from aifflorpave.aifflorpave_pave import CriticPAVE, pave_terms
'''

README_MD = '''# aifflorpave —— PAVE 式 critic 几何正则（验证臂）

## 1. 它要修什么（本仓库实测）

| 指标（基线 16/100 @ ckpt 28000） | 实测 | 文献参考（Lee et al. 2026 的 Base） |
|---|---|---|
| 相邻决策 ∇_aQ 符号翻转率 | **33–35%**（ckpt 10000 处 41%） | 9.0–35.5% |
| 5 台机组 ∇_aQ 同号率 | **45.7%**（≈随机） | — |
| 全 6×6 ∇²_aaQ 负定性率 | **4.5%**（256/256 仅 0.5%） | 14–47% |
| `slopeSD` 随训练增长 | **2.7×**（2.93→7.84） | M_sup 增长约 4× |

理论：策略敏感度 `L ≤ M/μ`（M=‖∇²_sa Q‖，μ=−λ_max(∇²_aa Q)）。

## 2. 三项辅助损失（只作用于 critic）

```
L_MPR  = E_ε ‖∇_a Q(s+ε,a) − ∇_a Q(s,a)‖²       (M 的有限差分代理)
L_VFC  = E   ‖∇_a Q(s,a)   − ∇_a Q(s',a)‖²       (轨迹相邻状态的场一致)
L_Curv = E_v max(0, vᵀ∇²_aa Q(s,a) v + δ)        (保曲率，防过平滑塌陷)
```

## 3. 与论文的两处显式差异

1. **附加优化步**而非联合损失：`super().update_policy()` 走完基线后，再对 profit 双
   critic 走一次几何梯度（`--pave_every k` 可降频）。好处：基线路径逐位不变、可审计、
   单因子消融；代价：critic 优化步数 +1。
2. 只作用于 **profit 双 critic**（actor 梯度主通道）；cost 通道与 L 层完全不碰。

## 4. 口径与开关

| CLI | 默认 | 说明 |
|---|---|---|
| `--critic_act` | `relu` | `relu`=与基线逐位同构；`silu`=C²（Curv 项的前提） |
| `--pave_w_mpr` / `--pave_w_vfc` | 0.0 | 已按 ‖∇_aQ‖² 归一化（无量纲） |
| `--pave_w_curv` | 0.0 | 原始量纲，**仅 act=silu 有效**（ReLU 的 ∇²_aaQ≡0） |
| `--pave_sigma` / `--pave_delta` / `--pave_nv` | 0.05 / 0.0 / 1 | MPR 扰动 std / Curv 严格凹裕度 / Hutchinson 采样数 |
| `--pave_report_only 1` | 0 | **只测量不施加**，写 `pave_stats.csv`（标定权重用） |
| `--pave_log_every` | 50 | `pave_stats.csv` 记录间隔（0=关） |

**标定流程**：先跑 `--critic_act silu --pave_report_only 1`（或 relu 下只测 MPR/VFC），
从 `<output>/pave_stats.csv` 读三项量级与 `gscale`，再给权重。

## 5. 不变量

`python agentutil/test_aifflorpave_contract.py`：
① `CriticPAVE(act='relu')` 与 `model.Critic` **数值逐位相同**（同 seed）；
② 关掉 PAVE 时 `SP_TD3_PAVE` 与 `SP_TD3` 的行动作逐位相同；
③ 几何步的梯度只落在 profit 双 critic（actor / cost 的梯度为 None 或 0）；
④ `pave_terms` 三项有限且非负；ReLU 下 Curv ≡ 0。

## 6. 目录

* `aifflorpave_pave.py` —— CriticPAVE + MPR/VFC/Curv
* `aifflorpave_ddpg.py` —— `SP_TD3_PAVE(SP_TD3)`
* `aifflorpave_main.py` —— 由 `agentutil/port_aifflorpave.py` 从 `aifflor_main.py` 派生
  （`python agentutil/port_aifflorpave.py --check` 证明只差 7 处）
'''


def main():
    apply = '--apply' in sys.argv
    check = '--check' in sys.argv
    src = open(SRC, encoding='utf-8').read()
    dst, hits = build(src)
    print('=' * 78)
    print('port_aifflorpave: %s -> %s' % (SRC, DST))
    ok = True
    for tag, n in hits:
        flag = 'OK' if n == 1 else '**命中 %d 次 ⇒ 失败**' % n
        if n != 1:
            ok = False
        print('  %-26s %s' % (tag, flag))
    extra = [(k, dst.count(k)) for k in ('SP_TD3_PAVE', 'pave_', 'AIFFLORPAVE', 'aifflorpave')]
    print('  生成文件自检: ' + '  '.join('%s×%d' % (k, v) for k, v in extra))
    if not ok:
        print('** 有替换未精确命中，整体不写盘 **')
        return 1
    if check:
        if not os.path.isfile(DST):
            print('** --check 需要先 --apply **')
            return 1
        got = open(DST, encoding='utf-8').read()
        rev = revert(got)
        same = (rev == src)
        print('  --check 反向还原: %s' % ('✅ 逐位等于 %s' % SRC if same else '**FAIL**'))
        if not same:
            import difflib
            for l in list(difflib.unified_diff(src.splitlines(), rev.splitlines(),
                                               'src', 'reverted', lineterm=''))[:40]:
                print('    ' + l)
            return 1
        return 0
    if apply:
        os.makedirs(os.path.dirname(DST), exist_ok=True)
        open(DST, 'w', encoding='utf-8', newline='\n').write(dst)
        open('aifflorpave/__init__.py', 'w', encoding='utf-8', newline='\n').write(INIT_PY)
        open('aifflorpave/README.md', 'w', encoding='utf-8', newline='\n').write(README_MD)
        print('  ✅ 已写入 %s / __init__.py / README.md' % DST)
        return 0
    print('  (dry-run; 加 --apply 写盘, 加 --check 做可逆审计)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
