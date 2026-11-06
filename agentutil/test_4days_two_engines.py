#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四固定风速日 × 两模型（AIFFLOR-run11 / PF1-AIFFLOR-run1）× 两引擎评测。

问题
----
同一套策略（`aifflor` 的 HSRL 分支）在两个**物理引擎**上各跑一遍：
    `aiffloris`     = FLORIS（纯 Python，`calibrated=False` + `power_scale=LEVEL_COEF`，
                      与 `aifflor/aifflor_main.py` 的生产默认**逐字相同**）
    `matlabwfsim`   = WFSim（MATLAB，`env.WindBattery` → `WFSim_port`，
                      即 `aifflor` 的**跨引擎迁移**评测臂）
两条臂**只换引擎**：策略权重、动作语义（轴向因子 `a∈[0,0.5]` + 第 6 维块总放电量 `g`）、
观测归一化、SOC 起点、风速序列、电价日程全部相同 ⇒ 差异只可能来自引擎。

四天的"固定"含义
----------------
`DAY_SPECS` 定义**怎么选**（日均风速落在哪个桶、取桶中心最近的列），
`FROZEN_DAYS` 记录**选完的结果**（`wind_iloc` 列号）。默认用 `FROZEN_DAYS`
（= 已固定），并逐列核验日均风速仍落在对应桶内；数据被改动过就会报错，
此时用 `--rebuild-days` 按规则重选（会打印可直接粘回源码的 `FROZEN_DAYS` 行）。
四个桶来自用户口径：① 5~6 m/s ② 10 m/s ③ 13 m/s ④ 15 m/s。

记录的三项日内指标（每个模型 × 每个引擎 × 每一天）
--------------------------------------------------
    viol_count       日内违例次数      Σ 1[ P_F > 3.0 + 1e-4 ]          （与 aifflor_main 的窗口计数同阈值）
    viol_excess_mw   日内违例功率和    Σ max(P_F − 3.0, 0)               [MW]，只在越限步累加
    profit_rmb       日内盈利          Σ (P_G·pr − 490·max(−pbrule,0))·Δt [元]
                                      ≈ `aifflor_main` 的 `episode_profit`（同式，Δt=1/60 h）

两次运行的结果**分开储存**
--------------------------
    <out>/fixed_days.csv                     四天清单（两引擎共用，只写一次）
    <out>/aiffloris/<模型>/{days,steps}.csv   FLORIS 臂（逐日汇总 + 逐步明细）
    <out>/matlabwfsim/<模型>/{days,steps}.csv WFSim 臂
    <out>/summary_all.csv, summary_by_run.csv, pivot_*.csv   跨引擎对照
    <out>/tensorboard/_wind                  四天**日内风速**（step = 一天内第几分钟）
    <out>/tensorboard/aiffloris              该引擎两模型的指标
    <out>/tensorboard/matlabwfsim

（TB 用"每个引擎一个 run + 一个公共 _wind run"，一条命令看全：
 `tensorboard --logdir <out>/tensorboard`；CSV 侧则是完全分开的目录。）

与 `test_five_days.py` 的关系
----------------------------
* 复用它的 `_detect_hidden` / `_detect_nb_states` / `ensure_engine` /
  `quit_engine` / `load_wind_pool`（不重复造轮子，保证口径一致）；
* **不复用** `ProjectRunner`：`test_five_days` 的 `aifflor` 条目是**非 HSRL**
  路径（5 维蒸馏 L 观测 + SOC 目标信号 + 残余动作），而这两个 run 都是
  `--hsrl` 分支（6 维 L 观测 + 绝对 pb + 块总放电量 g），推理路径不同。
  因此本文件的 `HsrlRunner.run_day` 按 `aifflor_main.train_hsrl` 的逐步口径
  重写为**纯前向**版本（不更新网络、不加探索噪声、不做对偶）。

每个模型需要哪些推理期开关（自动探测，不靠手填）
----------------------------------------------
    scalar_wt          `--scalar_wt`（五台风机共用一个 a[0]）。从 `train.log` 的
                       `[baseline] 动作空间 = 1 个自由度` 横幅判定；也可 `--scalar-wt 0|1` 强制。
                       实测 AIFFLOR-run11 = 开、PF1-AIFFLOR-run1 = 关
                       （`df_hsrl.csv` 里前 5 维跨维最大差 0.0000 vs 0.4999）。
    nb_states          从 `actor_6.pkl` / `actor_1.pkl` 的 `fc1.weight` 反推（都是 4 / 6）
    hidden1/hidden2    从 actor 权重形状反推（都是 16 / 100）
    critic_arch/feat   `model.infer_arch` / `model.feat_from_sd` 从 critic 权重反推 ——
                       **必须**，否则 `load_weights` 的形状断言会失败
                       （run11 = v1 无 feat；PF1 的 H = v2 + feat_power、L = v2 无 feat）。
    phase_feat         由 `nb_states − 4` 反推（两支都是 0 ⇒ 无相位特征）

用法
----
    # 全跑（默认 2 模型 × 2 引擎 × 4 天 = 16 个 episode；WFSim 很慢，先看 --dry-run）
    python agentutil/test_4days_two_engines.py --dry-run
    python agentutil/test_4days_two_engines.py

    # 只跑 FLORIS 臂（不启 MATLAB，几分钟）
    python agentutil/test_4days_two_engines.py --engines aiffloris

    # 只跑 WFSim 臂 / 只跑某个模型
    python agentutil/test_4days_two_engines.py --engines matlabwfsim --models run11

    # 冒烟（每"天"只跑 20 步，指标只统计这 20 步）
    python agentutil/test_4days_two_engines.py --episode-length 20

    # 自检：权重形状 + 引擎锚点（u=10、a=1/3 的出力），不跑 episode
    python agentutil/test_4days_two_engines.py --selftest

    # 重新选天（数据变了 / 想换桶宽）
    python agentutil/test_4days_two_engines.py --rebuild-days --dry-run

固定动作基线（`--fixed-a`，**不需要模型目录**）
--------------------------------------------
`--fixed-a A` 把五台轴向因子全程钉在 `A`、第 6 维（块总放电量 `g`）钉在 `--goal`（默认 0），
下层走 HSRL 的**摊还执行器**规则 `pb = clip(−(g−Σ)/n_rem, 物理域)`：
    g = 0  ⇒  need ≡ 0  ⇒  pb ≡ 0  ⇒  储能完全不动作（SOC 恒定）
也就是说这条臂 = **纯固定桨距风场 + 空电池**，不带任何学习成分，用来当"引擎/口径"的参照点。
    python agentutil/test_4days_two_engines.py --fixed-a 0.3333 --goal 0 --engines matlabwfsim
结果落在 `<out>/<engine>/FIXED-A0.333/`，与其它模型臂并列进同一套汇总表；不同 `--fixed-a`
值会得到不同目录（不会互相覆盖）。

环境
----
    MATLAB 引擎要求本机装有 MATLAB 且 `matlab.engine` 可用；受限沙箱（named pipe
    被禁）里起不来 ⇒ WFSim 臂请用**普通终端**跑。FLORIS 臂不需要 MATLAB。
    已知慢：WFSim 约 0.3~1 s/步 ⇒ 一天 1440 步 ≈ 7~24 min；4 天 × 2 模型 ≈ 1~3 h。
"""
import argparse
import csv
import os
import sys
import time
from copy import deepcopy

# ---------------------------------------------------------------------------
# 路径 bootstrap（AGENTS.md 约定：可被任意 CWD / 任意路径调用）
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)                                         # 相对数据/输出路径以仓库根为基准
sys.path.insert(0, ROOT)                               # 项目模块
sys.path.insert(0, _HERE)                              # agentutil 内互相 import

# Windows 中文控制台默认 GBK：个别符号无法编码会抛 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors='replace')
    except Exception:
        pass

import numpy as np                                                # noqa: E402
import pandas as pd                                              # noqa: E402

# ---------------------------------------------------------------------------
# 口径常量（与 aifflor_main / env / hsrl 一致，不另立一套）
# ---------------------------------------------------------------------------
DT = 1.0 / 60.0                     # 步长 (小时)；1 步 = 1 min
MAX_E = 0.50                        # BESS 容量 (MWh)
NB_TURBINE = 5                      # 风机台数
EPISODE_LENGTH = 1440               # 1 天 = 24 h × 60 min
FLUCT_LIMIT = 3.0                   # 波动限值 (MW)
VIOL_TOL = 1e-4                     # 越限判定容差（= aifflor_main 的 `P_F > 3.0 + 1e-4`）
RATED_POWER = 0.5                   # BESS 额定功率 (MW)
ETA_C = ETA_D = 0.98
BESS_DEG_RMB_PER_MWH = 490.0        # 放电退化成本（profit 口径里那一项）
SOC0_DEFAULT = 0.25                 # 初始 SOC (MWh) = hsrl_warmup.SOC_START / env.reset()
WIND_CSV = 'FFC_speed_noise_1.csv'

DAY_METRICS = ('viol_count', 'viol_excess_mw', 'profit_rmb')

# ---------------------------------------------------------------------------
# ① 四个风速日的**选择规则**（桶 + 桶中心）
# ---------------------------------------------------------------------------
#: (tag, 桶下界, 桶上界, 桶中心)；规则 = 桶内取"日均风速最接近桶中心"的列，
#: 并列时取列号小的。桶区间 = [lo, hi)。
DAY_SPECS = (
    ('d1_u05-06', 5.0, 6.0, 5.5),      # ① 平均风速 5~6
    ('d2_u10', 9.5, 10.5, 10.0),       # ② 平均风速 10
    ('d3_u13', 12.5, 13.5, 13.0),      # ③ 平均风速 13
    ('d4_u15', 14.5, 15.5, 15.0),      # ④ 平均风速 15
)

# ---------------------------------------------------------------------------
# ② 已固定的四天（`FFC_speed_noise_1.csv` 的 iloc 列号；第 0 列是行号索引）
# ---------------------------------------------------------------------------
# 选法见 `DAY_SPECS`；这四行就是"挑选完成之后固定它们"的落点。
# 核验：日均风速仍在对应桶内 + 与 `days_fixed.csv` 记录一致，否则报错（不静默换天）。
# 重新选：`--rebuild-days`（会打印新的 FROZEN_DAYS 字面量）。
FROZEN_DAYS = {
    'd1_u05-06': 31,     # 日均 5.967 m/s（列标签 '35'）
    'd2_u10': 132,       # 日均 10.057 m/s（列标签 '137'）
    'd3_u13': 160,       # 日均 13.004 m/s（列标签 '165'）
    'd4_u15': 134,       # 日均 14.850 m/s（列标签 '139'）
}

# ---------------------------------------------------------------------------
# ③ 两个模型（同一算法家族、同一 HSRL 分支；差别见 docstring 的"自动探测"一节）
# ---------------------------------------------------------------------------
MODELS = (
    dict(tag='AIFFLOR-run11', dir='output/AIFFLOR-run11',
         note='critic v1 / 无 feat / 标量动作(--scalar_wt 1)'),
    dict(tag='PF1-AIFFLOR-run1', dir='output/PF1/AIFFLOR-run1',
         note='critic v2 / H 带 feat_power / 五维独立动作'),
)

# ---------------------------------------------------------------------------
# ④ 两个引擎（顺序 = 用户要的顺序：先 aiffloris，再 matlabwfsim）
# ---------------------------------------------------------------------------
ENGINES = ('aiffloris', 'matlabwfsim')
ENGINE_DESC = {
    'aiffloris': 'FLORIS 纯 Python（calibrated=False + power_scale=LEVEL_COEF）',
    'matlabwfsim': 'WFSim MATLAB 引擎（env.WindBattery -> WFSim_port）',
}


# ===========================================================================
# 风速日：选 / 固定 / 核验
# ===========================================================================
def load_wind_pool(csv_path=WIND_CSV):
    """返回 (df, means)。

    列 1..194 是 194 条训练用风速日（列 0 是 pandas 写出的行号）。

    ⚠ `means` 的索引 = **列位置**（1..194），不是列标签 ——
      `env.py` 取天用的是 `self.wind_noise.iloc[:, j]`（位置），
      而 CSV 的列标签并不连续（存在跳号），两者不可混用。
    """
    df = pd.read_csv(os.path.join(ROOT, csv_path))
    vals = df.iloc[:, 1:].mean().to_numpy(dtype=float)
    means = pd.Series(vals, index=np.arange(1, len(vals) + 1))
    return df, means


def select_days(df, means):
    """按 `DAY_SPECS` 的规则选天；返回 [{tag, iloc, col, mean, lo, hi}]。"""
    picked = []
    for tag, lo, hi, center in DAY_SPECS:
        cand = means[(means >= lo) & (means < hi)]
        if len(cand) == 0:
            raise SystemExit(
                '风速池里没有日均落在 [{}, {}) 的列 —— 请放宽 DAY_SPECS 的桶宽'.format(lo, hi))
        # 最接近桶中心；并列 → 列号小（保证可复现）
        order = sorted(cand.items(), key=lambda kv: (abs(float(kv[1]) - center), int(kv[0])))
        iloc, m = order[0]
        picked.append(dict(tag=tag, iloc=int(iloc), col=str(df.columns[int(iloc)]),
                           mean=float(m), lo=float(lo), hi=float(hi)))
    return picked


def resolve_days(df, means, out_dir, rebuild=False):
    """确定本次要用的四天，并把清单落到 `fixed_days.csv`。

    默认 = `FROZEN_DAYS`（已固定）⇒ 每次都核验日均仍在桶内，防止"数据换了、天悄悄变了"。
    `--rebuild-days` = 按规则重选，并打印可直接粘回源码的 `FROZEN_DAYS` 字面量。
    """
    if rebuild:
        picked = select_days(df, means)
        print('[rebuild] 按 DAY_SPECS 规则重新选天（本次生效，并已写入 fixed_days.csv）:')
        print('[rebuild] FROZEN_DAYS = {')
        for d in picked:
            print("[rebuild]     '%s': %4d,     # 日均 %.3f m/s"
                  % (d['tag'], d['iloc'], d['mean']))
        print('[rebuild] }')
    else:
        picked = []
        for (tag, lo, hi, center) in DAY_SPECS:
            if tag not in FROZEN_DAYS:
                raise SystemExit('FROZEN_DAYS 缺少 %r（用 --rebuild-days 重选）' % tag)
            iloc = int(FROZEN_DAYS[tag])
            if iloc < 1 or iloc >= df.shape[1]:
                raise SystemExit('FROZEN_DAYS[%r] = %d 超出风速池列范围 [1, %d)'
                                 % (tag, iloc, df.shape[1]))
            m = float(means.iloc[iloc - 1])
            if not (lo <= m < hi):
                raise SystemExit(
                    '风速池已变化：%s（iloc=%d）日均 %.3f 不在桶 [%g, %g) 内。\n'
                    '  ⇒ 数据被换过？确认后用 --rebuild-days 重选，或恢复原数据。'
                    % (tag, iloc, m, lo, hi))
            picked.append(dict(tag=tag, iloc=iloc, col=str(df.columns[iloc]),
                               mean=m, lo=float(lo), hi=float(hi)))

    # ---- 落盘（两引擎共用同一份清单，也作为"固定"的书面记录）----
    path = os.path.join(out_dir, 'fixed_days.csv')
    os.makedirs(out_dir, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['tag', 'wind_iloc', 'wind_col', 'wind_mean', 'wind_min', 'wind_max',
                    'bucket_lo', 'bucket_hi'])
        for d in picked:
            s = df.iloc[:, d['iloc']].reset_index(drop=True)
            w.writerow([d['tag'], d['iloc'], d['col'], '%.6f' % d['mean'],
                        '%.6f' % float(s.min()), '%.6f' % float(s.max()),
                        d['lo'], d['hi']])
    return picked


# ===========================================================================
# 模型配置：权重形状 / critic 结构 / 推理期开关（全部自动探测）
# ===========================================================================
def _read_train_log(model_dir):
    """读 train.log（容错：不存在/读不动就返回空串）。"""
    for name in ('train.log', 'train_.log'):
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            try:
                with open(p, 'r', encoding='utf-8', errors='replace') as fh:
                    return fh.read()
            except Exception:
                pass
    return ''


def detect_scalar_wt(model_dir):
    """`--scalar_wt` 是否开启 —— 由 main 打印的 baseline 横幅判定。

    横幅原文（`aifflor_main.py:2183`）：
        [baseline] 动作空间 = 1 个自由度: 五台风机轴向因子全部 = a[0]
    这是**推理期唯一的模型级开关**（其余 --critic_* / --c1_* 只影响训练）。
    """
    log = _read_train_log(model_dir)
    if not log:
        return None
    return bool('动作空间 = 1 个自由度' in log)


def detect_model_cfg(model_dir, na_H=6, na_L=1):
    """从权重文件反推构造网络所需的全部形状参数。"""
    import torch
    from model import infer_arch, feat_from_sd

    cfg = dict(dir=model_dir, na_H=na_H, na_L=na_L)

    # ---- 观测维数（actor 第一层的输入宽）----
    def _ns(na):
        p = os.path.join(model_dir, 'actor_%d.pkl' % na)
        if not os.path.isfile(p):
            raise SystemExit('缺少权重文件: %s' % p)
        sd = torch.load(p, map_location='cpu')
        for k in ('fc1.weight', '0.weight'):
            if k in sd and getattr(sd[k], 'shape', None) is not None and len(sd[k].shape) == 2:
                return int(sd[k].shape[1])
        raise SystemExit('%s 里找不到 fc1.weight，无法反推观测维数' % p)

    def _hidden(na):
        p = os.path.join(model_dir, 'actor_%d.pkl' % na)
        sd = torch.load(p, map_location='cpu')
        if 'fc1.weight' not in sd or 'fc2.weight' not in sd:
            raise SystemExit('%s 里找不到 fc1/fc2，无法反推隐层宽度' % p)
        return int(sd['fc1.weight'].shape[0]), int(sd['fc2.weight'].shape[0])

    cfg['ns_H'] = _ns(na_H)
    cfg['ns_L'] = _ns(na_L)
    cfg['hidden_H'] = _hidden(na_H)
    cfg['hidden_L'] = _hidden(na_L)
    if cfg['hidden_H'] != cfg['hidden_L']:
        raise SystemExit('H/L 隐层宽度不同(%s vs %s)：本脚本共用一个 --hidden1/2 口径，'
                         '请确认该 run' % (cfg['hidden_H'], cfg['hidden_L']))
    cfg['hidden'] = cfg['hidden_H']

    # ---- 相位特征：nb_states = 4 + PHASE_DIM[mode] ----
    _dim2mode = {0: 0, 2: 1, 5: 2}
    d = cfg['ns_H'] - 4
    if d not in _dim2mode:
        raise SystemExit('H 观测维数 %d 无法解释为 4+相位(0/2/5)' % cfg['ns_H'])
    cfg['phase_mode'] = _dim2mode[d]

    # ---- critic：arch + feat_power 必须与权重一致，否则 load_weights 报形状错 ----
    def _critic(na, ns):
        p = os.path.join(model_dir, 'critic_p_%d.pkl' % na)
        if not os.path.isfile(p):
            raise SystemExit('缺少权重文件: %s' % p)
        sd = torch.load(p, map_location='cpu')
        arch = str(infer_arch(sd, ns, na))
        feat = bool(feat_from_sd(sd, ns, na, cfg['hidden'][0], cfg['hidden'][1]))
        return arch, feat

    cfg['critic_H'] = _critic(na_H, cfg['ns_H'])
    cfg['critic_L'] = _critic(na_L, cfg['ns_L'])
    return cfg


def build_args(cfg, device, seed=0, critic_arch='v2', critic_feat_power=False,
               scalar_wt=0):
    """构造 `SP_TD3` 需要的 args 命名空间（纯推理，训练专用项给安全默认）。

    复用 `test_five_days._build_args` 的骨架（它已覆盖 SP_TD3 直接读的全部字段），
    再补上本分支特有的 actor/critic 结构开关。
    """
    import test_five_days as T
    ns = T._build_args(device, seed, hidden=cfg['hidden'])
    ns.nb_turbine = NB_TURBINE
    ns.na_box = NB_TURBINE
    ns.phase_feat = int(cfg['phase_mode'])
    # ---- critic 结构（必须与 checkpoint 一致）----
    ns.critic_arch = str(critic_arch)
    ns.critic_feat_power = int(bool(critic_feat_power))
    ns.critic_action_first = 0
    ns.cost_clip = 1
    # ---- 动作空间开关（推理期生效）----
    ns.scalar_wt = int(bool(scalar_wt))
    ns.farm_level = 0
    ns.aux_model = 0
    # ---- 让 SP_TD3 的 seed() 不动全局 RNG（评测必须可复现且无副作用）----
    ns.seed = 0
    return ns


# ===========================================================================
# 单个模型 × 单个引擎的"一天"执行器（= aifflor_main.train_hsrl 的纯前向版）
# ===========================================================================
class HsrlRunner:
    """加载一份模型，在一个引擎上按它自己训练时的推理路径跑完整 episode。

    与 `train_hsrl` 的差异**只有三处**（其余逐步同序）：
      1. 不更新网络（`is_training=False` ⇒ `select_action` 无探索噪声）；
      2. 不做对偶/EMA/curriculum/20 步测试（不影响动作）；
      3. 风速日由外部给定（四天固定），不是 `env.reset()` 的固定序列。
    """

    def __init__(self, cfg, engine_kind, device, day_runner_args, verbose=True):
        import aifflor.aifflor_hsrl as hsrl
        import aifflor.aifflor_myfunc as myf
        from rbb.rbb_rule_bess import RuleBasedBESS

        self.hsrl = hsrl
        self.myf = myf
        self.engine_kind = engine_kind
        self.cfg = cfg
        self.verbose = verbose
        self.episode_length = int(day_runner_args['episode_length'])
        self.soc0 = float(day_runner_args['soc0'])
        # ---- 固定动作基线（`--fixed-a`）：无模型臂 ----
        self.fixed_a = day_runner_args.get('fixed_a')
        self.fixed_g = float(day_runner_args.get('goal', 0.0) or 0.0)
        self.phase_mode = int(cfg['phase_mode']) if cfg else 0
        self.phase_table = (myf.phase_daily_table() if self.phase_mode >= 2 else None)

        # ---- 物理后端 ----
        if engine_kind == 'aiffloris':
            from aifflor.aifflor_env import AifflorPhysics, WindBatteryAIF
            from aifflor import aifflor_turbine as AT
            # 与 aifflor_main.py:2236 逐字相同：FLORIS 随包默认尾流参数 + 出口电平系数
            physics = AifflorPhysics(nt=NB_TURBINE, calibrated=False,
                                     power_scale=AT.LEVEL_COEF)
            self.env = WindBatteryAIF(MAX_E, NB_TURBINE, False, None, physics=physics)
        elif engine_kind == 'matlabwfsim':
            import env as envmod
            self.env = envmod.WindBattery(MAX_E, NB_TURBINE, False,
                                          day_runner_args['engine'])
        else:
            raise ValueError('未知引擎: %r' % (engine_kind,))

        # ---- RBB：HSRL 下 fluctuation_limit=inf（退化为"纯 SOC+额定功率裁剪"）----
        self.rbb = RuleBasedBESS(max_E=MAX_E, min_E_factor=0.0,
                                 rated_power=RATED_POWER, eta_c=ETA_C, eta_d=ETA_D,
                                 time=DT, fluctuation_limit=float('inf'))

        # ---- 智能体（H: 6 动作 / L: 1 动作）----
        # `--fixed-a` 时**不加载任何模型**：动作是写死的常量，下层走摊还规则
        import torch
        torch.set_grad_enabled(False)
        self.agent_H = self.agent_L = None

        if self.fixed_a is not None:
            if verbose:
                print('    [{}] 固定动作臂：a_i ≡ {:.4f}（五台），g ≡ {:g} MW·min，'
                      '下层 = 摊还规则（g=0 ⇒ pb ≡ 0，储能不动作）'
                      .format(engine_kind, float(self.fixed_a), self.fixed_g))
            return

        from aifflor.aifflor_ddpg import SP_TD3

        args_H = build_args(cfg, device, critic_arch=cfg['critic_H'][0],
                            critic_feat_power=cfg['critic_H'][1],
                            scalar_wt=day_runner_args['scalar_wt'])
        args_L = build_args(cfg, device, critic_arch=cfg['critic_L'][0],
                            critic_feat_power=cfg['critic_L'][1], scalar_wt=0)

        # H 层 λ1 的区间/初值单一真源 = `lam1_phys_bounds()`（与 aifflor_main:2317 同）
        _lo1, _hi1, _init1 = hsrl.lam1_phys_bounds()
        self.agent_H = SP_TD3(nb_states=cfg['ns_H'], nb_actions=6, args=args_H,
                              lam_max=_hi1, lam_min=_lo1,
                              ema_lam=True, ema_alpha=0.02, ema_lam_lr=0.01,
                              record_update_stats=False,
                              policy_delay=args_H.policy_delay,
                              policy_noise=args_H.policy_noise,
                              noise_clip=args_H.noise_clip,
                              cost_critic_mode='max',
                              wt_noise=0.05, sig_noise_max=0.2, sig_noise_min=0.05,
                              wt_noise_floor=0.0, nstep=1, target_agg='min')
        self.agent_H.lam = float(_init1)
        self.agent_H.is_training = False
        self.agent_H.load_weights(cfg['dir'])

        self.agent_L = SP_TD3(nb_states=cfg['ns_L'], nb_actions=1, args=args_L,
                              lam_max=hsrl.HSRL_L_LAM_MAX, lam_min=hsrl.HSRL_L_LAM_MIN,
                              ema_lam=True, ema_alpha=0.02,
                              ema_lam_lr=hsrl.HSRL_L_EMA_LAM_LR,
                              record_update_stats=False,
                              policy_delay=args_L.policy_delay,
                              policy_noise=args_L.policy_noise,
                              noise_clip=args_L.noise_clip,
                              cost_critic_mode='max',
                              nstep=1, target_agg='min')
        self.agent_L.lam = hsrl.HSRL_L_LAM_INIT
        self.agent_L.cost_limit = hsrl.HSRL_L_COST_LIMIT
        self.agent_L.is_training = False
        self.agent_L.load_weights(cfg['dir'])

        if verbose:
            print('    [{}] H ns={} critic={}{} | L ns={} critic={}{} | scalar_wt={} | '
                  'hidden={}/{} | phase={}'
                  .format(engine_kind, cfg['ns_H'], cfg['critic_H'][0],
                          '+feat' if cfg['critic_H'][1] else '',
                          cfg['ns_L'], cfg['critic_L'][0],
                          '+feat' if cfg['critic_L'][1] else '',
                          int(day_runner_args['scalar_wt']),
                          cfg['hidden'][0], cfg['hidden'][1], self.phase_mode))

    # -- 观测归一化（H 层 4 维 + 可选相位）----------------------------------
    def _norm_H(self, obs, t):
        return self.myf.Normalize_H(obs, int(t) % self.myf.DAY_MINUTES,
                                    self.phase_mode, self.phase_table)

    # -- 一个 episode -------------------------------------------------------
    def run_day(self, wind_series, day_idx, wind_iloc):
        hsrl = self.hsrl
        env = self.env
        agent_H, agent_L = self.agent_H, self.agent_L
        CheckH = hsrl.check_action_H_hsrl

        # ---- episode 起点：复刻 env.reset()，再把风速日换成抽中的那一天 ----
        obs = deepcopy(env.reset())
        env.wind_day = wind_series
        env.count_wind = 0
        env.u = float(wind_series.iloc[0])
        obs[1] = env.u
        a_nom = env.nominal()
        temp = env.PowerOutput(env.u, a_nom)
        env.P_G_prev = temp                    # 波动基准 = 当日首分钟出力
        obs[-1] = temp
        env.E = self.soc0                      # 初始 SOC（两支同值 ⇒ 可比）
        obs[2] = self.soc0

        norm_obs = self._norm_H(obs, 0)
        if agent_H is not None:
            agent_H.reset(norm_obs)

        # ---- 累计量 ----
        viol_count = 0
        viol_excess = 0.0
        profit = 0.0
        soc_sum = 0.0
        soc_end = self.soc0
        max_pf = 0.0
        discharge_mwh = 0.0
        charge_mwh = 0.0
        a_sum = 0.0            # Σ 前 5 维执行动作（诊断：是否贴 0 下界 / 是否五台同值）
        a_zero_steps = 0       # 前 5 维全 ≤0 的步数（两个引擎在 a=0 处结构性分歧，见 README）
        step_rows = []

        action_H = applied_H = None
        g_applied = 0.0
        pw_tru = 0.0
        sum_disch = 0.0

        for t in range(self.episode_length):
            # ---- 块首（每 5 步）：H 决策 + 块级可行域 ----
            if t % 5 == 0:
                sum_disch = 0.0
                E_block_start = float(obs[2])
                if self.fixed_a is not None:
                    # 固定动作臂：前 5 维恒为 `--fixed-a`，第 6 维恒为 `--goal`（默认 0）
                    action_H = np.concatenate([np.full(5, float(self.fixed_a)),
                                               np.array([self.fixed_g])])
                else:
                    # 注意：`select_action` 内部已做 `--scalar_wt` 投影（aifflor_ddpg:692），
                    # 因此这里**不需要**再调 `_apply_scalar_wt`。
                    action_H = np.asarray(agent_H.select_action(norm_obs),
                                          dtype=float).reshape(-1)
                applied_H, _pen5, _pen6 = CheckH(action_H, E_block_start)
                g_applied = float(applied_H[5])
                pw_tru = env.PowerOutput(obs[1], applied_H)

            ptur = float(pw_tru)
            pg_prev = float(obs[-1])
            E_now = float(obs[2])
            n_rem = 5 - (t % 5)

            # ---- L 观测（6 维，含 c_L 的参照点 need）----
            need = hsrl.plan_need(g_applied, sum_disch, n_rem)
            obs_L = hsrl.build_L_obs_hsrl(obs[1], obs[0], E_now, pg_prev, g_applied, need)
            if t % 5 == 0 and agent_L is not None:
                agent_L.reset(obs_L)

            # ---- L 动作 = 绝对 pb（正=充电），只按物理域裁剪 ----
            if self.fixed_a is not None:
                # 固定动作臂的下层 = **摊还执行器**（HSRL 的"零成本动作"规则）：
                #   pb = clip(−(g−Σ)/n_rem, 物理域) ⇒ g≡0 时 pb≡0（储能不动作，SOC 恒定）
                pb_cmd = float(hsrl.amortized_rule_action(
                    g_applied, sum_disch, n_rem, E_now))
            else:
                action_L = np.asarray(agent_L.select_action(obs_L),
                                      dtype=float).reshape(-1)
                pb_cmd = float(action_L[0])

            # ---- RBB 硬裁剪（HSRL 下 ≡ clip 到物理域）----
            pbrule, _rbb_info = self.rbb.compute_pbrule(
                pb_original=pb_cmd, ptur=ptur, pg_prev=pg_prev, E=E_now)

            action = np.concatenate([applied_H, np.array([pb_cmd])])
            action_rbb = action.copy()
            action_rbb[-1] = pbrule

            # ---- 环境一步 ----
            pw_tru2, obs2, aL_env, _pen_L, _done, _done_L, info = \
                env.step(action_rbb[:6], action_rbb[6:], pw_tru)
            pb_applied = float(np.asarray(aL_env, dtype=float).reshape(-1)[0])
            a_L_disch = hsrl.discharge_of(pb_applied)
            sum_disch += a_L_disch

            # ---- 三项指标 ----
            P_F = float(info['P_F'])
            excess = max(P_F - FLUCT_LIMIT, 0.0) if P_F > FLUCT_LIMIT + VIOL_TOL else 0.0
            if excess > 0.0:
                viol_count += 1
                viol_excess += excess
            profit_step = (float(obs2[-1]) * float(obs[0])
                           - BESS_DEG_RMB_PER_MWH * max(-float(pbrule), 0.0)) * DT
            profit += profit_step
            soc_sum += float(obs2[2])
            soc_end = float(obs2[2])
            max_pf = max(max_pf, P_F)
            if pb_applied > 0:
                charge_mwh += pb_applied * ETA_C * DT
            else:
                discharge_mwh += -pb_applied * DT / ETA_D

            step_rows.append({
                'day_idx': day_idx,
                'wind_iloc': wind_iloc,
                'step': t,
                'price': float(obs[0]),
                'wind': float(obs[1]),
                'P_tur': ptur,
                'P_G': float(obs2[-1]),
                'P_F': P_F,
                'viol': int(excess > 0.0),
                'excess_mw': excess,
                # 前 5 维实际执行的轴向因子（run11 五维恒相等 = 标量臂的直接证据）
                'a0': float(applied_H[0]), 'a1': float(applied_H[1]),
                'a2': float(applied_H[2]), 'a3': float(applied_H[3]),
                'a4': float(applied_H[4]),
                'g': g_applied,
                'need': need,
                'pb_cmd': pb_cmd,
                'pbrule': float(pbrule),
                'SOC': float(obs2[2]),
                'profit_step': profit_step,
            })
            a_sum += float(applied_H[:5].sum())
            a_zero_steps += int(float(applied_H[:5].max()) <= 0.0)

            # ---- 推进 ----
            norm_obs = self._norm_H(obs2, t + 1)
            obs = obs2
            pw_tru = pw_tru2

        n_steps = self.episode_length
        day = {
            'day_idx': day_idx,
            'wind_iloc': wind_iloc,
            'wind_mean_actual': float(wind_series.iloc[:n_steps].mean()),
            'viol_count': viol_count,
            'viol_excess_mw': viol_excess,
            'viol_excess_mwh': viol_excess * DT,
            'profit_rmb': profit,
            'soc_init': self.soc0,
            'soc_mean': soc_sum / n_steps,
            'soc_end': soc_end,
            'charge_mwh': charge_mwh,
            'discharge_mwh': discharge_mwh,
            'max_PF': max_pf,
            # 动作诊断：a_mean = 五台 × 全程的平均轴向因子；frac_a_zero = 贴 0 下界的步占比
            'a_mean': a_sum / (5.0 * n_steps),
            'frac_a_zero': a_zero_steps / float(n_steps),
            'n_steps': n_steps,
        }
        return day, step_rows


# ===========================================================================
# 自检：权重形状 + 引擎锚点
# ===========================================================================
def run_selftest(engine_kinds, models, device):
    """两件事：① 两个模型的 agent 都能按反推的形状加载；② 引擎在锚点上对得上。

    锚点（u=10 m/s、五台 a=1/3，见 aifflor/README.md §3 与 analyse_engine_u10_forms）：
        WFSim  3.4462 MW
        FLORIS(with LEVEL_COEF) 3.559146 MW
    容差 1%：FLORIS 侧 3.5591/3.4462 = +3.3%，两边各自对各自的锚点。
    """
    ok = True
    print('\n' + '=' * 88)
    print('  自检 ①：模型配置探测 + 权重加载')
    print('=' * 88)
    cfgs = {}
    for m in models:
        d = os.path.abspath(m['dir'])
        if not os.path.isdir(d):
            print('  [跳过] %s 目录不存在: %s' % (m['tag'], d))
            ok = False
            continue
        cfg = detect_model_cfg(d)
        sw = detect_scalar_wt(d)
        if sw is None:
            print('  [警告] %s 没有 train.log，scalar_wt 无法探测' % m['tag'])
        cfgs[m['tag']] = (cfg, sw)
        print('  %-18s ns_H=%d ns_L=%d hidden=%s critic_H=%s critic_L=%s '
              'phase=%d scalar_wt=%s'
              % (m['tag'], cfg['ns_H'], cfg['ns_L'], cfg['hidden'],
                 cfg['critic_H'], cfg['critic_L'], cfg['phase_mode'], sw))
        for kind in engine_kinds:
            try:
                runner = HsrlRunner(cfg, kind, device,
                                    dict(episode_length=5, soc0=SOC0_DEFAULT,
                                         scalar_wt=int(bool(sw)), engine=None),
                                    verbose=False)
                # 动作空间接线自检：--scalar_wt 开 ⇒ 前 5 维**必须**逐位相等
                probe = np.array([300.0, 10.0, SOC0_DEFAULT, 2.5])
                raw = np.asarray(runner.agent_H.select_action(runner._norm_H(probe, 0)),
                                 dtype=float).reshape(-1)
                spread = float(raw[:5].max() - raw[:5].min())
                if bool(sw):
                    good = spread < 1e-12
                    extra = '前5维跨维差=%.2e（scalar_wt 开 ⇒ 必须为 0）' % spread
                else:
                    good = True
                    extra = '前5维=%s（五维独立）' % np.round(raw[:5], 3).tolist()
                ok &= good
                print('     [%s] %-12s 权重加载/前向通过；%s'
                      % ('ok' if good else 'FAIL', kind, extra))
            except Exception as exc:                       # noqa: BLE001
                ok = False
                print('     [FAIL] %-12s %s: %s' % (kind, type(exc).__name__, exc))

    print('\n' + '=' * 88)
    print('  自检 ②：引擎锚点（u=10, a=1/3×5）')
    print('=' * 88)
    a = np.full(NB_TURBINE, 1.0 / 3.0)
    if 'aiffloris' in engine_kinds:
        from aifflor.aifflor_env import AifflorPhysics, WindBatteryAIF
        from aifflor import aifflor_turbine as AT
        e = WindBatteryAIF(MAX_E, NB_TURBINE, False, None,
                           physics=AifflorPhysics(nt=NB_TURBINE, calibrated=False,
                                                  power_scale=AT.LEVEL_COEF))
        p = float(e.PowerOutput(10.0, a))
        ref = 3.559146
        good = abs(p / ref - 1.0) <= 0.01
        ok &= good
        print('  aiffloris     P=%.6f MW  锚点 %.6f  %+0.2f%%  %s'
              % (p, ref, 100 * (p / ref - 1), 'ok' if good else 'MISMATCH'))
    if 'matlabwfsim' in engine_kinds:
        import test_five_days as T
        eng = T.ensure_engine()
        import env as envmod
        e = envmod.WindBattery(MAX_E, NB_TURBINE, False, eng)
        p = float(e.PowerOutput(10.0, a))
        ref = 3.4462
        good = abs(p / ref - 1.0) <= 0.01
        ok &= good
        print('  matlabwfsim   P=%.6f MW  锚点 %.6f  %+0.2f%%  %s'
              % (p, ref, 100 * (p / ref - 1), 'ok' if good else 'MISMATCH'))
    print('\n  自检结论: %s' % ('全部通过' if ok else '有失败项（见上）'))
    return 0 if ok else 1


# ===========================================================================
# TensorBoard
# ===========================================================================
class Tb:
    """薄封装：没有 tensorboard 时静默降级（`--no-tensorboard` 亦然）。"""

    def __init__(self, tb_root, enabled=True):
        self.writers = {}
        self.root = tb_root
        self.enabled = bool(enabled)
        if not self.enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._SW = SummaryWriter
        except Exception as exc:                           # noqa: BLE001
            print('  [警告] tensorboard 不可用(%s)，只写 CSV' % exc)
            self.enabled = False

    def _w(self, run_name):
        if not self.enabled:
            return None
        if run_name not in self.writers:
            # 注意：**追加**写（不归档、不清空）。同一引擎后续再加别的臂（如 `--fixed-a` 基线）
            # 时，新旧 tag 互不重叠、各自成曲线；只有"同一臂重跑"才会在该 tag 上多出一个点，
            # 而 CSV 侧是幂等的（`days.csv` 覆盖写 + `summary_all.csv` 按 (engine,model,day) 去重）。
            # 想彻底重来就手工删掉 `<out>/tensorboard/<engine>`。
            self.writers[run_name] = self._SW(os.path.join(self.root, run_name))
        return self.writers[run_name]

    def scalar(self, run_name, tag, value, step):
        if not self.enabled or value is None:
            return
        v = float(value)
        if not np.isfinite(v):
            return
        self._w(run_name).add_scalar(tag, v, int(step))

    def flush(self):
        for w in self.writers.values():
            w.flush()

    def close(self):
        for w in self.writers.values():
            try:
                w.close()
            except Exception:
                pass
        self.writers = {}


def _has_events(run_dir):
    """该 TB run 目录里是否已有 event 文件。"""
    try:
        return any(n.startswith('events.out.tfevents') for n in os.listdir(run_dir))
    except OSError:
        return False


def write_wind_tb(tb, days, df_wind):
    """四天**日内风速**（step = 一天内第几分钟）写入公共 run `_wind`。

    风与引擎无关 ⇒ 只写一次：若 `_wind` 里已有 event 文件（前一次调用写过），直接跳过，
    否则"两个引擎分两次命令跑"会在同一个 run 里塞进两套 step 0..1439 的同名曲线。
    """
    if tb.enabled and _has_events(os.path.join(tb.root, '_wind')):
        print('TensorBoard: `_wind` 已存在（四天日内风速已记录），跳过重写')
        return
    for i, d in enumerate(days, start=1):
        s = df_wind.iloc[:, d['iloc']].reset_index(drop=True)
        tag = 'wind/day{}_{}'.format(i, d['tag'])
        for t, u in enumerate(s.values):
            tb.scalar('_wind', tag + '/speed', float(u), t)
        tb.scalar('_wind', tag + '/mean', float(s.mean()), 0)
        tb.scalar('_wind', tag + '/std', float(s.std()), 0)
        # 天级水平对照（x = 天序号）
        tb.scalar('_wind', 'wind_mean_by_day/all', float(s.mean()), i)
    tb.flush()


# ===========================================================================
# 汇总 / 打印
# ===========================================================================
def _pivot(df, value, days):
    """行 = 引擎/模型，列 = 四天（+ 平均）。"""
    order = [d['tag'] for d in days]
    piv = df.pivot_table(index=['engine', 'model'], columns='day_tag',
                         values=value, aggfunc='first')
    for tag in order:
        if tag not in piv.columns:
            piv[tag] = np.nan
    piv = piv[order]
    piv['mean4'] = piv.mean(axis=1)
    return piv


def collect_all_days(out_dir):
    """扫描 `<out>/<engine>/<model>/days.csv`，把**历次调用**的结果合并成一张表。

    为什么不在内存里攒：两个引擎的耗时差一个量级（FLORIS 分钟级、WFSim 小时级），
    实际用法是**分开两次命令**各跑一个引擎。若汇总只取本次内存里的行，第二次运行
    就会把第一次的结果从 `summary_all.csv` 里抹掉。⇒ 汇总一律从磁盘并集重建
    （同一 (engine, model, day) 重复时取**最后写入**的那份）。
    """
    frames = []
    for eng in ENGINES:
        d = os.path.join(out_dir, eng)
        if not os.path.isdir(d):
            continue
        for model in sorted(os.listdir(d)):
            p = os.path.join(d, model, 'days.csv')
            if os.path.isfile(p):
                try:
                    frames.append(pd.read_csv(p))
                except Exception as exc:                   # noqa: BLE001
                    print('[警告] 读取历史结果失败，跳过 %s (%s)' % (p, exc))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True, sort=False)
    keys = [c for c in ('engine', 'model', 'day_tag') if c in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys, keep='last')
    return df.reset_index(drop=True)


def print_report(df, days):
    order = [d['tag'] for d in days]
    print('\n' + '=' * 100)
    print('  日内指标明细（每个 引擎 × 模型 × 天）')
    print('=' * 100)
    print('  {:<12s} {:<20s} {:<10s} {:>8s} {:>8s} {:>12s} {:>7s} {:>8s}'
          .format('引擎', '模型', '天', '违例次数', '违例MW和', '盈利(元)', 'a均值', '贴0占比'))
    print('  ' + '-' * 96)
    for _, r in df.iterrows():
        print('  {:<12s} {:<20s} {:<10s} {:>8d} {:>8.4f} {:>12.2f} {:>7.3f} {:>7.1f}%'
              .format(r['engine'], r['model'], r['day_tag'],
                      int(r['viol_count']), float(r['viol_excess_mw']),
                      float(r['profit_rmb']), float(r['a_mean']),
                      100.0 * float(r['frac_a_zero'])))

    print('\n' + '=' * 100)
    print('  跨引擎对照（行 = 引擎/模型，列 = 四天；末列 = 四天均值）')
    print('=' * 100)
    for name, value, fmt in (('日内违例次数', 'viol_count', '{:7.2f}'),
                             ('日内违例功率和 (MW)', 'viol_excess_mw', '{:7.4f}'),
                             ('日内盈利 (元)', 'profit_rmb', '{:10.2f}')):
        piv = _pivot(df, value, days)
        print('\n  【%s】' % name)
        head = '  {:<12s} {:<20s} '.format('引擎', '模型') + \
               ' '.join('{:>7s}'.format(t.split('_')[0]) for t in order) + ' {:>9s}'.format('均值')
        print(head)
        print('  ' + '-' * (34 + 8 * (len(order) + 1)))
        for (eng, model), row in piv.iterrows():
            cells = ' '.join(fmt.format(row[t]) for t in order)
            print('  {:<12s} {:<20s} ' .format(eng, model) + cells +
                  ' ' + fmt.format(row['mean4']))
    print('\n  说明: 违例 = Σ 1[P_F > 3.0 MW]（阈值与 aifflor_main 的窗口计数一致，容差 1e-4）；')
    print('        违例功率和 = Σ max(P_F − 3.0, 0) [MW]（只在越限步累加）；')
    print('        盈利 = Σ (P_G·pr − 490·max(−pb,0))·Δt [元]，Δt = 1/60 h（≈ episode_profit）。')
    print('        同一天、同一模型下 aiffloris 与 matlabwfsim 的**绝对功率电平本不相同**，')
    print('        跨引擎只能比"趋势/越限结构"，不要当成同一物理量的两次重复测量。')
    print('  ⚠ 贴 0 下界（上表"贴0占比"高）时两引擎**结构性分歧**：WFSim 按 '
          'Cp=4a(1−a)² ⇒ a=0 时出力恰为 0；')
    print('    FLORIS 侧 `AifflorPhysics.a_clip` 会把 a 抬到 `A_PHYS_LO=0.025`（否则高斯尾流 0/0 '
          '出 NaN），出力不为 0。')
    print('    例：AIFFLOR-run11 的策略输出 a≈0.00（训练日志 post-warmup 均值 0.023）⇒ '
          '它的两引擎读数差异主要来自这条口径，而非物理差异。')
    print('=' * 100)


# ===========================================================================
# 主流程
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description='四固定风速日 × 两模型 × 两引擎（aiffloris / matlabwfsim）评测',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models', nargs='*', default=None, metavar='TAG|DIR',
                   help='只测这些模型（按 tag 或目录名匹配）；默认 = 全部 %s'
                        % ', '.join(m['tag'] for m in MODELS))
    p.add_argument('--model-dir', nargs='*', default=None, metavar='TAG=DIR',
                   help='覆盖模型目录，如 AIFFLOR-run11=output/AIFFLOR-run11')
    p.add_argument('--engines', nargs='*', default=list(ENGINES), choices=list(ENGINES),
                   help='要跑的引擎（默认两个都跑，顺序 aiffloris -> matlabwfsim）')
    p.add_argument('--episode-length', type=int, default=EPISODE_LENGTH,
                   help='每个 episode 的步数（默认 1440 = 完整一天；调小仅用于冒烟）')
    p.add_argument('--soc0', type=float, default=SOC0_DEFAULT,
                   help='初始 SOC [MWh]（默认 %.2f，= 两支训练时的重置值）' % SOC0_DEFAULT)
    p.add_argument('--viol-limit', type=float, default=FLUCT_LIMIT,
                   help='波动越限阈值 MW（默认 3.0）')
    p.add_argument('--deg-cost', type=float, default=BESS_DEG_RMB_PER_MWH,
                   help='放电退化成本 元/MWh（默认 490）')
    p.add_argument('--scalar-wt', type=int, default=None, choices=[0, 1],
                   help='强制 --scalar_wt（默认 None = 从 train.log 自动探测）')
    p.add_argument('--fixed-a', type=float, default=None, metavar='A',
                   help='**不要模型**：五台轴向因子全程固定为 A（如 0.3333），'
                        '第 6 维固定为 --goal；下层走摊还规则（g=0 ⇒ 储能不动作）。'
                        '用来跑"固定动作 + 无储能"的基线臂')
    p.add_argument('--goal', type=float, default=0.0,
                   help='配合 --fixed-a：块总放电量 g 的固定值 [MW·min]（默认 0 = 储能不动作）')
    p.add_argument('--output', default=os.path.join('output', 'test_4days_engine'),
                   help='输出目录（默认 output/test_4days_engine）')
    p.add_argument('--seed', type=int, default=0, help='torch/numpy 种子（默认 0）')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--rebuild-days', action='store_true',
                   help='按 DAY_SPECS 规则重新选天（覆盖 FROZEN_DAYS，仅本次生效）')
    p.add_argument('--no-tensorboard', action='store_true', help='不写 TensorBoard')
    p.add_argument('--no-save-steps', action='store_true', help='不写逐步明细 CSV')
    p.add_argument('--selftest', action='store_true',
                   help='只做自检（权重形状 + 引擎锚点），不跑 episode')
    p.add_argument('--dry-run', action='store_true', help='只打印计划与四天清单')
    return p.parse_args()


def pick_models(args):
    """应用 --models / --model-dir；返回 [(tag, abs_dir, note)]。"""
    dirs = {m['tag']: m['dir'] for m in MODELS}
    notes = {m['tag']: m.get('note', '') for m in MODELS}
    if args.model_dir:
        for item in args.model_dir:
            if '=' not in item:
                raise SystemExit('--model-dir 需要 TAG=DIR 形式，收到: %s' % item)
            k, v = item.split('=', 1)
            dirs[k.strip()] = v.strip()
            notes.setdefault(k.strip(), '')
    tags = list(dirs.keys())
    if args.models:
        want = set()
        for w in args.models:
            hit = [t for t in tags if w == t or w.lower() in t.lower()]
            if not hit:
                raise SystemExit('--models %r 未匹配到任何模型（可选 %s）'
                                 % (w, ', '.join(tags)))
            want.update(hit)
        tags = [t for t in tags if t in want]
    out = []
    for t in tags:
        d = os.path.abspath(dirs[t])
        if not os.path.isdir(d):
            print('[警告] %s 目录不存在，跳过: %s' % (t, d))
            continue
        if not os.path.isfile(os.path.join(d, 'actor_6.pkl')):
            print('[警告] %s 缺少 actor_6.pkl，跳过: %s' % (t, d))
            continue
        out.append((t, d, notes.get(t, '')))
    if not out:
        raise SystemExit('没有任何可用模型目录')
    return out


def main():
    args = parse_args()

    global EPISODE_LENGTH, FLUCT_LIMIT, BESS_DEG_RMB_PER_MWH
    EPISODE_LENGTH = int(args.episode_length)
    FLUCT_LIMIT = float(args.viol_limit)
    BESS_DEG_RMB_PER_MWH = float(args.deg_cost)

    import torch
    if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()):
        device = torch.device('cuda:0')
        print('设备: GPU %s' % torch.cuda.get_device_name(device))
    else:
        device = torch.device('cpu')
        print('设备: CPU')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    models = [] if args.fixed_a is not None else pick_models(args)
    engines = list(args.engines)

    if args.selftest:
        try:
            rc = run_selftest(engines, [dict(tag=t, dir=d) for t, d, _ in models], device)
        finally:
            try:
                import test_five_days as T
                T.quit_engine()
            except Exception:
                pass
        return rc

    t_start = time.time()
    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. 四天（固定）----
    df_wind, means = load_wind_pool()
    days = resolve_days(df_wind, means, out_dir, rebuild=args.rebuild_days)

    print('\n' + '=' * 88)
    print('  四固定风速日（口径：日均风速；列 = %s 的 iloc[:, 1..]）' % WIND_CSV)
    print('=' * 88)
    print('  {:<10s} {:>10s} {:>8s} {:>9s} {:>9s} {:>9s} {:>12s}'
          .format('天', 'wind_iloc', '列名', '日均', '最小', '最大', '桶'))
    for i, d in enumerate(days, start=1):
        s = df_wind.iloc[:, d['iloc']]
        print('  {:<10s} {:>10d} {:>8s} {:>9.3f} {:>9.3f} {:>9.3f} {:>12s}'
              .format(d['tag'], d['iloc'], d['col'], float(s.mean()),
                      float(s.min()), float(s.max()),
                      '[%g,%g)' % (d['lo'], d['hi'])))
    print('  清单已固化到: %s' % os.path.join(out_dir, 'fixed_days.csv'))

    # ---- 2. 模型配置（自动探测）；`--fixed-a` 时只建一个无模型的常量臂 ----
    runs = []
    if args.fixed_a is not None:
        a_val, g_val = float(args.fixed_a), float(args.goal)
        runs.append(dict(
            tag='FIXED-A%g' % a_val, dir='', cfg=None,
            note='固定动作基线: a_i≡%.4f (五台), g≡%g MW·min, 下层=摊还规则(储能不动作)'
                 % (a_val, g_val)))
    else:
        for tag, d, note in models:
            cfg = detect_model_cfg(d)
            sw = args.scalar_wt if args.scalar_wt is not None else detect_scalar_wt(d)
            if sw is None:
                sw = 0
                print('[警告] %s 无 train.log，scalar_wt 按 0 处理（如该 run 是标量臂请 --scalar-wt 1）'
                      % tag)
            cfg['scalar_wt'] = int(bool(sw))
            runs.append(dict(tag=tag, dir=d, note=note, cfg=cfg))

    n_ep = len(runs) * len(engines) * len(days)
    print('\n' + '=' * 88)
    print('  计划：%d 个模型 × %d 个引擎 × %d 天 = %d 个 episode（每 episode %d 步）'
          % (len(runs), len(engines), len(days), n_ep, EPISODE_LENGTH))
    print('=' * 88)
    for r in runs:
        if r['cfg'] is None:
            print('  %-18s %s' % (r['tag'], r['note']))
        else:
            print('  %-18s %-26s scalar_wt=%d | H%s%s L%s%s'
                  % (r['tag'], r['dir'], r['cfg']['scalar_wt'],
                     r['cfg']['ns_H'], r['cfg']['critic_H'][0], r['cfg']['ns_L'],
                     r['cfg']['critic_L'][0]))
    for e in engines:
        print('  %-18s %s' % (e, ENGINE_DESC[e]))
        if e == 'matlabwfsim':
            n_wf = len(runs) * len(days)
            print('  %-18s ⚠ 慢：约 0.3~1 s/步 ⇒ 一天 7~24 min；'
                  '本计划 WFSim 部分 %d 个 episode ≈ %.1f~%.1f 小时'
                  % ('', n_wf, n_wf * 7 / 60.0, n_wf * 24 / 60.0))
    if args.dry_run:
        print('\n[dry-run] 不执行。去掉 --dry-run 即开跑。')
        return 0

    # ---- 3. 结果目录（**按引擎分开**）----
    for e in engines:
        os.makedirs(os.path.join(out_dir, e), exist_ok=True)
    tb = Tb(os.path.join(out_dir, 'tensorboard'), enabled=not args.no_tensorboard)
    if tb.enabled:
        print('\nTensorBoard: %s   (tensorboard --logdir "%s")'
              % (os.path.join(out_dir, 'tensorboard'), os.path.join(out_dir, 'tensorboard')))
        write_wind_tb(tb, days, df_wind)

    day_rows = []          # 全部 (engine, model, day) 的汇总行
    engine_obj = None
    t_run0 = time.time()
    try:
        for engine_kind in engines:
            if engine_kind == 'matlabwfsim' and engine_obj is None:
                import test_five_days as T
                engine_obj = T.ensure_engine()
            for r in runs:
                tag, cfg = r['tag'], r['cfg']
                model_out = os.path.join(out_dir, engine_kind, tag)
                os.makedirs(model_out, exist_ok=True)
                print('\n' + '-' * 88)
                print('  引擎 %s | 模型 %s  -> %s' % (engine_kind, tag, model_out))
                print('-' * 88)
                runner = HsrlRunner(cfg, engine_kind, device,
                                    dict(episode_length=EPISODE_LENGTH,
                                         soc0=float(args.soc0),
                                         scalar_wt=(cfg['scalar_wt'] if cfg else 0),
                                         fixed_a=args.fixed_a,
                                         goal=float(args.goal),
                                         engine=engine_obj),
                                    verbose=True)
                run_days, run_steps = [], []
                for i, d in enumerate(days, start=1):
                    series = df_wind.iloc[:, d['iloc']].reset_index(drop=True)
                    t0 = time.time()
                    day, rows = runner.run_day(series, i, d['iloc'])
                    day.update(dict(engine=engine_kind, model=tag, day_tag=d['tag'],
                                    note=r['note']))
                    run_days.append(day)
                    for row in rows:
                        row.update(dict(engine=engine_kind, model=tag, day_tag=d['tag']))
                    run_steps.extend(rows)
                    day_rows.append(day)
                    print('    %-10s 日均%6.2f | 违例 %4d 次 | 违例功率和 %8.4f MW | '
                          '盈利 %12.2f 元 | a均 %5.3f (贴0 %4.1f%%)  [%.0fs]'
                          % (d['tag'], float(series.iloc[:EPISODE_LENGTH].mean()),
                             day['viol_count'], day['viol_excess_mw'],
                             day['profit_rmb'], day['a_mean'],
                             100.0 * day['frac_a_zero'], time.time() - t0))

                    # ---- TB: 日内曲线（step = 分钟）+ 天级指标（step = 天序号）----
                    for row in rows:
                        tb.scalar(engine_kind,
                                  '%s/%s/P_F' % (tag, d['tag']), row['P_F'], row['step'])
                        tb.scalar(engine_kind,
                                  '%s/%s/SOC' % (tag, d['tag']), row['SOC'], row['step'])
                    for m in DAY_METRICS:
                        tb.scalar(engine_kind, '%s/%s' % (tag, m), day[m], i)
                    tb.flush()

                # ---- 逐日 / 逐步 CSV（每个引擎、每个模型各一份 ⇒ 结果分开存）----
                pd.DataFrame(run_days).to_csv(
                    os.path.join(model_out, 'days.csv'), index=False, encoding='utf-8-sig')
                if not args.no_save_steps and run_steps:
                    pd.DataFrame(run_steps).to_csv(
                        os.path.join(model_out, 'steps.csv'), index=False,
                        encoding='utf-8-sig')
                if hasattr(runner.env, 'summary'):
                    try:
                        print('    ' + runner.env.summary())
                    except Exception:
                        pass
    finally:
        tb.close()
        try:
            import test_five_days as T
            T.quit_engine()
        except Exception:
            pass

    if not day_rows:
        print('\n没有产生任何结果（都被跳过了）。')
        return 1

    # ---- 4. 汇总（长表 + 每 run + 透视）——从磁盘并集重建，跨调用累加 ----
    df_disk = collect_all_days(out_dir)
    df = df_disk if df_disk is not None else pd.DataFrame(day_rows)
    if df_disk is not None:
        have = sorted(df['engine'].unique().tolist())
        print('\n汇总来源: 磁盘并集（已有引擎: %s）' % ', '.join(have))
    cols = ['engine', 'model', 'day_tag', 'day_idx', 'wind_iloc', 'wind_mean_actual',
            'viol_count', 'viol_excess_mw', 'viol_excess_mwh', 'profit_rmb',
            'soc_init', 'soc_mean', 'soc_end', 'discharge_mwh', 'charge_mwh',
            'max_PF', 'a_mean', 'frac_a_zero', 'n_steps', 'note']
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(os.path.join(out_dir, 'summary_all.csv'), index=False, encoding='utf-8-sig')

    agg = (df.groupby(['engine', 'model'], as_index=False)
           .agg(days=('day_idx', 'count'),
                viol_count_mean=('viol_count', 'mean'),
                viol_excess_mw_mean=('viol_excess_mw', 'mean'),
                profit_rmb_mean=('profit_rmb', 'mean'),
                profit_rmb_sum=('profit_rmb', 'sum')))
    agg.to_csv(os.path.join(out_dir, 'summary_by_run.csv'), index=False,
               encoding='utf-8-sig')
    for value, fname in (('viol_count', 'pivot_viol_count.csv'),
                         ('viol_excess_mw', 'pivot_viol_excess_mw.csv'),
                         ('profit_rmb', 'pivot_profit_rmb.csv')):
        _pivot(df, value, days).reset_index().to_csv(
            os.path.join(out_dir, fname), index=False, encoding='utf-8-sig')

    print_report(df, days)

    print('\n输出文件:')
    print('  四天清单        : %s' % os.path.join(out_dir, 'fixed_days.csv'))
    print('  长表(全指标)    : %s' % os.path.join(out_dir, 'summary_all.csv'))
    print('  每run均值       : %s' % os.path.join(out_dir, 'summary_by_run.csv'))
    for e in engines:
        print('  %-15s : %s' % (e, os.path.join(out_dir, e)))
    if tb.enabled:
        print('  TensorBoard     : %s' % os.path.join(out_dir, 'tensorboard'))
        print('                    tensorboard --logdir "%s"'
              % os.path.join(out_dir, 'tensorboard'))
    print('  总耗时          : %.1f min' % ((time.time() - t_start) / 60.0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
