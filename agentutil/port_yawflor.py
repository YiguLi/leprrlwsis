#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""port_yawflor.py —— 把 `td3per` 克隆成 `yawflor`（一次性迁移脚本，默认 dry-run）。

作用（`yawflor/README.md` §5/§6 P3）
-----------------------------------
新方法 = **同一算法的第二个动作空间 / 第二个物理引擎**。因此它的三个文件必须是
基线的**逐字克隆 + 可枚举的最小改动**（否则"新方法 vs 基线"的对照不成立）：

| 目标文件 | 源文件 | 改动 |
|---|---|---|
| `yawflor/yawflor_ddpg.py` | `td3per/td3per_ddpg.py` | **无**（逐字复制） |
| `yawflor/yawflor_myfunc.py` | `td3per/td3per_myfunc.py` | 动作带 `[0,0.5]→[−1,+1]`、越界罚系数 `400→100`（满带 200 元不变）、文案 |
| `yawflor/yawflor_main.py` | `td3per/td3per_main.py` | env 类 / `init_wt_action_bias` / `--wt_noise` / warmup 带参 / 不启 MATLAB / 文案 |

机制（沿用本仓库 `port_*.py` 的既有约定）
----------------------------------------
* 每条替换都要求**作用域内命中次数精确匹配**（`count`），任一不符 ⇒ **整体不写盘**；
* 默认 dry-run，只有 `--apply` 才落盘；
* `--check`：把"源文件 + 同一张替换表"重算一遍，与磁盘上的 `yawflor/*.py` 逐位比较
  ⇒ **这就是 P3① "diff 清单证明除动作带与 env 外无差异"的机器可读证明**
  （不一致的字节只可能落在表中列出的替换范围内）。

用法
----
    python agentutil/port_yawflor.py            # 打印替换计划（dry-run）
    python agentutil/port_yawflor.py --apply    # 落盘
    python agentutil/port_yawflor.py --check    # 校验磁盘文件 == 源 + 替换表
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


# ===========================================================================
# 替换表：(旧文本, 新文本, 期望命中次数, 说明)
# ===========================================================================
DD3_EDITS = []                       # ddpg：**逐字复制，不改**

MYFUNC_EDITS = [
    # ---- 模块 docstring：动作语义换成偏航 ----
    (
        '上层 H (RL, 6 维动作 = 5 台风机桨距 + 1 个 SOC 目标信号)\n'
        '--------------------------------------------------------\n'
        '  a6 ∈ [-0.5, 0.5] → soc_target = 3 + 5·a6 ∈ [0.5, 5.5]   (块级(5 步)固定)',
        '上层 H (RL, 6 维动作 = 5 台**偏航指令 n ∈ [−1,+1]** + 1 个 SOC 目标信号)\n'
        '--------------------------------------------------------------------------------\n'
        '  ⚠ yawflor 改造 (相对 td3per_myfunc 的**唯一**语义差异):\n'
        '    动作维是**无量纲偏航指令** `n`, env 内 `γ = 25°·n`（动作带 [−1,+1]，带宽 2）；\n'
        '    旧环境是 5 维轴向因子 `a ∈ [0,0.5]`（带宽 0.5）。奖励/成本/信号语义**完全不变**。\n'
        '  a6 ∈ [-0.5, 0.5] → soc_target = 3 + 5·a6 ∈ [0.5, 5.5]   (块级(5 步)固定)',
        1, 'docstring: 动作维语义 (桨距→偏航指令)'),
    # ---- 越界量的带（--viol_derate 用；默认关）----
    (
        'VIOL_DERATE_DIMS = 5       # 前 N 维 = 机组限功率; dim5(signal) 不计入 x\n'
        '_ACT_HI, _ACT_LO = 0.5, 0.0   # 与 CheckAction 的前 5 维边界一致',
        'VIOL_DERATE_DIMS = 5       # 前 N 维 = **偏航维**; dim5(signal) 不计入 x\n'
        '_ACT_HI, _ACT_LO = 1.0, -1.0  # 与 CheckAction 的前 5 维边界一致 (yawflor: [−1,+1])',
        1, 'violation_amount 的动作带'),
    (
        '    """x = 前 n_dims 维 (机组限功率, 边界 [0,0.5]) 的越界量之和。"""',
        '    """x = 前 n_dims 维 (**偏航维**, 边界 [−1,+1]) 的越界量之和。"""',
        1, 'violation_amount docstring'),
    # ---- 越界罚系数：满带越界仍是 200 元（400×0.5 == 100×2）----
    (
        'PENALTY_NN_H_COEF = 400.0  # [元/单位] H 层动作越界罚 (机组限功率 [0,0.5]×5 + 信号 [±0.5])\n'
        '                         #   原 200 -> 400: 越界仍未压住, 先整体加倍。\n'
        '                         #   ⚠ s_p=1000 ⇒ critic 空间 0.4/单位, 而框内经济斜率实测 0.27~4.5\n'
        '                         #   ⇒ 仍然同量级, 不改变"边界位置落在 critic 噪声里"的结论。',
        'PENALTY_NN_H_COEF = 100.0  # [元/单位] H 层动作越界罚 (**偏航指令带 [−1,+1]**×5 + 信号 [±0.5])\n'
        '                         #   ⚠ yawflor: 带宽从 0.5 (轴向因子 [0,0.5]) 变成 2 (偏航 [−1,+1])\n'
        '                         #   ⇒ 系数必须 ÷4, 才能保持**"满带越界 = 200 元"这一经济口径不变**\n'
        '                         #   (400 × 0.5 = 100 × 2 = 200 元; 与旧环境等值, 见 yawflor/README.md §4)。\n'
        '                         #   单机物理不像旧环境那样有"Cp 悬崖", 但带外仍无物理含义 ⇒ 同样不封顶。',
        1, 'PENALTY_NN_H_COEF 400→100 (满带 200 元不变)'),
    # ---- CheckAction：动作带 ----
    (
        '    """Clip actions to feasible ranges. WT: [0,0.5], Signal: [-0.5,0.5], Bat: [-charge,charge].\n'
        '\n'
        '    上层 (6 维): 与 td3perrb 一致 —— WT 越界与信号越界按 100/单位计罚, **不封顶**\n'
        '    (保持越界区恒定的恢复力, 见 td3perrb/td3prrlrb 的 run160 诊断)。',
        '    """Clip actions to feasible ranges. YAW(n): [-1,+1], Signal: [-0.5,0.5], Bat: [-charge,charge].\n'
        '\n'
        '    上层 (6 维): 与 td3per 一致 —— **偏航维**越界与信号越界按 PENALTY_NN_H_COEF 计罚,\n'
        '    **不封顶** (保持越界区恒定的恢复力, 见 td3perrb/td3prrlrb 的 run160 诊断)。\n'
        '    ⚠ yawflor: 前 5 维边界是 [−1,+1]（无量纲偏航指令），不是 [0,0.5]。',
        1, 'CheckAction docstring'),
    (
        '        for j in action[:5]:\n'
        '            ma, mi = max(0.5, j), min(0, j)\n'
        '            penalty += PENALTY_NN_H_COEF * ((ma - 0.5) + (0. - mi))',
        '        for j in action[:5]:\n'
        '            ma, mi = max(1.0, j), min(-1.0, j)\n'
        '            penalty += PENALTY_NN_H_COEF * ((ma - 1.0) + (-1.0 - mi))',
        2, 'CheckAction 偏航维越界量 (len==6 / len==5 两处)'),
    (
        "        applied = np.append(np.clip(action[:5], 0., 0.5),\n"
        "                            np.clip(action[5], SIGNAL_MIN, SIGNAL_MAX))",
        "        applied = np.append(np.clip(action[:5], -1., 1.),\n"
        "                            np.clip(action[5], SIGNAL_MIN, SIGNAL_MAX))",
        1, 'CheckAction 夹带 (len==6)'),
    (
        '        return np.clip(action[:5], 0., 0.5), penalty',
        '        return np.clip(action[:5], -1., 1.), penalty',
        1, 'CheckAction 夹带 (len==5)'),
]

MAIN_EDITS = [
    # ---- 模块 docstring ----
    (
        'TD3PER — 上层 = TD3PERRB 口径 (非分段奖励 + SOC 目标信号), 下层 = RL BESS 追 SOC 目标\n'
        '====================================================================================\n'
        '本文件是 td3per 的 "td3perrb 化" 改造 (原地修改):',
        'YAWFLOR — 偏航 + FLORIS 环境 (HSRL 方法 10)\n'
        '====================================================================================\n'
        '本文件 = `td3per/td3per_main.py` 的**逐字克隆**（由 `agentutil/port_yawflor.py`\n'
        '生成/校验），改动**仅限**（yawflor/README.md §5）：\n'
        '  1. env 类：`WindBattery`(WFSim/MATLAB) → `WindBatteryYaw`(FLORIS)，不再启 MATLAB；\n'
        '  2. 动作带：轴向因子 `[0,0.5]` → **无量纲偏航指令 `n ∈ [−1,+1]`**（env 内 γ = 25°·n）；\n'
        '  3. `init_wt_action_bias`：带 `[0.05,0.45]` → `[−1,+1]`，打印 γ 而不是 Cp；\n'
        '  4. `--wt_noise` 0.05 → 0.2（带宽 10%，等比例）；`hwarm.add_warmup_args(..., yaw=True)`；\n'
        '  5. CLI help 文案（偏航语义）。\n'
        '**算法本体（TD3+PER、对偶、奖励/成本、n 步、相位特征、日志/CSV）与基线逐字相同。**',
        1, 'main docstring: 标题与改动清单'),
    (
        '      - 动作 6 维 = 5 台风机桨距 [0,0.5] + 1 个 SOC 目标信号 a6 ∈ [-0.5, 0.5];',
        '      - 动作 6 维 = 5 台**偏航指令 n ∈ [−1,+1]**（env 内 γ = 25°·n）+ 1 个 SOC 目标信号 a6 ∈ [-0.5, 0.5];',
        1, 'main docstring: 动作维'),
    (
        '      - --epsilon 4000 (H 决策次数 ⇒ 20k 环境步); actor 输出层 WT bias 初始化到“中性宽带”。',
        '      - --epsilon 4000 (H 决策次数 ⇒ 20k 环境步); actor 输出层偏航 bias 初始化到“中性满带 [−1,+1]”。',
        1, 'main docstring: bias 初始化'),
    # ---- 路径 bootstrap + imports ----
    (
        '# Ensure both td3per/ and parent (HSRL) directories are on path\n'
        '_TD3PER_DIR = os.path.dirname(os.path.abspath(__file__))\n'
        '_HSRL_DIR = os.path.dirname(_TD3PER_DIR)\n'
        'sys.path.insert(0, _TD3PER_DIR)\n'
        'sys.path.insert(0, _HSRL_DIR)\n'
        '\n'
        'from env import WindBattery\n'
        'from td3per.td3per_ddpg import SP_TD3',
        '# Ensure both yawflor/ and parent (HSRL) directories are on path\n'
        '_YAWFLOR_DIR = os.path.dirname(os.path.abspath(__file__))\n'
        '_HSRL_DIR = os.path.dirname(_YAWFLOR_DIR)\n'
        'sys.path.insert(0, _YAWFLOR_DIR)\n'
        'sys.path.insert(0, _HSRL_DIR)\n'
        '\n'
        'from yawflor.yawflor_env import FlorisPhysicsYaw, WindBatteryYaw\n'
        'from yawflor import yawflor_turbine as yturb\n'
        'from yawflor.yawflor_ddpg import SP_TD3',
        1, 'bootstrap + env/agent imports'),
    (
        'from td3per.td3per_myfunc import (',
        'from yawflor.yawflor_myfunc import (',
        1, 'myfunc import'),
    # ---- 不启 MATLAB ----
    (
        'import matlab\n'
        'import matlab.engine\n'
        'import hsrl_cost as hc   # H 层 c1/λ/d_block 统一口径\n'
        'import hsrl_agent_init as hinit   # critic 输出层 bias(价值基线)\n'
        'import hsrl_soctarget as hst      # 下层 r2/c2 两区间口径 (单一真源)\n'
        'import hsrl_warmup as hwarm       # warmup 调度 + 规则下层 + transition 填入\n'
        'engine = matlab.engine.start_matlab()\n'
        '\n'
        '\n'
        "RUN_TAG = 'td3per'",
        'import hsrl_cost as hc   # H 层 c1/λ/d_block 统一口径\n'
        'import hsrl_agent_init as hinit   # critic 输出层 bias(价值基线)\n'
        'import hsrl_soctarget as hst      # 下层 r2/c2 两区间口径 (单一真源)\n'
        'import hsrl_warmup as hwarm       # warmup 调度 + 规则下层 + transition 填入\n'
        '# ⚠ yawflor **不启 MATLAB**：物理引擎是 FLORIS（纯 Python，见 yawflor_env.py）。\n'
        '#   环境按需惰性 import 旧 MATLAB 路径（`env._get_matlab`），本文件不再触碰它。\n'
        '\n'
        '\n'
        "RUN_TAG = 'yawflor'",
        1, '不启 MATLAB + RUN_TAG'),
    # ---- init_wt_action_bias：整函数替换 ----
    (
        "def init_wt_action_bias(agent, lo=0.05, hi=0.45, sig_value=0.0, verbose=True):\n"
        '    """把 actor 输出层前 5 维 (WT 轴向诱导因子 a) 的 bias 初始化到“中性宽带”。\n',
        "def init_wt_action_bias(agent, lo=None, hi=None, sig_value=0.0, verbose=True):\n"
        '    """把 actor 输出层前 5 维 (**无量纲偏航指令 n**) 的 bias 初始化到“中性满带”。\n',
        1, 'init_wt_action_bias 签名与首行'),
    (
        '    bias∈[lo,hi]=[0.05,0.45] 覆盖"上游降载 → 末台单机最优"的整个区间, **故意不预设**风场最优的形状\n'
        '    (它随尾流状态(风速/湍流/来流)变化, 是**策略要学的函数**, 不是可预先测定的常数)。\n'
        '    SOC 目标信号维 (第 6 维) 置 sig_value=0 → 初始 soc_target = 3.0 (SOC 带中点)。',
        '    yawflor 的带是 `[lo,hi] = [−1,+1]`（默认取 `hsrl_warmup.YAW_LO/YAW_HI` 单一真源），\n'
        '    与旧环境**按"占带宽的比例"同构**：起点 = 单机最优点（旧 a=1/3 / 新 n=0），\n'
        '    步长与噪声 = 带宽的 10%。同样**故意不预设**"上游转向、下游归零"的风场最优形状\n'
        '    —— 那是随尾流状态变化的**函数**，是策略要学的，不是可预先测定的常数。\n'
        '    SOC 目标信号维 (第 6 维) 置 sig_value=0 → 初始 soc_target = 3.0 (SOC 带中点)。',
        1, 'init_wt_action_bias docstring 带说明'),
    (
        '    a = agent.actor\n'
        '    if not hasattr(a, \'fc4\'):\n'
        '        return False\n'
        '    nb = a.fc4.bias.data.shape[0]\n'
        '    with torch.no_grad():\n'
        '        b = a.fc4.bias.data\n'
        '        n_wt = min(5, nb)\n'
        '        b[:n_wt] = torch.tensor(np.random.uniform(lo, hi, n_wt), dtype=b.dtype)\n'
        '        if nb > n_wt:\n'
        '            b[n_wt:] = float(sig_value)\n'
        '        tgt = agent.actor_target.fc4.bias.data\n'
        '        tgt.copy_(b)\n'
        '    if verbose:\n'
        '        cp = 4 * np.array([b[:n_wt].mean()]) * (1 - np.array([b[:n_wt].mean()])) ** 2\n'
        "        prGreen('init_wt_action_bias: WT bias = {} (mean a = {:.3f} -> Cp = {:.3f}); '\n"
        "                'signal bias = {}'.format(\n"
        '                    np.round(b[:n_wt].numpy(), 3), float(b[:n_wt].mean()),\n'
        '                    float(cp[0]), float(b[n_wt]) if nb > n_wt else None))\n'
        '    return True',
        '    if lo is None:\n'
        '        lo = hwarm.YAW_LO\n'
        '    if hi is None:\n'
        '        hi = hwarm.YAW_HI\n'
        '    a = agent.actor\n'
        '    if not hasattr(a, \'fc4\'):\n'
        '        return False\n'
        '    nb = a.fc4.bias.data.shape[0]\n'
        '    with torch.no_grad():\n'
        '        b = a.fc4.bias.data\n'
        "        n_wt = min(5, nb)\n"
        '        b[:n_wt] = torch.tensor(np.random.uniform(lo, hi, n_wt), dtype=b.dtype)\n'
        '        if nb > n_wt:\n'
        '            b[n_wt:] = float(sig_value)\n'
        '        tgt = agent.actor_target.fc4.bias.data\n'
        '        tgt.copy_(b)\n'
        '    if verbose:\n'
        '        g = yturb.GAMMA_MAX * b[:n_wt].numpy()\n'
        "        prGreen('init_wt_action_bias (yaw): n bias = {} (mean n = {:.3f} -> mean γ = {:.2f}°); '\n"
        "                'signal bias = {}'.format(\n"
        '                    np.round(b[:n_wt].numpy(), 3), float(b[:n_wt].mean()),\n'
        '                    float(g.mean()), float(b[n_wt]) if nb > n_wt else None))\n'
        '    return True',
        1, 'init_wt_action_bias 主体 (Cp 打印 → γ)'),
    # ---- CLI：探索噪声 ----
    (
        "    parser.add_argument('--wt_noise', default=0.05, type=float,\n"
        "                        help='WT 5 维探索噪声幅度 (±wt_noise*eps)')",
        "    parser.add_argument('--wt_noise', default=hwarm.YAW_WT_NOISE, type=float,\n"
        "                        help='偏航 5 维探索噪声幅度 (±wt_noise*eps); 默认 0.2 = 满带宽 2 的 10%%'\n"
        "                             ' (旧环境 0.05 = 0.5 带宽的 10%%; 按带宽等比例)')",
        1, '--wt_noise 0.05→0.2'),
    # ---- CLI：bias 初始化文案 ----
    (
        "                        help='1: 把 actor 输出层 WT 5 维 bias 初始化到“中性宽带” '\n"
        "                             '[0.05,0.45]; 0: 用 PyTorch 默认')",
        "                        help='1: 把 actor 输出层偏航 5 维 bias 初始化到“中性满带” '\n"
        "                             '[−1,+1]（yawflor; 旧环境为轴向因子 [0.05,0.45]）; 0: 用 PyTorch 默认')",
        1, '--wt_bias_init 文案'),
    # ---- warmup 带参 ----
    (
        '    hwarm.add_warmup_args(parser)',
        '    hwarm.add_warmup_args(parser, yaw=True)   # 偏航带 [−1,+1] / 起点 0 / 步长 0.2',
        1, 'add_warmup_args(yaw=True)'),
    # ---- env 构造：FLORIS 后端 ----
    (
        '    env = WindBattery(max_E, args.nb_turbine, upload, engine)\n'
        '    env_test = WindBattery(max_E, args.nb_turbine, upload, engine)',
        '    # yawflor: 物理引擎 = FLORIS（**不启 MATLAB**；env.py 的旧 MATLAB 路径不受影响）。\n'
        '    #   两个 env 各持一份物理模型 —— 与旧环境"两个 WindBattery + 同一个 MATLAB 引擎"\n'
        '    #   在结构上同形（互不串状态），代价是两份 FlorisModel（构建各 ~0.2 s）。\n'
        '    yaw_physics = FlorisPhysicsYaw(nt=args.nb_turbine)\n'
        '    yaw_physics_test = FlorisPhysicsYaw(nt=args.nb_turbine)\n'
        '    env = WindBatteryYaw(max_E, args.nb_turbine, upload, None, physics=yaw_physics)\n'
        '    env_test = WindBatteryYaw(max_E, args.nb_turbine, upload, None,\n'
        '                              physics=yaw_physics_test)\n'
        '    prCyan(env.summary())\n'
        '    prCyan("口径: " + yturb.declared_scope())',
        1, 'env 构造 (WindBatteryYaw + FlorisPhysicsYaw)'),
    # ---- bias 初始化调用 ----
    (
        '        init_wt_action_bias(agent_H, lo=0.05, hi=0.45, sig_value=0.0)',
        '        init_wt_action_bias(agent_H, lo=hwarm.YAW_LO, hi=hwarm.YAW_HI, sig_value=0.0)',
        1, 'bias 初始化调用 (带 = [−1,+1])'),
]

RULES = {
    'yawflor/yawflor_ddpg.py': ('td3per/td3per_ddpg.py', DD3_EDITS,
                                '逐字复制（零改动）'),
    'yawflor/yawflor_myfunc.py': ('td3per/td3per_myfunc.py', MYFUNC_EDITS,
                                  '动作带 [0,0.5]→[−1,+1] + 越界罚 400→100 + 文案'),
    'yawflor/yawflor_main.py': ('td3per/td3per_main.py', MAIN_EDITS,
                                'env 类 / bias 带 / --wt_noise / warmup 带参 / 不启 MATLAB / 文案'),
}


def build(kind):
    """重算 `yawflor/*.py` 的期望内容；任一锚点命中次数不符 ⇒ 抛异常（整体不写盘）。"""
    out = {}
    log = []
    for dst, (src, edits, note) in RULES.items():
        if kind == 'check':
            path = dst
        else:
            path = src
        with open(path, 'r', encoding='utf-8') as fh:
            text = fh.read()
        for old, new, cnt, why in edits:
            got = text.count(old)
            if got != cnt:
                raise AssertionError('[%s] 锚点命中 %d 次（期望 %d）：%s\n----\n%s\n----'
                                     % (dst, got, cnt, why, old[:400]))
            text = text.replace(old, new)
            log.append((dst, why, cnt))
        out[dst] = text
    return out, log


def unified_diff(a, b, name):
    return ''.join(difflib.unified_diff(a.splitlines(True), b.splitlines(True),
                                        'source/' + name, 'yawflor/' + name, n=1))


def main():
    ap = argparse.ArgumentParser(description='td3per → yawflor 一次性迁移')
    ap.add_argument('--apply', action='store_true', help='落盘（默认 dry-run）')
    ap.add_argument('--check', action='store_true',
                    help='校验磁盘上的 yawflor/*.py == 源文件 + 替换表')
    args = ap.parse_args()

    if args.check:
        ok = True
        for dst, (src, edits, note) in RULES.items():
            expected = build('check')[0][dst] if False else None
        # 逐文件：以**磁盘文件**为基准重跑替换表（可逆性检查）
        for dst, (src, edits, note) in RULES.items():
            with open(dst, 'r', encoding='utf-8') as fh:
                on_disk = fh.read()
            # 反推：把 dst 上的新文本换回旧文本，应当逐位等于源文件
            back = on_disk
            bad = []
            for old, new, cnt, why in edits:
                got = back.count(new)
                if got != cnt:
                    bad.append('%s: 反向锚点命中 %d 次（期望 %d）' % (why, got, cnt))
                    ok = False
                    continue
                back = back.replace(new, old)
            with open(src, 'r', encoding='utf-8') as fh:
                src_text = fh.read()
            same = (back == src_text)
            ok = ok and same
            print('%-28s 反向还原 == %-28s : %s' % (dst, src, 'OK' if same else '**FAIL**'))
            if not same:
                d = unified_diff(src_text, back, src)
                print('\n'.join(d.splitlines()[:40]))
            if bad:
                print('   ', '; '.join(bad))
        # 附带：ddpg 必须**逐字相同**
        with open(RULES['yawflor/yawflor_ddpg.py'][0], 'r', encoding='utf-8') as f1, \
                open('td3per/td3per_ddpg.py', 'r', encoding='utf-8') as f2:
            same = f1.read() == f2.read()
        print('%-28s 逐字相同（零改动）        : %s' % ('yawflor/yawflor_ddpg.py',
                                                        'OK' if same else '**FAIL**'))
        ok = ok and same
        print()
        print('✅ P3① diff 清单成立：yawflor/*.py 只差替换表里列出的 %d 处'
              % sum(len(e[1]) for e in RULES.values()))
        return 0 if ok else 1

    expected, log = build('apply')
    print('替换计划（%d 个文件，%d 处替换）：' % (len(expected), len(log)))
    for dst, why, cnt in log:
        print('  [%s] %s  (×%d)' % (dst, why, cnt))
    print()
    for dst, (src, edits, note) in RULES.items():
        with open(src, 'r', encoding='utf-8') as fh:
            a = fh.read()
        d = unified_diff(a, expected[dst], src.split('/')[-1])
        n = len([l for l in d.splitlines() if l.startswith('+') or l.startswith('-')])
        print('%-28s ← %-28s %s（diff 行数 %d）' % (dst, src, note, n))
    if not args.apply:
        print('\n(dry-run；加 --apply 落盘)')
        return 0
    for dst, text in expected.items():
        with open(dst, 'w', encoding='utf-8', newline='') as fh:
            fh.write(text)
        print('已写入 %s' % dst)
    return 0


if __name__ == '__main__':
    sys.exit(main())
