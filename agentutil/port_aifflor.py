# -*- coding: utf-8 -*-
"""`td3per` → `aifflor` 的一次性迁移（替换表 + `--check` 可逆性校验）。

`aifflor` 是什么
----------------
**`td3per` 把风场仿真器从 WFSim 换成 FLORIS，其余完全一致。**

    aifflor/aifflor_main.py   = td3per/td3per_main.py   + 下面的 MAIN_EDITS
    aifflor/aifflor_myfunc.py = td3per/td3per_myfunc.py + **零改动（逐字复制）**

与 `yawflor` 的关键区别（这是本包能更小改动的原因）
---------------------------------------------------
`yawflor` 把动作带从轴向因子 `[0, 0.5]` 换成了无量纲偏航 `[−1, +1]`，
所以它**必须**改 `myfunc`（`CheckAction`、`PENALTY_NN_H_COEF` 400→100、
`violation_amount` 边界等都要跟着带宽变）。

`aifflor` 的**动作语义与 `td3per` 完全相同**（同为轴向因子 `a ∈ [0,0.5]`，
航向恒为 0）—— 只换引擎 ⇒ **`myfunc` 逐字复用，一处不改**。
这一点由 `aifflor_turbine.py` 顶部的 I/O 映射表逐项论证。

新增（**不是克隆**）的两个模块
------------------------------
* `aifflor_turbine.py` —— 口径/I-O 映射表/模型构造/向量化求值（单一真源）
* `aifflor_env.py`     —— `AifflorPhysics`（非有限值守卫 + 预取）+ `WindBatteryAIF`

用法
----
    python agentutil/port_aifflor.py            # dry-run，打印 diff
    python agentutil/port_aifflor.py --apply    # 落盘
    python agentutil/port_aifflor.py --check    # 校验磁盘文件 == 源 + 替换表（双向）
"""
import argparse
import difflib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

# ===========================================================================
# 替换表（main）
# ===========================================================================
MAIN_EDITS = [
    # ---- 包 docstring：改动清单 ----
    (
        'TD3PER — 上层 = TD3PERRB 口径 (非分段奖励 + SOC 目标信号), 下层 = RL BESS 追 SOC 目标',
        'AIFFLOR — **td3per 的 FLORIS 引擎版**：上层 TD3PERRB 口径, 下层 RL BESS 追 SOC 目标',
        1, 'main docstring: 标题'),
    (
        '# Ensure both td3per/ and parent (HSRL) directories are on path\n'
        '_TD3PER_DIR = os.path.dirname(os.path.abspath(__file__))\n'
        '_HSRL_DIR = os.path.dirname(_TD3PER_DIR)\n'
        'sys.path.insert(0, _TD3PER_DIR)\n'
        'sys.path.insert(0, _HSRL_DIR)\n'
        '\n'
        'from env import WindBattery\n'
        'from td3per.td3per_ddpg import SP_TD3',
        '# Ensure both aifflor/ and parent (HSRL) directories are on path\n'
        '_AIFFLOR_DIR = os.path.dirname(os.path.abspath(__file__))\n'
        '_HSRL_DIR = os.path.dirname(_AIFFLOR_DIR)\n'
        'sys.path.insert(0, _AIFFLOR_DIR)\n'
        'sys.path.insert(0, _HSRL_DIR)\n'
        '\n'
        '# aifflor 与 td3per 的**唯一**区别：风场仿真器 WFSim(MATLAB) → FLORIS(纯 Python)\n'
        'from aifflor.aifflor_env import AifflorPhysics, WindBatteryAIF\n'
        'from aifflor import aifflor_turbine as aturb\n'
        'from aifflor.aifflor_ddpg import SP_TD3',
        1, 'bootstrap + env/agent imports'),
    # ---- myfunc 仍来自 td3per（逐字复用）：**不改 import** ----
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
        '# ⚠ aifflor **不启 MATLAB**：物理引擎是 FLORIS（纯 Python，见 aifflor_env.py）。\n'
        '#   `env.py` 按需惰性 import 旧 MATLAB 路径（`env._get_matlab`），本文件不再触碰它。\n'
        '\n'
        '\n'
        "RUN_TAG = 'aifflor'",
        1, '不启 MATLAB + RUN_TAG'),
    # ---- env 构造：FLORIS 后端 ----
    (
        '    env = WindBattery(max_E, args.nb_turbine, upload, engine)\n'
        '    env_test = WindBattery(max_E, args.nb_turbine, upload, engine)',
        '    # aifflor: 物理引擎 = FLORIS（**不启 MATLAB**；env.py 的旧 MATLAB 路径不受影响）。\n'
        '    #   动作语义与 td3per **逐字相同**（轴向因子 a ∈ [0,0.5]，航向恒 0），\n'
        '    #   所以 `td3per_myfunc` 的 CheckAction / 越界罚 / 动作带全部原样可用。\n'
        '    #   两个 env 各持一份物理模型 —— 与旧环境"两个 WindBattery + 同一个 MATLAB 引擎"\n'
        '    #   在结构上同形（互不串状态），代价是两份 FlorisModel（构建各 ~0.2 s）。\n'
        '    #   尾流参数 = `agentutil/floris_calibrated_defaults.py` 的**标定值**\n'
        '    #   （对 WFSim 的验证集相对误差 66.9% → 1.1%）。\n'
        "    # aifflor 的物理配置（**由 `agentutil/port_aifflor.py` 的替换表决定**）：\n"
        "    #   `calibrated=False` = **FLORIS 随包默认尾流参数**（可审计：不改任何参数）；\n"
        "    #   `power_scale=AT.LEVEL_COEF` = 出口电平系数 1/1.6696 ≈ 0.59895，\n"
        "    #     把 FLORIS 的电场出力整体抬到 WFSim 电平。\n"
        "    #   依据（实测，`agentutil/analyse_level_coef_modes.py`）：\n"
        "    #     · 全域 350 点（3–20 m/s，随机动作）原始 MAPE 68.476% → **7.189%**；\n"
        "    #     · 重标定后 MAPE 66.962% → **0.001%**（即电平偏差被**完全**吸收）；\n"
        "    #     · 波动：限幅后波动均值相对 WFSim +22.50%（标定参数为 −28.11%）；\n"
        "    #     · 锚点 u=10, a=1/3：由 5.942350 变为 3.559146 MW"
        "（WFSim 参考 3.4462，+3.28%）。\n"
        "    #   ⇒ 物理用原厂默认、只加一个显式系数，功率与波动都对齐且更易审计。\n"
        "    aif_physics = AifflorPhysics(nt=args.nb_turbine, calibrated=False,\n"
        "                                 power_scale=aturb.LEVEL_COEF)\n"
        "    aif_physics_test = AifflorPhysics(nt=args.nb_turbine, calibrated=False,\n"
        "                                      power_scale=aturb.LEVEL_COEF)\n"
        '    env = WindBatteryAIF(max_E, args.nb_turbine, upload, None, physics=aif_physics)\n'
        '    env_test = WindBatteryAIF(max_E, args.nb_turbine, upload, None,\n'
        '                              physics=aif_physics_test)\n'
        '    prCyan("口径: " + aturb.declared_scope())',
        1, 'env 构造 (WindBatteryAIF + AifflorPhysics)'),
    # ---- 训练收尾：打印物理调用统计 ----
    (
        '              use_csv=use_csv)\n'
        '        if upload:\n'
        '            writer.close()',
        '              use_csv=use_csv)\n'
        '        if upload:\n'
        '            writer.close()\n'
        "        prCyan('[aifflor] 训练结束 —— 物理调用统计:')\n"
        '        prCyan(env.summary())\n'
        '        prCyan(env_test.summary())',
        1, '训练收尾打印物理调用统计'),
    # ---- 输出目录名 / TensorBoard run 名：TD3PER → AIFFLOR ----
    #   ⚠ 为什么必须改：TB 的 **scalar tag** 全是克隆来的通用名（`r1`/`c1`/`P_F`/`test9/...`），
    #   `aifflor` 与 `td3per` 在 tag 上**完全同名**（这正是跨 run 对齐比较所要的）。
    #   ⇒ 方法身份只能靠 **TensorBoard 的 run 名**（= 日志目录名
    #   `get_output_folder(output, args.env)`）与 **CSV 文件名**（`RUN_TAG='aifflor'`）。
    #   若沿用 `--env` 默认 `'TD3PER'`，不传参启动会落进 `output/TD3PER-runN`，
    #   TB 左侧那一栏就叫 TD3PER ⇒ 与真 td3per 无法分辨。
    #   大小写取 `AIFFLOR` 与 `yawflor` 的 `YAWFLOR` 一致。
    #   ⚠ 下游分析脚本若按 `output/TD3PER-*` 匹配，需要显式传目录名。
    (
        "    parser.add_argument('--env',\n"
        "                        default='TD3PER',\n"
        "                        type=str,\n"
        "                        help='open-ai gym environment')",
        "    # aifflor: 输出目录 / **TensorBoard run 名** 前缀改为 'AIFFLOR'\n"
        "    #   （产物 output/AIFFLOR-runNN，与 td3per 的 output/TD3PER-runNN 分开）。\n"
        "    #   ⚠ TB 的 scalar tag 与 td3per 逐字同名（刻意的，便于跨 run 对齐比较）\n"
        "    #   ⇒ 方法身份靠这个 run 名 + CSV 名 df_aifflor.csv。\n"
        "    parser.add_argument('--env',\n"
        "                        default='AIFFLOR',\n"
        "                        type=str,\n"
        "                        help='run 身份 (决定日志目录 output/<env>-runN, 即 TensorBoard 的 run 名);'\n"
        "                             ' aifflor: TB 的 scalar tag 与 td3per 逐字同名(便于跨 run 对齐),'\n"
        "                             ' 故方法区分靠此目录名 + CSV 名 (df_aifflor.csv)')",
        1, "--env 默认 TD3PER → AIFFLOR（输出目录名 = TB run 名）"),
]

#: `myfunc` **逐字复制、零改动**（见模块 docstring 的论证）
MYFUNC_EDITS = []

RULES = {
    'aifflor/aifflor_ddpg.py': ('td3per/td3per_ddpg.py', [], '逐字复制（零改动）'),
    'aifflor/aifflor_myfunc.py': ('td3per/td3per_myfunc.py', MYFUNC_EDITS,
                                  '逐字复制（动作语义与 td3per 相同 ⇒ 一处不改）'),
    'aifflor/aifflor_main.py': ('td3per/td3per_main.py', MAIN_EDITS,
                                'env 类 / 不启 MATLAB / 文案（动作带与 myfunc 不变）'),
}


def build(kind='apply'):
    """重算期望内容；任一锚点命中次数不符 ⇒ 抛异常（整体不写盘）。"""
    out, log = {}, []
    for dst, (src, edits, note) in RULES.items():
        path = dst if kind == 'check' else src
        with open(path, 'r', encoding='utf-8') as fh:
            text = fh.read()
        for old, new, cnt, why in edits:
            got = text.count(old)
            if got != cnt:
                raise AssertionError(
                    '[%s] 锚点命中 %d 次（期望 %d）：%s\n----\n%s\n----'
                    % (dst, got, cnt, why, old[:400]))
            text = text.replace(old, new)
            log.append((dst, why, cnt))
        out[dst] = text
    return out, log


def unified_diff(a, b, name):
    return ''.join(difflib.unified_diff(a.splitlines(True), b.splitlines(True),
                                        'td3per/' + name, 'aifflor/' + name, n=1))


def main():
    ap = argparse.ArgumentParser(description='td3per → aifflor 一次性迁移',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument('--apply', action='store_true', help='落盘（默认 dry-run）')
    ap.add_argument('--check', action='store_true',
                    help='校验磁盘上的 aifflor/*.py == 源文件 + 替换表（双向）')
    ap.add_argument('--quiet', action='store_true', help='不打印 diff')
    args = ap.parse_args()

    if args.check:
        ok = True
        for dst, (src, edits, note) in RULES.items():
            with open(dst, 'r', encoding='utf-8') as fh:
                on_disk = fh.read()
            back, bad = on_disk, []
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
            ok = ok and same and not bad
            print('  %-28s %s  (%s)%s'
                  % (os.path.basename(dst), 'OK' if (same and not bad) else '**不一致**',
                     note, ('' if not bad else '\n      ' + '\n      '.join(bad))))
        print('\n%s' % ('可逆性校验通过：磁盘文件 == 源 + 替换表 ✅' if ok
                        else '**校验失败** ❌'))
        return 0 if ok else 1

    built, log = build('apply')
    for dst, (src, edits, note) in RULES.items():
        with open(src, 'r', encoding='utf-8') as fh:
            src_text = fh.read()
        if not args.quiet:
            d = unified_diff(src_text, built[dst], os.path.basename(dst))
            print('\n' + '=' * 90)
            print('=== %s  ←  %s   (%s)' % (dst, src, note))
            print('=' * 90)
            print(d if d else '  (无差异：逐字复制)')
    if not args.apply:
        print('\n[dry-run] 未写盘。确认后加 --apply。')
        print('落盘文件：%s' % ', '.join(sorted(built)))
        return 0
    for dst, text in built.items():
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, 'w', encoding='utf-8', newline='') as fh:
            fh.write(text)
        print('已写入 %s' % dst)
    print('\n下一步: python agentutil/port_aifflor.py --check')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
