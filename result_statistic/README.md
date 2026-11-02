# result_statistic/ —— 四固定风速日的评测脚本

本目录是用户在仓库根**显式指定**的目录（`AGENTS.md` 的"项目无关脚本进 `agentutil/`"
的**已登记例外**）：里面的脚本只做评测，不做训练。

## 文件

| 文件 | 用途 |
|---|---|
| `test_four_days.py` | **四固定风速日评测**：env 式（发出状态 → 接收动作 → 推进环境 → 记录结果），只推理不训练 |

## 这四天是固定的（不随机抽、不可换）

来自 `agentutil/select_4_wind_days.py` 的选天结果（`FFC_speed_noise_1.csv` 的
**iloc 列位置**，与 `env.py` 的 `iloc[:, j]` 同口径）：

| tag | 含义 | iloc | 日均 | std | CV |
|---|---|---|---|---|---|
| `d1_low_lowcv` | 低风速·低方差 | 24 | 8.968 | 1.666 | 0.186 |
| `d2_low_highcv` | 低风速·高方差 | 21 | 8.950 | 2.542 | 0.284 |
| `d3_high_lowcv` | 中高风速·低方差 | 164 | 11.974 | 1.415 | 0.118 |
| `d4_high_highcv` | 中高风速·高方差 | 2 | 11.288 | 3.253 | 0.288 |

每次运行都会核验这四列的日均/std 仍与上表一致（容差 0.02）；数据被换过就报错，
`--no-verify-days` 才强制按 iloc 跑。

## 它做什么 / 不做什么

**做**：与 `env.py::WindBattery` 同一套契约 —— `reset()` 给 `[pr, u, E, P_G_prev]`、
`step(H, L, pw_tru)` 推进电价/风速/储能并回 `P_F`/`P_G`、`PowerOutput(u, a)` 算风机出力；
逐 (方法 × run × 天) 记录三项指标（越限次数 / 越限总量 / 利润）与 SOC、充放电、峰值。

**不做**：不更新网络、不写 replay、不做 PER 采样 / λ 对偶 / curriculum / checkpoint；
`agent.is_training=False` ⇒ 探索噪声恒为 0，动作完全确定。

推理路径**不重写**：直接复用 `test_five_days.py` 的 `PROJECTS` 注册表（11 个方法）与
`ProjectRunner`（每方法自己的 H/L 观测、残余动作或规则下层、RBB 硬裁剪、piecewise
权重、相位特征），只把"取天"换成固定四天 ⇒ 不会与生产口径漂移。要加新方法，去
`test_five_days.PROJECTS` 注册。

## 用法

```bash
# 一个方法一组模型（完整四天 × 1440 步；WFSim 约 0.3~1 s/步 ⇒ 约 0.5~1.6 h/模型）
python result_statistic/test_four_days.py --td3per output/TD3PER-run19

# 多方法 / 多组模型（同一 KEY 重复 = run1/run2/...）
python result_statistic/test_four_days.py \
    --td3per output/TD3PER-run18 output/TD3PER-run19 \
    --aifflor output/AIFFLOR-run15

# 冒烟（每天 60 步，指标只统计这 60 步）
python result_statistic/test_four_days.py --td3per output/TD3PER-run19 --episode-length 60

# 只看计划：不加载权重、不推进仿真
python result_statistic/test_four_days.py --td3per output/TD3PER-run19 --dry-run

# 无 MATLAB 环境下的管线自检（指标无物理意义）
python result_statistic/test_four_days.py --td3per output/TD3PER-run19 \
    --episode-length 60 --stub-engine

python result_statistic/test_four_days.py --list-projects
```

## 前置

* **解释器**：项目自己的环境（本机 = `anaconda3/envs/llmenhancedwsis`，torch 2.12）。
  系统 PATH 上的 `python` 是 Microsoft Store 占位符，不能直接用。
* **MATLAB**：走 WFSim 的方法需要 `matlab.engine`；受限沙箱里起不来
  （表现为 `UnicodeDecodeError`，见 `agentutil/README.md` 的记录）⇒ 请用普通终端。
* **纯 FLORIS 方法**（`yawflor`）不需要 MATLAB。
* `--stub-engine` 用确定性纯 Python 物理替代 WFSim，只用于验证"状态→动作→推进→记录"
  这条管线（本次在沙箱里就是这样自检的）；其指标**不可**当 WFSim 读数引用。

## 输出（`--output` 默认 `output/result_statistic/`）

```
four_day_days.csv      每 (方法 × run × 天) 一行：三项指标 + SOC/充放电/峰值 + 四天标识
four_day_steps.csv     逐步明细（--no-save-steps 关闭）
four_day_summary.csv   每 (方法 × run) 的四天均值±std + 合计     ★主表
four_day_by_day.csv    每天 × 每方法（跨 run 均值）—— 看哪种风况谁更好
four_day_ranking.csv   排名（越限次数 → 越限总量 → 利润）
tensorboard/           可选（--no-tensorboard 关闭）
```
