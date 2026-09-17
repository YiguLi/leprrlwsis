# agentutil/ —— agent 自建的**项目无关**工具与测试

本目录是 HSRL 项目的**约定目录**（见根目录 `AGENTS.md`）：

> 所有**项目无关**的临时/自建文件（一次性脚本、探针、分析工具、回归测试、
> 临时草稿）一律放这里，不要散落在仓库根目录或各项目子目录里。
> 需要跑训练/评测的**生产模块**（`hsrl_*.py`、`myfunc.py`、`util.py`、
> `env.py`、`*_main.py`、`*_ddpg.py`、`test_scenario.py` …）必须留在根目录，
> 因为 9 个 main 与各 `*_myfunc.py` 直接 import 它们。

## 目录内脚本

| 脚本 | 用途 | 命令 |
|---|---|---|
| `test_c1_uniform.py` | H 层 c1 口径不变量（8 项目逐位相同 / legacy 分支 / `Σ₅fluct_cost(3)=15`） | `python agentutil/test_c1_uniform.py` |
| `test_critic_bias.py` | critic 输出层 bias（价值基线）初始化 | `python agentutil/test_critic_bias.py` |
| `test_warmup_schedule.py` | warmup 调度 + bias 触发点（含旧 `step==warmup` 不可达 bug 回归） | `python agentutil/test_warmup_schedule.py` |
| `test_value_scale.py` | H 层 `--value_scale` 归一化：覆盖性 + 两臂数值不变量 | `python agentutil/test_value_scale.py` |
| `test_l_lower_unit.py` | RL-BESS 下层口径 ÷100：系数/预算/步长 + 物理量守恒 | `python agentutil/test_l_lower_unit.py` |
| `analyse_q_scale.py` | 从 `update_stats_H.csv` 量化 Q 尺度瞬态与相对 TD 误差 | `python agentutil/analyse_q_scale.py [csv|通配 ...]` |
| `ab_critic_bias.py` | 同 seed 短程 A/B 驱动（bias 初始化 / 值函数尺度） | `python agentutil/ab_critic_bias.py --project td3perrb --seed 101 --steps 20000 [--dry-run]` |
| `probe_action_symmetry.py` | 已训练 critic 的 ∂Q/∂a 动作对称性探针 | `python agentutil/probe_action_symmetry.py output/<run1> [run2 ...]` |
| `probe_value_scale.py` | 归一化两臂的学习曲线/隐层激活量级诊断 | `python agentutil/probe_value_scale.py` |
| `audit_coverage.py` | 九方法覆盖审计（归一化/bias 接线逐项核对）+ H 层 η1 分析 | `python agentutil/audit_coverage.py` |

留在根目录的评测主脚本（非一次性工具）：`test_five_days.py`、`test_simple.py`、
`test_compare_projects.py`、`test_scenario.py`、`compare_*.py`、
`extract_training_wind_days.py`、`generate_day_sequence.py`。

## 写新脚本时的两条硬性约定

1. **路径 bootstrap**：脚本可能从任意 CWD、以任意路径被调用，因此开头必须写

   ```python
   import os
   import sys

   _HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
   ROOT = os.path.dirname(_HERE)                          # 仓库根
   os.chdir(ROOT)                                         # 相对数据/输出路径以仓库根为基准
   sys.path.insert(0, ROOT)                               # 项目模块 (hsrl_*, myfunc, util, td3*/...)
   sys.path.insert(0, _HERE)                              # agentutil 内互相 import
   ```

   **不要**写 `ROOT = dirname(__file__)`（那是本目录，不是仓库根），更不要硬编码
   `C:\Users\...` 绝对路径。

2. **不写生产依赖到本目录**：项目代码（main / myfunc / hsrl_*）**不得** import 本目录
   下的任何模块，否则训练路径会被工具目录绑架。反过来（本目录 import 项目模块）是允许的。

## 运行前置

* 需要环境：`C:\Users\yigul\anaconda3\envs\llmenhancedwsis\python.exe`（torch 2.12 CPU / numpy / pandas）。
* 除 `test_*` 与 `probe_*` 外，涉及真实训练的脚本需要 **MATLAB**（`env.py` 启动
  `matlab.engine`）；单元测试**不需要** MATLAB，可在任何机器上跑。
