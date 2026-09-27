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
| `test_csv_incremental.py` | 日志表**增量落盘**回归：6 个 main 静态核查 + `flush_csv_append` 与全量写逐位一致/写盘量不随行数增长 | `python agentutil/test_csv_incremental.py` |
| `test_test9_settlement.py` | test9 **存货结算**：`profit_econ = profit + ΔSOC·580` 的列/算术/望远镜一致性 + 净充放电方向 + main 接线契约（桩 env + 真 RuleBasedBESS 真跑 `nine_case_test`） | `python agentutil/test_test9_settlement.py` |
| `test9_compare.py` | 多 run test9 结果对比（**只用 `OVERALL` 行**；每次评测 10 行 = 9 场景 + 1 汇总）：同 step 对齐 + 同窗口稳定性 + **SOC 结算拆解**（`econ 差 = raw 差 + 结算差`）+ 动态范围（step480 未训练 vs 最好）+ **`[6] r1 vs profit 分解`**（二者是同一笔钱：`r1 = revenue − degradation − penalty_NN_H` 为每步量，`profit` 为 45 步总和 ⇒ `profit/45 − r1 = 越界罚/步 ≥ 0`；**r1 抖而 profit/econ 稳 = 抖的是越界罚，经济稳定**，并自动检出恒等式被破坏）+ `--settle` 自洽校验 | `python agentutil/test9_compare.py [--window LO HI] [--tail 24] [--scenarios] run12 run13` |
| `audit_policy_stability.py` | 跨 run **H 层策略稳定性 / 值层自洽**同口径审计。核心方法：**共同面板**（各 run 回放状态的均衡并集，`--panel-from union`）——单 run 各自的面板状态分布不同，`D_pi` 绝对值不可跨 run 比。输出: [A] `D_pi首/D_pi邻/M越界`、[B] `Qbar/锚点/自洽比(滚动锚+固定锚)/|dQp/da|/lam1`、[C] 候选 vs 基线改善摘要（含 sd 与**锚点稳健过滤** `--anchor-min`，小分母相除会虚高）、[D] `--check-gamma` 反解 `γ_eff`（n=1 → γ；n=5 → γ⁵）、[E] 各 run 自身末点。**不 import `*_myfunc`（避免启 MATLAB）**，`Normalize_H` 为逐位复刻并用日志 `phase_sin/cos` 列自校验（`--check-phase`） | `python agentutil/audit_policy_stability.py run9 run10 run13 [--per-run 1000] [--steps ...] [--check-gamma] [--check-phase]` |
| `test_l_lower_unit.py` | RL-BESS 下层口径：真源接线 + **命令值(两段 clip)** + λ2 兑换率/对偶律 + 预算守恒 | `python agentutil/test_l_lower_unit.py` |
| `test_soctarget_uniform.py` | 下层 SOC 目标四项目（td3per/td3prrbb/prle/td3prle）逐位等价 | `python agentutil/test_soctarget_uniform.py` |
| `test_l_residual.py` | 残余动作 `pb_cmd = 规则基线 + Δ` 的不变量（基线逐位 ≡ `SocTracker.act` / `Δ≡0 ⇒ r2≡0 ∧ c2≡0` / 零初始化输出层 / 三 main 接线 / 事后填入核对） | `python agentutil/test_l_residual.py` |
| `test_l_track_range.py` | 下层 r2/c2 两区间分治不变量（F=∅ ⇒ 单点竭力值 / `n_rem` 时间表 / 单位符号） | `python agentutil/test_l_track_range.py` |
| `test_phase_uniform.py` | 24h 相位口径统一回归：五统一模块的 `Normalize_H`/`phase_daily_table`/`day_price_schedule` 逐位相同 + `rep_minute_for_price_level` 两种调用约定（档位名/数值电价）+ `td3leprrlrb` 旧口径**语义等价** + 六方法接线契约（`--phase_feat`、`nb_states_H`、`--bsize 128`、两个接收方的 `nine_case_test` 与 main、`test_five_days.py` 的 `h_obs`） | `python agentutil/test_phase_uniform.py` |
| `test_nstep_h.py` | H 层 **n 步回报**（`--nstep`，**滑窗 stride-1**，CLI 默认 5）与 twin-target 聚合（`--target_agg`）实现契约：**显式 `nstep=1`（复现 run1~run10 的历史口径）**与输入逐位相同 + 滑窗条数 `=块数−n+1`、起点逐块覆盖、`Σγ^k r_k`/第 n 后继/`γ^n`/done 截断/episode 尾结算 + 一整集条数 == 块数（不丢状态覆盖）+ target 公式（`r+γv` vs `Σγ^k r_k+γ^n v`、min 与 mean 聚合） | `python agentutil/test_nstep_h.py` |
| `analyse_q_scale.py` | 从 `update_stats_H.csv` 量化 Q 尺度瞬态与相对 TD 误差 | `python agentutil/analyse_q_scale.py [csv|通配 ...]` |
| `ab_critic_bias.py` | 同 seed 短程 A/B 驱动（bias 初始化 / 值函数尺度） | `python agentutil/ab_critic_bias.py --project td3perrb --seed 101 --steps 20000 [--dry-run]` |
| `probe_action_symmetry.py` | 已训练 critic 的 ∂Q/∂a 动作对称性探针 | `python agentutil/probe_action_symmetry.py output/<run1> [run2 ...]` |
| `probe_value_scale.py` | 归一化两臂的学习曲线/隐层激活量级诊断 | `python agentutil/probe_value_scale.py` |
| `probe_l_fempty_point.py` | 用真实 run 数据验证 `F=∅` 参考集合是零宽单点（只有竭力值免费） | `python agentutil/probe_l_fempty_point.py [csv]` |
| `probe_l_trackerr_worst.py` | 定位 trackerr 最大的步并归因（下层惰速/带宽失配指纹） | `python agentutil/probe_l_trackerr_worst.py [csv] [topN]` |
| `probe_wfsim_memoryless.py` | **`WFSim_port` 是否无记忆**（决定 test9 能否当确定性收敛判据）：同参数恒定 / 穿插后复原 / 两实例共享引擎互不干扰 | `python agentutil/probe_wfsim_memoryless.py`（**需 MATLAB**，约 2 min） |
| `probe_bsize_scan.py` | `--bsize` 扫描：用**固定样本量 vs 固定步数**两口径把"梯度估计更准"与"多学了几遍"分开（离线有监督代理目标） | `python agentutil/probe_bsize_scan.py`（约 6 min） |
| `probe_init_convention.py` | `fanin_init` 口径 A/B（**含 `stock` 纯噪声对照臂 + 逐 seed 配对差**）：`current`(现状) / `true_fanin` / `old_scoped`(把 fan_out 口径也套到 fc4/fc5) / `fix_fc1`，测 500/1500/6000 步 val R²·MSE·死 ReLU | `python agentutil/probe_init_convention.py`（约 10 min） |
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
* **例外**：`probe_wfsim_memoryless.py` 必须启 MATLAB（它测的就是 MATLAB 侧的
  `WFSim_port`）；其余 `probe_*` 只读日志/checkpoint，不需要。
