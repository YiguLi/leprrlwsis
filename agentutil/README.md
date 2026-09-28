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
| `test9_compare.py` | 多 run test9 结果对比（**只用 `OVERALL` 行**；每次评测 10 行 = 9 场景 + 1 汇总）：同 step 对齐 + 同窗口稳定性 + **SOC 结算拆解**（`econ 差 = raw 差 + 结算差`）+ 动态范围（step480 未训练 vs 最好）+ **`[6] r1 vs profit 分解`**：`profit/45 − r1 = (penalty_NN_H + derate_loss)/45`，其中 `derate_loss = turbine_profit·(1−f)` 是 `--viol_derate` 的**乘性削减**（c=0 时恒 0）。**有 `turbine_profit`/`derate_loss`/`penalty_NN_H` 三列时分开报（并自校验分解恒等式）**，历史 run 无此列则退回"未分离"（此时合计即纯罚）+ `--settle` 自洽校验。容忍 BOM 与只有 OVERALL 行的 CSV | `python agentutil/test9_compare.py [--window LO HI] [--tail 24] [--scenarios] run12 run13` |
| `audit_policy_stability.py` | 跨 run **H 层策略稳定性 / 值层自洽**同口径审计。核心方法：**共同面板**（各 run 回放状态的均衡并集，`--panel-from union`）——单 run 各自的面板状态分布不同，`D_pi` 绝对值不可跨 run 比。输出: [A] `D_pi首/D_pi邻/M越界`、[B] `Qbar/锚点/自洽比(滚动锚+固定锚)/|dQp/da|/lam1`、[C] 候选 vs 基线改善摘要（含 sd 与**锚点稳健过滤** `--anchor-min`，小分母相除会虚高）、[D] `--check-gamma` 反解 `γ_eff`（n=1 → γ；n=5 → γ⁵）、[E] 各 run 自身末点、**[F] 位移–间隔标度律**（`rms(k)=sqrt(mean_s‖a_{i+k}−a_i‖²)`，拟合 `rms∝k^α`：**α≈1 匀速漂移=未收敛 / α≈0.5 定点附近随机游走=已收敛 / α≈0 冻结**，附 `rms(1)/a_scale` 噪声地板与相邻 `D_pi` 趋势）。**不 import `*_myfunc`（避免启 MATLAB）**，`Normalize_H` 为逐位复刻并用日志 `phase_sin/cos` 列自校验（`--check-phase`） | `python agentutil/audit_policy_stability.py run9 run10 run13 [--per-run 1000] [--steps ...] [--check-gamma] [--check-phase]` |
| `test_l_lower_unit.py` | RL-BESS 下层口径：真源接线 + **命令值(两段 clip)** + λ2 兑换率/对偶律 + 预算守恒 | `python agentutil/test_l_lower_unit.py` |
| `test_soctarget_uniform.py` | 下层 SOC 目标四项目（td3per/td3prrbb/prle/td3prle）逐位等价 | `python agentutil/test_soctarget_uniform.py` |
| `test_l_residual.py` | 残余动作 `pb_cmd = 规则基线 + Δ` 的不变量（基线逐位 ≡ `SocTracker.act` / `Δ≡0 ⇒ r2≡0 ∧ c2≡0` / 零初始化输出层 / 三 main 接线 / 事后填入核对） | `python agentutil/test_l_residual.py` |
| `test_l_track_range.py` | 下层 r2/c2 两区间分治不变量（F=∅ ⇒ 单点竭力值 / `n_rem` 时间表 / 单位符号） | `python agentutil/test_l_track_range.py` |
| `test_phase_uniform.py` | 24h 相位口径统一回归：五统一模块的 `Normalize_H`/`phase_daily_table`/`day_price_schedule` 逐位相同 + `rep_minute_for_price_level` 两种调用约定（档位名/数值电价）+ `td3leprrlrb` 旧口径**语义等价** + 六方法接线契约（`--phase_feat`、`nb_states_H`、`--bsize 128`、两个接收方的 `nine_case_test` 与 main、`test_five_days.py` 的 `h_obs`） | `python agentutil/test_phase_uniform.py` |
| `test_nstep_h.py` | H 层 **n 步回报**（`--nstep`，**滑窗 stride-1**，CLI 默认 5）与 twin-target 聚合（`--target_agg`）实现契约：**显式 `nstep=1`（复现 run1~run10 的历史口径）**与输入逐位相同 + 滑窗条数 `=块数−n+1`、起点逐块覆盖、`Σγ^k r_k`/第 n 后继/`γ^n`/done 截断/episode 尾结算 + 一整集条数 == 块数（不丢状态覆盖）+ target 公式（`r+γv` vs `Σγ^k r_k+γ^n v`、min 与 mean 聚合） | `python agentutil/test_nstep_h.py` |
| `test_viol_derate.py` | **越界乘性惩罚**（`--viol_derate`，**默认 0 = 关闭**；`r1` 的风机收益 `× exp(−c·Σ前5维越界量)`）契约：`x` 只含机组维（**dim5 signal 不计**）+ 关闭/无越界时 `f` **恰为 1.0**（逐位复现历史 run）+ 只缩放风机项（`Δr1 == turbine_profit·(f−1)`）+ **`r1` 单调不增**（越界永不改善奖励）+ **`details` 里 `turbine_profit`/`revenue`/`bess_profit`/`degradation` 不受影响**（记录列不变）+ main/`nine_case_test` 接线与其它 8 个 `*_myfunc` 未被牵连 + **端到端（桩 env 真跑 `nine_case_test`，不需 MATLAB）**：`c=0` 与 `c=5` 的 `profit`/`dsoc`/`profit_econ` **逐位相同**、分解恒等式 `profit/N − r1 == (罚 + 削减)/N` 成立、`coef=None` ≡ 模块默认（动态跟随常量）+ 断言默认值 = 0（c=5.0 经 run14 实测有害已回退） | `python agentutil/test_viol_derate.py` |
| `analyse_q_scale.py` | 从 `update_stats_H.csv` 量化 Q 尺度瞬态与相对 TD 误差 | `python agentutil/analyse_q_scale.py [csv|通配 ...]` |
| `ab_critic_bias.py` | 同 seed 短程 A/B 驱动（bias 初始化 / 值函数尺度） | `python agentutil/ab_critic_bias.py --project td3perrb --seed 101 --steps 20000 [--dry-run]` |
| `probe_q_profile_boundary.py` | **逐维动作边界剖面**：固定其它维、只扫一维得到 `Q(a_j)` 面板均值切片 → 峰位、界内/界外签名斜率、"反差 = 越界深度 × \|界外斜率\|" vs critic TD 误差。**区分两类越界**：(A) 角点最优 + 平滑策略 → 峰在界内但界外回正力不足；(B) critic 偏好在界外 → **Q 峰落在界外或界外斜率符号错**。训练/评估路径不 import `*_myfunc`（不启 MATLAB） | `python agentutil/probe_q_profile_boundary.py run13 [--steps 34000 40000] [--dims 4 5] [--td-error 644] [--which p\|c] [--grid -1 1 0.05]` |
| `probe_action_hist.py` | **动作分布探针**（把"策略在逼近角点"与"tanh 饱和 / 整排翻面"分开）：面板状态复用 `probe_q_profile_boundary`（不启 MATLAB），统计**确定性策略输出**（eval，无探索噪声）→ 逐维 `mean/std/p05…p95`、越界上/下占比、贴边占比、**端部倍率**（两端各 5% 宽度内质量 ÷ 均匀应有的 0.10）+ 单行密度串直方图 + **"贴边维数"分布**与贴边指示量互相关。签名：端部倍率>1 且多贴边占比高且贴边相关>0 ⇒ "堆两端 + 整排翻面"。run15 ckpt38000 实测该签名**不成立**（多贴边 0.5%、贴边相关 −0.079、只有 dim2 单边堆上界，均值稳定在 0.48~0.50）⇒ 其 `M` 摆动是平滑网络在折点 `a2=0.5` 附近的落点抖动，不是策略 churn；run13 的 dim4/dim5 则是**下界轨道**堆积（病，见 Q 剖面峰在界外） | `python agentutil/probe_action_hist.py run15 [--steps 28000 38000] [--dims 2] [--rail-eps 0.02] [--bins 20]` |
| `probe_a_induction_power.py` | **∂(风场功率)/∂a_j 的日志回归**（把"critic 偏好"与"env 真实物理"对照）：`ptur` 由 `turbine_profit = w_wind·pr·ptur/60` 反解（脚本打印它与 `obs[3]` 的相关做对齐自校验——`obs[3]` 是 `P_G`（机组+储能，`env.py:270`），**不是**纯机组功率，相关 0.53 属正常）；按 u 分箱内 OLS 控制其余四维，**外加"u 取 2 位小数严格配对"估计量**消掉 u 的一切泄漏（策略对风速有阈值行为时，仅靠线性控制 u 会伪造 MW 级斜率）；同时给出 `r1`/`c1_step`/`bess_profit` 的同类斜率以区分"风功率通道"与"储能/波动通道"。实测：`ptur` 是 (u,a) 的确定性函数（R²=0.97~0.99，残差 0.4~0.6 MW）⇒ 对齐成立；run15 dim2 的 +2.4 MW/单位 与 run13 的 −0.21 **符号相反** | `python agentutil/probe_a_induction_power.py run15 run13 [--act raw] [--ubins 0.5] [--dec 2]` |
| `probe_a_power_sweep.py` | **直接扫 WFSim 的 风场功率 vs a_j**（判定"推到 a=0.5"对不对；**需 MATLAB**）：固定其余四维（默认取 run15 日志均值），只扫一维，每点一次 `WFSim_port` → 打印总功率、局部斜率、峰值位置，`--hysteresis` 倒序再扫做**历史依赖检验**。动机：按 MATLAB 侧公式 a_j→0.5 应**双输**（`Actuator.m:69` `Power = powerscale·½ρA·CT·mean(Ue³)` 且 `CT = input.CT_prime = 4a(1−a)²` ⇒ 单机峰在 a=1/3、0.5 处仅 84%；`Actuator.m:64` `Fthrust = ½ρUe²·4a(1−a)` ⇒ 推力在 0.5 最大、尾流亏损更深），而日志回归给出的是**正**斜率 ⇒ 二者必须用仿真裁决 | `python agentutil/probe_a_power_sweep.py --dim 2 [--u 8 10 12] [--others ...] [--hysteresis]` |
| `probe_a_per_turbine.py` | **逐台机组 `CT_prime`/`power` 读出**（确认 5 个动作维是否真接进 env；**需 MATLAB**）：`meshing.m:196-197` 的 `zeros(NT,1)+Control.c_prime`（`Control.c_prime` 是 Python 传来的 **1×5 行**）在 MATLAB 隐式扩展下成为 **NT×5 矩阵**，而 `Actuator.m:14` 取标量 struct、`:55 CT(kk)=input.CT_prime(kk)` / `:61 a(kk)=input.a(kk)` **线性索引只取第一列** ⇒ **每台机组都拿到 `c_prime(a0)`，dim1~4 与 env 无关**。脚本用 `eng.eval` 直接跑 `WFSim`，对若干组动作打印逐台 CT/power：逐台 CT 全同 ⇒ 机制成立；逐台不同 ⇒ 接入正常。**不改任何项目文件**（训练中改 `*.m` 会被 MATLAB 按时间戳重载，可能影响运行中的实验）。**2026-09-28 实测确认**：u=10、其余 0.20 时，`a2=0.50` 与 `a4=0.50` 两组的 `CT_prime`（`0.512×5`）与逐台 `power` 与 `all 0.20` **逐位相同**（total 3.717941 MW ×3），`ramp 0.1..0.5` 只取 `0.324=c_p(0.1)` 复制给 5 台 ⇒ dim1~4 无效、dim0 是全体共用的集体诱导因子 | `python agentutil/probe_a_per_turbine.py [--u 10]` |
| `patch_a_induction_wiring.py` | **把 5 维轴向因子逐台接进 WFSim 的一次性补丁**（默认 dry-run）：改 `WFSim_port.m`（截到 NT=5 并**转列向量**，使 `zeros(NT,1)+Control.a` 变逐元素；第 6 维信号顺带丢掉，收到不足 NT 维则 `error` 报错）+ `env.py::PowerOutput`（clip 前先截到 `self.num_turbine`，防多传维再次隐式扩展）。不采用"改 `meshing.m`"是因为它有 4 处同样模式 + 2 处硬写 `2*ones(NT,1)`；改入口一处即可覆盖。脚本对锚点要求"恰好命中 1 次"、保留原换行风格，**训练中禁止 `--apply`**（MATLAB 按时间戳重载 `*.m`） | `python agentutil/patch_a_induction_wiring.py [--apply]` |
| `test_nstep_uniform.py` | **H 层 n 步机制的跨项目一致性**（六个 TD3 方法）：绕过 `__init__` 装配最小实例 + 桩 memory, 对每个项目真跑 `observe_jump`/`_emit_nstep`/`_push_transition` —— `nstep=1` ⇒ 逐条直接落库（历史 1-step 同形）；`nstep=3` 滑窗喂 6 块 ⇒ 恰好 6 条、`r_sum=Σγ^k r_k`、起点覆盖每个块（stride-1）、`boot=γ^n`、窗口内含 done ⇒ `boot=0` 且存 done=1、episode 末 pending 清空；两个 PRLE 项目的 **context 顺序拼接**、非 PRLE 不传 context。**不需 MATLAB** | `python agentutil/test_nstep_uniform.py` |
| `port_nstep.py` | **一次性迁移脚本（已执行完毕，保留作审计记录）**：把 td3per 的 H 层 n 步机制同步到其余五个 TD3 方法（`*_ddpg.py` 加 `nstep`/`target_agg` kwarg + init 块 + target 聚合 + 按条 `_disc` + `_push_transition`/`_emit_nstep`/滑窗 `observe_jump`；`*_main.py` 加 `--nstep`(默认5)/`--target_agg`(默认min)、`--rmsize` 改 20000、构造接线；两个 PRLE 项目额外做 context 拼接 + `relabel_all(discount=γ)` 按块 γ 加权）。作用域解析支持类与方法, 每个变换要求"恰好命中 1 次"否则不写盘; 默认 dry-run | `python agentutil/port_nstep.py [--apply]` |
| `test_viol_derate_uniform.py` | **越界乘性惩罚的跨项目一致性**（六个 TD3 方法）：helper **逐位同源** + 六个包装函数（`Cal_Rew_RB`/`Cal_Rew_RLRB`/`Cal_Rew_PRRBB`）的 `Δr1 == turbine_profit·(f−1)` 与记录列不变 + **端到端（桩 env 真跑 `nine_case_test`）**：`c=0`/`c=5` 的 `profit`/`profit_econ`/`dsoc` **逐位相同**、分解残差**与 c 无关**（`wT=wB=1` 的两个项目残差 ≡0；四个 PRRRB 系项目因收益重加权残差非零，属正常）+ 六个 main 的静态接线。**不需 MATLAB** | `python agentutil/test_viol_derate_uniform.py` |
| `port_viol_derate.py` | **一次性迁移脚本（已执行完毕，保留作审计记录）**：把 td3per 的奖励侧改动（helper 块**从 td3per 逐字提取**、r1 乘系数、包装函数透传、`nine_case_test` 分解三列、main 的 CLI/接线/df_test9 三列）同步到其余五个 TD3 方法。每个变换要求"作用域内恰好命中 1 次"，不满足则整体不写盘；默认 dry-run，`--apply` 才写 | `python agentutil/port_viol_derate.py [--apply]` |
| `flip_viol_derate_default.py` | **一次性脚本（已执行）：把 `--viol_derate` 默认从 5.0 改回 0（关闭）**。run14 实测该乘性罚有害（奖励电平被策略自身越界率绑架 ⇒ critic 高估约 2x ⇒ 正反馈），故只回退默认值、**保留机制/开关/全部测试**供日后受控实验；带断言、默认 dry-run。**另见 README 的 `--viol_derate` 条目里的失败模式** | `python agentutil/flip_viol_derate_default.py [--apply]` |
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
