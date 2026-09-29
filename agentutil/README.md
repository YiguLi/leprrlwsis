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
| `test_phase_uniform.py` | 24h 相位口径统一回归：五统一模块的 `Normalize_H`/`phase_daily_table`/`day_price_schedule` 逐位相同 + `rep_minute_for_price_level` 两种调用约定（档位名/数值电价）+ `td3leprrlrb` 旧口径**语义等价** + 六方法接线契约（`--phase_feat`、`nb_states_H`、`--bsize 256` / `--rmsize 40000`、两个接收方的 `nine_case_test` 与 main、`test_five_days.py` 的 `h_obs`） | `python agentutil/test_phase_uniform.py` |
| `test_nstep_h.py` | H 层 **n 步回报**（`--nstep`，**滑窗 stride-1**，CLI 默认 5）与 twin-target 聚合（`--target_agg`）实现契约：**显式 `nstep=1`（复现 run1~run10 的历史口径）**与输入逐位相同 + 滑窗条数 `=块数−n+1`、起点逐块覆盖、`Σγ^k r_k`/第 n 后继/`γ^n`/done 截断/episode 尾结算 + 一整集条数 == 块数（不丢状态覆盖）+ target 公式（`r+γv` vs `Σγ^k r_k+γ^n v`、min 与 mean 聚合） | `python agentutil/test_nstep_h.py` |
| `test_viol_derate.py` | **越界乘性惩罚**（`--viol_derate`，**默认 0 = 关闭**；`r1` 的风机收益 `× exp(−c·Σ前5维越界量)`）契约：`x` 只含机组维（**dim5 signal 不计**）+ 关闭/无越界时 `f` **恰为 1.0**（逐位复现历史 run）+ 只缩放风机项（`Δr1 == turbine_profit·(f−1)`）+ **`r1` 单调不增**（越界永不改善奖励）+ **`details` 里 `turbine_profit`/`revenue`/`bess_profit`/`degradation` 不受影响**（记录列不变）+ main/`nine_case_test` 接线与其它 8 个 `*_myfunc` 未被牵连 + **端到端（桩 env 真跑 `nine_case_test`，不需 MATLAB）**：`c=0` 与 `c=5` 的 `profit`/`dsoc`/`profit_econ` **逐位相同**、分解恒等式 `profit/N − r1 == (罚 + 削减)/N` 成立、`coef=None` ≡ 模块默认（动态跟随常量）+ 断言默认值 = 0（c=5.0 经 run14 实测有害已回退） | `python agentutil/test_viol_derate.py` |
| `analyse_q_scale.py` | 从 `update_stats_H.csv` 量化 Q 尺度瞬态与相对 TD 误差 | `python agentutil/analyse_q_scale.py [csv|通配 ...]` |
| `ab_critic_bias.py` | 同 seed 短程 A/B 驱动（bias 初始化 / 值函数尺度） | `python agentutil/ab_critic_bias.py --project td3perrb --seed 101 --steps 20000 [--dry-run]` |
| `probe_q_profile_boundary.py` | **逐维动作边界剖面**：固定其它维、只扫一维得到 `Q(a_j)` 面板均值切片 → 峰位、界内/界外签名斜率、"反差 = 越界深度 × \|界外斜率\|" vs critic TD 误差。**区分两类越界**：(A) 角点最优 + 平滑策略 → 峰在界内但界外回正力不足；(B) critic 偏好在界外 → **Q 峰落在界外或界外斜率符号错**。训练/评估路径不 import `*_myfunc`（不启 MATLAB） | `python agentutil/probe_q_profile_boundary.py run13 [--steps 34000 40000] [--dims 4 5] [--td-error 644] [--which p\|c] [--grid -1 1 0.05]` |
| `probe_action_hist.py` | **动作分布探针**（把"策略在逼近角点"与"tanh 饱和 / 整排翻面"分开）：面板状态复用 `probe_q_profile_boundary`（不启 MATLAB），统计**确定性策略输出**（eval，无探索噪声）→ 逐维 `mean/std/p05…p95`、越界上/下占比、贴边占比、**端部倍率**（两端各 5% 宽度内质量 ÷ 均匀应有的 0.10）+ 单行密度串直方图 + **"贴边维数"分布**与贴边指示量互相关。签名：端部倍率>1 且多贴边占比高且贴边相关>0 ⇒ "堆两端 + 整排翻面"。run15 ckpt38000 实测该签名**不成立**（多贴边 0.5%、贴边相关 −0.079、只有 dim2 单边堆上界，均值稳定在 0.48~0.50）⇒ 其 `M` 摆动是平滑网络在折点 `a2=0.5` 附近的落点抖动，不是策略 churn；run13 的 dim4/dim5 则是**下界轨道**堆积（病，见 Q 剖面峰在界外） | `python agentutil/probe_action_hist.py run15 [--steps 28000 38000] [--dims 2] [--rail-eps 0.02] [--bins 20]` |
| `probe_a_induction_power.py` | **∂(风场功率)/∂a_j 的日志回归**（把"critic 偏好"与"env 真实物理"对照）：`ptur` 由 `turbine_profit = w_wind·pr·ptur/60` 反解（脚本打印它与 `obs[3]` 的相关做对齐自校验——`obs[3]` 是 `P_G`（机组+储能，`env.py:270`），**不是**纯机组功率，相关 0.53 属正常）；按 u 分箱内 OLS 控制其余四维，**外加"u 取 2 位小数严格配对"估计量**消掉 u 的一切泄漏（策略对风速有阈值行为时，仅靠线性控制 u 会伪造 MW 级斜率）；同时给出 `r1`/`c1_step`/`bess_profit` 的同类斜率以区分"风功率通道"与"储能/波动通道"。实测：`ptur` 是 (u,a) 的确定性函数（R²=0.97~0.99，残差 0.4~0.6 MW）⇒ 对齐成立；run15 dim2 的 +2.4 MW/单位 与 run13 的 −0.21 **符号相反** | `python agentutil/probe_a_induction_power.py run15 run13 [--act raw] [--ubins 0.5] [--dec 2]` |
| `probe_a_power_sweep.py` | **直接扫 WFSim 的 风场功率 vs a_j**（判定"推到 a=0.5"对不对；**需 MATLAB**）：固定其余四维（默认取 run15 日志均值），只扫一维，每点一次 `WFSim_port` → 打印总功率、局部斜率、峰值位置，`--hysteresis` 倒序再扫做**历史依赖检验**。动机：按 MATLAB 侧公式 a_j→0.5 应**双输**（`Actuator.m:69` `Power = powerscale·½ρA·CT·mean(Ue³)` 且 `CT = input.CT_prime = 4a(1−a)²` ⇒ 单机峰在 a=1/3、0.5 处仅 84%；`Actuator.m:64` `Fthrust = ½ρUe²·4a(1−a)` ⇒ 推力在 0.5 最大、尾流亏损更深），而日志回归给出的是**正**斜率 ⇒ 二者必须用仿真裁决 | `python agentutil/probe_a_power_sweep.py --dim 2 [--u 8 10 12] [--others ...] [--hysteresis]` |
| `probe_a_farm_structure.py` | **风场总功率的"动作结构"**（定量描述被控对象；**需 MATLAB**）。动机：修好逐台接线后动作空间第一次真正逐台起作用，而当时的动作先验（warmup 带 `[0.20,0.45]`、`init_wt_action_bias(U(0.25,0.40))`）全部基于"**单机** `C_p=4a(1−a)²` 保持最优"。四臂：`uniform`（共同水平 0.00→0.50 扫描 **+ 平台宽度**：`tol=1/2/5/10%` 下"总功率 ≥ 峰的 (1−tol)"的连续区间）、`ramp`（上游降载斜坡 + 镜像 + 只降上游）、`marginal`（只动一维 ±0.10 的**宽区间割线**——⚠ 被曲率主导而非局部梯度，只用来判峰在哪一侧）、`opt`（逐维细扫 0.05~0.40）。**2026-09-28 实测（MATLAB R2025a）**：**均匀族**峰全在 a=0.20（u=6/8/10/12/14 → 0.8031/1.9036/3.7179/6.4246/10.2020 MW，精确 **∝ u³**），±2% 平台 [0.20,0.25]、±5% 平台 [0.15,0.30] 也全同；全体单机最优 `1/3` 比 0.20 低 **7.3%**（u=10: 3.4462 vs 3.7179）⇒ 由此**唯一站得住**的结论是"**单机 C_p 最优 ≠ 风场总功率最优**"。⚠⚠ **本脚本的输出不是联合最优**，**不得**再据此定先验：`opt` 是**坐标轮询**（其余维钉在 0.20）、`ramp` 只试了几条固定斜坡，曾据此错误宣称"最优解接近常向量、不需要空间结构"（"拼接值与单独最优一致"是**循环论证**）。**真正的联合最优是随尾流状态变化的单调上升斜坡**——最下游机组背风侧无机组，其功率 ∝ `CT(a₅)·Ue³` 且上游动作固定时 `Ue` 与 `a₅` 无关 ⇒ 应推向单机最优 ≈1/3，需要降载留风的是上游；这个形状**是策略要学的函数**。用途收敛为"描述结构（峰位量级/平台/曲率）"，**不用于**定最优；带以 `hsrl_warmup` 为单一真源读出 | `python agentutil/probe_a_farm_structure.py [--arms uniform ramp marginal opt] [--u 6 8 10 12 14]` |
| `probe_a_per_turbine.py` | **逐台机组 `CT_prime`/`power` 读出 + 接线回归测试**（确认 5 个动作维是否真接进 env；**需 MATLAB**）。机制：`meshing.m:196-197` 的 `zeros(NT,1)+Control.c_prime` 在 `Control` 是 **1×K 行**（Python 侧 `matlab.double([list(act)])`）时经隐式扩展成为 **NT×K 矩阵**，而 `Actuator.m:14` 取标量 struct、`:55 CT(kk)=input.CT_prime(kk)` / `:61 a(kk)=input.a(kk)` 是**线性索引**，`kk=1..NT` 只落在**第一列** ⇒ 每台机组都拿到 `c_prime(a₀)`。**四个臂**：`port`（**生产入口** `WFSim_port.m`，判定"**逐台按位置响应**"——实测变化的机组集合必须恰为 `{d : c_p(a_d)≠c_p(0.20)}`，其余台与 all-0.20 基线**逐位相同**，且 Δtotal 非零）、`raw`（`meshing.m` 直连、绕过入口，**永远**打印 5 个相同 CT —— 只打印不断言，用于解释机制而非判据）、`env`（**整链** `WindBattery.PowerOutput(6 维)` 必须与 `WFSim_port(u,5 维前缀)` **逐位相同**，且第 6 维取 +0.25/−0.40 功率不变 ⇒ 信号维未泄漏）、`guard`（**入口维度守卫**：只接受**恰好 NT 维**——5 维通过，而 4 维（少传）与 6/10 维（多传）都必须**响亮报错**；多传曾是静默"只取前 NT 个"，实测 10 维 `[.2]*5+[.5]*5` 与 5 维逐位相同 3.717941 MW，与"只有 dim0 生效"是**同一类静默失效**）。任一断言失败退出码 1，故可直接当回归测试。**不改任何项目文件**。**2026-09-28 修前/修后对照（u=10）**：`raw` 臂 `a2=0.50`/`a4=0.50` 的逐台 CT（`0.512×5`）与 total 与 all-0.20 **逐位相同**（3.717941 MW ×3）、`ramp` 只取 `0.324=c_p(0.1)` ⇒ 修前 dim1~4 无效；打补丁后 `port` 臂 `a2=0.50`→CT `[.512 .512 .5 .512 .512]`（只动 dim2，Δtotal −0.178 MW）、`a4=0.50`→只动 dim4（−0.092 MW）、`ramp`→`[.324 .512 .588 .576 .5]`（五台各取 `c_p(自己的 a)`，Δtotal −0.308 MW），`env` 臂 6 组 `Δ=0.00e+00` 全 OK、第 6 维 ±0.25/−0.40 功率逐位相同 | `python agentutil/probe_a_per_turbine.py [--arms port raw env] [--u 10]` |
| `patch_a_induction_wiring.py` | **把 5 维轴向因子逐台接进 WFSim 的一次性补丁**（默认 dry-run；**2026-09-28 已 `--apply` 执行完毕，保留作审计记录**）：改 `WFSim_port.m`（**要求恰好 NT=5 维**并**转列向量**，使 `zeros(NT,1)+Control.a` 变逐元素；维度守卫**对称**——少传与多传都 `error` 报错，多传曾是静默取前 5 个；策略侧第 6 维信号由 `env.py::PowerOutput` 截掉后再下传）+ `env.py::PowerOutput`（clip 前先截到 `self.num_turbine`，防多传维再次隐式扩展）。不采用"改 `meshing.m`"是因为它有 4 处同样模式 + 2 处硬写 `2*ones(NT,1)`；改入口一处即可覆盖所有读 `Control` 的分支。脚本对锚点要求"**恰好命中 1 次**"、保留原换行风格，**训练中禁止 `--apply`**（MATLAB 按时间戳重载 `*.m`）；已打补丁后重复执行会因锚点命中 0 次而**安全中止、不写盘** | `python agentutil/patch_a_induction_wiring.py [--apply]` |
| `port_wt_prior.py` | **一次性迁移（2026-09-28 已 `--apply`；⚠ 已被 `relax_wt_prior.py` 取代，仅作审计记录）**：把**动作先验**从"单机 Cp 最优"改到"对准 `probe_a_farm_structure.py` 测出的均匀族峰位" —— `WT_START 0.33→0.20`、`WT_LO,WT_HI 0.20,0.45→0.15,0.25`、6 个 main 的 `init_wt_action_bias 0.25,0.40→0.17,0.23`、`WT_BIAS_LOW/HIGH`、`env.py` 三处 `a_temp 0.3333→0.20`、三个评测主脚本同项、`test_warmup_schedule.py` 断言。**这一轮的判据是错的**（把均匀族上的坐标轮询结果当成联合最优，见 `probe_a_farm_structure.py` 与 `relax_wt_prior.py` 的记录）；锚点要求命中数精确匹配、任一不符整体不写盘、按文件保留换行风格、默认 dry-run 这些机制仍值得复用 | `python agentutil/port_wt_prior.py [--apply]` |
| `relax_wt_prior.py` | **一次性迁移（2026-09-28 已 `--apply`，动作先验的最终口径）**：把先验**放宽为中性宽带**——`hsrl_warmup.WT_START 0.20→1/3`（单机 `C_p` 最优点，只讲单机物理）、`WT_LO,WT_HI → [0.05,0.45]`（只排除 a<0.05 停机悬崖与 a>0.45 低效区）、6 个 main 的 `init_wt_action_bias(lo,hi) → (0.05,0.45)`（与带同区间 ⇒ "warmup 与策略起步同分布"精确保留）、`env.py` 三处 `a_temp 0.20→0.3333`、三个评测主脚本同项、判据文字与 CLI help/`⚠` 记录（写明第一轮为何是错的）、`test_warmup_schedule.py` 判据退回**纯契约**（起点/界内/步长/截断/两端覆盖，**不**断言任何"最优/效率"）。锚点要求命中数精确匹配，任一不符整体不写盘；默认 dry-run | `python agentutil/relax_wt_prior.py [--apply]` |
| `test_nstep_uniform.py` | **H 层 n 步机制的跨项目一致性**（六个 TD3 方法）：绕过 `__init__` 装配最小实例 + 桩 memory, 对每个项目真跑 `observe_jump`/`_emit_nstep`/`_push_transition` —— `nstep=1` ⇒ 逐条直接落库（历史 1-step 同形）；`nstep=3` 滑窗喂 6 块 ⇒ 恰好 6 条、`r_sum=Σγ^k r_k`、起点覆盖每个块（stride-1）、`boot=γ^n`、窗口内含 done ⇒ `boot=0` 且存 done=1、episode 末 pending 清空；两个 PRLE 项目的 **context 顺序拼接**、非 PRLE 不传 context。**不需 MATLAB** | `python agentutil/test_nstep_uniform.py` |
| `port_nstep.py` | **一次性迁移脚本（已执行完毕，保留作审计记录）**：把 td3per 的 H 层 n 步机制同步到其余五个 TD3 方法（`*_ddpg.py` 加 `nstep`/`target_agg` kwarg + init 块 + target 聚合 + 按条 `_disc` + `_push_transition`/`_emit_nstep`/滑窗 `observe_jump`；`*_main.py` 加 `--nstep`(默认5)/`--target_agg`(默认min)、`--rmsize` 改 20000、构造接线；两个 PRLE 项目额外做 context 拼接 + `relabel_all(discount=γ)` 按块 γ 加权）。作用域解析支持类与方法, 每个变换要求"恰好命中 1 次"否则不写盘; 默认 dry-run | `python agentutil/port_nstep.py [--apply]` |
| `test_viol_derate_uniform.py` | **越界乘性惩罚的跨项目一致性**（六个 TD3 方法）：helper **逐位同源** + 六个包装函数（`Cal_Rew_RB`/`Cal_Rew_RLRB`/`Cal_Rew_PRRBB`）的 `Δr1 == turbine_profit·(f−1)` 与记录列不变 + **端到端（桩 env 真跑 `nine_case_test`）**：`c=0`/`c=5` 的 `profit`/`profit_econ`/`dsoc` **逐位相同**、分解残差**与 c 无关**（`wT=wB=1` 的两个项目残差 ≡0；四个 PRRRB 系项目因收益重加权残差非零，属正常）+ 六个 main 的静态接线。**不需 MATLAB** | `python agentutil/test_viol_derate_uniform.py` |
| `port_viol_derate.py` | **一次性迁移脚本（已执行完毕，保留作审计记录）**：把 td3per 的奖励侧改动（helper 块**从 td3per 逐字提取**、r1 乘系数、包装函数透传、`nine_case_test` 分解三列、main 的 CLI/接线/df_test9 三列）同步到其余五个 TD3 方法。每个变换要求"作用域内恰好命中 1 次"，不满足则整体不写盘；默认 dry-run，`--apply` 才写 | `python agentutil/port_viol_derate.py [--apply]` |
| `flip_viol_derate_default.py` | **一次性脚本（已执行）：把 `--viol_derate` 默认从 5.0 改回 0（关闭）**。run14 实测该乘性罚有害（奖励电平被策略自身越界率绑架 ⇒ critic 高估约 2x ⇒ 正反馈），故只回退默认值、**保留机制/开关/全部测试**供日后受控实验；带断言、默认 dry-run。**另见 README 的 `--viol_derate` 条目里的失败模式** | `python agentutil/flip_viol_derate_default.py [--apply]` |
| `probe_action_symmetry.py` | 已训练 critic 的 ∂Q/∂a 动作对称性探针 | `python agentutil/probe_action_symmetry.py output/<run1> [run2 ...]` |
| `probe_value_scale.py` | 归一化两臂的学习曲线/隐层激活量级诊断 | `python agentutil/probe_value_scale.py` |
| `probe_l_fempty_point.py` | 用真实 run 数据验证 `F=∅` 参考集合是零宽单点（只有竭力值免费） | `python agentutil/probe_l_fempty_point.py [csv]` |
| `probe_l_trackerr_worst.py` | 定位 trackerr 最大的步并归因（下层惰速/带宽失配指纹） | `python agentutil/probe_l_trackerr_worst.py [csv] [topN]` |
| `probe_floris_yaw_structure.py` | **FLORIS 偏航环境（新方法）的功率结构与可行性探针**（**只读、不需 MATLAB**；需 `pip install floris`，实测 4.6.6）。物理口径逐项对齐 `WFSim_port.m`/`meshing.m` windTunnel：D=100 m、单排 5 台 5D 间距、ρ=1.2、a≡1/3 ⇒ `Cp=4a(1−a)²=0.59259`、`Cp_eff=0.56296`、`CT_eff=8/9`、无量定上限；`wind_directions=270`（FLORIS 用气象来向，270=沿 +x）。八节：①单机标定（隔离机组 vs 解析式，实测相对误差 **1.9e-16**）②均匀偏航族扫描（±5…±60°）③逐台串行精化求风场最优 ④带内可达功率区间（**削减能力**）⑤镜像对称/可重复性 ⑥调用耗时拆解 ⑦尾流模型×偏航损失指数敏感性 ⑧真实风速序列下的波动量级；另加 ⑨一天最优轨迹+速率限制代价 ⑩输运延迟代理 ⑪波动-能量折中 ⑫**引擎默认值的来源类型与杠杆**（读随包源码把默认值分五类：①论文标定 ②历史惯性值 ③代码里无引用 ④标准常数 ⑤数值折中；实测杠杆：`crespo` 换论文值 **−21.2 pp**、`turbine_grid_points` 1/2/3 = **+58.2/+29.0/+38.0%**（非单调 ⇒ 离散化特征）、`ka` ±8pp、`kb` ±4pp、`air_density` **−0.10 pp**）。**2026 实测（u=10，γ=0）**：单机 2.6529 MW（=解析），5 台总 **5.9918 MW**（同 u 的 WFSim 全体 a=1/3 为 3.4462 MW ⇒ 引擎绝对电平差 1.74×）；最优偏航 **+38~47%**（形态 = 上游 +16~25°、下游 0，正是"随尾流状态变化的斜坡"）；**带内最小出力只比 γ=0 低 2.5~3%**（最劣模式 = 只转末台）；`gauss` 挠曲模型在**全场同时 ≥45°** 时 `sqrt` 出 NaN（`wake_deflection/gauss.py:189`）⇒ env 必须守卫；GCH 默认三个开关里 `enable_transverse_velocities` 让 `P(+γ)≠P(−γ)` 约 +2%（关掉后**精确**为 0，但转向增益从 +43% 掉到 +10%）；jensen+jimenez 只给 **+17%**（模型依赖大）；单条调用 40 ms、批量 0.6 ms/条。口径开关 `--wind-shear`：**0.12（官方 `gch.yaml`/`default_inputs.yaml`，主臂）→ +37.4%；0 → +42.8%**（一阶项） | `python agentutil/probe_floris_yaw_structure.py [--skip-opt] [--skip-sens] [--skip-diag] [--skip-series] [--skip-day] [--skip-delay] [--skip-fluct] [--wind-shear 0.12] [--json output/x.json] [--log output/x.log]` |
| `probe_bsize_scan.py` | `--bsize` 扫描：用**固定样本量 vs 固定步数**两口径把"梯度估计更准"与"多学了几遍"分开（离线有监督代理目标） | `python agentutil/probe_bsize_scan.py`（约 6 min） |
| `test_yawflor_config.py` | **`yawflor` 新环境的"口径冻结契约"**（只读、不需 MATLAB）：把 `yawflor/README.md` §2 的口径表变成**可执行断言**（42 项：引擎/四模型名/三开关/求解器/机型表/布局/密度/切变/TI/动作域 + `table.power ≡ 解析 Cp 表` 逐点 0 偏差），任何改动在此响警报；外加 6 条**静默失效陷阱**检查（`reference_wind_height ≠ hub_height` 在切变下整场电平偏移；`air_density ≠ table.ref_air_density` 使功率静默 ×ρ/ρ_ref；功率表未覆盖 env 守卫 `[3.5,25]` 会被 `interp1d(fill_value=0)` 截成 0；单机标定；可重复性；数值锚 u=10/γ=0 = 5.9423 MW），并用 `--json` 落盘口径快照作复现性 artifact。**首跑当场抓出两件事**：① 我写错的版本号断言；② **真实口径陷阱** —— 官方 `wind_shear=0.12` 下隔离机组功率**不再**逐位等于轮毂风速解析式（比值 0.995555 = `u_eff/u_hub` 0.998516）：FLORIS 在转子盘上按 `u_eff=cbrt(mean(u³))` 平均切变剖面，而解析式用轮毂风速 ⇒ 该不变量只在 `shear=0` 成立，`shear=0.12` 下只能报告不能断言（表以 `u_eff` 为自变量，故表自洽性仍 0 偏差） | `python agentutil/test_yawflor_config.py [--json output/yawflor_config_snapshot.json]` |
| `probe_init_convention.py` | `fanin_init` 口径 A/B（**含 `stock` 纯噪声对照臂 + 逐 seed 配对差**）：`current`(现状) / `true_fanin` / `old_scoped`(把 fan_out 口径也套到 fc4/fc5) / `fix_fc1`，测 500/1500/6000 步 val R²·MSE·死 ReLU | `python agentutil/probe_init_convention.py`（约 10 min） |
| `audit_coverage.py` | 九方法覆盖审计（归一化/bias 接线逐项核对）+ H 层 η1 分析 | `python agentutil/audit_coverage.py` |

留在根目录的评测主脚本（非一次性工具）：`test_five_days.py`、`test_simple.py`、
`test_compare_projects.py`、`test_scenario.py`、`compare_*.py`、
`extract_training_wind_days.py`、`generate_day_sequence.py`。

## 研究笔记（非脚本）

| 文件 | 内容 |
|---|---|
| `floris_reproducibility_survey.md` | **11 篇已发表论文如何报告 FLORIS 仿真口径**（逐篇字段对照表 + 众数值 + "最常被省略的字段"排序 + "理想化常数 Cp 有无先例"专项 + "偏航速率约束的已发表做法"专项 + 读取/受阻台账）。结论摘要：配置表是例外不是惯例（读完的 4 篇经典论文里只有 1 篇给了 FLORIS 版本号，湍流模型/叠加模型/三开关/空气密度**从未被点名**）；模态取值 = Gaussian+GCH、TI 6%/10%、8 m/s、NREL 5MW + **真实查表曲线**、单排 3~5 台 **6~8D**、偏航上限 20°/25°；**常数 `Cp=4a(1−a)²`（a≡1/3）在 FLORIS 文献里查无先例**；**没有任何 FLORIS 论文在模型内部建模偏航速率**（约束一律在控制器层；WFCRL 原话"FLORIS 里 θᵢ 恒等于执行器当前值"）。用途：为 `yawflor/README.md` §2.3/§2.4 提供依据 |

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

* 需要环境：`C:\Users\yigul\anaconda3\envs\llmenhancedwsis\python.exe`（torch 2.12 CPU / numpy 2.4 / pandas 3.0）。
  FLORIS 相关脚本额外需要该环境里的 `floris`（**2026-09 实测装入 4.6.6**，一并带来
  scipy/matplotlib/numexpr/shapely/coloredlogs/pathos/attrs/pyyaml；numpy/torch/pandas
  未降级）。⚠ `floris` 在 import 时会 import matplotlib，后者要写 `~/.matplotlib`
  字体缓存 —— 在受限沙箱下会报 `Could not save font_manager cache ... Permission denied`，
  脚本已 `os.environ.setdefault('MPLCONFIGDIR', <ROOT>/output/_mplcache)` 规避。
* 除 `test_*` 与 `probe_*` 外，涉及真实训练的脚本需要 **MATLAB**（`env.py` 启动
  `matlab.engine`）；单元测试**不需要** MATLAB，可在任何机器上跑。
* **例外**：`probe_wfsim_memoryless.py` 必须启 MATLAB（它测的就是 MATLAB 侧的
  `WFSim_port`）；其余 `probe_*` 只读日志/checkpoint，不需要。
* **MATLAB 需要工作区之外的写入权限**：MATLAB 启动时要写自己的偏好/临时目录，在
  受限文件沙箱（workspace-write）内会以
  `Fatal Startup Error: System Error: File system inconsistency` 失败（约 1 s 内退出），
  `matlab.engine.start_matlab()` 则表现为 `UnicodeDecodeError`（引擎把 GBK 的中文启动
  横幅按 UTF-8 解码，**掩盖了真实错误**）。此时需以更宽的权限运行，或改在普通终端里跑。
  仅读日志/checkpoint 的测试不受影响。
