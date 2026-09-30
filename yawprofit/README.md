# YAWPROFIT — **只调度 FLORIS 的 5-min 发电利润最大化智能体**

> 由 `yawflor` 派生（"从 yawflor 里改出来一个"）。目的不是提出新算法，而是给
> "**yawflor 的学习不稳定**"找一个**最小对照臂**：把层次结构（上层 H + 下层 RL BESS）、
> 电池耦合、可行域硬裁剪、约束对偶**一次性全部摘掉**，只留

> **每 5 min 决策一次的 5 维偏航 + "发多少电就卖多少钱"的纯利润奖励。**

---

## 1. 为什么要有这一臂

`yawflor` 的诊断链已经排除了不少东西（双通道对偶 E1、κ/软通道 E2a/E2c），但仍有多条
互相纠缠的通道同时作用在同一个奖励上：

| 可能的"不稳定来源" | 在 yawflor 里 | 在 yawprofit 里 |
|---|---|---|
| 下层 RL BESS（1 维残余动作 + 每块 10 次更新） | 有 | **没有** |
| 上层 → 下层的信息通道（SOC 目标信号 `a6`） | 有（动作第 6 维） | **没有**（动作只有 5 维） |
| RBB 硬裁剪 / 可行窗口 / `unfeasible_gap` | 有 | **没有** |
| 块末 SOC gap 罚（`W_GAP`，进 r1） | 有 | **没有** |
| 电池收益与退化（`bess_profit`、`490 元/MWh`） | 有 | **没有**（`P_G ≡ P_tur`） |
| 波动成本通道 `c1` + λ 对偶上升 | 有 | **c1 ≡ 0 ⇒ λ = 0，通道关闭** |
| 观测里的 SOC 与上一分钟电网功率 | 有（4 维基） | **没有**（2 维基） |

于是只剩一个问题：**"在 FLORIS 上做 5 min 级偏航调度、只最大化发电利润"这件事，
本身稳不稳？**

* 若这一臂**稳定** ⇒ 不稳定来自被摘掉的电池/层次/约束机制（回到 yawflor 逐条复现）；
* 若这一臂**仍不稳定** ⇒ 问题在更底层（奖励尺度、TD3+PER 的超参、偏航动作带的
  边界解、探索噪声、`--nstep` 自助传播等），与 BESS 无关。

---

## 2. 口径（单一真源 = `yawprofit_myfunc.py`）

| 项 | 定义 | 备注 |
|---|---|---|
| **state** | `[pr/1000, u/10]` + 24h 相位特征 | `--phase_feat 1`（默认）⇒ **4 维**；**无 SOC、无上一分钟功率** |
| **action** | 5 维无量纲偏航指令 `n ∈ [−1, +1]` | env 内 `γ = 25°·n`；**块级（5 步 = 5 min）固定** |
| **reward** | `r1 = pr·P_tur(u, γ)·dt − penalty_NN_H` | `dt = 1/60 h`，单位 = 元；`pr` 取当前分钟电价 |
| **cost** | `c1 ≡ 0` | 无波动约束、无可行域缺口；`λ = 0` ⇒ actor 目标 = `−Q_p` |
| **越界罚** | `penalty_NN_H = 100 · Σ max(|n_j|−1, 0)` | 满带越界 = 200 元（与 yawflor 等值）；**不封顶** |
| **物理** | `yawflor.yawflor_turbine`（FLORIS 4.6.6 + GCH） | **口径表已冻结 ⇒ 不复制、不分叉** |
| **电网功率** | `P_G ≡ P_tur`（无电池） | `P_F = |ΔP_tur|` **只作诊断**，不进任何奖励/成本 |

**state 的充分性**：FLORIS 是**稳态**模型（无时间记忆），执行器动态（速率限制 / 一阶延迟 /
运输延迟）默认全关 ⇒ `(pr, u)` 已经是这个 MDP 的充分统计量，奖励只依赖
`(pr, P_tur(u, γ))`。丢掉的 `E` 与 `P_G_prev` 在两个项目里的信息价值本来也不存在
（`P_G_prev` 在无电池时 = 上一分钟的风机出力，而它由 `u` 与块内动作决定）。

---

## 3. 与 `yawflor` 的差异（逐项）与文件来源

### 3.1 差异表

| 项 | `yawflor` | `yawprofit` |
|---|---|---|
| 动作维 | 6 = 5 偏航 + **1 SOC 目标信号** | **5 = 只偏航** |
| 观测基维 | 4 = `[pr, u, E, P_G]` | **2 = `[pr, u]`** |
| 下层 agent | RL BESS（TD3+PER+对偶+残余动作） | **无** |
| RBB | `RuleBasedBESS`（可行窗口 + 硬裁剪） | **无** |
| r1 | `pr·P_G·dt − 490·max(−pb,0)·dt − penalty` | `pr·P_tur·dt − penalty` |
| c1 | `P_F + 20(P_F−3)⁺`（+ 可行域缺口） | **≡ 0** |
| 块末 gap 罚 | `10·max(0,|soc_target−SOC|−0.1)` | **无** |
| λ 对偶 | EMA 实际块成本（每块一次） | **无**（`λ ≡ 0`） |
| 更新频率 | H 每块 2 次 + **L 每块 10 次** | 每块 2 次 |
| 九例测试 | H + L + RBB（含 SOC 结算列） | **只 H**，另附**无偏航参考臂** `profit_nominal` |
| `--viol_derate` 乘性罚 | 有（默认 0 = 关） | **不设**（与默认 0 逐位等价） |
| 随机 warmup 臂的动作带 | `random_action()` 给 `U(0,0.5)`（旧轴向因子带，**yawflor 里是死代码**） | `U(−1,+1)`（偏航带） |

**刻意保持相同**（保证"只有口径在变"）：TD3（twin critics + 延迟策略更新 + target
smoothing）、PER（`PER_buffer.py` 原样）、n 步回报滑窗（`--nstep 5`）、块级 `observe_jump`
接口、warmup 期 `WarmupSchedule`（起点 0、每块 ±0.2、带 `[−1,+1]`）、critic 输出层 bias 的
价值基线初始化、`--value_scale 1000`、24h 相位特征、CSV 增量落盘、每 2000 步 checkpoint、
每 480 环境步一次九例测试、`--hidden1 16 / --hidden2 100 / --rate 0.01 / --prate 1e-4 /
--bsize 128 / --rmsize 20000`。

### 3.2 文件来源

| 文件 | 来源 | 说明 |
|---|---|---|
| `yawprofit_env.py` | **新写** | `WindTurbineYaw(WindBatteryYaw)`：只覆写观测与步进；`GetBatteryActions()` 抛 `NotImplementedError` |
| `yawprofit_myfunc.py` | **新写** | 2 维观测归一化 / 5 维 `CheckAction` / 纯利润 `Cal_Rew_WT` / 无 BESS 九例测试 |
| `yawprofit_main.py` | **新写**（结构照搬 `yawflor_main.py`） | 单层训练循环；无 RBB / 无下层 / 无对偶 / 无 gap 罚 |
| `yawprofit_ddpg.py` | **克隆** `yawflor_ddpg.py` | 差异只有两处：加 docstring、`random_action()` 用偏航带（见 3.1 最后一行） |
| `yawflor.yawflor_turbine` | **import（不复制）** | FLORIS 口径表 / `γ = 25°·n` / 功率表 —— 冻结资产，分叉即失去可比性 |
| `yawflor.yawflor_env.FlorisPhysicsYaw` | **import（不复制）** | 逐块预取 + NaN 守卫的后端 |
| `yawflor.yawflor_myfunc` | **import（不复制）** | 电价日程 / 24h 相位表 / 风况-电价分档 / 九例"代表性时刻" |
| `hsrl_warmup` | import | 偏航带 `YAW_*` 与 warmup 调度器（单一真源） |
| `hsrl_agent_init` / `hsrl_cost` | import | critic bias 价值基线 / 值函数回归空间缩放 |

---

## 4. 运行

```bash
# 契约测试（37 项，不需 MATLAB，冷启动约 1 min）
python agentutil/test_yawprofit_nobess.py

# 短程 smoke（约 3 个 episode = 4320 环境步）
python yawprofit/yawprofit_main.py --env YAWPROFIT_SMOKE --max_steps 4320

# 正式训练（默认 1000 个 episode = 1.44M 环境步；输出 output/YAWPROFIT-runN）
python yawprofit/yawprofit_main.py
```

常用开关（与 yawflor 同名同义）：

```bash
--nstep 1                 # 复现 1-step 回报的对照臂（默认 5）
--phase_feat 0            # 关掉相位特征 ⇒ 观测 2 维
--critic_bias_init none   # 关掉 critic 价值基线初始化
--warmup_sched random     # warmup 用随机动作而非 ±0.2 游走
--wt_noise 0.1 / --epsilon 2000   # 探索噪声幅度 / 衰减长度（决策次数）
```

> ⚠ 控制台是 GBK（Windows 默认）而 stdout 被重定向时，本仓库文案里的 `−`/`γ`/`⇒`
> 会触发 `UnicodeEncodeError`。`yawprofit_main.py` / 本目录的契约测试已在开头做
> `sys.stdout.reconfigure(errors='replace')` 降级（不动编码），因此
> `python ... > log.txt` 不会再被一句日志打断。

---

## 5. 该看什么（学习稳定性的判读口径）

| 文件 / 信号 | 看什么 | 稳定长什么样 |
|---|---|---|
| `df_yawprofit.csv`（逐步） | `r1`、`profit_step`、`penalty_NN_H`、`action_excess`、`yaw_deg_mean`、`ptur` | 块级 `r1` 的**滚动中位数**单调不降；`action_excess` 随 `ε` 衰减后趋于 0 |
| `update_stats_H.csv`（每块） | `td_p_mean`、`q_p_mean`、`target_p_mean`（**原始口径**） | `q_p_mean ≈ target_p_mean`（自洽比 → 1）；`td_p_mean` 不随训练步增大 |
| `df_test9.csv`（每 480 步） | `profit` vs **`profit_nominal`**（无偏航参考臂）、`gain`、`yaw_deg_mean` | `gain > 0` 且**不再漂移**；`yaw_deg_mean` 收敛到一个稳定的形状而非在 0°↔25° 间来回 |
| `df_episode_yawprofit.csv` | `episode_profit`、`episode_energy_MWh`、`episode_yaw_deg_mean` | episode 间波动收窄 |
| TensorBoard | `episode/profit`、`test9/overall/gain`、`action/a_H_*`、`r1` | 与上表一致 |

**关键判据（本项目独有）**：`gain = profit − profit_nominal` —— 无偏航基线是**同一条
物理路径**（零策略下 `gain` 必须**恰好为 0**，见 §7 的断言），所以它可以直接回答
"**策略到底有没有比不转向更好**"，这是纯利润臂唯一有意义的性能指标。

---

## 6. 已知口径细节与坑（诚实清单）

1. **越界罚记在"带噪动作"上**（与 yawflor 同）：`--epsilon` 期的探索噪声可以直接把
   `|n|` 推出 1（初始 bias 是 `U(−1,+1)`，很多维本来就在边缘），于是**探索期每步都
   在付罚**（实测早期 ~12 元/步 ≈ 块收益的 20%）。这是**两个臂共同的**机制，因此
   不破坏对照；但它是"奖励电平随探索而变"的一个来源。候选改动（未做）：把罚改成
   算在**确定性动作**（噪声之前）上 —— 那才是梯度能控制、也才是罚想塑造的量。
2. **`c1 ≡ 0` 但 cost 通道仍在**：`critic_c/critic_c2` 照常更新（回归到 0），`λ = 0`
   使 actor 目标精确等于 `−Q_p`。保留它们的唯一理由是"与 yawflor 的 ddpg 逐字同源"；
   代价 = 每批多算两张网。PER 优先级 = `|TD_p|/mean_p + |TD_c|/mean_c`（自归一化，
   不受影响）。
3. **相位自检在每个电价段边界有 5 分钟不一致**：`env.CalPrice()` 的窗口是
   `count_price ∈ [1, 419]`（= 分钟 0..418）而 `day_price_schedule()` 的 LOW 是
   `m ∈ [0, 419]` ⇒ 每天在 419/659/1019/1259/1379 这 5 个分钟上差 1 档
   （5/1440 = 0.35%）。这是 `env.py` 与相位表的**历史口径差，yawflor 与其余 8 个
   项目完全相同**，本臂按原样继承（改 `env.py` 会让所有历史 run 不可比）。
4. **`P_G` 的记账时点与父类不同**：`WindBattery.step` 的 `P_G(t+1)` 用的是**传进来**的
   `pw_tru`（步初的风机出力 ⇒ 滞后一步），那是电池"步内先动"的遗留约定。
   无电池时滞后没有物理含义，本环境取 `P_G(t+1) = P_tur(t+1)`；`P_F` **只作诊断**
   （`c1 ≡ 0`），reward 路径一字不差（`r1` 一直用当时的 `P_tur`）。
5. **`df_test9.csv` 没有 SOC 相关列**（`dsoc`/`soc_track_err`/`profit_econ`…）：
   不编造 NaN 列。因此 `agentutil/test9_compare.py` **不适用于本臂**（它的口径含电池
   结算），跨臂比较请只用 `profit` / `gain` / `yaw_deg_mean`。
6. **父类残留**：`WindBattery.reset()` 仍会写常数 `self.E = 3.`，父类也仍带着
   `GetBatteryActions` 等方法（`env.py` 是 9 个方法共用的，不改）。本环境**从不读写
   `self.E`**，并把 `GetBatteryActions()` 改成抛错 ⇒ "电池通道已抹去"是**结构性事实**，
   契约测试里有断言。

---

## 7. 验证与回归

`agentutil/test_yawprofit_nobess.py`（37 项断言，全部通过）：

| 组 | 判据 |
|---|---|
| ① state | 观测 3 维且**无 SOC**；`[pr,u]` 与 `WindBatteryYaw` 逐位相同；第 3 维 = 父类的 `P_G` |
| ② 步进 | 30 步同动作下与 `WindBatteryYaw(act_bat≡0)` **逐位相同**；该环境 SOC 全程不动 ⇒ "抹掉电池" = "删掉恒零项"，物理路径未变 |
| ③ action | 5 维 / 带 `[−1,+1]` / 罚 = 100×Σ越界且不封顶 / **6 维直接报错**（不静默截断）/ 与 yawflor 的 5 维 `CheckAction` 逐位相同 |
| ④ reward | `r1 = pr·P_tur·dt − penalty`；**`c1 ≡ 0`**；`details` 里没有 `bess_profit`/`degradation`/`soc_target`/`unfeasible_gap`/`soc_deviation` |
| ⑤ 电池通道 | `GetBatteryActions()` 抛 `NotImplementedError` |
| ⑥ 静态结构 | `yawprofit_main.py` / `yawprofit_myfunc.py` 的 **AST 标识符**里不存在 `RuleBasedBESS`/`rbb_system`/`soc_target`/`gap_penalty`/`agent_L`/`update_lam_ema`/`dual_update`/`hsrl_soctarget`/`matlab`（docstring 提到这些名词不算违规） |
| ⑦ 观测维度 | `nb_states = 2 + PHASE_DIM[mode]`（2/4/7）；`Normalize` 与 yawflor 前 2 维逐位相同 |
| ⑧ 九例测试 | 零策略（`n≡0`）下 **`gain = 0`**（参考臂与实跑同路径 ⇒ 它可信）、`Σr1 = profit`、无参考臂时记 NaN |
| ⑨ 克隆的 agent | `SP_TD3(nb_states=4, nb_actions=5)` 可构造；`random_action()` 落在 `[−1,+1]`（**不是**克隆源的 `[0,0.5]`） |

---

## 8. 下一步候选实验（如果这一臂仍不稳定 / 稳定）

* **仍不稳定** ⇒ 在**这一臂内**逐一排除（每个都是一次单变量 A/B）：
  1. `--nstep 1`（自助传播深度 50 → 1）：看是不是 n 步 + PER 的组合在放大 TD 噪声；
  2. `--critic_bias_init none`：价值基线错了会让早期 Q 跑偏；
  3. 越界罚改算在**确定性动作**上（§6.1）—— 去掉"奖励电平随探索而变"；
  4. `--wt_noise 0` / `--epsilon 500`：把探索单独摘出来看；
  5. `--phase_feat 0`：相位特征若与电价时钟不同步会成为噪声源。
* **稳定** ⇒ 回到 `yawflor`，按"上层看到的奖励"逐项加回：先加回 `c1`+λ（波动约束），
  再加回块末 gap 罚，最后加回下层 RL —— 每一步都看 `gain` 与 `td_p_mean` 的自洽比。
