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

配套诊断脚本（`agentutil/`，已入库；§9 的实测就是它们跑出来的）：

```bash
python agentutil/analyse_yawprofit.py output/<run>                     # 块级奖励/越界/九例/critic 自洽/动作分布
python agentutil/probe_action_state_sensitivity.py output/<run>        # 真实状态回放 → 每维"状态敏感度 sd" + 越界率
python agentutil/probe_action_state_sensitivity.py output/YAWFLOR_E2C-run29 --layout yawflor   # 同口径对照 yawflor
```

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

---

## 9. 首轮实测（2026-10-01，`--max_steps 4320` = 3 episode，CPU）

**运行**：`output/YAWPROFIT_SMOKE3-run32`；**诊断脚本**：
`agentutil/analyse_yawprofit.py`（CSV 轨迹）、
`agentutil/probe_action_state_sensitivity.py`（把真实状态回放给各 checkpoint 的 actor，
**不加噪声**）。

### 9.1 没崩，但也没有可测的进步

* **test9**（固定 9 例场景，跨时间可比）：`profit` 3528~3594 元/45 步，`profit_nominal`
  恒 **3384.8**，`gain` = **+143…+209**（+4.2%…+6.2%），`pf_viol` 全程 **0**；
  8 个评测点（step 480…3840）**看不出趋势**，都在 +5% 上下摆动。
  分档看 gain：wind HIGH 时最大（HIGH×HIGH 81 元），wind LOW 时几乎为 0。
* **critic 自洽比** `q_p/target_p` = **1.000 ± 0.012**（末 10%），`relTD` 0.05~0.10
  ⇒ **值函数没有发散**（但基本停在 bias 初始化给的水位上，见 `hsrl_agent_init` 的
  "水位时间常数 ≈ 2~3 个 run 长度"）。

### 9.2 唯一的非利润项（动作带罚）曾一度吃掉**全部**收益

块级轨迹（每 20 块一桶，`block 0..863`）：

| 块段 | `action_excess`/步 | 块内罚 | 块内收益 | **罚/收益** | 块 r1 |
|---|---|---|---|---|---|
| 0–180（warmup + 早期） | 0.000–0.029 | 0–18 | 100–200 | <20% | +73…+181 |
| 200–280（高价高风日） | 0.001–0.011 | 0–6 | 690–1310 | <1% | +681…+1307 |
| **280–520** | **0.06→0.38** | **30–190** | 70–600 | **28%…208%** | **−93…+535** |
| 540–700 | 0.056→0.006 | 28→3 | 137–1010 | 18%→0.3% | +135…+1007 |
| 700–863 | 0.002–0.043 | 1–22 | 480–1570 | <4% | +205…+1571 |

即：**块 280–520 期间，"越界罚"超过了发电收益本身**（107%~208%），块 r1 一度为负。
这正是 `yawflor/README` 在 `--viol_derate`（run14）里记录过的
"**奖励电平被策略自身的越界率绑架**"现象 —— 在纯利润臂里**同样出现**，
只是本次是暂态的、~300 块后自己恢复了。

### 9.3 确定性策略其实是"**常数偏航**"，而且这个常数会漂出动作带

把 4320 个真实训练状态回放给 checkpoint 的 actor（无噪声）：

| checkpoint | 每维 sd（对状态敏感度） | 越界维 |
|---|---|---|
| step 2000 | 0.007–0.054 | dim1 = **+1.197**(100%)、dim3 = **+1.047**(100%) |
| step 4000 | 0.002–0.009 | 无（全在带内） |
| final(4320) | 0.003–0.015 | dim2 = **−1.061**(100%) |

* **sd ≈ 0.002–0.054**，而动作带半宽 = 1 ⇒ 策略对状态**几乎不敏感**（是常数偏航，
  不是"按风速/电价调度"）。test9 的那 +5% gain 全部来自这个**常数偏移**。
* 常数本身在慢速游走并**穿越带边界**；越界的维被 env 夹回 ±1
  （有效策略 = 贴着边界饱和），于是 9.2 的罚项就出现了。

机制（可直接量）：`model.Actor` 把 `fc4.weight` 初始化到 ±3e-3，而
`init_wt_action_bias` 把 `fc4.bias` 放到 U(−1,1) ⇒ **初始动作 ≈ 常数 bias**。
实测末点 `|fc4.weight|` = mean 0.017 / max 0.082，而且 **4000 步时比 2000 步还小**
（0.014 vs 0.019）⇒ 状态通路没有长起来。往前看，隐层已经把输入方差压掉两个数量级：

    输入 sd [0.28, 0.32, 0.71, 0.71] → fc1relu 0.099 → fc2relu 0.016 → fc3relu 0.029
    → 动作 sd [0.003, 0.009, 0.013, 0.015, 0.005]

### 9.4 同一现象在 `yawflor` 上逐条复现（⇒ 与 BESS/层次/对偶无关）

同架构（hidden 16/100）的 `output/YAWFLOR_E2C-run29`（7101 步）：

| checkpoint | 每维 sd | 越界 |
|---|---|---|
| 2000 | 0.013–0.049 | 无 |
| 4000 | 0.012–0.034 | 无 |
| 6000 | 0.013–0.082 | dim3 = −1.055（**64%**） |
| final(7101) | 0.015–0.105 | dim3 = −1.195（**100%**） |

`|fc4.weight|` mean 0.013–0.019（同样没长起来）。对照 64/128 口径的
`YAWFLOR_E1A-run24`：sd 在 4000→6000→8000 步长到 0.10→0.44→0.78，越界率一度
达 55%~73%，10000 步后回落，14000~18000 步再次越界（5%~47%）——**反复出入动作带**。

> **结论**：`yawprofit` 复现了 yawflor 的两条主症 —— ① **策略对状态不敏感**
> （输出层常数项主导）、② **常数项漂出动作带并把奖励电平搅乱**；两者都不需要
> BESS 就能出现。⇒ 下一步应当在这一臂上先解决 ① ②，再谈"加回电池"。

### 9.5 决策偏好（未执行，等用户定）

| # | 单变量 A/B | 期望 | 依据 |
|---|---|---|---|
| 1 | `--wt_bias_init 0`（初始 bias 回到 PyTorch 默认 U(±0.25)） | 初始 `|n| ≤ 0.25 + 噪声 0.2 < 1` ⇒ **结构性不越界**，奖励电平恢复平稳 | 9.3 的常数项就是越界源；`fc4.weight` 仍是 ±3e-3 不变，**只动常数项** |
| 2 | `--warmup_wt_lo -0.3 --warmup_wt_hi 0.3` | warmup 与初始策略同分布且都在带内更窄处 | `hsrl_warmup` 的"起点=单机最优、带宽 10%"原则不变 |
| 3 | 若 1+2 后策略**仍状态盲** | 问题在 `model.Actor` 的 fan_out 初始化把状态信息压掉（fc2 16→100 处实测压 6 倍）⇒ 需要改 `model.py`（多项目共享，必须先做 A/B） | 9.3 的隐层方差表 |

### 9.6 已执行的 A/B：`--wt_bias_init 0`（1440 步 = 1 episode，其余全部相同）

| | 默认 `--wt_bias_init 1`（bias ~ U(−1,1)） | `--wt_bias_init 0`（bias ~ U(±0.25)） |
|---|---|---|
| `action_excess`（逐块均值） | 0.000 → 0.029 → 0.040（块 40~140） | **0.0000（全部 288 块）** |
| 一整个 episode 的越界罚 | **2346.7 元**（元/块 0~18） | **0.0 元**（= 收益，逐位相等） |
| `yaw|γ|mean`（episode） | 14.47° | **4.76°** |
| test9 `gain` @step480 / @960 | **+203.7（+6.0%）** / +174.6 | **+24.7（+0.7%）** / +8.7 |
| test9 `yaw_deg_mean` | 15.16° / 15.22° | 3.54° / 2.60° |
| episode0 收益（训练日） | 113374.7 元（reward 111028.1） | 109764.0 元（reward 同） |

两条结论（都可直接支撑下一步）：

1. **动作带罚 = 纯初始化效应**：把 `fc4.bias` 从 U(−1,1) 收到 U(±0.25) 之后，
   越界量在 288 个块里**恒为 0**（罚 0 元，`episode_reward == episode_profit`）。
   ⇒ 9.2 那个"罚吃掉全部收益"的现象**不需要改奖励、不需要改算法**，改初始化即可消失。
2. **默认臂 test9 的 +5%~6% gain 主要来自"常数偏航"而不是"学会调度"**：
   小 bias 臂的 `gain` 只有 +0.7%（且随训练下降，因为策略仍近似 γ≈0），
   而默认臂的 gain 与 `yaw|γ|mean`（15° vs 3.5°）同步缩放；
   结合 9.3（确定性策略 sd ≈ 0.01）⇒ 那段性能是 `init_wt_action_bias` 抽到的
   **随机常数偏航**带来的，不是状态相关策略。
   ⚠ 单 seed 结论，需要多 seed 复核（随机常数偏航的 sd 本身就是 run 间方差的大头：
   每维 mean|n| 的期望 0.5 ⇒ mean|γ| 的 run 间波动可达数度）。


