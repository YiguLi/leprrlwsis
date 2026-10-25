# testPg —— `aifflor` 的「固定风机行为 + 标量降载率」对照实验臂

> 一句话：把上层策略的动作从 **6 维** `[a₁…a₅, goal]`（5 台机组的轴向诱导因子 + SOC
> 目标信号）换成 **2 维** `[rate, goal]`；风机行为钉死在
> `[0.2, 0.2, 0.2, 0.2, 0.33]`，策略只学**整场降载率** `rate ∈ [0,1]`。
> 其余（FLORIS 物理、下层 RL BESS、r1/c1/gap、r2/c2、λ 对偶、TD3/PER、warmup、
> 9-case 测试）全部沿用 `aifflor` 的默认分支。

---

## 1. 它回答什么问题

`aifflor` 的上层同时学两件事：

1. **逐台风机的尾流协同**（改变 5 维 `a` 的**形状**：上游降载留风、下游推高）；
2. **整场出力水平**（5 维整体抬高/压低）。

`testPg` 把 (1) 钉死在一个已知的好形状上，只让策略学 (2)。因此：

> **`aifflor − testPg` 的差值 = "逐机尾流协同 + 连续降载"相对"固定尾流形状 + 单标量降载"
> 的增益。**

这是一个**单因子消融/对照臂**，不是新物理能力：`testPg` 能产生的功率轨迹，`aifflor`
原则上也能产生（把 5 维固定成同一组值即可），区别在于**搜索空间**从 5 维形状 + 水平
缩成 1 维水平。它回答的是"学尾流形状值不值"。

---

## 2. 与 `aifflor` 的单因子差异

| 项 | `aifflor`（基准） | `testPg`（本臂） |
|---|---|---|
| 上层动作 | 6 维 `[a₁…a₅, goal]` | **2 维** `[rate, goal]` |
| 风机行为 | 由策略学 `a ∈ [0,0.5]⁵` | **固定** `[0.2, 0.2, 0.2, 0.2, 0.33]` |
| 风机出力 | `ptur = P_FLORIS(u; a)` | `ptur = rate · Pmax(u)` |
| `Pmax` | —（隐含在 a 里） | `Pmax(u) = P_FLORIS(u; [0.2,0.2,0.2,0.2,0.33])`，**每步现算** |
| 越界罚（机组维） | `10·Σ_j[(a_j−0.5)₊+(−a_j)₊]` | `10·[(rate−1)₊+(−rate)₊]` |
| 越界乘性罚 | `x = Σ 前 5 维越界量` | `x = rate` 越界量 |
| `goal` 语义/映射 | `soc_target_from_signal(goal)` | **同式**（同一个函数） |
| 下层 L（BESS） | TD3 + PER，残余动作 `Δ` | **逐字复用**（`aifflor_ddpg.SP_TD3`） |
| L 观测 | `[pg(t−1)/10, ptur/10, SOC/10, target/10, n_rem/5]` | **同式**（`ptur` 换成 `rate·Pmax`） |
| r1 / c1 / gap | `pr·P_G·dt − 490·max(−pb,0)·dt − pen` 等 | **同式**（只换 `ptur` 的来源） |
| r2 / c2 | `hsrl_soctarget` 单一真源 | **逐字复用** |
| 物理引擎 | FLORIS stock + `LEVEL_COEF` | **同**（`calibrated=False, power_scale=LEVEL_COEF`） |
| 训练中测试 | 9-case 45 步 | **同协议**，CSV 多 `rate/pmax/ptur` 三列 |
| 观测布局 | `[pr, u, E, P_G_last]`（+ 相位） | **同**（刻意不动，保持可比） |

**唯一被改变的自变量 = 上层动作参数化。** 其余模块能 import 的就 import，不复制第二份
口径（见 §5）。

---

## 3. 动作 / 功率 / 奖励口径

### 3.1 每个环境步

```text
a_fixed = [0.2, 0.2, 0.2, 0.2, 0.33]                  # 固定风机行为
Pmax(u_t) = P_FLORIS(u_t; a_fixed)                     # 每步现算（u ∈ [3.5, 25]，否则 0）
ptur(t)   = rate_k · Pmax(u_t)                         # k = 当前 5 步块
P_G(t)    = ptur(t) − pbrule(t)                        # 并网功率（env 原语义）
```

`rate_k` 在块首由上层 actor 决定，块内（5 步）恒定；`u_t` 每步变化 ⇒ `Pmax` 每步现算。
**注意**：`P_G` 仍含 BESS 的 `−pbrule`，不是"N/A"式的净风电；`rate·Pmax` 是**风机侧**
实际出力 `ptur`。

### 3.2 上层动作检查

```text
rate ← clip(rate_raw, 0, 1)
goal ← clip(goal_raw, −0.5, 0.5)

penalty_NN_H = 10·[(rate_raw − 1)₊ + (−rate_raw)₊]                  # 机组维 → rate 维
             + 10·[(goal_raw − 0.5)₊ + (−0.5 − goal_raw)₊]          # 信号维保持同式（默认开）
```

* `--pg_goal_pen 0` 可把第二项关掉，得到**严格**"penalty_NN_H 只罚 rate"的变体；
* 默认保留 goal 项，因为 `aifflor` 的 `penalty_NN_H` 本来就同时含信号越界罚，"按语义
  替换机组维"不应顺手删掉信号维；
* 与 `aifflor` 一样**不封顶**（保持越界区恒定恢复力）。

### 3.3 奖励

```text
r1 = pr·P_G·dt − 490·max(−pbrule,0)·dt − penalty_NN_H
   = pr·ptur·dt + (−pr·pbrule·dt) − 490·max(−pbrule,0)·dt − penalty_NN_H
c1 = [10·(P_F−3)₊ + 0.1·P_F] + W_GAP·unfeasible_gap        # 与 aifflor 同式
   + [块末] 10·max(0, |soc_target − SOC_end| − tol)
L: r2 = −min(clip_amount, 20);  c2 = trackerr [MW]         # 与 aifflor 同式
```

`turbine_profit = pr·ptur·dt` 中的 `ptur` 是 `rate·Pmax` ⇒ `rate` 直接进入经济奖励；
`rate` 的越界乘性罚（`--viol_derate`，默认 0 关闭）乘在 `turbine_profit` 上，与
`aifflor` 的 `f = exp(−c·x)` 同式。

---

## 4. 为什么固定动作取 `[0.2, 0.2, 0.2, 0.2, 0.33]`

它就是仓库 warmup 的**结构先验** `hsrl_warmup.WT_PRIOR = (0.20, 0.20, 0.20, 0.20, 1/3)`
（末台的 `1/3` 按用户口径写成 `0.33`）：

* `probe_a_farm_structure.py` 实测：**均匀族**峰在 `a = 0.20`（±5% 平台 `[0.15,0.30]`）；
* 最下游机组背风侧无机组 ⇒ 其功率与自身 `a` 的关系回到**单机** `C_p` 峰 ≈ `1/3`；
* 需要降载留风的是**上游** ⇒ 形状是单调上升斜坡（前四台 0.20，末台 1/3）。

⚠ 这是**单个代表点**，不是联合最优（联合最优随来流/尾流状态变化，是策略要学的函数）。
因此 `testPg` 是"**强消融**"而不是"朴素基线"：它把 `aifflor` 的搜索先验固定住，只去掉
形状自由度。若连这个先验也不给（例如全 0.33），差值会更大，但那混入了"先验质量"。

---

## 5. 文件与单一真源

| 文件 | 作用 | 与 aifflor 的关系 |
|---|---|---|
| `testpg_env.py` | `WindBatteryPG`：固定动作 Pmax + rate 缩放 + 块首预取 | 继承 `WindBatteryAIF`，**只覆写 `PowerOutput`**；新增 `Pmax()` / `prefetch_pmax()` |
| `testpg_myfunc.py` | 2 维 `CheckAction`、rate 越界乘性罚、`cal_rew_rb_pg`、`build_L_obs_pg`、`nine_case_test_pg`、bias 初始化 | 奖励/L 观测从 `aifflor_myfunc` / `hsrl_soctarget` **原样导入**；只复制必须改的 3 个函数 |
| `testpg_ddpg.py` | `SP_TD3_PG(SP_TD3)`：2 维动作的噪声 / 随机动作 / cost 裁剪 | TD3 本体、PER、对偶**一行不动**，只覆写 4 个动作相关点 |
| `testpg_warmup.py` | `PGWarmup`：`rate ~ U(0,1)` + goal 价格条件游走 | 复用 `hsrl_warmup` 的信号游走常量与公式；不生成 5 个风机维 |
| `testpg_main.py` | 训练入口（由 `aifflor_main.py` 的**默认分支**派生） | 删除 HSRL 分支；动作块、env/reward 调用、日志列改成 2 维 |
| `../agentutil/test_testpg_contract.py` | 契约测试（只读，不需 MATLAB） | 见 §8 |

**刻意保持的复用**（避免口径漂移）：

* 物理：`aifflor.aifflor_env.AifflorPhysics`（stock + 电平系数），`aifflor_turbine`；
* 下层：`aifflor.aifflor_ddpg.SP_TD3`（nb_actions=1，5 维 obs，残余动作）；
* r1/c1：`aifflor_myfunc` 的 `Cal_r1_continuous` / `c1_apply` / 全部常量；
* L 层：`hsrl_soctarget` 的 `build_L_obs` / `lower_reward` / `lower_cost_soc_track` /
  `rule_action` / 对偶；
* warmup 下层：`hsrl_warmup` 的 `WarmupBessRule`、`BessFillStats`、`fill_bess_transition`；
* 双对偶与值域缩放：`hsrl_cost` / `hsrl_soctarget`。

---

## 6. 训练协议（与 `aifflor` 对齐的部分）

* H 每 5 步决策一次；L 每步决策；H 观测 `[pr, u, E, P_G_last]`（+ 可选 24h 相位）；
* 每块 H 更新 `--utd_h` 次、L 更新 `--utd_l` 次；`--nstep 5`（滑窗 stride-1）；
* λ1/λ2 的 EMA 对偶、`d_block` curriculum、`--value_scale` / `--cost_scale`；
* 训练中每 480 环境步跑一次 9-case（`df_test9.csv`）；
* warmup 时长口径：默认 7200 步（5 个完整 episode）；warmup 期零网络更新；
* H 探索噪声：`rate` 维**恒定** `--pg_rate_noise`（默认 0.05，与 aifflor 的 WT 噪声
  "恒定不衰减"同口径）；`goal` 维 `--sig_noise_max → --sig_noise_min`；
* 输出层 bias 初始化：`[rate=1.0, goal=0.0]`（**起点不降载**，SOC 目标在容量中点）；
  可用 `--pg_rate_bias 0.5` 改成"半载起点"做敏感性；
* warmup 上层：`rate ~ U(0,1)` 每块独立覆盖，`goal` 按 300↑/580→/980↓ 价格游走
  （`--warmup_sig fixed` 可退回恒 0）。

---

## 7. 运行与产物

```bash
# 正式训练（默认 1000 个 episode = 1.44M 环境步；可用 --max_steps 截断做 A/B）
python testpg/testpg_main.py --env TESTPG --seed 1

# 短程冒烟（warmup 30 步、训练 60 步）
python testpg/testpg_main.py --env TESTPG-SMOKE --seed 1 --max_steps 60 --warmup 30

# 关键开关
--pg_rate_noise 0.05        # rate 探索半宽（恒定）
--pg_rate_noise_floor 0.0   # 可选：让 rate 噪声随 ε 衰减
--pg_goal_pen 0             # 只罚 rate（严格按"penalty_NN_H 只罚 rate"）
--pg_bias_init 0            # 不覆写输出层 bias（PyTorch 默认起点）
--pg_rate_bias 0.5          # 起点半载（默认 1.0 = 不降载）
--pg_fixed_action 0.2,0.2,0.2,0.2,0.33   # 固定风机行为（默认 = 结构先验）
```

产物在 `output/TESTPG-runN/`：

| 文件 | 内容 |
|---|---|
| `df_testpg.csv` | 逐环境步：`rate / goal / Pmax / ptur / r1 / c1 / P_F / SOC / SOC target / λ / pbrule …` |
| `df_episode_testpg.csv` | 逐 episode 汇总 |
| `df_test9.csv` | 训练中 9-case（45 步）指标，含 `rate / pmax / ptur` |
| `update_stats_H.csv` | H 层 TD/Q/batch 统计 |
| `train.log` | 逐行详细日志（控制台只留启动行与汇总） |
| TensorBoard | scalar tag 与 aifflor 同名（`r1/c1/test9/*`），便于叠图；方法身份靠 run 名 |

TensorBoard 中 testPg 新增：`env/Pmax`、`env/ptur`、`action/rate_step`、
`action/goal_step`、`action/rate_original|applied`、`action/goal_original|applied`、
`test9/overall/{rate,pmax,ptur}`。

---

## 8. 对照实验怎么做（建议协议）

### 8.1 配对与重复

* **同一 seed** 跑 `aifflor` 与 `testPg`；两者的环境随机流（固定日序列、FLORIS 确定性）
  相同，但**动作维不同 ⇒ 探索噪声流不同**，轨迹不可能逐位配对。
* 因此建议 **≥3 个 seed**，报告 `profit_econ`、`c1`、`P_F>3` 步数、`SOC` 跟踪误差的
  **mean ± sd**，而不是单条曲线下结论。
* 对比窗口：同 step 的 test9（固定场景、配对）比训练期日收益噪声小。

### 8.2 指标（与 aifflor 逐列同名）

* 经济：`profit` / `profit_econ`（= `profit + ΔSOC·580`）；
* 波动：`c1`、`P_F`、`pf_viol`、`window_violation_count`；
* SOC 跟踪：`soc_track_err`、`gap_pen`、`SOC` 轨迹；
* 动作：`rate` 分布（均值/分位）、`Pmax`、`ptur`、`rate` 沿时间的行为（是否在高风/高价
  降载、是否用降载平抑波动）；
* 训练健康：`lam1/lam2`、`ema_c1_H`、`Qbar`（`update_stats_H.csv`）。

### 8.3 推荐消融

| 消融 | 命令/改动 | 回答 |
|---|---|---|
| 只罚 rate | `--pg_goal_pen 0` | goal 越界罚是否影响主结论 |
| 中性起点 | `--pg_rate_bias 0.5`（或 `--pg_bias_init 0`） | 结论是否依赖"起点=满发" |
| 探索幅度 | `--pg_rate_noise {0.02,0.05,0.1}` | 结论是否只是探索不足 |
| 衰减探索 | `--pg_rate_noise_floor 0` | 恒定噪声的必要性 |
| 先验质量 | `--pg_fixed_action 0.33,0.33,0.33,0.33,0.33`（或其它形状） | 差值中有多少来自先验 |
| 不降载 | 固定 `rate ≡ 1`（`--pg_rate_bias 1.0 --pg_rate_noise 0`） | 降载本身的价值 |

---

## 9. 契约测试与不变量

```bash
python agentutil/test_testpg_contract.py
```

覆盖：

1. `ptur = rate·Pmax`（rate=0/0.25/0.5/1 逐位）；rate=1 与 `aifflor` 固定动作**同值**；
2. reset 满发（`P_G_prev = Pmax`）、5 维标称路径、`u` 围栏不调引擎；
3. 预取缓存与直算逐位一致、无 NaN；
4. 2 维 `CheckAction` 的 clip/罚、`--pg_goal_pen 0`、1 维 L 委托；
5. `rate` 越界乘性罚只算 rate、`coef=0 ⇒ f=1.0`；
6. `cal_rew_rb_pg` 的 `r1/c1/details` 与 `aifflor_myfunc.Cal_Rew_RB` **逐位相同**；
7. L 观测与 `hsrl_soctarget.build_L_obs` 逐位相同；
8. `SP_TD3_PG` 的 2 维随机动作/噪声/target smoothing/cost 裁剪/bias；
9. `PGWarmup` 覆盖 [0,1]、goal 在带内、`random` 返回 None；
10. 9-case 可跑通且 `ptur ≈ rate·Pmax`；
11. 不 import `matlab`。

---

## 10. 已知局限 / 注意事项

1. **`test9` 的历史协议**：`aifflor` 的 9-case 在 `E = 5.0` 起步（> `max_E = 0.5`），且
   L actor 的残余动作在该测试里直接当命令用。本臂**原样保留**（为了与 aifflor 的
   `df_test9.csv` 逐曲线可比），因此 test9 的 L 指标**不应**单独解释为"策略质量"；
   要看 L 层请用训练期 `df_testpg.csv` 的逐步行或修好协议后再比。
2. **动作空间不是新增能力**：testPg 的功率轨迹是 aifflor 动作空间的子集；差值只衡量
   "学习 5 维形状"的收益，不是"testPg 物理更强"。
3. **固定先验是强先验**：`[0.2,0.2,0.2,0.2,0.33]` 来自 warmup 的结构先验；若换成更差
   的先验，testPg 会显著变差，差值会混入先验质量。做敏感性时必须固定同一先验。
4. **`test_five_days.py` 尚未接线**：该脚本按 `actor_6.pkl` / 6 维动作硬编码；testPg 的
   H 权重是 `actor_2.pkl`。当前评测入口 = 训练中 9-case + `df_*` 日志；需要 5 天正式
   评测时，应扩展 `test_five_days.py` 的动作分派（读 `actor_2.pkl`、用
   `WindBatteryPG`、把 `rate·Pmax` 接到 `ptur`）。
5. **rate 的观测**：H 观测里没有显式 `rate`；策略从 `P_G_last`、`u`、`SOC` 推断。这是
   刻意保持与 aifflor 同观测布局；若要让策略直接看 `rate`，需要扩观测维（会破坏
   单因子对照）。
6. **`Pmax` 缓存**：块首预取 `u_{t+1..t+5}`（同 aifflor 的 `PREFETCH_K=5`），块内命中；
   `Pmax` 只是 `u` 的确定性函数，缓存不改变数值（契约测试断言）。首步/跨 episode 仍有
   少量直算，`env.summary()` 会打印批量/单条/命中统计。
7. **FLORIS 版本**：与 aifflor 相同，使用 `calibrated=False` 的 stock 尾流参数 +
   `LEVEL_COEF=1/1.6696`；换 FLORIS 版本或改该系数会改变 `Pmax` 的绝对电平，但不改变
   testPg 的动作语义。

---

## 11. 与 aifflor 的差异清单（可审计）

| # | 位置 | 改动 |
|---|---|---|
| 1 | `testpg_env.py` | 新增 `WindBatteryPG`：`PowerOutput = clip(rate,0,1)·Pmax(u)`；`Pmax` 用固定动作；`prefetch_pmax` |
| 2 | `testpg_myfunc.py` | `CheckAction` 增加 2 维分支（rate [0,1] + goal [-0.5,0.5]）；1 维委托 aifflor |
| 3 | 同上 | `violation_amount_pg` / `violation_derate_pg`：x = rate 越界量 |
| 4 | 同上 | `cal_rew_rb_pg`：与 `Cal_Rew_RB` 同式，signal 取 `action[1]`，details 加 rate/pmax |
| 5 | 同上 | `build_L_obs_pg`：与 `build_L_obs_distilled` 同式，goal 取 `action[1]` |
| 6 | 同上 | `nine_case_test_pg`：2 维 H + `ptur=rate·Pmax`，CSV 加 rate/pmax/ptur |
| 7 | 同上 | `init_pg_action_bias`：输出层 bias = `[rate, goal]` |
| 8 | `testpg_ddpg.py` | `SP_TD3_PG`：nb_actions=2、amp=[0.5,0.5]、rate 恒定噪声、random_action、`_clip_box` 只夹 rate |
| 9 | `testpg_warmup.py` | `PGWarmup`：rate~U(0,1) + goal 价格游走（同 `sig_delta` 公式/常量） |
| 10 | `testpg_main.py` | H 动作块 2 维；`env.step(action[:2], action[2:])`；env/奖励/L 观测改用 testPg 版本；日志列/CSV/TB 改 2 维；删除 HSRL 分支；CLI 改 PG 参数 |
| 11 | 未改 | `aifflor_turbine` / `AifflorPhysics` / `SP_TD3`（L）/ `hsrl_cost` / `hsrl_soctarget` / `hsrl_warmup` 下层 / `rbb_rule_bess` |

---

## 12. 相关文档

* `aifflor/README.md` —— 基准臂的完整口径（引擎、标定、奖励、HSRL 分支、run 记录）；
* `hsrl_warmup.py` —— `WT_PRIOR = (0.20,0.20,0.20,0.20,1/3)` 的来源（probe 证据见注释）；
* `agentutil/test_testpg_contract.py` —— 本臂的可执行不变量；
* `agentutil/README.md` —— 契约测试的运行前置（FLORIS 解释器守卫）。
