# `td3per` vs `aifflor`：除引擎（WFSim vs FLORIS）之外的实质差异

> 审计日期：2026-11-05 ｜ 口径来源：`td3per/td3per_main.py`、`td3per/td3per_myfunc.py`、
> `td3per/td3per_ddpg.py`、`aifflor/aifflor_main.py`（`train_hsrl`）、`aifflor/aifflor_hsrl.py`、
> `aifflor/aifflor_ddpg.py`、`hsrl_cost.py`、`hsrl_warmup.py`
> 目的：回答"两个方法除了环境引擎，还有哪些不一样"——尤其是**能不能拿它们的性能数字直接横比**。

## 0. 一句话

两者共用同一套 TD3+PER+拉格朗日骨架（`SP_TD3` 基本同源，网络/优化器/更新节奏一致），
但**上层第 6 维动作语义、下层动作语义、奖励/成本口径、λ 对偶口径、训练日程、测试协议**
全都不同 ⇒ **它们不是同一个 MDP 的两种实现，性能数字不可直接横比**。

## 1. 完全一致的部分

| 项 | 值（两者相同） |
|---|---|
| 网络 | 2026-11-04 起默认 v7：critic `cat(s,a)→400→300→400→1`；H actor `s→400→300→400→a`；L actor `16/100/16` |
| TD3 | twin critic、`policy_delay=2`、`policy_noise=0.1`、`noise_clip=0.3` |
| 折扣 | H 块级 `γ=0.95`、L 步级 `γ=0.98` |
| n 步 | `--nstep 5`、`--target_agg min` |
| 优化 | `--rate 0.01`（critic）、`--prate 1e-4`（actor）、`--tau 1e-3` |
| PER | `α=0.6`、`β=0.4`、`rmsize=40000` |
| 更新比 | `utd_h=4`、`utd_l=10`（每块） |
| warmup | 7200 步、零网络更新、`resolve_warmup` 同一实现 |
| 归一化 | `value_scale=1000`（r1）、`cost_scale=100`（c1）、λ 自动重标 |
| H 观测 | `Normalize_H(obs)` 4 维、`--phase_feat 0` |
| H 动作 | 前 5 维风机轴向因子 `[0,0.5]` |

## 2. 实质差异

### 2.1 上层第 6 维：完全不同的动作语义 ★

| | `aifflor --hsrl` | `td3per` |
|---|---|---|
| 第 6 维 | **块总放电量 `g` [MW·min]**，放电为正 | **SOC 目标信号 `a6∈[−0.5,0.5]`** |
| 映射 | 可行域 `[−C_max, +D_max]`（由块首 SOC + 额定功率算出，**不含波动约束**） | `soc_target = 3 + 5·a6 ∈ [0.5, 5.5]`（块级固定） |
| 越界罚 | `pen6 = 10 元/MW·min`（不封顶） | `pen = 10 元/单位`（a6 出 `[−0.5,0.5]`） |
| 语义 | "这一块放多少电" | "这一块把 SOC 追到哪" |

### 2.2 上层奖励 `r1` ★

| | `aifflor --hsrl` | `td3per` |
|---|---|---|
| 公式 | `pr·P_G·dt − pen5 − pen6` | `pr·P_G·dt − 490·max(−pb,0)·dt − penalty_NN_H − gap_pen` |
| 退化项 `490·max(−pb,0)·dt` | **已删除** | **保留**（放电按 490 元/MWh 计退化） |
| 块末 SOC gap 罚 | **已删除**（执行偏差交给 `c_L`） | **保留**：`10·max(0, |soc_target−SOC_end| − 0.1)`，uniform 下进 **r1** |
| `pen5` 系数 | **100 元/单位**（`--pen_h` 默认 100） | **10 元/单位**（无覆盖） |

⇒ 同一越界量在 aifflor 里罚 10 倍；td3per 还额外背着 490 退化与 gap 追责。

### 2.3 上层成本 `c1`

| | `aifflor --hsrl` | `td3per` |
|---|---|---|
| 形状 | `P_F + 20·(P_F−3)⁺`，**默认平滑**（softplus β=100、ε=0.05） | 同式，**默认平滑**（2026-11-05 起已对齐） |
| 预算 | `d_block_H = 15`，**curriculum 可减半**（连续 50 块满足 ⇒ `d/2`，地板 0） | `d_block_H = 15`，**地板 = 15（= 初值）** ⇒ `max(d/2, 15)` 恒为 15，减半被禁用 |

### 2.4 λ1 对偶 ★

| | `aifflor --hsrl` | `td3per` |
|---|---|---|
| 分子 | **critic cost 的 Q**（本块 `update_policy` 的 `q_c_mean`，`update_batch`） | **实际块成本 EMA**（`hc.dual_update(agent_H, d_block_H)`） |
| 物理价区间 | **[2, 50] 元/(MW·step)**（actor 空间 `[0.2, 5.0]`） | **[0.967, 4.833] 元/(MW·step)**（actor 空间 `[0.0967, 0.4833]`） |
| 初值 | 2.0（= 下限） | 0.967（= 下限） |
| 效果 | 波动最多可被定价到 50 元/MW·step | 最多 4.83 元/MW·step（≈ 5 倍边际收益的一条线） |

### 2.5 下层（BESS）★

| | `aifflor --hsrl` | `td3per` |
|---|---|---|
| 动作 | **绝对 `pb`**（物理域裁剪） | **残余 `Δ`**：`pb = 规则基线 + Δ` |
| 观测 | **6 维** `[u/10, pr/1000, SOC/0.5, P_G/5, g/2.5, need/2.5]` | **5 维** `[pg(t−1)/10, ptur/10, SOC/10, soc_target/10, n_rem/5]` |
| 奖励 `r2` | `−200·越界 − 20·放电` | `−1·min(clip_amount, 20)` |
| 成本 `c2` | `c_L = 1 − exp(−|(g−Σ已放电)/n_rem − a_L|)` | 实际 `pb` 到 **SOC 目标参考集合**的距离（`hsrl_soctarget`） |
| warmup 下层 | 摊还执行器 `clip(−(g−Σ)/n_rem, 物理域)` | 规则/Soctracker 基线 |

### 2.6 训练日程与超参默认

> **2026-11-05 已对齐**：`--bsize`、`--epsilon`、`--critic_out_init`、`--critic_bias_init`、
> `c1` 平滑 —— 见 §4。下表里标 "✅ 已对齐" 的项两者现在相同。

| | `aifflor` | `td3per` |
|---|---|---|
| `--bsize` | 32 | 32 ✅ 已对齐（此前 256） |
| `--epsilon`（H 探索决策次数） | 10000 | 10000 ✅ 已对齐（此前 4000） |
| `--wt_noise_floor` | 0.0 | 0.01 |
| `--sig_noise_min` | 0.0 | 0.01 |
| warmup 风速调度 | `unified`（半 LHS + 半统一升降噪声） | `lhs` |
| `--critic_out_init` | `fanin` | `fanin` ✅ 已对齐（此前 `3e3`） |
| `--critic_bias_init` | `none` | `none` ✅ 已对齐（此前 `estimate`） |
| `--log_file` | 1（默认简化输出） | 0 |

### 2.7 测试协议（这也是"数据表现"不可横比的直接原因）

| | `aifflor` | `td3per` |
|---|---|---|
| 协议 | **test20**：20 步、5 块、固定风段（col=120/start=483）、电价恒 300、开局 SOC=0.25 | **nine_case_test**：9 个 case × 45 步，`seed=42` 固定场景 |
| 记录 | `df_test20_hsrl.csv` | `df_test9.csv` |
| 指标 | `profit/profit_econ/c1/pf_viol/cvm/dsoc` | 同名但**口径不同**（`profit` 是 45 步总和、`cvm/pf_viol` 是计数） |

### 2.8 独有开关

- `aifflor`：`--hsrl`、`--critic_wind5`、`--critic_feat_power`、`--lin_wt`、`--scalar_wt`、
  `--farm_level`、`--aux_model`、`--c1_bound_coef`、`--diag_*`、`--hsrl_*` 一整组。
- `td3per`：`--nolow`、`--l_critic_arch`、`--branch_lower/--branch_signal`、分段奖励家族（`td3perseg` 分支）。

## 3. 结论

1. **能横比的只有网络结构与 TD3 骨架**；奖励/成本/动作语义至少有 8 处实质差异（§2.1–2.5）。
2. 最容易被忽略、但影响最大的两条（**超参口径已于 2026-11-05 对齐**，见 §4）：
   - **第 6 维语义不同**（放电量 vs SOC 目标）⇒ 策略要学的东西完全不同；
   - **λ1 区间差 10 倍**（[2,50] vs [0.967,4.833]）⇒ 波动约束的价格上限完全不同。
3. 所以"v7 在 aifflor 上更好 ⇒ td3per 也该更好"这个推断**不成立**；td3per 的 v7 run
   必须与 **td3per 自己的 v6 对照**（同 seed、同 branch）才有意义。
4. 若要真正做到"同口径对比"，还需要把 §2.1–2.5 的项逐条对齐（第 6 维语义、r1 的 490/gap、
   λ1 的分子与区间、下层动作/观测）——每条都是行为变更，建议单独 A/B。

## 4. 2026-11-05 对齐记录（超参口径，用户指定）

| 项 | td3per 改前 | 改后（= aifflor） |
|---|---|---|
| `--bsize` | 256 | **32** |
| `--epsilon`（H 探索决策次数） | 4000（=20k 环境步） | **10000**（=50k 环境步） |
| `--critic_out_init` | `3e3` | **`fanin`** |
| `--critic_bias_init` | `estimate` | **`none`** |
| c1 形状 | 硬折点 | **默认平滑**（新增 `--c1_smooth 1` / `--c1_smooth_beta 100` / `--c1_smooth_eps 0.05`，并在 main 里调用 `hc.set_smooth`） |

⚠ 这 5 项都会改变训练行为：与 2026-11-05 之前的 td3per run **不再逐位可比**
（复现旧口径：`--bsize 256 --epsilon 4000 --critic_out_init 3e3 --critic_bias_init estimate --c1_smooth 0`）。

契约：`agentutil/test_critic_arch_v7.py` 会断言这 5 个默认值与 `hc.set_smooth` 的调用。

---

## 5. 2026-11-06：td3per 全面对齐 aifflor（新增 `--hsrl` 分支）

**验收口径**：`python td3per/td3per_main.py --hsrl ...` 与
`python aifflor/aifflor_main.py --hsrl ...` 相比，**除了物理引擎（WFSim ↔ FLORIS）之外
没有任何口径差异**；未加 `--hsrl` 时 td3per 的历史路径逐位不变。

### 5.1 新增/修改的文件

| 文件 | 内容 |
|---|---|
| `td3per/td3per_hsrl.py`（新） | `--hsrl` 的全部口径：与 `aifflor/aifflor_hsrl.py` **数值逐位一致**，只换了 `myfunc` 的包名与 docstring |
| `td3per/td3per_main.py` | 新增 `train_hsrl()`（873 行，与旧 `train()` 并列）+ `--hsrl` / `--hsrl_*` / `--noise_zero_step` / `--wt_noise_mode|min|max` / `--cost_clip` / `--pen_h` / `--diag_*` 开关 + `__main__` 装配 |
| `td3per/td3per_ddpg.py` | `SP_TD3` 新增 `wt_noise_mode/min/max`（wide 噪声）与 `cost_clip`（cost 塔输入裁剪）；**默认值保持历史行为** |
| `td3per/td3per_branch.py` | 新增 `BranchController.goal()`、`--branch_freeze_lower/--branch_goal_pm1`、`run_branch_block_hsrl()`（g 语义的 branch 块） |
| `env.py` | 评测开局 SOC 默认改为 `0.5·max_E`（旧值 5.0 见 §5.4） |

### 5.2 逐项对齐（编号对应本文件 §2）

| §2 的差异 | `--hsrl` 下的 td3per 实现 |
|---|---|
| 2.1 第 6 维语义 | **块总放电量 `g [MW·min]`**，可行域 `[−C_max,+D_max]`（`hsrl.feasible_total`） |
| 2.2 r1 | `pr·P_G·dt − pen5 − pen6`（无 bessdeg、无块末 gap 罚；`pen5` 系数 = `--pen_h` = 100） |
| 2.3 c1 | 同式同平滑；`d_block_H=15` 起、curriculum **可减半到 0** |
| 2.4 λ1 | 分子 = cost critic 的 Q（`update_batch`）、物理价 **[2, 50]**、初值 2 |
| 2.5 下层 | **绝对 `pb`**、6 维观测、`r2=−200·越界−20·放电`、`c_L=1−exp(−|need−a_L|)`、λ2∈[2,50] 初值 10、`d_block_L=0.1` |
| 2.6 训练日程 | `wt_noise=wide U[0,0.05]`、`--noise_zero_step 115200` ⇒ ε=**21600** 次决策、`warmup_wt_sched=unified`、bias 用 prior 混合、`cost_clip=1` |
| 2.7 测试协议 | **test20**（20 步、电价恒 300、固定风段 col=120/start=483、开局 SOC=0.25），写 `df_test20_hsrl.csv` |
| 2.8 独有开关 | `--hsrl_*` 一整组（pen6/viol/deg/sig_noise/d_block_l/dual_*/lam1/d_block_h_min/test_wind*） |
| §3.4 RBB 硬窗口 | `rbb_system.fluctuation_limit = inf`（与 aifflor 一致，波动只由 c1 软定价） |
| branch | 第 6 维 = `g`（`goal()`，冻结下层时 ≡0）；块成本走 `cal_rew_rb_hsrl`；`pb≡0` 严格冻结 |

### 5.3 仍然不同（**只有**这些）

1. **物理引擎**：WFSim（MATLAB）↔ FLORIS（`power_scale=LEVEL_COEF`）。跨引擎的
   功率电平/收益**不可直接横比**（`aifflor/__init__.py` 的声明：u=10、a=1/3 时
   FLORIS 3.294 MW vs WFSim 3.446 MW）。
2. 输出目录 / CSV 所在 run：`--env` 决定。
3. aifflor 独有的**动作流形**分支（`--lin_wt/--scalar_wt/--farm_level`）与
   `--critic_wind5/--critic_feat_power/--aux_model/--explore_mix_blocks/--diag_u/pr`：
   这些不是"对齐项"，td3per 侧未移植（默认全关，不影响 HSRL 主路径）。

> 说明：`--hsrl_test_wind_col/start` 的哨兵 `-1` 在 aifflor 侧会走到
> `np.clip(-1, 0, n-1) = 0`（run25 的 `df_test20_hsrl.csv` 里 `wind_col/wind_start`
> 实测就是 0/0），并不是 docstring 写的 120/483。td3per 侧**逐位沿用**同一行为
> （对齐优先于"修正"）；要真正固定在 120/483 请显式传
> `--hsrl_test_wind_col 120 --hsrl_test_wind_start 483`。

### 5.4 顺带修复：评测开局 SOC = 5.0 的历史 bug

`env.py` 的 `TestReset` / `ScenarioReset` 曾写死 `self.E = 5.`（`max_E=5 MWh` 时代），
而现在 `max_E=0.5` ⇒ 5.0 是 **10 倍容量**：RBB 的充电上限变成负数、可行域退化，
首步走不可行分支输出 `pbphysicsmax ≈ −275 MW` 的假放电（`P_F ≈ 275`、`c1 ≈ 5.7e3`）。
9-case 的 **case 0（LOW/LOW）** 因此被彻底污染：`profit_econ ≈ −3588`，
而其余 8 个 case 均值 `≈ +211`；`OVERALL`（9 case 均值）被拖到 `≈ −1901`。

- 现在默认 = `0.5·max_E`（= 0.25），并有 `env.test_soc_start()` 统一入口；
- 复现旧数据：`env.legacy_test_soc5 = True` 或 `env.test_soc0 = 5.0`；
- ⚠ 历史 `df_test9.csv` / 用 `TestReset` 的评测（`test_five_days.py` 等）在 2026-11-06
  之前都带这个首步偏差，跨日期比较时需注意。

### 5.5 如何验收

```bash
# 口径逐位一致（常量 + 纯函数 + 对偶 + reward，3446 项）
python agentutil/test_td3per_hsrl_parity.py
# 静态装配自检（argparse/运行期覆盖/模块属性，不启 MATLAB、不跑仿真）
python agentutil/check_td3per_hsrl_port.py
```

真正的"表现对齐"仍需跑训练：建议 `--hsrl` 与 aifflor 用**同一 `--max_steps`、同一 seed**
配对比较，并且**同时看 profit 与 c1**（只报 c1 会奖励"停场"策略，只报 profit 会奖励
"无视约束"）。
