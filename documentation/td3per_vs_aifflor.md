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
