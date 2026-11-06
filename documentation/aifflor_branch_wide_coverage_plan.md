# AIFFLOR 宽动作覆盖 + 同块首 branch 训练计划

> 状态：**已实施**（2026-10-29）—— 实施记录、与设计稿的偏差、以及"哪些地方刻意没按字面做"见文末 **§13**。
> 基准提交：`e116e8e`（宽动作覆盖默认值）+ `8906b36`（`--critic_wind5`）
> 适用：`aifflor` 上层 critic（v6，可选 `critic_wind5`）
> 预算口径：总训练约 **200k env steps**
> 新会话先读本文件，再读 `aifflor/README.md` 与 `git log --oneline -5`

---

## 0. 新会话启动清单

```text
1. pwd / git status --short
2. git log --oneline -5          # 确认基线 e116e8e 之后没有未提交改动
3. 读本文件
4. 读 aifflor/README.md 的“宽动作覆盖”和“--critic_wind5”两节
5. 未确认备份策略时先问用户；本仓库默认“每次修改前自动快照”
6. 任何仿真/训练/评测命令，先 ask_user_question 得到明确同意
7. 新脚本放 agentutil/，自带路径 bootstrap；项目代码不得 import agentutil/
```

**禁止**：未询问就跑 `*_main.py`、多步 rollout、episode 测试、FLORIS 批量仿真。

---

## 1. 背景与问题

### 1.1 已观察到的现象

`AIFFLOR-run14`（HSRL，v6 + `critic_wind5=1`，8 维 critic state）：

- `Q_c` level 从 0 长到 2.8–3.45；
- 但 `∂Q_c/∂a` 与物理 `∂c1/∂a` 的 sign 一致率约 **50%**，Pearson≈0；
- 3 MW 折点梯度比真值约 **37×**，critic 只有 ~1×；
- 一次 actor 步：物理 c1 下降样本仅 49.6%，均值反而 +2.1e-3；
- λ 顶到上限，cost 通道噪声占更新量 ~86%。

### 1.2 离线测试为什么“看起来可以”

离线 C1 回归（宽动作、直接目标、`logc1`）：

- v6 直接拼 obs：test R²≈0.987、梯度 r≈0.96；
- 但这验证的是**表示能力**，不是在线 TD3 的可辨识性。

在线失败的核心不是网络容量，而是：

1. actor 行为策略动作分布窄，同 state 动作变化少；
2. 状态/风速解释了 ~96% 的 value 方差，动作边际贡献 ~1%；
3. raw c1 + PER 重尾，折点/动作敏感样本被风主导；
4. v6 纯 concat，无 `u·g(a)` 乘性通路，`feat_power` 未开；
5. 6D 动作（5 风机 + SOC 信号）比离线 1D scalar 难得多。

### 1.3 本计划的目标

在不增加太多训练成本的前提下：

```text
每个 block：
  actor 主 transition 正常推进
  从同一块首 state branch 出 4 个测试动作
  所有 branch transition 都进 replay，当正常 TD 数据用
但：
  actor/lower 只用 actor 数据
  λ/EMA/curriculum/CSV/TB/test 只记录 actor 行为
```

目标验收：

```text
∂Q_c/∂a sign ≥65%；折点梯度比 ≥10；动作增量 R² ≥5–10%
```

---

## 2. 当前代码状态（基线）

| 项 | 状态 |
|---|---|
| `--critic_wind5` | 已实现：actor 4D，critic = 原 obs + `[u2..u5]/10`，asymmetric |
| `SP_TD3(critic_nb_states=...)` | 已实现：memory 存完整 state，actor 取前 `nb_states` |
| `q_cost_rate` / `policy_batch_cost_rate` | 已实现 actor 切片 |
| 宽动作覆盖默认 | 已实现：`DAY_REPEAT=3`；WT 噪声 `U[0.05,0.25]`（**2026-10-29 起收窄为 `U[0.00,0.05]`，见 §13.8**）、触边截断噪声、不衰减 |
| OU 噪声 | **未接线**：`random_process` 创建但 `select_action` 不用 |
| 信号维噪声 | 仍按 `sig_noise_max → sig_noise_min` 衰减；HSRL 覆写 `0.5 → 0.025` |
| branch / 多动作 transition | **未实现**，本计划核心 |
| replay source 分层 | **未实现** |
| actor/λ/EMA 只统计 actor | **未实现**（需加 source 过滤） |

关键文件：

```text
aifflor/aifflor_main.py      训练循环、观测、日志、对偶
aifflor/aifflor_ddpg.py      SP_TD3、replay、select_action、update_policy
aifflor/aifflor_env.py       WindBatteryAIF、DAY_REPEAT=3、_upcoming_wind
env.py                       DAY_REPEAT 基类钩子、reset
PER_buffer.py                replay / PER
hsrl_cost.py / aifflor_hsrl.py  λ/EMA/对偶
model.py                     v6 critic；可选 feat_power
```

---

## 3. 核心设计

### 3.1 块首 state 必须完整相同

branch 的前提不是“同一串风速”，而是**完整块首 state**：

```text
s_block = (price, u_idx, E, P_G_prev, phase, lower-BESS state)
wind5   = (u1,u2,u3,u4,u5)
```

- 同一 block 的所有 branch 共享 `s_block` 和 wind5；
- 每个 branch 只换 upper action；
- 若 c1 是 wind-only 简化口径，可只匹配 wind5 + action；但实际 AIFFLOR c1 走 `P_G`，含 BESS，必须匹配 E/P_G/lower 状态；
- branch 不推进主 episode：不更新 `episode_steps / E / P_G_prev / SOC`；
- 主轨迹只由 actor 分支推进。

### 3.2 动作集

每个 branch block 生成：

```text
a_actor  = actor(s)                         # 主分支，真正推进
a_prior  = [0.2,0.2,0.2,0.2,0.33] + per-dim U[-0.05,0.05]
a_low    = [0.05]*5 + per-dim U[0,0.05]     # 可选改成 0.0 ± 0.05 覆盖 [0,0.05]
a_high   = [0.45]*5 + per-dim U[0,0.05]
a_uniform= per-dim U[0,0.5]^5
```

- 第 6 维 goal：测试分支随机 `±1`；actor 分支保留 actor 自己的 goal；
- 所有动作先 `CheckAction` 合法化；
- 触边只截断**噪声**，不整体 clip actor 输出；
- 可选：每 N 个 branch 轮换 ramp/交替/单台扰动/折点定向动作（`P_hat` 预筛）。

### 3.3 Branch 执行

```text
actor 分支：
  env.step ×5（真实推进）
  记录 transition，source=actor
  更新 episode 统计

测试分支（4 个）：
  从 s_block 快照恢复（u_idx, E, P_G_prev, price, lower state）
  用同一 wind5，跑 5 步
  下层 BESS 冻结，只前向
  记录 (s,a,r1,c1,s',done)，source=branch
  恢复/丢弃 branch 环境，不影响主轨迹
```

- branch 的 `s'` 是该 action 自己的块末状态；
- `done` 只影响该 branch 的 TD bootstrap；
- branch 不参与 n-step 链；`n_boot=γ`；
- actor transition 仍按原 `nstep` 口径。

---

## 4. Replay 设计

### 4.1 单一 replay，带 source 标签

```text
Replay fields:
  s, a, r1, c1, s_next, done,
  source,          # actor / branch
  mode,            # actor / prior / low / high / uniform
  block_id,        # 同一块首的 branch 共享
  lower_version,   # 下层 BESS 版本
  n_boot           # actor: γ^n；branch: γ
```

- **所有 branch transition 都进 replay**，不浪费；
- `source/mode/block_id/lower_version` 只用于采样/诊断，**不进入 critic state**；
- branch 的 `s'` 合法，TD 目标 `r1 + γ Q_c(s', a')` 可用。

### 4.2 采样

```text
每个 critic batch：
  actor  50–60%
  branch 40–50%
  branch 内部按 mode 均衡
  PER 分池采样：actor 池、branch 池
```

不要自然比例采样（每块 5 动作时 branch 占 80%，actor 会被淹没）。

### 4.3 损失

| 用途 | 数据 |
|---|---|
| critic TD | **全部**（actor + branch） |
| critic 辅助 C1 监督 | branch（可选） |
| actor loss | **只用 actor**（或 branch ≤20% 权重） |
| lower BESS | **只用 actor** |
| λ / EMA / curriculum | **只用 actor** |
| CSV / TensorBoard / test | **只用 actor** |

### 4.4 行为记录只写 actor

所有行为层加 source 过滤：

```text
episode_reward/cost/profit/deg
ema_c1_H / λ1 / λ2 / dual
curriculum 连续满足块数
df_aifflor.csv / df_hsrl.csv / update_stats_H.csv
test9 / test20 / nine_case
BiasInitTracker 的块样本
```

branch 只写（可选）：

```text
buffer_branch/...
debug_branch_c1/...
```

若不需要，branch 不写任何行为日志。

### 4.5 n-step / done

```text
actor:  n_boot = γ^n（原口径）
branch: n_boot = γ（只有 1 个块）
```

- branch 不参与 n-step 链；
- branch `done` 按该 branch 是否到 episode 末判断，只用于它自己的 bootstrap；
- 主 episode 的 done 仍只由 actor 分支决定。

---

## 5. 成本与 branch 频率

总预算约 200k env steps。

| 方案 | 额外仿真 | 总仿真 | 说明 |
|---|---:|---:|---|
| 每块 5 动作（全程） | +760k | ~950k（~5×） | 太贵，不推荐 |
| 每 4 块 branch 一次 | +190k | ~380k（~1.9×） | 可接受上限 |
| 每 10 块 branch 一次 | +76k | ~266k（~1.3×） | 推荐起点 |
| 前 10k 每块 branch | +40k | 50k | pilot/预训练 |

### 阶段 A：pilot / 预训练（10k env steps）

```text
10k env steps = 2,000 个 block
每个 block：actor + 4 测试动作
→ 10,000 条 transition / (s,a,c1)
→ 约 50k 步仿真
下层 BESS：冻结/固定规则
```

用途：

- 验证 branch 管线；
- 预训练 cost critic（`log1p(C1)` / Huber）；
- 验证动作增量 R²、边界/折点覆盖。

### 阶段 B：online（剩余 190k env steps）

```text
每个 block：actor transition 正常进 replay
每 K 个 block：branch 4 个测试动作
K=10 起步（1.3×）；不够再降到 K=4（1.9×）
```

---

## 6. 训练流程

### 6.1 阶段 A：预训练

```text
数据：pilot 的 branch (s,a,C1)
目标：log1p(C1) 或 Huber(C1/s_c)
网络：v6 + critic_wind5 8D
可选特征：P_hat、PF_hat、hinge、feat_power
```

验收（不通过不进入 online）：

```text
动作增量 R² ≥5–10%
∂C1/∂a sign ≥65%
PF>3 折点梯度比 ≥10
Q_c level 与平稳参考同量级
```

### 6.2 阶段 B：online

```text
L = L_TD(all replay)
  + β·L_sup(branch only, 可选)
```

- β 从 1.0 线性降到 0.2；
- actor loss 只用 actor；
- λ/EMA/curriculum 只用 actor；
- lower BESS 只用 actor；
- branch 的 c1 不进入 dual。

### 6.3 checkpoint 门控

每 5k 步测一次：

```text
∂Q_c/∂a sign（actor 动作点）
折点梯度比
actor 物理 c1 是否下降
Q_c level 是否漂移
```

不达标：优先加 branch 频率/动作模式，不要盲目加训练步数。

---

## 7. 验收指标

| 指标 | 目标 |
|---|---|
| branch 同块动作数 | 5（actor + 4） |
| 动作增量 R² | `state+action` 比 state-only 高 ≥5–10% |
| 边界覆盖 | `a<0.05` / `a>0.45` 各 ≥5% |
| 折点覆盖 | `PF∈[2,4]` ≥10% |
| 梯度 sign | ≥65% |
| 折点梯度比 | ≥10 |
| actor 物理 c1 | 下降样本 >60% |
| Q_c level | 与平稳参考同量级 |
| NaN/越界 | 0 |
| branch/actor 比例 | branch ≤50% |

---

## 8. 实施清单

### 8.1 数据/环境

- [x] `aifflor_env.py`：提供块首 state 快照/恢复（`u_idx, E, P_G_prev, price_idx, lower state`）；
      → 实际落在 **`aifflor/aifflor_branch.py`** 的 `snapshot_env` / `restore_env`
      （**`aifflor_env.py` 一行未改**：这 6 个标量是 `env.WindBattery` 的公有属性，
      在环境层再加一层封装没有收益，反而让"哪个才是真源"变模糊；见 §13.2）；
- [x] `aifflor_main.py`：block 开始时保存快照；
- [x] branch 时恢复快照，跑 5 步，不推进主 episode；
- [x] 下层 BESS 在 branch 中冻结/固定规则（`--branch_freeze_lower`，默认 1 = `pb≡0`）；
- [x] `lower_version` 记录（`agent_H.lower_ver` → replay 的 `lower_ver` 字段 + `df_branch_hsrl.csv`）。

### 8.2 Replay

- [x] replay 增加 `source/mode/block_id/lower_version/n_boot`
      （`n_boot` 复用既有的 `SP_TD3._boot`，不新增数组）；
- [x] branch transition 写入同一 replay；
- [x] actor/branch 分池采样 + mode 均衡（`PER_buffer.sample_pool(..., by_mode=True)`）；
- [x] PER 分池，避免 branch 高误差淹没 actor（**池内**仍按 priority^α 比例抽）。

### 8.3 训练/日志

- [x] `update_policy`：critic TD 全部；actor 只用 actor（掩码切片，见 `actor_mask`）；
- [x] lower 只用 actor（`agent_L` 有**独立的** memory，branch 从不写进去 ⇒ 天然满足）；
- [x] λ/EMA/curriculum 过滤 actor（`recent_states` 按 `src` 过滤 + `q_c_mean` 只算 actor 行）；
- [x] CSV/TB/test 过滤 actor（branch 只写 `df_branch_hsrl.csv` 与 `branch/*` TB tag）；
- [x] nstep：branch 1-step（`n_boot=γ`），actor 原口径。

### 8.4 动作采样

- [x] actor/prior/low/high/uniform（+ 可选的 `ramp`/`alt`/`single`，共 7 种生成器）；
- [x] 第 6 维 goal ±1（`--branch_goal_pm1`，HSRL 语义下 = ±满可行量）；
- [x] 触边只截噪声（生成器**按构造**就在 `[0,0.5]` 内 ⇒ 这一步在 branch 里不需要）；
- [x] 可选：ramp/交替/单台（`--branch_actions prior,low,high,uniform,ramp,alt,single`）；
      **折点定向（`P_hat` 预筛）未实现**（见 §13.4）。

### 8.5 诊断脚本

- [x] 在 `agentutil/` 新建 `probe_cost_grad_hsrl_wind5.py`（路径 bootstrap）；
- [x] 支持 8D critic state + smooth/hard c1（自动推断 checkpoint 的 state 口径）；
- [x] 输出 sign、Pearson、折点比、K_c、actor/data 动作点；
- [x] 不跑仿真也能读 checkpoint/CSV 做静态分析（需要仿真时先问）。

---

## 9. 风险与边界

1. **branch 不是行为数据**  
   可以进 critic TD，但不能进 λ/EMA/curriculum/CSV/test；否则随机动作高 c1 会污染对偶。

2. **下层 BESS 版本漂移**  
   branch 用冻结 lower；若 online lower 继续更新，旧 branch 的 c1 标签会过时；采样时优先最近 `lower_version`。

3. **actor 数据比例**  
   branch 占 80% 时 actor 会欠采样；必须 source 分层。

4. **n-step**  
   branch 只有 1 个块；不能进 n-step 链；`n_boot=γ`。

5. **同块 state**  
   不仅要同一串风速，还要 E/P_G_prev/price/lower 状态一致；否则不是反事实。

6. **动作维度**  
   5D 风机 + 6D 动作；4 个测试动作仍偏少；建议轮换模式、加折点/异向。

7. **成本**  
   每块 5 动作 = 5×；用 K=10 起步（1.3×），不够再降到 K=4（1.9×）。

8. **raw c1 重尾**  
   预训练/辅助用 `log1p`/Huber；online TD 保持原口径，但监控折点。

---

## 10. 复现命令（实现后；未获同意不要跑）

```powershell
$PY = 'C:\Users\yigul\anaconda3\envs\llmenhancedwsis\python.exe'

# 基线：v6 + wind5，无 branch
& $PY aifflor\aifflor_main.py `
    --critic_arch v6 --critic_wind5 1 `
    --wt_noise_mode wide --wt_noise_min 0.0 --wt_noise_max 0.05 `
    ...

# branch 方案（实现后）
& $PY aifflor\aifflor_main.py `
    --critic_arch v6 --critic_wind5 1 `
    --branch_actions prior,low,high,uniform `
    --branch_every 10 `
    --branch_goal_pm1 1 `
    --branch_freeze_lower 1 `
    ...
```

> 任何上述命令运行前先 `ask_user_question`，写清命令、目的、预计耗时/资源；
> 用户拒绝后不要换命令绕过。

---

## 11. 完成标准

```text
1. branch 管线跑通，同块首 5 动作，transition 全进 replay；
2. branch 不推进主 episode，不污染行为记录；
3. screen：sign≥65%，折点比≥10，动作增量 R²≥5%；
4. online：actor 物理 c1 下降样本>60%，λ 不因 branch 顶格；
5. 成本 ≤1.9× baseline，且无 NaN/越界异常。
```

未达到时优先改：

```text
动作模式/折点覆盖 > branch 频率 > 物理特征/feat_power > 网络宽度
```

---

## 12. 一句话

**每个 block 从同一完整块首 state 生成 actor + prior/low/high/uniform 四个测试动作；所有 transition 都进 replay 当正常 TD 数据，但 actor/lower/λ/EMA/日志只使用 actor source；branch 每 10 块一次起步，pilot 10k 步验证，再按 sign/折点比决定是否加密。**

---

## 13. 实施记录（2026-10-29）

### 13.1 文件清单

| 文件 | 改动 |
|---|---|
| **`aifflor/aifflor_branch.py`** | **新增**。branch 的**口径单一真源**：CLI（`add_branch_args`）、`parse_modes`、7 种动作生成器、`snapshot_env`/`restore_env`、`run_branch_block`（5 步推进 + `finally` 还原）、`BranchController`（触发/goal/诊断/β 衰减）、`declared_scope()` |
| `PER_buffer.py` | **改**。`PrioritizedReplayBuffer` 增加**惰性**元数据（`src/mode/block_id/lower_ver`）、`set_meta`、`pool_idxes`、`sample_pool(pool,n,by_mode)`、`sample_by_idxes`。不调 `set_meta` 时 `sample()` 与历史**逐位相同** |
| `aifflor/aifflor_ddpg.py` | **改**。`branch_mix`/`branch_sup_beta` kwarg；`_tag_actor`；`push_branch_transition`（1 步、`n_boot=γ`、不进 `_pending`）；`_sample_batch`（分池）+ `_src_of`；`update_policy` 的 **actor 掩码**、可选辅助监督、`q_c_mean` 只算 actor 行 |
| `aifflor/aifflor_hsrl.py` | **改**。`recent_states(..., source=0)`：λ 的分子只在 `src==actor` 的转移里取近端 N 条（无元数据时逐位退回历史） |
| `aifflor/aifflor_main.py` | **改**。`abr.add_branch_args(parser)`；`agent_H` 构造传 `branch_mix`；`train_hsrl` 里块首快照 → 跑 branch → 写 replay；`lower_ver` 推进；`branch/*` TB；`df_branch_hsrl.csv`；结束时打印汇总 |
| `agentutil/test_aifflor_branch_contract.py` | **新增**。9 段契约，**28149 条断言**全绿（纯静态） |
| `agentutil/probe_cost_grad_hsrl_wind5.py` | **新增**。`∂Q_c/∂a` vs 物理 `∂c1/∂a` 的静态探针（见 §8.5） |

**`aifflor_env.py` / `env.py` / `model.py` / `hsrl_cost.py` 一行未改。**

### 13.2 与设计稿的**口径换算**（重要，别照字面读 §3.2）

设计稿写于"第 6 维 = SOC 目标信号 `a6 ∈ [−0.5,0.5]`"的年代。`--hsrl` 之后第 6 维是
**块总放电量 `g` [MW·min]**（`aifflor_hsrl.HSRL_*`），所以：

| 设计稿 | 实施 |
|---|---|
| `--branch_goal_pm1 1` ⇒ "第 6 维随机 ±1" | `g = +D_max`（满放电）/ `−C_max`（满充电），即**满可行量**的 ±；**默认关** |
| `--branch_freeze_lower 1` ⇒ 下层冻结 | `pb ≡ 0` ⇒ `P_G = P_tur`、`E` 不变 ⇒ 块内 `c1` **只由风机动作决定**（对上层 critic 最干净的反事实） |
| 冻结下层时的第 6 维 | **恒 0**：`pb≡0` 时 `g` 对物理没有任何作用，留着只污染动作维 |
| `a_low = [0.05]⁵ + U[0,0.05]` | **改为逐维 `U(0.00, 0.10)`**：字面写法对 `a<0.05` 的覆盖率是 **0%**，达不到 §7 的"边界覆盖 ≥5%"；`high` 同理取 `U(0.40,0.50)` |
| "branch 不行就加频率" | 保留 `--branch_every`/`--branch_pilot_blocks` 两个旋钮 |

**默认组合 = `--branch_freeze_lower 1 --branch_goal_pm1 0`**：上下层解耦，branch 只改 5 维风机动作。

### 13.3 三个实现上的**坑**（改代码前请看）

1. **`restore_env` 必须在 `make_state` 之后调用。**
   `s'` 里的 `u2..u5` 来自 `env._upcoming_wind(4)`，它读的是**块末**的 `env.count_wind`；
   先还原再算 `s'` 就会拿到块首的 4 个风速（而且不报错，只是悄悄错）。
   `run_branch_block` 里 `make_state(...)` 放在 `try` 内、`restore_env` 放在 `finally`。
2. **必须先 `restore_env` 再让主循环继续，且 `restore_env` 要把 `env._pf_state` 置 `None`。**
   `AifflorPhysics.prefetch` 是**整体替换** `_cache`（`aifflor_env.py`），branch 跑完后
   缓存里是 branch 动作的键；只把 `_pf_state` 恢复成旧值等于撒谎（"缓存有效"这个断言是假的）
   —— 正是 README §3.5 记载过的那类坑。置 `None` 的代价只是多一次批量求值。
3. **replay 是环形缓冲 ⇒ actor 必须显式覆盖写 `src`。**
   被复用到的槽上一次可能是 branch（`src=1`）；不覆盖写就会把 actor 数据误判成 branch
   （actor loss 漏掉它、branch 池混进 actor 样本）。`_tag_actor` 就是干这个的。

### 13.4 刻意**没有**做（以及为什么）

1. **`aifflor_env.py` 里加 `snapshot()/restore()` 方法**：快照只有 6 个公有标量，
   在 `aifflor_branch.py` 里做一次就够；加进 env 会让 `env.py` 的"单一真源"多一处分叉。
2. **折点定向动作（`P_hat` 预筛）**：需要把 `model.feat_power` 的 `P_hat` 搬到 branch 生成器里，
   且要先知道"当前 state 下哪个 a 让 `P_F≈3`"（要解方程）。现有 `low/high/uniform` 已经
   覆盖边界；折点覆盖由 `P_F∈[2,4]` 的**实测比例**监控（`branch/cover_fold`），不够再补。
3. **按 `lower_version` 加权采样**（§9.2）：`--branch_freeze_lower 1`（默认）时 branch 的
   `c1` **完全不含下层策略** ⇒ 标签不会过期，加权无意义。`lower_ver` 只作诊断记录。
4. **`actor loss` 里给 branch ≤20% 权重**（§4.3 的备选）：采用主口径"只用 actor"。
5. **branch 写行为日志**：完全不写（§4.4 的口径），只写 `df_branch_hsrl.csv` 与 `branch/*`。

### 13.5 验证证据

| 项 | 结果 |
|---|---|
| `python agentutil/test_aifflor_branch_contract.py` | **28149 条断言全绿**（含：`run_branch_block` 的 `Σr1/Σc1` 与**独立数组重算**逐位一致、`freeze_lower` 时 SOC/pbrule 恒 0、主轨迹逐位还原、PER 环形覆盖后计数正确、`update_policy` 里 **actor 每批只吃 actor 行**、`n_boot` branch=γ/actor=γ^n、`recent_states` 只取 actor） |
| `--branch_*` CLI | 15 个开关出现在 `--help` 的 `branch` 组里；模块 import 通过；`py_compile` 全绿 |
| 探针在 `AIFFLOR-run14` 上的读数（n=150，wind5 用**真实**未来 4 步） | sign **51.5%**、Pearson **+0.13**、折点放大比 **0.78**（真值 **21.7 ≈ 解析 21**）、动作增量 R² **+0.0022** ⇒ 三条验收全 FAIL —— **与 §1.1 对 run14 的判断逐条吻合**，说明探针量到的正是计划要修的东西 |
| 探针的物理侧自校验 | 折点放大比的"真值"实测 21.7，与 `c1 = P_F + 20(P_F−3)⁺` 的解析值 21 一致（偏差来自差分离散） |

### 13.6 复现命令

```powershell
$PY = 'C:\Users\yigul\anaconda3\envs\llmenhancedwsis\python.exe'

# 0) 契约测试（纯静态，随时可跑）
& $PY agentutil\test_aifflor_branch_contract.py

# 1) 基线读数（不跑训练）：先确认"要修的病"还在
& $PY agentutil\probe_cost_grad_hsrl_wind5.py --run output\AIFFLOR-run14 --n 150 --wind5_mode day

# 2) 阶段 A（pilot）：前 2000 块每块 branch（= 10k env steps）
& $PY aifflor\aifflor_main.py --hsrl --critic_arch v6 --critic_wind5 1 `
    --branch_actions prior,low,high,uniform --branch_pilot_blocks 2000 `
    --branch_every 0 --max_steps 10000 --seed 1

# 3) 阶段 B（online）：每 10 块 branch 一次
& $PY aifflor\aifflor_main.py --hsrl --critic_arch v6 --critic_wind5 1 `
    --branch_actions prior,low,high,uniform --branch_every 10 --seed 1

# 4) 阶段 B 的 checkpoint 门控（每 5k 步，见 §6.3）
& $PY agentutil\probe_cost_grad_hsrl_wind5.py `
    --run output\AIFFLOR-runNN --ckpt checkpoint_20000 --n 200 --wind5_mode day
```

> **2026-10-29 起 `--critic_arch` 的默认值已改为 `v6`**（`--critic_wind5` 仍默认 **0**）。
> 上面命令里的 `--critic_arch v6` 现在是**冗余但显式**的 —— 保留它便于与历史命令逐字对照；
> 若照 §13.5 的口径想复现 run14 的 8 维 critic，仍必须显式 `--critic_wind5 1`。

> **一切训练/评测命令跑之前仍须 `ask_user_question` 得到明确同意**（AGENTS.md 的强制规则）。

### 13.7 成本实测口径

一个 branch 块 = `len(--branch_actions) × 5` 个额外环境步（**另加 5 次下层 BESS 前向**，
但 BESS 是纯算术、`pb≡0` 时几乎零成本）。

| 配置 | 每 10 块的环境步 | 相对基线 |
|---|---:|---:|
| 基线（无 branch） | 50 | 1.00× |
| `--branch_every 10`（默认 4 个测试动作） | 50 + 20 = 70 | **1.40×** |
| `--branch_every 4` | 50 + 50 = 100 | **2.00×** |
| `--branch_every 1`（每块） | 50 + 200 = 250 | 5.00× |
| `--branch_pilot_blocks 2000 --branch_every 0`（只 pilot） | — | 前 2000 块 5.00× |

> 设计稿 §5 的"K=10 ⇒ 1.3×"是按"每块 5 动作"的近端口径估的；**按上式实际是 1.4×**。
> `branch_mix` 的额外算力 = 每个 critic 批多 45% 的 branch 行（同一批、同一次反向传播），
> 不增加前向次数。

### 13.8 配套改动：actor 行为噪声收窄到 `U[0.00, 0.05]`（2026-10-29）

**改了什么**

| 项 | 旧 | 新 |
|---|---|---|
| wide 的幅度带 | `U[0.05, 0.25]` | **`U[0.00, 0.05]`** |
| 下界开关 | `--wt_noise`（兼 legacy 起幅） | **`--wt_noise_min`**（默认 0.0） |
| `--wt_noise` | wide 下界 + legacy 起幅 | **只管 legacy**（起幅，默认仍 0.05） |
| `--wt_noise_max` | 0.25 | **0.05** |

**为什么**（与 branch 的设计直接相关）

1. **噪声幅度 ≈ 设计间距**：branch 的测试动作在带内的间距是 **0.20~0.25**
   （`prior≈0.20–0.33` / `low≈0–0.10` / `high≈0.40–0.50`），而旧噪声
   **std≈0.093、峰值 0.25** ⇒ 等于把"同块 5 动作"的配对结构抹平。
   新带 **std≈0.017、峰值 0.05 = 间距的 1/4**。
2. **删失堆料**：旧带幅度 > 半带，触边后噪声被压成单边、并可正好落到 0/0.5
   ⇒ run14 实测 `a>0.45` 占 **35.7%**、`a<0.05` 仅 **2.7%**。这既不是
   "actor+对称噪声"也不是"平衡覆盖"。覆盖率改由 `--branch_actions` 承担
   （平衡、无删失、不随 π 移动、样本干净）。
3. **λ 为噪声付费**：`c1=φ(P_F)` 凸 ⇒ `E[c1(π+ε)] > c1(π)`，而 wide 的风机维噪声
   **从不衰减** ⇒ 旧带让约束价格长期被探索噪声抬高（README §3.8 记录过同源的
   Jensen 缺口：0.317 vs 0.490）。

**保留了什么（为什么不直接关到 0）**

* branch 只覆盖 `1/--branch_every` 的块（K=10 ⇒ 只有 **10%** 的状态有兄弟动作）
  ⇒ 全关噪声会让 90% 的状态只剩"点质量动作"，`∂c1/∂a` 只能靠 10% 的状态外推；
* actor 噪声还兼着 **状态探索**（改动作 → 改 E/P_G/SOC 轨迹，branch 不推进 episode）
  与**边界处的局部光滑**（策略有向边界堆的倾向，那里正好是折点 = critic 最不准的地方）。

**未定项（需要 A/B）**

`0.05` 是"设计间距 1/4"的经验值，不是实测最优。建议三臂（`--branch_every 10`，
同 `--seed`，判据 = 探针的 sign / Pearson / 折点放大比 + `λ1` 轨迹 + `sum_c1` + 越限率）：

| 臂 | 噪声 | 预期 |
|---|---|---|
| A | `--wt_noise_min 0.05 --wt_noise_max 0.25`（旧口径） | 基线 |
| B | **默认 `U[0.00, 0.05]`** | 配对结构清晰，仍保留局部光滑 |
| C | `--wt_noise_min 0 --wt_noise_max 0`（全关） | 最"干净"但 90% 状态无动作约束 ⇒ **不看好** |
