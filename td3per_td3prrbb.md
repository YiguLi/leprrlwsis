# TD3PER 与 TD3PRRBB 说明

本文档按顺序描述本项目中的两个强化学习方案：

1. **TD3PER**（`td3per/`）— TD3 + PER + 拉格朗日成本约束（real-profit 变体）
2. **TD3PRRBB**（`td3prrbb/`）— TD3 + PRRBB 风速×电价分段奖励（机会/风险分离版）
3. 两者之间的差异对比

---

## 一、TD3PER（TD3 + PER）

### 1.1 定位与架构

- **算法层**：`SP_TD3` agent（`td3per/td3per_ddpg.py`）——TD3 的孪生批评器（利润通道 min 聚合、成本通道 clamp≥0 聚合）+ 延迟策略更新 + 目标策略平滑；结合 **PER**（优先经验回放，含重要性采样）与**拉格朗日成本约束**（profit − λ·cost 目标）。
- **奖励层**：real-profit 变体——L 层（BESS）用**真实电价**套利利润，而非影子利润。
- **分层双智能体**：
  - H 层（上层）：控制 5 台风机降载系数（0~0.5）+ 1 个影子价格信号（−0.5~0.5），每 5 步决策一次。
  - L 层（下层）：控制 BESS 功率（±10 MW），每步决策；Q 值跨 5 步块引导。
- **文件**：`td3per_main.py`（训练主循环）、`td3per_ddpg.py`（SP_TD3）、`td3per_myfunc.py`（奖励/观测/测试）。

### 1.2 奖励结构（训练，`td3per_main.py` 192–260 行）

**H 层奖励 r1**（`td3per_main.py:193`）：

```
r1 = pr × P_G × time − penalty_NN_H − 490 × max(−pbrule, 0) × time
```

- `pr × P_G × time`：电网售电收入（单一项，未分解风机/BESS；time = 1/60 h）
- `penalty_NN_H`：H 层动作越界软惩罚（`CheckAction`，每越界 1 单位罚 500）
- `490 × max(−pbrule, 0) × time`：BESS 放电退化成本（490 RMB/MWh）

**H 层成本 c1**（`td3per_main.py:208`）：

```
c1 = c + 0.1 × P_F + W_GAP × unfeasible_gap
   其中 c = 10 × (P_F − 3)   (仅 P_F > 3 MW 时生效, 来自 Cal_Rew)
```

- `c`：电网波动越限惩罚；`0.1 × P_F`：平滑软项（任意波动都计）
- `W_GAP = 50.0` × `unfeasible_gap`：波动约束 ∩ 物理约束为空的不可行缺口（MW）

**L 层奖励 r2**（`td3per_main.py:236`，real-profit）：

```
r2 = real_profit − bessdeg − 100 × clip_amount
real_profit = −pbrule × price / 20     （真实电价套利, 低谷充电/高峰放电为正）
bessdeg     = 490 × max(−pbrule, 0) / 20   （退化成本, 与利润项同分母, 盈亏平衡≈490）
```

**L 层成本 c2**（`td3per_main.py:239`）：

```
c2 = 1000 × soc_deviation     （SOC 越出固定安全带 [1.5, 4.5] 的偏移量）
```

### 1.3 分层交互与对偶更新

- **5 步块结构**：H 每 5 步决策一次；块内累计 `sum_r1` / `sum_c1`（H）与 `sum_c2`（L），块末 `observe_jump` 入库。
- **EMA 对偶上升**（实际成本驱动，`td3per_ddpg.py`）：
  - H：`lam1 ∈ [3, 50]`，初始 5.0；`lam1 ← clip(lam1 + 0.01 × (EMA − d_block_H), 3, 50)`
  - L：`lam2 ∈ [10, 100]`；`lam2 ← clip(lam2 + 0.01 × (EMA − d_block_L), 10, 100)`
  - EMA：`EMA ← (1−0.02)×EMA + 0.02×sum_c`，每 5 步块更新
- **课程学习（CL）**：EMA 成本连续 100 次满足预算则 `d_block` 减半：
  - `d_block_H`：初始 2.5（≈5 MW/步）→ 地板 1.25（≈2.5 MW/步）
  - `d_block_L`：初始 20.0（≈每步 4，按 c2 系数 1000 标定）→ 地板 2.5
- **学习目标**：策略损失 = `−(min_p − λ × min_c)`；成本 bootstrap 目标 `clamp(min=0)`。
- **PER 优先级**：`|TD_p|/mean_p + |TD_c|/mean_c`（去 λ、量纲归一化）。
- **L 层 episode**：整日 1440 步，仅日末 `done_L=True`（Q 值跨 5 步块引导）。

### 1.4 观测

- H 层：`Normalize([pr, u, E, P_G_last])` → `[pr/1000, u/10, E/10, P_G_last/10]`
- L 层（蒸馏 5 维）：`[pbrulemin/10, pbrulemax/10, price/100, SOC/10, u/10]`（第 3 维为**真实电价**；风速 u 供分段可观测，TD3PER 固定安全带下为冗余）

---

## 二、TD3PRRBB（TD3 + PRRBB）

### 2.1 定位与架构

- **算法层**：沿用 TD3PER 的 `SP_TD3`（`td3prrbb/td3prrbb_ddpg.py` 为 `td3per_ddpg.py` 的拷贝），完全相同。
- **奖励层**：采用 PRRBB 的 **风速×电价分段奖励（机会/风险分离版）**——把 P_G 收益分解为风机（ptur）与 BESS（pb）两部分，机会侧（利润）用收益权重编码，风险侧（波动/平抑可行性/SOC）全部由 c1/c2 编码，不再混入收益权重。
- **文件**：`td3prrbb_main.py`、`td3prrbb_ddpg.py`、`td3prrbb_myfunc.py`（`Cal_Rew_PRRBB` 等）、`td3prrbb_rule_bess.py`。

### 2.2 分段规则（查表权重）

- 电价三档：LOW ≤ 300、MEDIUM ∈ (300, 980)、HIGH ≥ 980（`get_price_level`）
- 风速三档：LOW < 7.0 m/s、MEDIUM ∈ [7.0, 11.0)、HIGH ≥ 11.0 m/s（`get_wind_level`）

| 权重 | 含义 | 取值 |
|---|---|---|
| wT_u | 风机收益×风速 | LOW=1.2 / MEDIUM=1.0 / HIGH=0.9（高风轻度降载） |
| wT_p | 风机收益×电价 | LOW=1.0 / MEDIUM=1.0 / HIGH=1.05（高价时段轻微加码） |
| wB_p | BESS 收益×电价 | LOW=0.8 / MEDIUM=1.0 / HIGH=1.2（低价时少罚充电→鼓励低谷充电） |
| wB_soc | BESS 收益×SOC 贴边 | 带内=1.0 / 带外=0.7（SOC 贴边抑制套利） |
| r_u | c1 风险系数×风速 | LOW=0.8 / MEDIUM=1.0 / HIGH=1.2 |

### 2.3 奖励结构（`Cal_Rew_PRRBB`，`td3prrbb_myfunc.py:225`）

**H 层奖励 r1**（机会侧，乘法分解）：

```
r1 = wT × (pr × ptur × time)                 ← 风机收益, wT = wT_u(u) × wT_p(pr)
   + wB × (−pr × pbrule × time)              ← BESS 收益, wB = wB_p(pr) × wB_soc(SOC, wind)
   − 490 × max(−pbrule, 0) × time            ← 退化（H 看 BESS 净收益）
   − penalty_NN_H
```

**H 层成本 c1**（风险侧）：

```
c1 = r_u(u) × [10×(P_F−3)⁺ + 0.1×P_F] + W_GAP × unfeasible_gap     （W_GAP = 50.0）
```

- 波动惩罚整体乘上风速风险系数 `r_u`（HIGH 风惩罚最重 ×1.2）

**L 层奖励 r2**（纯信号跟随，无真实利润项）：

```
r2 = 10 × shadow_profit − 100 × clip_amount
shadow_profit = −shadow_price × pbrule       （shadow_price = action[5] × 2 ∈ (−1, 1)）
```

**L 层成本 c2**（风速分段 SOC 安全带）：

```
c2 = 700 × soc_deviation
SOC 安全带: LOW 风 [0.5, 5.5] / MEDIUM 风 [1.0, 5.0] / HIGH 风 [1.5, 4.5]
```

### 2.4 分层交互与对偶更新

- 同 TD3PER：5 步块结构、EMA 对偶上升（`lam1 ∈ [3, 50]` init 5.0、`lam2 ∈ [10, 100]`，`ema_alpha=0.02`、`ema_lam_lr=0.01`）。
- **课程学习预算**（PRRBB 标定口径）：
  - `d_block_H`：初始 1.2 → 地板 0.3（c1 按 `r_u×[...]` 标定）
  - `d_block_L`：初始 14.0 → 地板 1.75（c2 按 700 系数标定）
- **L 层 episode**：每 5 步块结束 `done_L=True`（L 层 episode = 5 步，回归 PER 原口径）。
- **九例分段测试**：训练中每 480 episode step 执行一次 9 例（3 风速 × 3 电价）45 步测试，**固定随机种子 42**，保证相邻两次 test9 可点对点比较。

### 2.5 观测

- H 层：同 TD3PER（`Normalize` 4 维）。
- L 层（蒸馏 5 维）：`[pbrulemin/10, pbrulemax/10, shadow_price, SOC/10, u/10]`——第 3 维为 **H 层影子价格信号**（而非真实电价）；风速 u 使分段 SOC 安全带可观测。

---

## 三、TD3PER 与 TD3PRRBB 的差异

两者算法层完全相同（同一份 `SP_TD3`），差异全部在**奖励/成本结构与标定**：

| 维度 | TD3PER | TD3PRRBB |
|---|---|---|
| **收益分解** | 单一售电项 `pr × P_G × time` | 分解为 风机 `wT×pr×ptur` + BESS `wB×(−pr×pbrule)` |
| **分段权重** | 无（统一系数） | `wT_u / wT_p / wB_p / wB_soc` 风速×电价查表乘法分解 |
| **L 层利润信号** | 真实电价套利 `real_profit = −pbrule×price/20` | 影子价格跟随 `10 × shadow_profit = −10×shadow_price×pbrule` |
| **c1 波动风险系数** | 统一 ×1.0 | `r_u(u)` 分档：LOW 0.8 / MEDIUM 1.0 / HIGH 1.2 |
| **c2 SOC 安全带** | 固定 `[1.5, 4.5]`，系数 **1000**/MWh | 风速分档 `[0.5,5.5] / [1.0,5.0] / [1.5,4.5]`，系数 **700**/MWh |
| **L 观测第 3 维** | `price/100`（真实电价） | `shadow_price`（H 影子信号） |
| **d_block_H**（CL 预算） | 2.5 → 地板 1.25 | 1.2 → 地板 0.3 |
| **d_block_L**（CL 预算） | 20.0 → 地板 2.5 | 14.0 → 地板 1.75 |
| **L 层 episode 语义** | 整日 1440 步（Q 跨块引导） | 每 5 步块结束（回归 PER 口径） |
| **总奖励 r 口径** | 仅日志用 `Cal_Rew`（不参与训练） | `r = r1 + r2`、`c = c1`（供指标统计） |
| **九例测试** | 每 480 step，9-case（seed 42） | 每 480 step，9-case，seed 42 |
| **设计目标** | 真实套利利润最大化 + 固定 SOC 安全带 | 机会/风险分离：分段权重编码机会、c1/c2 编码风险，风速自适应安全带 |

### 3.1 差异要点解读

1. **L 层利润信号不同**：TD3PER 的 L 层直接看真实电价套利（`price/20` 分母与退化成本同量纲，盈亏平衡电价≈490）；TD3PRRBB 的 L 层跟随 H 层的影子价格信号（系数 10），机会完全交给 H 层编码。
2. **分段与否**：TD3PRRBB 将收益按风速/电价查表加权（低风多发电、低价多充电、SOC 贴边降权），风险系数 `r_u` 与 SOC 安全带也随风速收紧/放松；TD3PER 无任何分段，统一系数 + 固定安全带 [1.5, 4.5]。
3. **预算标定不同**：由于 c1/c2 公式与系数不同（1000 vs 700、统一 vs 分档），两个方案的 `d_block` 初值与地板按各自成本量级分别标定。
4. **设计哲学**：TD3PER 追求"真实利润 + 硬约束安全带"；TD3PRRBB 追求"机会/风险分离 + 环境自适应"——机会侧只进 r，风险侧只进 c1/c2，并通过风速分段让约束随风况松紧变化。
