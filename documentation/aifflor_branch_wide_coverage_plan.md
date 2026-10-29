# AIFFLOR 宽动作覆盖 + 同块首 branch 训练计划

> 状态：**设计稿 / 未实施**
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
| 宽动作覆盖默认 | 已实现：`DAY_REPEAT=3`；WT 噪声 `U[0.05,0.25]`、触边截断噪声、不衰减 |
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

- [ ] `aifflor_env.py`：提供块首 state 快照/恢复（`u_idx, E, P_G_prev, price_idx, lower state`）；
- [ ] `aifflor_main.py`：block 开始时保存快照；
- [ ] branch 时恢复快照，跑 5 步，不推进主 episode；
- [ ] 下层 BESS 在 branch 中冻结/固定规则；
- [ ] `lower_version` 记录。

### 8.2 Replay

- [ ] replay 增加 `source/mode/block_id/lower_version/n_boot`；
- [ ] branch transition 写入同一 replay；
- [ ] actor/branch 分池采样 + mode 均衡；
- [ ] PER 分池，避免 branch 高误差淹没 actor。

### 8.3 训练/日志

- [ ] `update_policy`：critic TD 全部；actor 只用 actor；
- [ ] lower 只用 actor；
- [ ] λ/EMA/curriculum 过滤 actor；
- [ ] CSV/TB/test 过滤 actor；
- [ ] nstep：branch 1-step，actor 原口径。

### 8.4 动作采样

- [ ] actor/prior/low/high/uniform；
- [ ] 第 6 维 goal ±1；
- [ ] 触边只截噪声；
- [ ] 可选：ramp/交替/单台/折点定向。

### 8.5 诊断脚本

- [ ] 在 `agentutil/` 新建 `probe_cost_grad_hsrl_wind5.py`（路径 bootstrap）；
- [ ] 支持 8D critic state + smooth/hard c1；
- [ ] 输出 sign、Pearson、折点比、K_c、actor/data 动作点；
- [ ] 不跑仿真也能读 checkpoint/CSV 做静态分析（需要仿真时先问）。

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
    --wt_noise_mode wide --wt_noise 0.05 --wt_noise_max 0.25 `
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
