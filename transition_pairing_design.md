# Transition 分流设计说明（Q_p / Q_c 双通道动作-标量配对）

> 状态：**仅概念设计，未实现**（2026-08-27 会话决定：先只实现 RBB 回正机制，
> 本设计仅以文档形式保留，供后续评估）。
>
> 提出背景：L 层 `r2 = 10·shadow_profit − 100·clip_amount` 存在动作-奖励错配，
> 本设计试图从"哪个 Q 通道与哪个动作配对"入手根治。

---

## 1. 背景：动作-奖励错配问题

当前训练循环（PER / PRRBB / TD3PRRBB 系列）：

```python
pb_original = agent_L.select_action(...)          # NN 意图动作
pbrule = rbb_system.compute_pbrule(pb_original, ...)  # RBB 硬裁剪
r2 = 10.0 * shadow_profit(pbrule) - 100.0 * clip_amount   # clip = |pb_original - pbrule|
agent_L.observe_jump(s, pb_original, r2, c2, s', done)
```

问题：**一个标量 r2 混了两种动作语义**。

- `shadow_profit = −shadow_price·pbrule` 是**执行动作** pbrule 的函数（环境真实结果）；
- `clip_amount = |pb_original − pbrule|` 是**意图动作** pb_original 的函数（意图与可行域的距离）。

于是 Q_p 学到的"利润是动作的函数"实际上是一个混淆函数——同一个标量既依赖意图又依赖执行，
梯度方向被拧在一起。历史参考：`446d2f4`（feat: ...修复动作-奖励错配）只修了一半
（回放里存的仍是原始动作 + 基于 pbrule 的奖励），随后被 `a477774` 整体 revert。

## 2. 概念：按通道分流

利用已有的双 critic 结构（profit 通道 + cost 通道），**每个通道存语义自洽的 (s, a) 对**：

| 通道 | 存储的 action | 存储的标量 | 语义 |
|---|---|---|---|
| Q_p（利润） | `a_executed = pbrule` | 纯利润项（如 `shadow_profit`，不含 clip） | 环境真实结果 |
| Q_c（成本） | `a_intended = pb_original` | 全部惩罚（`700·soc_dev`、`100·clip_amount`、硬惩罚） | 意图偏离可行域要付多少 |

策略目标不变：

```
max Q_p(s, π) − lam·Q_c(s, π)
```

actor 梯度仍用**原始输出**评估两个 critic；只是两个 critic 各自的训练数据
（TD target 用的 action）分开了。

## 3. 为什么成立

1. **惩罚梯度终于指向意图**：`d(clip_amount)/d(pb_original) = ±1`，解析上干净——
   "意图偏了就往可行域推"这个信号通过 Q_c 回归后传给 actor，方向正确。
   对比现状：clip 混在 r2 里，与 shadow_profit（依赖 pbrule）共用一条梯度，方向拧巴。
2. **Q_p 学真实映射**：利润是执行动作的函数，不再被意图污染。
3. **与 pbrule 回正机制是天作之合**：恢复期执行动作 = 恢复点（带边缘）。cost 通道存
   `(s, 意图, clip=巨大)` → Q_c 学到"越界意图很贵" → actor 的最优响应是**直接把输出
   放到恢复点**（clip→0）——NN 把安全规则内化成自己的策略，安全层从"强制覆盖"
   变成"老师"（安全 RL 的理想终态）。
4. **补完 PRRBB 的"机会/风险分离"哲学**：H 层 r1/c1 早已分离；L 层的 r2 一直混着
   clip（一个惩罚项）。本设计把 L 层也补完，整个框架终于名副其实。

## 4. 实现要点（概念层，未实现）

- **buffer 存双 action**：`memory.py` / `PER_buffer.py` 的 append/sample 增加一个
  字段（`a_intended` 给 cost critic，`a_executed` 给 profit critic），
  `update_policy` 里按通道取不同的 action 构造 TD target。
- **恢复期 clip 归属**：按本设计，clip 进 cost 通道；r2 变成纯利润
  （`r2' = 10·shadow_profit`），不再混 clip。
- **c2' 量级变化 → 重新标定**：c2 加进 `100·clip_amount` 后量级变化（恢复期可达
  2000/步），`d_block_L`、lam2 上下界、curriculum 判据（连续满足次数）都要重标。
  不是公式问题，是超参问题。
- **Q_p 外推 transient**：actor 梯度要评估 `Q_p(s, π_intended)`，而 Q_p 只见过执行
  动作。训练早期（意图≠执行时）Q_p 的梯度不可靠。缓解：cost 通道会把意图拉向执行
  （收敛后两者一致，外推自然消失）；TD3 的 twin critics 压高估；可选升级——actor
  利润侧用 straight-through 投影评估 `Q_p(s, 投影(π))`（参考 `pernewact` 项目
  `f885853` 的 clamp/straight-through 探索）。

## 5. 风险与待决

- Q_p 在意图动作上的外推 transient 的严重程度（需实验观察）；
- 恢复期 clip 尖峰对 PER 采样优先级的影响（高 TD error 可能主导回放）；
- H 层同样存在错配（`penalty_NN_H` 在 r1 里、H 存 original_action 但 r1 依赖
  applied_action_H）——建议二期处理（H 的 CheckAction 很少触发，错配轻微）；
- 是否保留 r2 中的 clip 项作为即时教学信号（与 Q_c 双轨并行）还是完全移除；
- 与"恢复期 clip 留在 r2"（当前已实现的 RBB 回正机制）的取舍：本设计是把 clip
  从 r2 迁往 Q_c，两者并存会双重计罚，实现时需明确归属。

## 6. 状态

- [x] 概念设计（本文档）
- [ ] buffer 双 action 字段
- [ ] update_policy 按通道取 action
- [ ] r2 去 clip、c2 并入 clip
- [ ] d_block_L / lam2 重新标定
- [ ] H 层错配（二期）

实现需单独提交，不影响 RBB 回正机制。
