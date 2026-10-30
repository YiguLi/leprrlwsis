# aifflor 线性行为分支（`--lin_wt`，结构版）

> 一句话：**actor 输出层只有 3 个神经元 `a0` / `a4` / `g`**；中间三台由 `a0` 到 `a4`
> 线性插值展开（保留两位小数）；critic 仍吃 6 维动作，其余训练口径不变。
>
> 开关：`--lin_wt 1`（默认 0）；小数位：`--lin_wt_decimals 2`（默认 2）。
> 契约测试：`python agentutil/test_aifflor_lin_wt.py`。

---

## 1. 动作定义

上层 H 的动作是 6 维：5 台风机 + 1 个 `g`（HSRL 的块总放电量）。

```
actor 输出（3 维） = [a0, a4, g]

展开到 6 维：
    Δ = a4 − a0
    a1 = round(a0 + 0.25Δ, 2)
    a2 = round(a0 + 0.50Δ, 2)
    a3 = round(a0 + 0.75Δ, 2)
    action = [a0, a1, a2, a3, a4, g]
```

例：`a0=0.12, a4=0.25, g=-0.7` ⇒ `[0.12, 0.15, 0.19, 0.22, 0.25, -0.7]`。

* **actor 没有中间三个输出单元**：中间维是 `a0/a4` 的确定性函数
  （`model.expand_lin_wt`），因此不存在"中间三台各自的 actor 行为/梯度"。
* 首尾维**不取整、不 clip**：越界仍由 `check_action_H_hsrl` / `CheckAction`
  裁 + `pen5` 罚。首尾越界会连带推出中间维越界，`pen5` 按实际越界维数累计。
* 第 6 维 `g` 是 actor 的第 3 个输出，**完全独立**。

### 1.1 有没有"中间 actor 行为 / 梯度"？

**没有。** 这是结构版与"6 输出 + 投影覆盖"版的关键区别：

| | 结构版（当前） | 投影覆盖版（旧提案） |
|---|---|---|
| actor 输出层 | **3 个神经元** `a0/a4/g` | 6 个神经元，第 1..3 维被覆盖 |
| 中间维来源 | `expand_lin_wt(a0,a4)` | `proj_lin_wt` 覆盖 |
| 中间输出单元 | **不存在** | 存在但永远不被使用 |
| `actor_*.pkl` | **`actor_3.pkl`** | `actor_6.pkl`（形状其实 3） |
| 中间维梯度 | 通过 `∂a_i/∂a0`、`∂a_i/∂a4` 回到 **a0/a4** | 同左，但另有 3 个死输出单元 |

即：中间动作值存在（要送进 env / critic），但**不是 actor 的自由输出**；它们对 Q 的
影响按链式法则全部记到 `a0/a4` 头上。actor 学的是"两个端点如何决定整条线性型线"。

---

## 2. 为什么是"首尾 + 线性插值"

`agentutil/probe_turbine_wake_attribution.py`（单排 5 台、FLORIS 真值）给出的
动作灵敏度结构：

| 机组 | 净 `∂P/∂a` | 自身项 / 下游耦合 | 真实 `∂c1_block/∂a` 符号稳定率 |
|---|---:|---|---:|
| T1（首端） | **+1.506** | 9.252 / −7.746（抵消 84%） | **84%** |
| T2 | −0.069 | 自身与耦合近抵消 | 43% |
| T3 | −0.034 | 耦合 **超过**自身 ⇒ 符号反 | 48% |
| T4 | −0.187 | — | — |
| T5（末端） | **+2.957** | 下游耦合恒 0 | **90%** |

⇒ 中间三台对"动作 → 波动成本"几乎是**近零/符号不可辨识方向**；两端才是稳定、
量级大的通道。线性行为分支把 5 维压成"两端 2 维 + 整排线性型线"：

1. 去掉中间三个近零方向，降低 actor 要辨识的维数；
2. 保留整排动作的**连贯形状**（不是 `--scalar_wt` 的"五台全相等"）；
3. 型线是 actor 输出的确定性函数 ⇒ critic 仍可用同一个 6 维动作接口，
   不需要改 critic 结构、奖励或日志口径。

---

## 3. 与现有动作流形开关的区别

| 开关 | 流形 | actor 输出维 | 中间维 |
|---|---|---:|---|
| 默认 | 5 维独立 | 6 | 自由 |
| `--farm_level 1` | `a_i = s·c_i`（`FARM_CENTRE`） | 6 | 射线决定，保留形状 |
| `--scalar_wt 1` | `a_i = a_0` | 6 | 全部等于首端 |
| **`--lin_wt 1`** | **首尾 + 线性插值** | **3（a0/a4/g）** | **线性型线，两位小数** |

三者**互斥**（`aifflor_main.py` 里 assert）：同时开会让"到底投影到哪条流形"
不可判读。

---

## 4. 实现落点（缺一处就会被绕过）

| 位置 | 作用 |
|---|---|
| `model.expand_lin_wt` / `expand_lin_wt_np` | **3 维 actor 输出 → 6 维动作**（中间线性插值 + 两位小数） |
| `model.proj_lin_wt` / `proj_lin_wt_np` | 把**完整动作**（warmup 的 5/6 维）投影回首尾 + 线性型线 |
| `model._round_half_away_ste` / `_round_half_away_np` | 四舍五入 + straight-through |
| `SP_TD3.actor_nb_actions` | `2 + (nb_actions − na_box)`（上层 = 3）；下层 L 自动回退 1 |
| `SP_TD3._expand` | 所有 actor 前向的统一出口：lin 展开 / 其余走原投影 |
| `SP_TD3.select_action` | 只在 `[a0,a4,g]` 上加探索噪声，再展开 |
| `SP_TD3._target_smoothed_action` | 只在 `[a0,a4,g]` 上加 TD3 target smoothing 噪声，再展开 |
| `SP_TD3.random_action` | warmup 直接采 `[a0,a4]∈[0,0.5]²`、`g∈[-0.5,0.5]`，再展开 |
| `SP_TD3.save/load/save_checkpoint` | actor 文件名用 `actor_tag`（lin = **3**） |
| `aifflor_main._apply_lin_wt` | warmup 调度 / `diag_fix` / `explore_mix` 之后把完整动作投影回线性型线 |
| `aifflor_main.init_wt_action_bias` | **3 输出映射**：`a0/a4` ← `U(lo,hi)`，`g` ← `sig_value` |
| `train()` / `train_hsrl()` | 两条主循环的 warmup 与 actor 路径都接线 |

**下层 BESS 不受影响**：L 层只有 1 维动作，`SP_TD3` 构造时按
`nb_actions >= na_box` 自动把 `lin_wt` 置 `False`；否则会把 1 维动作误扩成 5 维。

**critic 不变**：仍按 6 维动作构造；`actor_3.pkl` 只影响 actor 的 checkpoint，
`critic_*_6.pkl` 文件名与形状都不变。

---

## 5. 两位小数：前向量化 + STE

直接 `round` 会把中间维对 `a0/a4` 的梯度贡献清零（`round/floor` 导数几乎处处 0）。
因此展开用 **straight-through**：

```python
y = sign(x) * floor(|x| * 10^d + 0.5 + 1e-6) / 10^d
return x + (y - x).detach()          # 前向 = y，反向 = 1
```

* **前向**：actor 评估点、replay 动作、实际下发 env 的动作都是两位小数；
* **反向**：按连续线性插值传梯度 ⇒ `a0/a4` 仍可学；
* `1e-6`（缩放后 = 动作量 `1e-8`）只吸收 float32 在 `.xx5` 边界上的表示抖动
  （`0.12` 实际是 `0.1199999973…`），不改变正常四舍五入；
* 插值与取整中间量用 **float64** 计算再写回原 dtype ⇒ torch 与 numpy 对同一
  输入给出同一位小数。

> 取舍：量化使"动作 → 奖励"在 `0.01` 网格上分段常值。STE 是让梯度仍然可用的
> 标准做法；它**不改变执行的动作**，只改变反向传播的近似。

---

## 6. 训练期行为

* **探索**：actor 输出 3 维 → 只在这 3 维上加噪声 → 展开；中间维没有独立噪声。
  wide 模式的两个端点仍用 `U[wt_noise_min, wt_noise_max]` 幅度、按 `[0,0.5]`
  截断噪声；`g` 仍用 `sig_noise_max → sig_noise_min`。
* **target smoothing**：同样只对 3 个自由维加 clipped Gaussian 噪声，再展开
  ⇒ 目标动作始终在线性流形上。
* **warmup**：`random_action` 直接采 3 维；`WarmupSchedule` 产生的完整 6 维动作
  在 `_apply_lin_wt` 处投影回首尾 + 线性型线（`unified` / `lhs` 的端点分布不变）。
* **初始化**：`init_wt_action_bias` 把 3 个输出 bias 映射为
  `a0,a4 ~ U(0.05,0.45)`、`g = 0.0`。
* **两条主循环都接**：旧 `train()` 与 `train_hsrl()` 都在动作检查前调用；
  不开 `--lin_wt` 时该函数立即返回，旧路径逐位不变。
* **checkpoint**：`actor_3.pkl`；critic 仍 `critic_*_6.pkl`。旧分析工具若写死
  `actor_6.pkl`，需要把文件名改成 `actor_3.pkl` 并注意 actor 只有 3 个输出。
* **`--branch_*`**：branch 的动作集是独立生成的，**不**经过线性投影
  （"其他不变"）。同时开 `--branch_every` 时 critic 会同时看到流形外的 branch
  动作；只想验证线性分支建议 `--branch_every 0`。

---

## 7. 已知局限（刻意保留）

1. **型线被强制为线性**：真实最优型线未必线性（`FARM_CENTRE` 就不线性）。
   这条分支测的是"两端 + 线性"这个结构假设，不是"最优型线"。
2. **只有 2 个自由度**：中间三台无法独立微调；需要非单调型线时这条臂会系统性吃亏。
3. **量化**：两位小数 ⇒ 中间维在带内是 51 级格点；首尾维仍连续。
   `--lin_wt_decimals` 可改（`<0` 关闭取整），但改小数位会改变动作集，
   跨 run 不可直接比。
4. **首尾越界的连带罚**：首尾维越界会推出中间维越界，`pen5` 可能一次计 3~5 维；
   这是"2 维驱动 5 台"的必然结果，不是 bug。
5. **checkpoint 不兼容**：`actor_3.pkl` 不能与 6 输出 actor 互相 warm-start
   （`fc4` 形状不同）。

---

## 8. 用法与验证

```powershell
# 线性行为分支（结构版，HSRL 口径）
python aifflor/aifflor_main.py --hsrl --lin_wt 1 --seed 0

# 短程冒烟
python aifflor/aifflor_main.py --hsrl --lin_wt 1 --max_steps 300

# 契约测试（只读；单点 FLORIS 求值，不跑 episode）
python agentutil/test_aifflor_lin_wt.py
```

契约测试覆盖：

* 展开/投影语义（首尾原样、中间两位小数、`g` 不动、不 clip、幂等、无 `g` 的 5 维动作）；
* actor 结构（3 输出 / 3447 参数 / 自由维索引 / 默认臂仍 6 输出 / 下层 1 维自动跳过）；
* 梯度（`a0/a4/g` 三路都有梯度；没有中间输出单元；STE 前向取整/反向恒等）；
* agent 层（探索噪声、`random_action`、target smoothing 全在流形上）；
* checkpoint（`actor_3.pkl` + `critic_*_6.pkl` 保存/加载）；
* 物理（同首尾、其余维任改 ⇒ 出力逐位相同）；
* 初始化映射（3 输出：`a0/a4` 带内、`g=0`；默认 6 输出不变）；
* 接线（旧 `train()` 与 `train_hsrl()` 都调用 `_apply_lin_wt`，CLI/互斥/结构版接线存在）。
