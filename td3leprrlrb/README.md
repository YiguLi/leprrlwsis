# TD3LEPRRLRB — LLM 分段微调 + 分段奖励 RL + 规则 BESS

`td3leprrlrb` 是 `td3prrlrb` 的 **LLM 增强版**，两者的关系与
`td3prrbb → td3prle` 完全对应：

| 对比 | 基线 | 增强版 |
|---|---|---|
| 分段奖励 + 学习型下层 | `td3prrbb` | `td3prle` |
| 分段奖励 + **规则**下层 | `td3prrlrb` | **`td3leprrlrb`（本项目）** |

与 `td3prrlrb` 的差异共两处：

1. **风速分段阈值动态化**（本项目主体）：`LOW_MAX / HIGH_MIN` 不再是写死的常量，而是
   **可变模块状态** —— 启动时由 `--seg` 定起点，训练中每隔一个窗口由 LLM 依据运行统计
   **在线微调**（硬约束：只能在原设定 ±1 m/s 内移动），阈值一变就立刻用新阈值
   **重算回放池里历史转移的 (r1, c1)**。
2. **`wB` 的 SOC 贴边折减取消**（`WB_SOC_OUTSIDE`: 0.7 → 1.0，与 `td3prle` 对齐）：
   上层 BESS 盈利不再随 SOC 位置打折，改由 SOC 目标机制承担该约束。

除此之外，分层结构、动作语义、奖励/成本公式（除第 2 点）、对偶上升、规则下层 —— 全部与
`td3prrlrb` 逐字一致，`td3prrlrb` 本体未做任何改动。

```
td3leprrlrb/
├── td3leprrlrb_main.py       训练主循环 (+ LLM 窗口 / relabel 触发 / 日志 / CLI)
├── td3leprrlrb_myfunc.py     动态阈值奖励 + relabel 辅助 + 9-case 测试
├── td3leprrlrb_llm.py        LLM 元控制器 (WindowStats + 提示词 + provider 工厂)
├── td3leprrlrb_buffer.py     LEPRRLRBBuffer (PER + relabel 上下文存储)
├── td3leprrlrb_ddpg.py       SP_TD3_LEPRRLRB (TD3, 唯一学习器, 供 H 层使用)
├── td3leprrlrb_rule_bess.py  RuleBasedBESS + SocTracker (与 td3prrlrb 逐字一致)
└── llm_config.json           LLM provider / model / api_key
```

## 1. 架构（与 td3prrlrb 一致）

| 层 | 实现 | 动作 | 频率 |
|---|---|---|---|
| H (上层) | TD3 (`SP_TD3_LEPRRLRB`, nb_states=4, nb_actions=6) | 5 台风机桨距 `[0,0.5]` + 1 个 **SOC 目标信号** `a6∈[-0.5,0.5]` | 每 5 步(块)一次 |
| L (下层) | **规则** `SocTracker` | BESS 功率 `pb` (MW, 正=充电) | 每一步 |

- SOC 目标：`soc_target = 3 + 5·a6 ∈ [0.5, 5.5]`（块内固定）。
- L 层的"物理可行 + 平抑波动"由 RBB 可行窗口硬保证，因此下层不需要 RL。

## 2. 新增的 LLM 分段微调层

### 2.1 阈值如何影响学习

风速档 `wind_level(u)` 同时决定两件事：

```
收益侧:  wT = WT_WIND[档] × WT_PRICE[电价]
              WT_WIND: LOW=1.2  MEDIUM=1.0  HIGH=0.9   ← HIGH 抑制激进
         wB = WB_PRICE[电价] × w_soc
              w_soc ≡ 1.0（SOC 贴边折减已取消，WB_SOC_OUTSIDE = 1.0）
              ⇒ wB 只由电价档决定，与 SOC 位置、风速档都无关
风险侧:  c1 的波动项乘 C1_RISK[档]
              C1_RISK: LOW=0.8  MEDIUM=1.0  HIGH=1.2
```

所以阈值本质上是一个**风险/收益的旋钮**：

- 调高 `HIGH_MIN` → 更少样本归入 HIGH → 风机收益权重抬升、波动风险系数下降
  → 盈利空间变大，风险敞口也变大；
- 调低 `HIGH_MIN` → 更保守，盈利空间缩小，但更安全；
- 调高 `LOW_MAX` → 更多样本归入 LOW（收益 ×1.2、风险 ×0.8）→ 中等风速被鼓励多发。

> **SOC 贴边折减已取消**（`WB_SOC_OUTSIDE`: 0.7 → 1.0，与 `td3prle` 对齐）：
> 原机制会让上层在 SOC 贴边时低估套利收益（软刹车）。现在这套约束由 SOC 目标机制
> 接管 —— H 给目标、下层规则跟踪、H 在块末按 `|目标−实际|` 计 `gap_pen` 付费，
> 收益侧不再额外打折。安全带判定 `SOC_SAFE_RANGES` 与其分支保留，把常量改回 `0.7`
> 即一键恢复（改回后 relabel 仍精确，因为 context 里保留了每子步的决策时刻 SOC）。

### 2.2 调整循环（默认每 7200 步 = 5 天一次）

```
窗口内逐步累积 WindowStats
   ↓  (step % llm_window == 0 且 episode >= --llm_min_episodes)
组装中文提示词: 当前阈值 + 阈值作用机理 + 调整带 + 窗口统计 + 最近 5 次调整历史
   ↓
LLM 返回一行 JSON: {"adjust_low_max": Δ, "adjust_high_min": Δ, "reasoning": "..."}
   ↓
钳制: 先夹到 ±1 m/s，再夹到【原设定 ±1 m/s】调整带，最后过 set_wind_thresholds 物理钳制
   ↓  (阈值真的变了才继续)
agent_H.memory.relabel_all(relabel_r1_substep, relabel_c1_substep)
   ↓
阈值/增量/窗口统计/relabel 条数 落盘 df_llm_adjust.csv + TensorBoard(llm/*)
   ↓
window_stats.reset()  进入下一个窗口
```

窗口统计 `WindowStats` 含：`r1_avg`、**真实利润 `profit_avg`**、越限率、不可行事件数、
**块末 SOC 目标罚（每块均值 / 出现罚的块占比）**、**块末 SOC 残差均值**，以及
LOW/MEDIUM/HIGH 三档的步数占比、`r1_avg`、`profit_avg`、越限率。

> 其中"块末 SOC 目标罚与残差"是**本项目特有**的指标：它刻画"上层下达的 SOC 目标是否被
> 下层规则在 5 步块内做到"。它不由风速阈值直接决定，但持续偏高说明上层在不可达目标上
> 反复试探，是提示词里的一类风险信号。

### 2.3 为什么必须 relabel

`r1` 与 `c1` 都依赖风速分档。阈值一动，同一条历史轨迹的档位就变了 —— 回放池里的
`(r, c)` 与"当前奖励函数"不再一致。若不重算，critic 会拿两套口径混在一起的目标去拟合
（非平稳奖励）。

`LEPRRLRBBuffer` 为每条 H 层转移（一个 5 步块）保存 5 个**子步上下文**：

```python
{'u', 'pr', 'turbine_raw', 'bess_raw', 'degradation', 'penalty_NN_H',
 'SOC', 'P_F', 'unfeasible_gap', 'gap_pen'}
```

其中 `turbine_raw = pr·ptur·dt`、`bess_raw = −pr·pbrule·dt`、`degradation`、
`penalty_NN_H`、`P_F` 项、`unfeasible_gap`、`gap_pen` 都与阈值无关（原样带回），
只有**分档权重 `wT`** 与**风险系数 `C1_RISK`** 需要重算（`wB` 自贴边折减取消后
也已与阈值无关，但重算 `wT` 仍然必要）。

`SOC`（决策时刻）字段保留在 context 里：当前 `WB_SOC_OUTSIDE = 1.0` 下它对结果没有
影响，但**不要删** —— 一是与 `td3prle` 的 relabel 签名保持一致，二是让"把折减改回
0.7"这条回退路径依然精确（那时 `wB` 会重新依赖 SOC 落在哪一档安全带上）。

relabel 之后所有优先级重置为 `max_priority` —— 原来存在 PER 树里的 TD error 是在旧奖励
下算出来的，已失效，重置后先均匀采样，等 TD error 在新奖励下重新收敛。

> **与 `td3prle` 的差异（本项目修正）**：`td3prle` 把块末的 SOC 跟踪罚写在块级、
> 没有放进 context，于是 `relabel_all()` 重算成本时会**静默丢掉这一项**。本项目把
> `gap_pen` 回填进块内最后一个子步的上下文并原样带回，保证 relabel 前后成本口径逐项一致。
> `LEPRRLRBBuffer.relabel_check()` 可做该不变量的自检（阈值未变时，重算与存档之差应 ≈0）。

### 2.4 LLM 不可用时的行为

任何一步失败都**不会**打断训练，而是退化为"占位模式"（增量 = 0，阈值不动）：

- 未装 provider SDK（如 `anthropic`）→ 建厂函数抛错 → 占位模式；
- 未配 api_key / 网络不通 / 鉴权失败 → 启动连通性自检失败 → 占位模式；
- 返回内容不是合法 JSON → `_parse_llm_response` 容错解析失败 → 本次增量 = 0。

此时整个项目的行为**等价于 `td3prrlrb`**（只是阈值起点可能是 `--seg` 给的值）。

## 3. 运行

```bash
cd HSRL
python td3leprrlrb/td3leprrlrb_main.py                 # 默认 train, 1000 episode
python td3leprrlrb/td3leprrlrb_main.py --seg expert    # 默认; 也可 percentile / kmeans
python td3leprrlrb/td3leprrlrb_main.py --llm_provider deepseek   # 覆盖配置文件
```

参数（在 `td3prrlrb` 的基础上新增）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--seg` | `expert` | 分段**起点**方法：`percentile`(33/66 分位) / `kmeans`(簇中心中点) / `expert`(8.5/12.0)。起点确定后由 LLM 在 ±1 m/s 内微调 |
| `--llm_provider` | 配置文件 | `anthropic` / `openai` / `deepseek` |
| `--llm_config` | `td3leprrlrb/llm_config.json` | 配置 JSON 路径 |
| `--llm_window` | 7200 | LLM 调整窗口长度（步）。7200 = 5 天；调小可用于快速验证 LLM 层 |
| `--llm_min_episodes` | 3 | 冷启动保护：episode 数达到该值后才允许 LLM 调整阈值 |

其余参数（`--warmup`、`--discount`、`--policy_delay`、`--soc_tol`、
`--rbb_fluctuation_limit`、`--rbb_rated_power` 等）与 `td3prrlrb` 完全一致。

### 3.1 启动自检（LLM 验证输出）

`__main__` 在解析完参数后**立刻**打印 LLM 分段微调层的自检结果（在分段计算、env/agent
构造、TensorBoard 建目录之前），所以你一启动就能确知"这次 run 到底会不会调阈值"：

```
==============================================================================
 LEPRRLRB: LLM 分段微调层 · 启动自检
==============================================================================
 LEPRRLRB: 配置 provider=deepseek model=deepseek-v4-flash api_key=sk-e0a...0747 (来源: llm_config.json)
 LEPRRLRB: LLM ready (provider=from config)
 LEPRRLRB: Testing LLM API connectivity ...
 LEPRRLRB: LLM API test PASSED (2.4s). Response: "OK"
==============================================================================
```

自检**不通过也不会中断训练**，只会降级为占位模式并给出排查顺序：

```
 LEPRRLRB: LLM API test FAILED — ImportError: No module named 'anthropic'
 LEPRRLRB: 训练将继续, 但【阈值调整会被跳过】(占位模式, 等价于 td3prrlrb)。
 LEPRRLRB: 排查顺序: --llm_provider / --llm_config 路径 → llm_config.json 的 api_key → provider SDK 是否安装 → 网络。
```

`train()` 通过 `llm_call_fn` 参数接收自检结果；传 `None` 即占位模式（阈值恒为起点值）。
`api_key` 只打印掩码（前 6 位 + 后 4 位）。

> 注意：MATLAB 引擎是在**模块 import 时**启动的（沿用兄弟项目的写法），所以引擎启动的
> 十几秒仍会排在自检之前；自检之后才轮到分段计算与 env 构造。

> `--seg expert` 给的是 8.5/12.0，与 `td3prrlrb` 写死的 7.0/11.0 不同。
> 想复现基线口径，用 `--seg percentile`（约 9.15/11.79）或直接把
> `td3leprrlrb_myfunc.WIND_LOW_MAX/WIND_HIGH_MIN` 改回 7.0/11.0（模块默认值即 7.0/11.0）。

## 4. 记录文件（输出目录 `output/TD3LEPRRLRB-runN/`）

| 文件 | 内容 |
|---|---|
| `df_td3leprrlrb.csv` | 每步全量（与 td3prrlrb 相同），**新增** `wind_low_max` / `wind_high_min` 两列，可逐步回溯分档依据 |
| `df_episode_td3leprrlrb.csv` | 每 episode：`episode_reward/cost/gap_pen/profit`、`deg_charge/discharge`、**当前阈值** |
| `df_test9.csv` | 每 480 步的 9-case 测试，**新增** `project_wind_levels`（该 case 在**当前**阈值下实际落入的档）与两列阈值 |
| `update_stats_H.csv` | H 层每块 2 批更新的 TD/Q/target/batch 统计均值 |
| **`df_llm_adjust.csv`** | **每次阈值调整一行**：`old/new_low_max`、`old/new_high_min`、`adjust_low_max`、`adjust_high_min`、`n_relabeled`、以及触发出该决策的窗口统计 |

TensorBoard 新增标量：`seg/wind_low_max`、`seg/wind_high_min`（逐步阈值轨迹）、
`llm/*`（窗口边界上的阈值、增量、是否变更、窗口统计）、`test9/wind_low_max|high_min`。

> 注意：CSV 采用**增量追加**落盘（`_flush_csv`），其中 `df_llm_adjust.csv` 与
> `update_stats_H.csv` 是"有事件才写"的表。

## 5. 与三个近亲项目的差异

| 项目 | 下层 | 风速分段 | LLM |
|---|---|---|---|
| `td3prrbb` | TD3 (5 维蒸馏观测) | 动态阈值（`--seg` 定起点） | 每 7200 步调阈值 + relabel |
| `td3prle` | TD3 (5 维蒸馏观测) | 动态阈值 | 同上 |
| `td3prrlrb` | **规则** SocTracker | **写死 7.0/11.0** | 无 |
| **`td3leprrlrb`** | **规则** SocTracker | **动态阈值**（`--seg` 定起点，LLM 在线微调） | 每 `--llm_window` 步调阈值 + relabel |

`td3leprrlrb` 与 `td3prrlrb` 的差异 = `td3prle` 与 `td3prrbb` 的差异（LLM 分段微调层），
另有两处刻意对齐/修正：

1. `WB_SOC_OUTSIDE` 0.7 → 1.0（取消 wB 的 SOC 贴边折减，与 `td3prle` 对齐，见 §2.1 注）；
2. relabel 的块末 SOC 罚口径（`gap_pen` 进 context，见 §2.3 注；`td3prle` 会丢这一项）。

## 6. 待标定 / 风险点

1. **窗口长度 vs 墙钟时间**：单步要调两次 WFSim（本项目量级 ~1–4 s/步），7200 步 ≈ 6–8 小时
   才会触发一次 LLM。若要观察 LLM 层的效果，请用 `--llm_window 200 --llm_min_episodes 0`
   之类的短窗口先跑通，再回到 7200。
2. **LLM 只看窗口统计，不看梯度**：阈值调整是"慢回路"，与 H 的"快回路"（每块 2 次更新）
   时间尺度相差 3 个数量级，短期内看不出效果属于正常。
3. **±1 m/s 调整带**：这是硬约束（提示词与代码双重钳制）。若起点用 `expert`(8.5/12.0)，
   可探索区间是 LOW∈[7.5,9.5]、HIGH∈[11,13]；换个 `--seg` 起点即换搜索区间。
4. **api_key 明文**：`llm_config.json` 里当前是明文 key（沿用 `td3prle` 的既有约定）。
   建议后续改为环境变量（`DEEPSEEK_API_KEY` / `OPENAI_API_KEY`，`load_llm_config` 已支持
   回退）并把该文件加入 `.gitignore`。
5. **relabel 成本**：每次阈值变更要遍历整个回放池（`--rmsize` 条 × 5 子步）重算，
   20000 条量级约数十毫秒，相对 WFSim 的单步开销可忽略。
6. **`set_wind_thresholds` 的极端输入**：该函数从 `td3prrbb`/`td3prle` 逐字继承，
   两个钳制是顺序施加的（先 `low_max ∈ [3, high_min-1]`，再 `high_min ∈ [low_max+1, 22]`），
   因此对**极端离谱的请求**（如 `set(30, 31)`）可能返回 `low_max > high_min`。
   经 `--seg` + ±1 m/s 调整带这条链路不可达，故保留与兄弟项目一致的行为以便横向对比。

## 7. 已验证内容

验证分两层（均可离线运行，第二层用**合成 MATLAB 引擎**跑通真实 `train()` 主循环，
不需要 WFSim）。

### 7.1 单元 / 不变量层

- **relabel 精确性**：`relabel_r1_substep` / `relabel_c1_substep` 在 400 组随机
  (u, pr, E, P_G, ptur, pb, penalty) 上与在线奖励逐位一致（|Δ| < 1e-9）；
  阈值移动引起的 LOW↔MEDIUM↔HIGH 换档，其权重变化与手算一致
  （wT 1.2→1.0、`C1_RISK` 0.8→1.0）。
- **`wB` 与 SOC/风速档解耦**（贴边折减取消后的新不变量）：遍历 3 个电价档 × 3 个
  风速档 × SOC 全量程，`get_wB` 只随电价档变化；`relabel_r1_substep` 传入不同 SOC
  得到完全相同的结果。
- **上下文完备性**：真实 5 步块（目标 SOC 刻意不可达 → `gap_pen = 15.83 > 0`）入池后，
  `relabel_check()` 报 `|Δr| = 3.4e-13`、`|Δc| = 0`；阈值移动后能检出陈旧
  （`|Δr| = 137.2`、`|Δc| = 0.04`）；`relabel_all()` 后重新一致。
  **这正是 `td3prle` 会丢项的地方**（它把块末 SOC 跟踪罚留在块级、未入 context）。
- **±1 m/s 硬约束**：反复施加 +0.4/−0.5 的增量，阈值序列
  8.5→8.9→9.3→9.5(saturate) 与 12.0→11.5→11.0(saturate) 完全落在带内；
  超大请求被夹回对侧边界。
- **解析容错**：合法 JSON / ```json 围栏 / 前后夹带解释文字 / 数值越界 / 垃圾文本 /
  空串 / `None` / 调用抛异常 —— 全部确定行为（多为 (0,0)），不打断训练。
- **provider 工厂**：`deepseek`/`openai` 可建客户端；未装 SDK（`anthropic`）
  抛 `ImportError` → 主循环退化为占位模式；未知 provider 抛 `ValueError`。

### 7.2 主循环端到端层

- **链路 A（LLM 可用）**：桩 LLM 每次请求 (+0.4, −0.5) →
  两个窗口边界（step 25/50）各触发一次真实调整；
  阈值链、`df_llm_adjust.csv`（含窗口统计与 `n_relabeled`）、
  逐步/每 episode 的阈值列、`df_test9` 的 `project_wind_levels` 全部正确；
  `relabel_all()` 分别在 5 / 10 条转移上执行；
  跑完后回放池**在新阈值下自洽**：全池 12 条 `|Δr| ≤ 7.6e-06`、`|Δc| ≤ 5.6e-17`。
- **链路 B（LLM 不可用）**：建厂函数抛 `ImportError` → 占位模式：
  阈值恒为 8.5/12.0、`relabel_all` 从不调用、不产生 `df_llm_adjust.csv`、
  训练正常跑完（行为等价于 `td3prrlrb`）。
- **CLI 默认值**：`--env TD3LEPRRLRB`（**不会写进 `TD3PRRLRB-runN`**）、
  `--seg expert`、`--llm_window 7200`、`--llm_min_episodes 3`，
  父项目参数（`--warmup/--rmsize/--policy_delay/--policy_noise/--soc_tol/--rbb_*`）原样保留。

### 7.3 一个继承来的数值细节（非本项目 bug）

链路 A 里 `|Δr|` 的量级是 `1e-6`，而不是单元层的 `1e-13`。原因**不是** relabel 少算了项：

- `CheckAction` 的输入是 actor 输出（float32）；在 NumPy ≥ 2 的弱标量规则下，
  `penalty_NN_H` 保持 **float32**（只要没顶到 50 的上限）；
- 于是 `Cal_r1_PRRBB` 末尾的 `… - penalty_NN_H` 把在线 `r1` 也变成 **float32**，
  写进回放池的奖励因此带 ~1e-5 的量化；
- 而 relabel 走 float64（context 里的 `penalty_NN_H` 以 Python float 存），
  所以两者相差 ~1e-6 ≈ **不到该量级下 1 个 float32 ulp**（实测 7.6e-6 < ulp 1.5e-5），
  相对误差 3e-8。

该行为在 `td3prrlrb`（以及同一族的 `td3prrbb`/`td3prle`）中完全相同，属继承细节。
奖励量级 O(10~1000)，量化步长对 TD 学习无实质影响；本项目**保持不变以维持与父项目可比**。


