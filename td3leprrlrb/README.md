# TD3LEPRRLRB — LLM 分段微调 + 分段奖励 RL + 规则 BESS

`td3leprrlrb` 是 `td3prrlrb` 的 **LLM 增强版**，两者的关系与
`td3prrbb → td3prle` 完全对应：

| 对比 | 基线 | 增强版 |
|---|---|---|
| 分段奖励 + 学习型下层 | `td3prrbb` | `td3prle` |
| 分段奖励 + **规则**下层 | `td3prrlrb` | **`td3leprrlrb`（本项目）** |

与 `td3prrlrb` 的差异共六处：

1. **风速分段阈值动态化**（本项目主体）：`LOW_MAX / HIGH_MIN` 不再是写死的常量，而是
   **可变模块状态** —— 启动时由 `--seg` 定起点，训练中每隔一个窗口由 LLM 依据运行统计
   **在线微调**（硬约束：只能在原设定 ±1 m/s 内移动），阈值一变就立刻用新阈值
   **重算回放池里历史转移的 (r1, c1)**。
2. **`wB` 的 SOC 贴边折减取消**（`WB_SOC_OUTSIDE`: 0.7 → 1.0，与 `td3prle` 对齐）：
   上层 BESS 盈利不再随 SOC 位置打折，改由 SOC 目标机制承担该约束。
3. **P0.1 训练起始点**：把 H 层 actor 输出层 bias 覆写到 Cp 高效区（`--wt_bias_init`，默认 1）。
4. **P0.2 罚项不封顶**：`CheckAction` 里 `penalty = min(penalty, 50.0)` 已删除。
5. **P0.3 24h 相位特征**：H 观测从 4 维扩到 4/6/9 维（`--phase_feat`，默认 1）。
6. **SOC 死区归属 + 自愿功率限幅**：下层 `SocTracker` 不再有跟踪死区
   （`--tracker_dead_zone` 默认 **0.0 = 尽力追随**，残差 0.01 MWh 也追）；0.1 MWh 的豁免
   **只留给上层** gap 罚（`--soc_tol`）；同时下层自愿指令限幅到波动限值
   （`--tracker_pb_cap` 默认 auto=3 MW），以消除"自致过冲"。详见 §9。

3~5 的动机 / 实现 / 验证见文末 **§8 P0 三项改动**，第 6 项见 **§9**。
除此之外，分层结构、动作语义、对偶上升 —— 全部与 `td3prrlrb` 逐字一致，
`td3prrlrb` 本体未做任何改动。

```
td3leprrlrb/
├── td3leprrlrb_main.py       训练主循环 (+ LLM 窗口 / relabel 触发 / 日志 / CLI)
├── td3leprrlrb_myfunc.py     动态阈值奖励 + relabel 辅助 + 9-case 测试
├── td3leprrlrb_llm.py        LLM 元控制器 (WindowStats + 提示词 + provider 工厂)
├── td3leprrlrb_buffer.py     LEPRRLRBBuffer (PER + relabel 上下文存储)
├── td3leprrlrb_ddpg.py       SP_TD3_LEPRRLRB (TD3, 唯一学习器, 供 H 层使用)
├── td3leprrlrb_rule_bess.py  RuleBasedBESS (逐字一致) + SocTracker (跟踪死区默认 0)
└── llm_config.json           LLM provider / model / api_key
```

## 1. 架构

| 层 | 实现 | 动作 | 频率 |
|---|---|---|---|
| H (上层) | TD3 (`SP_TD3_LEPRRLRB`, nb_states=4/6/9, nb_actions=6) | 5 台风机桨距 `[0,0.5]` + 1 个 **SOC 目标信号** `a6∈[-0.5,0.5]` | 每 5 步(块)一次 |
| L (下层) | **规则** `SocTracker`（**无跟踪死区**，尽力追随） | BESS 功率 `pb` (MW, 正=充电) | 每一步 |

- SOC 目标：`soc_target = 3 + 5·a6 ∈ [0.5, 5.5]`（块内固定）。
- L 层的"物理可行 + 平抑波动"由 RBB 可行窗口硬保证，因此下层不需要 RL；
  0.1 MWh 的**豁免只给上层** gap 罚（见 §9）。

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
| `--wt_bias_init` | **1** | **P0.1** 起始点：1 = 把 H 层 actor/actor_target 输出层 bias 覆写为 `WT∈U(0.25,0.40)` + 信号维 0（Cp 高效区）；0 = 保留 PyTorch 默认 `U(±0.25)`（复现旧 run） |
| `--phase_feat` | **1** | **P0.3** 24h 相位：0 = 无相位（H 观测 4 维）/ 1 = `[sin, cos]`（6 维）/ 2 = 再 +`[段内剩余比例, 剩余小时/8, 下一段电价/1000]`（9 维） |
| `--soc_tol` | 0.1 | **上层** gap 罚的豁免带（MWh）：H 只对超过它的目标残差付费（与下层跟踪死区**已解耦**） |
| `--tracker_dead_zone` | **0.0** | **下层** `SocTracker` 的跟踪死区（MWh）。0 = 尽力追随（残差 0.01 也追，δ=0 时只做窗口允许的最小被迫吞吐）；>0 复现旧的"最小吞吐"行为，但必须 ≤ `--soc_tol` |

其余参数（`--warmup`、`--discount`、`--policy_delay`、`--soc_tol`、
`--rbb_fluctuation_limit`、`--rbb_rated_power` 等）与 `td3prrlrb` 完全一致。

> ⚠️ `--phase_feat` 会改变 H 观测维数 → **换档后旧 run 的 `.pkl` 权重不可直接复用**
> （`actor` 第一层形状不同）。`--phase_feat 0` 才是与 td3prrlrb / 旧 run 完全同构的档位。

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
| `df_td3leprrlrb.csv` | 每步全量（与 td3prrlrb 相同），**新增** `wind_low_max` / `wind_high_min` 两列（回溯分档依据）+ **P0.3 的 6 列相位**：`phase_minute`、`phase_sin`、`phase_cos`、`seg_remain_frac`、`seg_remain_hours`、`next_price` |
| `df_episode_td3leprrlrb.csv` | 每 episode：`episode_reward/cost/gap_pen/profit`、`deg_charge/discharge`、**当前阈值** |
| `df_test9.csv` | 每 480 步的 9-case 测试，**新增** `project_wind_levels`（该 case 在**当前**阈值下实际落入的档）、两列阈值、`phase_minute`（该 case 用的代表相位，OVERALL 行为 -1） |
| `update_stats_H.csv` | H 层每块 2 批更新的 TD/Q/target/batch 统计均值 |
| **`df_llm_adjust.csv`** | **每次阈值调整一行**：`old/new_low_max`、`old/new_high_min`、`adjust_low_max`、`adjust_high_min`、`n_relabeled`、以及触发出该决策的窗口统计 |

TensorBoard 新增标量：`seg/wind_low_max`、`seg/wind_high_min`（逐步阈值轨迹）、
`llm/*`（窗口边界上的阈值、增量、是否变更、窗口统计）、`test9/wind_low_max|high_min`、
以及 **P0.3 的 `phase/*`**（`minute/sin/cos/seg_remain_frac/seg_remain_hours/next_price/n_states`）。

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

> P0.2（罚项不封顶）之后，`penalty_NN_H` 在深越界时可达千级，r1/c1 的 float32 量化
> 绝对幅度会同步变大，但相对误差量级不变（仍是 ~1e-8）。

---

## 8. P0 三项改动（起始点 / 罚项不封顶 / 24h 相位）

三项改动针对三个**独立病灶**，全部落在 `td3leprrlrb/` 内；`env.py`、`model.py`、
`td3prrbb/`、`td3prrlrb/` 零改动。

### 8.1 P0.1 训练起始点（`--wt_bias_init`，默认 1）

**病因**：`model.Actor.init_weights()` 只初始化 `fc1/fc2/fc3` 的 weight 与 `fc4.weight`
（±`init_w`=0.003），`fc4.bias` 仍是 PyTorch 默认 `U(±1/√fan_in)`，`fan_in = hidden1 = 16`
⇒ `U(±0.25)`。而 `WFSim_port.m` 给出 `Cp = 4a(1−a)²`，**a=0 就是停机（Cp=0）**。
初始策略的"基准工作点"≈ bias 本身，于是初始 ΣCp 只有 ~0.4 甚至为负（负 bias 被
`env.PowerOutput` 的 `clip(a,0,0.5)` 夹成 0），实测 5 个种子里 2~4 台机永久停机。

**处方**：构造 agent 后覆写 `actor` 与 `actor_target` 的 `fc4.bias` —— 前 5 维（桨距）
取 `U(0.25, 0.40)`（Cp 高效区，a=1/3 时最优 Cp=16/27≈0.5926，该区间正好夹住它），
第 6 维（SOC 目标信号）置 0 ⇒ 初始目标 SOC = 3.0，中性且不泄露策略。只改 bias，
`fc4.weight` 仍为 ±0.003，所以"策略强度"依然随机，只是"基准工作点"落在高效区。

**启动即打印**（诊断口径：a 先按 `clip(0,0.5)` 还原成风机实际拿到的桨距，再算 Cp）：

```
 LEPRRLRB[P0.1]: WT bias <- U(0.25,0.40) x5 + 信号维 0; 初始 sum_Cp = 2.904
   (效率 98.0%, 最优 2.963), 死机 0/5 台; 逐台 Cp = [0.5923, 0.5898, 0.5712, 0.5864, 0.5646]
```

实测：8 个种子下初始效率 **98.1%~99.7%**、死机 **0/5**；而 PyTorch 默认起点在
不同种子上效率 0%~14%、死机 3~5/5 台。`--wt_bias_init 0` 回到默认起点（并打印同样的
诊断行，便于 A/B）。

> 副作用（正面）：起点落在界内，warmup 之后 `penalty_NN_H` 基本恒为 0
> （旧起点下几乎每步都在越界）。

### 8.2 P0.2 罚项不封顶

**病因**：`CheckAction` 的 H 分支原为 `penalty = min(penalty, 50.0)`。封顶后"越界区"在
**收益通道**（r1 里 `clip(action, 0, 0.5)` 后出力饱和，梯度已是平的）与**成本通道**
（`min` 封顶后罚值恒定，梯度为 0）**两侧同时失去梯度** ⇒ 越界维度没有任何恢复力，
被共享隐层"顺带拖走"成随机游走（run160 实测 `a2/a3/a4` 漂到 −3.9/−4.7/−6.4，
三台机永久停机、ΣCp 仅 30.8%，而真实越界罚 1600~2400 只被记成 50）。

**处方**：删掉那一行 `min`，系数 100/MW 不变。每单位越界恒定产生 100 的恢复梯度，
任意深度都不衰减。实测同一动作族在不同越界深度下罚值：

| 越界深度 a | 0.6 | 0.8 | 1.0 | 2.0 | 4.0 | 6.9 |
|---|---|---|---|---|---|---|
| `penalty_NN_H` | 310 | 430 | 550 | 1150 | 2350 | 4090 |

（深度差 → 罚差恒等于 `100 × 6 × Δa`，斜率不打折；界内动作罚 0。）

**代价**：越界期 r1/c1 出现千级尖峰 ⇒ λ1 会短暂顶格、PER 会优先重放这些样本。
**没有开关**；如需回退，`git revert` 该次提交中对 `CheckAction` 的改动即可。

### 8.3 P0.3 24h 相位特征（`--phase_feat`，默认 1）

**病因**：套利里"方向/持留"只需当前电价（run159 的"持留—倾泻"就是相位盲学出来的），
但**择时** —— 在低价窗最后一刻把电充满 —— 必须知道"本段还剩多久、下一段涨跌"。
算术：下层规则速率 ≈0.245 MWh/块 ⇒ 充满 5 MWh 需 1.7 h，而低价窗长达 7~8 h。
贴现比较：提前 1.7 h 充电只需 γ ≥ 0.951，而相位盲的"见 300 就充"（提前 ~7 h）
需要 γ ≥ 0.988。

**处方**：H 观测 = `Normalize(obs) ⊕ 相位特征`。

| `--phase_feat` | 特征 | H 观测维数 |
|---|---|---|
| 0 | 无（复现旧 run） | 4 |
| 1（默认） | `[sin(2π·m/1440), cos(2π·m/1440)]` | 6 |
| 2 | 1 的全部 + `[段内剩余比例, 剩余小时/8, 下一段电价/1000]` | 9 |

`m = episode_steps % 1440`（已核对：`env.reset()` 里 `count_price=0`、`step()` 恒
`done=False` ⇒ episode 恰好 1440 步）。**为什么用 sin/cos 而不是 1 维锯齿**：锯齿跨零点
不连续，同结构网络拟合 MSE 0.047 vs 0.0008（差 60 倍）、判对率 82.5% vs 99.7%。

相位日程与 `env.CalPrice` 的判档规则一致，且**跨零点同价段合并**（23:00–24:00 与
0:00–7:00 都是 300 ⇒ 合成一段 8 h）：

| 当日分钟 m | 段内剩余比例 | 剩余小时/8 | 下一段电价 |
|---|---|---|---|
| 0 (00:00) | 420/480 = 0.875 | 7/8 = 0.875 | 980 |
| 420 (07:00) | 1.0 | 4/8 = 0.5 | 580 |
| 1380 (23:00) | 1.0 | 8/8 = 1.0 | 980 |
| 1439 (23:59) | 421/480 | 421/480 | 980 |

**9-case**：每步按该 case 的**价格档**映射到该档内的代表相位（300→03:30、580→14:00、
980→09:00），避免"价格档 = LOW 却给 14:00 相位"这种训练中不会出现的 OOD 组合。

**电价时钟自检**（启动打印）：用 `env.CalPrice` 走满一天与日程表比对。结论是
**1439/1440 一致，唯一例外在第 1440 分钟** —— `env.CalPrice` 是"先自增再判档"，
`count_price == 24*60` 时落入 `else` 档 → 980（而不是 300），且该值恰好落在 episode
的最后一步。相位特征仍按**人类时钟**（index = 当日第几分钟）给出，因此相对 env 时钟
领先 1 分钟（1/480 的段内比例偏差，对择时无实质影响）：

```
 LEPRRLRB[P0.3]: 电价时钟自检 1439/1440 分钟与日程表一致; 唯一例外在第 1440 分钟
   (env.CalPrice 在 count_price==1440 时落入 else 档 -> 980, 属 env 已知怪癖, 落在 episode 最后一步)
 LEPRRLRB[P0.3]: 相位特征用人类时钟 (index = 当日第几分钟); env 时钟领先 1 分钟
 LEPRRLRB[P0.3]: --phase_feat=1 -> 相位特征 2 维, H 观测 6 维 (代表相位: {300: 210, 580: 840, 980: 540})
```

> 与原始改动说明的一处差异：说明里写"自检 520/520 步 mismatch = 0"。按上面实测，
> 若自检**逐分钟严格比价**，人类时钟表在 6 个边界分钟上会与 env 差 1 分钟、外加第 1440
> 分钟那处怪癖 —— 这两个事实不可能同时给出"0 mismatch"。本项目选择：**特征用人类时钟**
> （从而复现 23:00→8h / 7:00→4h&580 / 0:00→7h 三条语义），**自检按 env 的 +1 分钟口径
> 对齐并如实报出那一处怪癖**。若你那边的实现是严格 0 mismatch，很可能是把日程表也按
> `classify(m+1)` 建的表 —— 代价是段长会变成 59 分钟而非 8 小时。

### 8.4 未改动的部分

`λ1∈[3,100]` 初值 5、`d_block_H` 50/地板 5、`warmup 200`、PER α=0.6、退化口径
（490 对全部吞吐计费）、H 每块 2 次更新、RBB 参数、`env.py`、`model.py`、
`td3prrbb/`、`td3prrlrb/` —— 一律未动。P1（输出有界化）未做，作为后手
（它与 P0.2 是替代关系，且 sigmoid 会顺带解决 P0.1）。

> 例外：下层 `SocTracker` 的跟踪死区默认值在 §9 中由 0.1 改为 0（这是独立于 P0 的
> 另一处设计修正），`RuleBasedBESS` 本体仍与 `td3prrlrb` 逐字一致。

### 8.5 必看指标

| 指标 | 改前（run159/160） | 期望 |
|---|---|---|
| 死机台数 / ΣCp 效率 | 0~3/5，31%~57% | 稳态 ~0/5，>90% |
| `penalty_NN_H` | 恒顶格（50），53%~93% 的步数 | 均值 <5；越界时出现尖峰但能衰减回落 |
| 低价段 `soc_target` | ~1.5 MWh | ≥3 MWh |
| 7:00 前 1–2 h 目标是否上冲 | 无（相位盲） | 上冲 = 择时套利生效的直接签名 |
| 980 段前 1 h 平均 `pbrule` | ≈0 | 显著 > 0 |

两点提醒：① 早期阶段与旧 run **不可直接对比**（风机从"半停机"变成正常工作点 ⇒
出力、波动、被迫 BESS 吞吐都会更大）；② 不封顶罚项使 r1/c1 在越界期出现千级尖峰，
λ1 会短暂顶格、PER 会优先重放这些样本 —— 若恢复慢，再把 `--PER_alpha` 从 0.6 降到 0.4~0.5。

### 8.6 回退开关

| 项 | 回退方式 |
|---|---|
| P0.1 | `--wt_bias_init 0` |
| P0.3 | `--phase_feat 0` |
| P0.2 | 无开关，`git revert` 该次提交中对 `CheckAction` 的改动 |

### 8.7 建议的下一步 A/B

用 `--wt_bias_init 0` 跑 3–5 个种子，看坏种子的尾巴是否消失 —— 消失则说明 P0.2 的
恢复力已足够、起点降为二阶因素，可永久撤掉 P0.1；仍有种子卡死则说明需要 P1 有界化。

---

## 9. SOC 死区归属 与 自愿功率限幅（下层行为的两条硬约束）

### 9.1 规则

```
下层 SocTracker:
  跟踪死区 = --tracker_dead_zone, 默认 0.0        ← 尽力追随
      δ ≠ 0 → pb_need = δ/gain
              pb_vol  = clip(pb_need, ±cap)      ← cap = --tracker_pb_cap (默认 = 波动限值 3 MW)
              pb      = clip(pb_vol, 窗口)        ← 被迫部分 = pb − pb_vol
      δ = 0 → clip(0, 窗口)                       ← 不主动吞吐; 窗口不含 0 时取最小被迫量
上层 gap 罚:
  豁免带 = --soc_tol, 默认 0.1 MWh               ← H 不必为零头付费
      gap_pen = 10·max(0, |soc_target − SOC_末| − 0.1)
```

**0.1 MWh 是上层的记账约定，不是下层的停止条件。** 不对称方向是**安全**的：下层能消掉的
残差 ⊇ 上层豁免的（`0 ≤ 0.1`），所以不会出现"规则停住不动、上层却照罚"的幽灵税。
⚠ 只有把 `--tracker_dead_zone` 调成**大于** `--soc_tol` 时幽灵税才会回归（启动会告警）。

### 9.2 为什么自愿指令要限幅到波动限值（防"自致过冲"）

窗口位置 = `ptur − pg_prev ± 3`，而本步 pb 会成为下一步 `pg_prev` 的一部分：

```
pg_prev'      = ptur − pb
下一步窗口下沿 = ptur' − pg_prev' − 3 = pb + (ptur' − ptur) − 3
```

⇒ **只要 `|pb| ≤ 3`（= 波动限值）且风速无阶跃，下一步窗口必含 0**，规则下一步总能停在
目标上。反之 `|pb| > 3` 会**把自己的窗口推离 0**，下一步被逼继续同向吞吐，于是过冲。

实测（δ₀ = 0.30 的块，ptur = pg₀ = 20）：

| k | 窗口 | 需要功率 | 实际 pb | 被迫量 |
|---|---|---|---|---|
| 0 | [−3, 3] | 18.37 | 3.0 | 0 |
| 1 | [0, 6] | 15.37 | 6.0 | 0 |
| 2 | [3, 9] | 9.37 | **9.0** | 0 |
| 3 | **[6, 10]** | 0.37 | **6.0** | **+5.63** |
| 4 | [3, 9] | −5.63 | 3.0 | **+8.63** |

第 2 步自愿充 9 MW（> 波动限值）⇒ 第 3 步窗口变成 [6,10]（不含 0）⇒ 只需 0.37 MW 却被
逼充 6 MW ⇒ 块末残差从 0.006 涨到 **0.141** ⇒ 收 gap 0.41。**这就是"为 0.01 的差值用
大功率、结果 SOC 跳 0.16"这类现象的成因**（0.16 = 10 MW × 0.98/60）。加 cap 后每一步
`|pb_forced| = 0`，SOC 平滑爬升，残差 0.055 → gap **0**。

> 注意：0.16 MWh 的 SOC 跳变还有**另一个纯外部来源** —— 风阶跃使 `|ptur − pg_prev| > 13`
> 时窗口被压成单点 {10}（例如 `ptur=13, pg_prev=0` → 窗口 [10,10]）。此时波动约束
> `|ΔP_G| ≤ 3` **要求** pb ≥ 13 − 3 = 10 MW，`pb = 0` 会造成 13 MW 波动违约。
> 这种被迫吞吐**无法由规则消除**，只能在"波动优先"与"SOC 优先"之间重新排优先级。

### 9.3 实测：cap 把 gap 从"毛刺"变成"线性"

`--tracker_pb_cap 0`（auto = 3 MW，默认）vs `-1`（不限，旧行为）：

| δ₀ | 不限幅：残差 / gap / max\|pb_forced\| | cap=3：残差 / gap / max\|pb_forced\| |
|---|---|---|
| 0.01 | 0 / 0 / 0 | 0 / 0 / **0** |
| 0.20 | 0 / 0 / **7.0** | 0 / 0 / **0** |
| 0.30 | 0.141 / **0.410** | 0.055 / **0** / **0** |
| 0.40 | 0.057 / **0** | 0.155 / 0.550 / **0** |
| 0.50 | 0.072 / **0** | 0.255 / 1.550 / **0** |
| 0.80 | 0.179 / 0.793 | 0.555 / 4.550 / **0** |
| 1.00 | 0.379 / 2.793 | 0.755 / 6.550 / **0** |

- 不限幅：gap **非单调**（0.30 罚 0.41 而 0.40/0.50 免罚）⇒ H 面对一个毛刺状代价面；
- cap=3：`gap = 10·(δ₀ − 0.345)`，**严格线性**；自由带 = `可达跨度 0.245 + 豁免 0.1`。

**代价（明确的取舍）**：单块可达 SOC 跨度从 ~0.62 MWh 降到 **0.245 MWh**
（= 5 × 3 MW × η_c·dt，与 `d_cap_c` 诊断完全一致 —— cap 之后 `df` 里的
`target_feasible / d_cap_c / d_cap_d` 不再是乐观估计，已按 `min(窗口, cap)` 计算）。
0.245 MWh/块 ≈ 70 MWh/天，对日内套利（几 MWh 级）绰绰有余，但 H 必须学会"小步走"。

### 9.4 记录与开关

| 项 | 说明 |
|---|---|
| CSV 新增 2 列 | `pb_voluntary`（限幅后的自愿指令）、`pb_forced`（被窗口额外强加的量，MW） |
| TB 新增 | `rbb/pb_voluntary`、`rbb/pb_forced` |
| `--tracker_dead_zone` | 0 = 尽力追随（默认）；>0 = 旧"最小吞吐"行为（必须 ≤ `--soc_tol`） |
| `--tracker_pb_cap` | 0 = auto 取 `--rbb_fluctuation_limit`（默认，防自致过冲）；负 = 不限（旧行为）；正 = 用该值 |

> 仍存在的外部强制吞吐（风阶跃 > 13 MW/min）会把残差推过 0.1 并让 H 付费。若要把这段
> 也剔除，需要给 gap 罚加"被迫吞吐反事实"口径（用 `pb_forced` 的 SOC 贡献修正残差）——
> 已具备数据基础（`pb_forced` 每步可查），但会改动奖励，需要单独决策。




