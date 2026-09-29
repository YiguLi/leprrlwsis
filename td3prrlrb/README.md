# TD3PRRLRB — Piecewise Reward RL + Rule-Based BESS

`td3prrlrb` 是 `td3prrbb` 的重构版本：**上层保留 TD3 学习，下层改为规则控制器**。
本目录全部为新建文件，`td3prrbb/` 的代码未被修改。

```
td3prrlrb/
├── td3prrlrb_main.py      训练主循环 (只训练 H 层)
├── td3prrlrb_myfunc.py    奖励/成本、SOC 目标映射、动作检查、9-case 测试
├── td3prrlrb_rule_bess.py RuleBasedBESS (RBB 可行窗口) + SocTracker (规则下层)
├── td3prrlrb_ddpg.py      SP_TD3 (与 td3prrbb 相同, 供 H 层使用)
└── __init__.py
```

## 1. 架构与职责

| 层 | 实现 | 动作 | 频率 |
|---|---|---|---|
| H (上层) | TD3 (`SP_TD3`, nb_states=4, nb_actions=6) | 5 台风机桨距 `[0,0.5]` + 1 个 **SOC 目标信号** `a6∈[-0.5,0.5]` | 每 5 步(块)一次 |
| L (下层) | **规则** `SocTracker` | BESS 功率 `pb` (MW, 正=充电) | 每一步 |

- SOC 目标：`soc_target = 3 + 5·a6 ∈ [0.5, 5.5]`（块内固定，`soc_target_from_signal`）。
- H 是**唯一决策者**：既要收益，也要为一个"不可行的 SOC 目标"付费。
- L 的任务只有一个是学习不来的：**追踪目标**；而"物理可行 + 平抑波动"由 RBB 可行窗口
  `[pbrulemin, pbrulemax]` 硬保证，因此下层不需要 RL（no actor/critic/λ2/c2/PBRS）。

### 下层规则行为（`SocTracker.act`）
1. **死区**：`|soc_target − E| ≤ SOC_TOL`（默认 0.1 MWh = 6 MW·min）→ 不主动吞吐；
   但若该时刻窗口不含 0（风速阶跃 > 3 MW/min 被迫吞吐），则取窗口内**离 0 最近**的点，
   由规则自己完成裁剪（不把裁剪量留给 RBB）。
2. **约束死拍**：死区外按 `pb = (target−E)/gain` 一步补齐（`gain_c=η_c·dt`，`gain_d=dt/η_d`），
   先按额定 ±10 MW 限幅。
3. **窗口饱和**：把 `pb` 夹进 `[pbrulemin, pbrulemax]` = 用满可用权限；
   窗口为空（`pbrulemin > pbrulemax`，物理不可行）时把死拍指令交给 RBB 兜底。

## 2. 奖励 / 成本（仅 H）

```
r1 = wT·(pr·ptur·dt) + wB·(−pr·pbrule·dt) − 490·max(−pbrule,0)·dt − penalty_NN_H
     wT = wT_u(u)·wT_p(pr)      wB = wB_p(pr)·wB_soc(SOC, wind)

c1 = C1_RISK[wind]·[10·(P_F−3)⁺ + 0.1·P_F]        ← 波动越限 + 平滑
   + W_GAP·unfeasible_gap                         ← 平抑可行性缺口 (W_GAP=10)
   + [块末结算] 10·max(0, |soc_target − SOC_end| − SOC_TOL)   ← 上层可行性惩罚
```

- **退化费 490 元/MWh 归 H**（`max(−pbrule,0)`，放电侧）：吞吐是 H 下达的目标引发的，
  按因果归属由 H 付费（原版本把 `c_deg` 放在下层，与"谁引发谁付费"不符）。
- 0.1 MWh 死区**必须与下层死区同源**：否则规则停在死区边缘时上层会被收一笔永远消不掉的"幽灵税"。
- 原版本的 `r2/c2`（`clip_amount` 惩罚 + `(SOC−target)²` 跟踪罚）、影子利润项、下层观测蒸馏
  全部删除。

### 对偶上升（只对 H）
```
EMA  ← (1−0.02)·EMA + 0.02·sum_c1        (块实际成本, warmup 后开始记)
lam1 ← clip(lam1 + 0.01·(EMA − d_block_H), 3, 100), 初值 5.0
d_block_H: 初值 50.0, 地板 5.0, 连续 100 次 EMA ≤ d_block_H 则减半
H 每块更新 2 次; H 层越界罚 penalty_NN_H 系数 100/上限 100 (CheckAction)
```

## 3. 已验证的结构性不变量

用 `rule_bess` 窗口 + 规则控制器可**证明并实测**（见下方验证命令）：

| 不变量 | 结论 |
|---|---|
| 平抑波动 | `P_F ≤ 3 MW` 在所有**可行**步成立；仅当窗口为空（风速阶跃 > 13 MW/min 之类物理不可行）才越限 |
| 可行性 | `clip_amount ≡ 0`（可行步）；不可行步 `clip_amount>0` 表示 RBB 兜底接管 |
| 规则最优性（块内） | SOC 为纯积分器、代价对 `|target−E|` 单调 ⇒ 死拍+窗口饱和 = 块内 best effort |

推论（设计前提）：**块末残差 `|soc_target − SOC_end|` ≈ 该目标的不可行量**，
所以上层的块末罚只惩罚真正的不可行。注意该等式是近似的：风速阶跃造成的**被迫吞吐**也会
改变 SOC，而它不完全由 H 控制（H 只能通过桨距间接影响 `ptur` 与 `P_F`），
因此 `gap_pen` 中包含一部分"H 只能部分预判的风险费"。

## 4. 记录文件（输出目录 `output/TD3PRRLRB-runN/`）

| 文件 | 内容 |
|---|---|
| `df_td3prrlrb.csv` | 每步全量：`r1/c1_step/gap_raw/gap_pen/soc_signal/soc_target/soc_delta`、RBB 窗口量、`clip_amount`、`d_cap_c/d_cap_d`、收益分解、`lam1`、`d_block_H` |
| `df_episode_td3prrlrb.csv` | 每 episode：`episode_reward/cost/gap_pen/profit`、`deg_charge/deg_discharge` |
| `df_test9.csv` | 每 480 步的 9-case 测试：`r1/c1/gap_pen/cvm/pf_viol/profit/soc_track_err` |
| `update_stats_H.csv` | H 层每块 2 批更新的 TD/Q/target/batch 统计均值 |

CSV 采用**增量追加**落盘（`util.flush_csv_append_batch`：本表按"攒一批 → flush → 清空"写），避免高迭代数时整表重写的卡顿。

`d_cap_c = pbrulemax·η_c·dt·5`、`d_cap_d = |pbrulemin|·(dt/η_d)·5` 是本块规则的最大能力，
即自由目标带宽 ≈ `SOC ± (cap + 0.1)`（典型值 ±0.25 MWh/块）。

## 5. 与 td3prrbb 的差异

| 项目 | td3prrbb | td3prrlrb |
|---|---|---|
| 下层 | TD3 (nb_states=5 蒸馏观测) | **规则** SocTracker |
| 下层奖励 | `r2=−100·(clip+rated_clip)`, `c2=10·max(0,\|δ\|−0.1)²` | 无 |
| 下层对偶 | `lam2`, `d_block_L`, EMA | 无 |
| 退化费归属 | 出现在 H 的 r1 与 L 的 profit 统计 | 明确由 H 承担（唯一学习器） |
| 记录文件 | `df_td3prrbb.csv` (+r2/c2/lam2) | `df_td3prrlrb.csv` (去 r2/c2，加 gap/cap 诊断) |
| 9-case 测试 | 驱动 agent_L | 驱动 SocTracker |

## 6. 运行

```bash
cd HSRL
python td3prrlrb/td3prrlrb_main.py          # 默认 train, 1000 episode (1440 步/episode)
```

依赖与其它 main 相同（`env.py`/WFSim 的 MATLAB engine、torch、tensorboard、pandas）。
参数：`--warmup 200`、`--discount 0.99`、`--policy_delay 2`、`--soc_tol 0.1`、
`--phase_feat 1`（H 观测 4→6 维，24h 相位）、
`--wt_bias_init 1`（WT 输出层 bias 初始化到 Cp 高效区 [0.25,0.40]）、
`--epsilon 4000`（探索 ε 衰减 1→0 的**决策次数**；H 每 5 步决策一次 ⇒ 20k 环境步）、
`--wt_noise 0.05`、`--sig_noise_max 0.2`、`--sig_noise_min 0.05`（逐维探索幅度，
信号维 = `max(0.2·ε, 0.05)`）、
`--rbb_fluctuation_limit 3.0`、`--rbb_rated_power 10.0`。

## 7. 待标定 / 风险点

1. **λ1 放大**：块末罚写在 `c1` 内，等效权重 = `10·lam1`（lam1 可达 100 ⇒ 1000 元/MWh
   ≫ 退化 490 元/MWh）。若发现 H 过度保守（不敢动 SOC），可把 gap 罚改放 `r1` 固定权重，
   或下调 `d_block_H` 的收敛路径。
2. **free-band 很窄**：规则每块只能动约 ±0.25 MWh，而 H 的信号输出跨度是 5 MWh
   ⇒ 有用动作区间只占 6%~10%，梯度信号弱。当前用"地板 5.0 / 死区 0.1"把收费限在真实
   不可行量上；warmup 的随机信号 `U(−0.5,0.5)` 目标恒在合法带内（`random_action` 已硬修复），
   不会再出现 run157 那种病态目标。若 H 难以学到目标，可进一步把 warmup 目标初始化到
   `soc_target ≈ E`（当前 SOC 附近）。
3. **被迫吞吐进入 gap**：见第 3 节末的"近似"说明；若后续给下层加入"额外平滑"的自主余量，
   则 `gap_pen` 的口径必须重算（残差将不再等于不可行量）。
4. `[0.5, 5.5]` 的目标带是隐性软 SOC 下限：SOC 很低时结构性地罚 gap，
   相当于把 SOC 下限塞给了 H。

## 8. 标定记录

### R1（run159/160 诊断后，本次改动）

诊断：run160 的 SOC 目标信号 a6 漂到 −2.5…−3.9（100% 越界 → `soc_target≡0.5`，套利功能归零），
run159 漂到 −0.57（67% 越界，目标仍随价格变化 → 部分套利）；
两 run 的 `penalty_NN_H` 分别有 53%/93.5% 的时间**顶在封顶值上**，未截断的原始罚最高 2384
→ 越界区梯度≈0，没有恢复力。

| 项 | 旧 | 新 | 目的 |
|---|---|---|---|
| `--discount` | 0.98（半衰期 34 块≈2.8 h） | **0.99**（69 块≈5.8 h） | 覆盖 4~7 h 的"低价充→高价放"周期 |
| `CheckAction` H 罚上限 | 50 | **100** | 扩大有梯度恢复力的区间 |
| 信号维探索幅度 | ±0.05·ε（且 `[0]` 使 6 维共用一个标量） | **±max(0.2·ε, 0.05)**，逐维独立抽样 | 探索跨度 > 1 个块的跟踪能力（0.245 MWh），才可能跳出越界区/找到套利目标 |
| 探索衰减完成 | 50000 决策次（=250k 环境步） | **4000 决策次（=20k 环境步）** | 尽早进入"地板探索"阶段 |
| `--epsilon` 语义 | 决策次数 | 决策次数（help 已写明 = 5× 环境步） | 避免误读 |

其余（λ1 范围、d_block_H、warmup=200、PER α、退化口径、H 观测维度）本次**未改**。

### R2（run160 轴向因子/停机诊断后，本次改动）

**物理事实**（`code/HSRL(2)/HSRL/WFSim_port.m` L10）：WT 动作 = 轴向诱导因子 a，功率系数
`Cp = 4a(1−a)²` ⇒ **a=0 → Cp=0（该机零出力/停机）**，a=1/3 → 0.593（Betz 最大），a=0.5 → 0.5。

run160 实测：施加后 a = [0.364, **0.006, 0.005, 0.003**, 0.439] ⇒ 2/3/4 号机 96–98% 的步数
被夹到 0（长期停机），1/5 号机长期顶在 0.5 ⇒ 全场 **ΣCp = 0.913 / 2.963 = 30.8%**，
同风况本可多发 3.24 倍（turbine_profit 4.6 万/天 vs 理想 14.9 万/天）。

根因：`model.py` 的 `Actor` 只初始化 `fc4.weight`（±0.003），**`fc4.bias` 仍是 PyTorch 默认
`U(−1/√fan_in)=U(±0.25)`**（fan_in = hidden1 = 16，为 run160 当时口径；2026-09-29 起默认
hidden1 = 64 ⇒ `U(±0.125)`）⇒ 初始输出 ≈ bias，实测 5 个种子死机 2~4 台、
效率仅 16%~24%；而 a<0 区被 `CheckAction` 裁掉后梯度为 0，唯一能提供恢复力的越界罚又被
`min(penalty, 50)` 封顶（5 维各越界 0.1 即触顶）⇒ 越界区梯度全为 0 → 维度被共享隐层拖走随机游走
（run160 漂到 −3.9/−4.7/−6.4/+6.4，原始罚达 1600~2400）→ 三台机永久停机，无法恢复。

| 项 | 旧 | 新 | 实测效果 |
|---|---|---|---|
| **P0.1** `--wt_bias_init` | fc4.bias = PyTorch 默认 U(±0.25) | **构造后覆写前 5 维为 U(0.25, 0.40)**（Cp 高效区），第 6 维置 0（初始 soc_target = 3.0） | 初始 ΣCp 0.784 → **2.917**（效率 26.5% → **98.4%**），死机 2/5 → **0/5** |
| **P0.2** `CheckAction` H 罚上限 | `min(penalty, 100)`（原 50） | **完全不封顶** | 惩罚随越界深度线性增长（1590/1730/2130），**每单位恒定 100 的恢复梯度**，任意深度不衰减 |

- P0.1 在**构造后覆写** `actor`/`actor_target` 的 `fc4.bias`，因此 `model.py`（多项目共享）**不改动**，
  `td3prrbb` 不受影响。
- ⚠️ 副作用：初始风机出力从"半停机"变成"正常工作点"，早期波动与被平抑约束强制的 BESS 吞吐都会
  大于 run159/160，因此**早期阶段与旧 run 不可直接对比**（这正是修复的目的）。
- 与 P1（输出有界化）的关系：二者是**替代方案**——不封顶罚项给出的是恒定恢复力，而有界化
  （`0.5·sigmoid`）是结构上删掉越界区，但代价是边界处 `∂a/∂z→0`（梯度饱和），且 sigmoid 会把
  初始 z≈±0.25 映射到 a≈0.22~0.28（**顺带自动解决 P0.1**）。故本轮**不做 P1**，留作后手。

### R3（24h 相位特征，本次改动）

**动机**：H 原本看不到"一天中的位置"。套利里"方向/持留"这一半只需当前电价（马尔可夫，无需相位，
run159 的"持留—倾泻"就是这么学的），但"择时"那一半——**在低价窗的哪一刻把电充满**——必须要知道
"本段还剩多久、下一段是涨还是跌"：规则速率 0.245 MWh/块 ⇒ 充 5 MWh 需 1.7 h，而低价窗长达 7~8 h。
相位盲策略只能做持留套利（~1 MWh/天），无法做择时套利（3~5 MWh/天）。

| `--phase_feat` | 特征 | 维数 | 说明 |
|---|---|---|---|
| **0** | 无 | 4 | 复现旧 run |
| **1**（默认） | `[sin(2π·m/1440), cos(2π·m/1440)]` | **6** | 24h 相位；用 sin/cos 而非 1 维锯齿 `m/1440`——后者跨零点不连续，拟合 MSE **0.047 vs 0.0008**（差 60 倍）、判对率 82.5% vs 99.7% |
| 2 | mode 1 + `[段内剩余比例, 剩余小时/8, 下一段电价/1000]` | 9 | 小样本预算下 precision 0.51→0.74（更"好教"）；需要日程表；留作 A/B |

- **相位来源**：`episode_steps % 1440`，与 `env.CalPrice` 的日程表对齐。已核对 `env.py`：
  `reset()` 里 `count_price = 0`（L71）、`step()` 恒 `done=False`（L207）⇒ 每个 episode 恰好 1440 步，
  **时钟与 `episode_steps` 严格对齐**。
- **日程表循环处理**：23:00–23:59 与 0:00–6:59 同为 300 ⇒ 属于**同一个 8 h 段**，因此 0:00 的
  "本段剩余"是 7 h 而 23:00 是 8 h（`phase_daily_table()` 用 `(j±1) % 1440` 循环扫描）。
- **9-case 测试的相位映射**：场景电价每 5 步就变一次，直接喂 wall-clock 相位会 OOD；因此按 case 的
  电价档映射到该档**代表性时刻**（`REP_MINUTE`: LOW→210 (3:30)、MEDIUM→840 (14:00)、HIGH→540 (9:00)，
  即各段中点），使测试与训练同分布且跨 run 可比；next_price/段内特征同样从日程表取（不取场景序列）。
- **一致性自检**：每步检查 `env` 的实际电价是否等于 `day_price_schedule()[episode_steps % 1440]`，
  记录 `phase/check_mismatch_frac`（TB）并在训练末尾打印；**实测 520/520 步 mismatch = 0**。
  若哪天 episode 长度/时钟被改动，这条会立刻报警（相位特征会静默失效，必须防）。
- 记录：CSV 增 `phase_minute / phase_sin / phase_cos`（mode 2 另有 `phase_rest_frac / phase_rest_h /
  phase_next_pr`），TB 增 `phase/hour`（+ mode 2 的 `rest_frac/rest_hours/next_price` 与自检值）。
- 代价：输入维变化 ⇒ **不能 warm-start 历史 checkpoint**（本来也要重训）。`env.py`/`td3prrbb` 不动。

### R4（下层规则两项修正：死区归属分离 + 一步填平夹窗口）

参考 `td3leprrlrb/td3leprrlrb_rule_bess.py` 修改 `SocTracker`
（**只改下层规则；上层 `gap_penalty` 的 0.1 MWh 豁免原样保留**）：

| 项 | 旧 | 新 |
|---|---|---|
| 下层跟踪死区 `tracker.soc_tol` | 0.1（与上层同源） | **0.0**（残差 0.01 MWh 也照追） |
| 下层动作形式 | `clip(ΔSOC/g, 窗口)` 但含 0.1 死区（死区内→贴窗口 0） | **`clip(ΔSOC/g, pbrulemin, pbrulemax)`** —— 一步填平所需功率夹进窗口，无死区、无功率上限 |
| 新 CLI | — | `--tracker_soc_tol 0.0`；`--soc_tol` 只管**上层** gap 罚（默认仍 0.1）；`--pb_cap -1`（≤0 ⇒ **不限制**，仅实验用） |
| 新记录 | — | CSV `pb_voluntary`/`pb_forced`、TB `target/pb_voluntary`/`target/pb_forced` |

**① 死区归属分离**：0.1 MWh 的豁免只是**上层记账约定**（H 不必为零头付费），不是下层动作规则。
下层死区 0 < 上层豁免 0.1 ⇒ 只会"消得更多"，不会出现"规则停在死区不动、上层却照罚"的幽灵税。
⚠ 若把 `tracker.soc_tol` 调回 >0 做 A/B，**必须 `tracker.soc_tol ≤ --soc_tol`**（main 里有 assert）。

**② 机制（定稿，只有这一句）**：

```
delta = soc_target − E
pb    = clip( delta / g , pbrulemin , pbrulemax )     # g = η·dt (充) 或 dt/η (放)
```

即"**一步填平 SOC 所需功率**"夹进可行窗口 —— 这就是"在 pbrule 内对着目标最大化"的最小充分形式。
它天然满足"能用非顶格功率就不顶格"：目标近则所需功率小，**不会去顶窗口边界**。
**不取窗口边界的最大功率**，因为顶格会把下一步窗口推离 0（`pg_prev' = ptur − pb` ⇒ 下一步窗口中心
= `ΔP_tur + pb`），逼出"自致过冲"。

实测（`soc_tol=0`、无上限）：

| δ (MWh) | 所需功率 δ/g | 窗口 | 实际 pb | voluntary / forced |
|---|---|---|---|---|
| 0.00 | 0 | [−3,3] | **0.000** | 0 / 0 |
| 0.01 | +0.61 | [−3,3] | **+0.612** | +0.61 / 0（旧版死区会返回 0） |
| 0.03 | +1.84 | [−3,3] | **+1.837** | +1.84 / 0（不顶格到 3） |
| 0.30 | +18.4 | [−8,8] | +8.00 | +10.0 / **−2.0**（被窗口限制） |
| −0.10 | −5.88 | [−8,8] | −5.88 | −5.88 / 0 |

`pb_forced` 的符号约定：**>0 = 被窗口/兜底被迫多做**，**<0 = 被窗口限制掉**。

- **残留局限（属外部几何，不是规则的错）**：若 `|δ| > 3g ≈ 0.05 MWh`，一步填平本身就要求 >3 MW，
  下一步窗口可能不含 0 而被逼补一点，过冲 ≈ `max(0, |δ| − 3g)`；上层 0.1 MWh 豁免可吸收一部分。
- **能力是涌现的、随风况变化**（本轮更正）：约束只加在**并网功率** `|ΔP_G|≤3` 上，因此电池功率的
  步变化允许量是 **`3 + ΔP_tur`**，不是常数 3（实测 run159：`|ΔP_tur|` p50 0.90 / p90 3.14 /
  p99 6.23 MW ⇒ 允许量 p50≈3.9、p90≈6.1 MW；11.1% 的步 `|ΔP_tur|>3` 即窗口不含 0 → 被迫吞吐）。
  中性风下每块名义能力 ≈0.5 MWh（爬坡 3→6→9→10→10），额定上界 `10·g·5 ≈ 0.82 MWh/块`。
  ⇒ `block_cap_charge/discharge` 与 `target_feasible` 只能当**名义估计**，不是硬上限。
- RBB 兜底不变：规则恒在窗口内，全流程 smoke 实测 `clip_amount max=mean=0.000000`。

**本轮记录的三处更正**（前两版文档/结论有误，此处为准）：
① ~~"pb_cap 把每块能力标准化为 0.245 MWh"~~ —— 那是**副作用且是能力损失**，已撤销该限制；
② ~~"电池自身功率每步最多变化 3 MW"~~ —— 正确表述是上面那条"约束在并网功率，电池允许量 = 3+ΔP_tur"；
③ 曾提出的"停车距离法则"（`−R+√(R²+2Rδ/g)`）**未采用** —— 用户方案（一步填平夹窗口）更简单且已足够。
