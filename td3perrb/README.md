# TD3PERRB — 非分段奖励 + Rule-Based BESS

`td3perrb` = **`td3prrlrb` 的非分段奖励版本**：上层 TD3 学习、下层规则控制器（SocTracker），
**唯一差异是不使用分段奖励函数** —— 收益权重与 c1 风险系数全部取固定 `1.0`，不移植任何分段系数。

```
td3perrb/
├── td3perrb_main.py      训练主循环 (只训练 H 层)
├── td3perrb_myfunc.py    非分段奖励/成本、SOC 目标映射、动作检查、9-case 测试
├── td3perrb_rule_bess.py RuleBasedBESS (RBB 可行窗口) + SocTracker (规则下层)
├── td3perrb_ddpg.py      SP_TD3 (twin critic + 延迟策略更新 + 目标平滑 + PER)
└── __init__.py
```

本目录全部为新建文件；`td3prrlrb/`、`td3prrbb/`、`td3per/` 的代码均未被修改。

## 1. 架构与职责

| 层 | 实现 | 动作 | 频率 |
|---|---|---|---|
| H (上层) | TD3 (`SP_TD3`, nb_states=6, nb_actions=6) | 5 台风机桨距 `[0,0.5]` + 1 个 **SOC 目标信号** `a6∈[-0.5,0.5]` | 每 5 步(块)一次 |
| L (下层) | **规则** `SocTracker` | BESS 功率 `pb` (MW, 正=充电) | 每一步 |

- SOC 目标：`soc_target = 3 + 5·a6 ∈ [0.5, 5.5]`（块内固定，`soc_target_from_signal`）。
- H 是**唯一决策者**：既要收益，也要为一个"不可行的 SOC 目标"付费。
- L 只做一件事——**追踪目标**；"物理可行 + 平抑波动"由 RBB 可行窗口 `[pbrulemin, pbrulemax]`
  硬保证，因此下层不需要 RL（无 actor/critic/λ2/c2/PBRS）。
- H 观测 = `Normalize(obs)` 4 维 + `--phase_feat 1` 的 24h 相位 2 维 = 6 维。
- `--wt_bias_init 1`：构造后把 actor 输出层前 5 维 bias 覆写到 Cp 高效区 `U(0.25,0.40)`。

## 2. 奖励 / 成本（仅 H，非分段）

```
r1 = 1.0·(pr·ptur·dt) + 1.0·(−pr·pbrule·dt) − 490·max(−pbrule,0)·dt − penalty_NN_H
   = pr·P_G·dt − 490·max(−pbrule,0)·dt − penalty_NN_H          (P_G = ptur − pbrule)

c1 = 1.0·[10·(P_F−3)⁺ + 0.1·P_F]        ← 波动越限 + 平滑
   + W_GAP·unfeasible_gap               ← 平抑可行性缺口 (W_GAP = 10)
   + [块末结算] 10·max(0, |soc_target − SOC_end| − SOC_TOL)   ← 上层可行性惩罚
```

与 `td3prrlrb` 的逐项对照（分段 → 非分段）：

| 系数 | td3prrlrb (分段) | td3perrb (非分段) |
|---|---|---|
| 风机收益权重 `wT` | `WT_WIND[u]·WT_PRICE[pr]`，`{LOW 1.2, MED 1.0, HIGH 0.9}×{1.0,1.0,1.05}` | `W_TURBINE = 1.0` |
| BESS 收益权重 `wB` | `WB_PRICE[pr]·wB_soc(SOC, wind)`，`{0.8,1.0,1.2}×{1.0 或 0.7}` | `W_BESS = 1.0` |
| c1 风险系数 | `C1_RISK[wind] ∈ {LOW 0.8, MED 1.0, HIGH 1.2}` | `C1_RISK_COEF = 1.0` |
| SOC 安全带（**仅日志**） | 按风况分档 `[0.5,5.5]/[1.0,5.0]/[1.5,4.5]` | 单一档 `[1.5, 4.5]` |

- `wT=wB=1.0` 时 `r1` 退化为 `pr·P_G·dt − 490·deg − penalty_NN_H`，与 `td3per` 的非分段 r1 口径**代数等价**。
- 风况/电价档位标签（`get_wind_level`/`get_price_level`）**仍然计算**，但只写入日志与
  CSV/TB 的诊断列（`wind_level/price_level/w_wind/w_bess`），**不进入任何系数**。
- 与分段版的关键差别：不再出现"同一次放电在不同风况档下收益权重不同"的激励扭曲；
  代价是失去"风况越差/风险越高 → c1 系数越大"的风险分段（如果那是想要的行为）。
- ⚠️ 因为 c1 量级相对分段版变化约 ±20%，`d_block_H = 50/5` 与 `λ1 ∈ [3,100]` 的标定
  需要在 td3perrb 上重新观察（本轮按指示沿用 td3prrlrb 的取值，未改）。

### 对偶上升（只对 H）
```
EMA  ← (1−0.02)·EMA + 0.02·sum_c1        (块实际成本, warmup 后开始记)
lam1 ← clip(lam1 + 0.01·(EMA − d_block_H), 3, 100), 初值 5.0
d_block_H: 初值 50.0, 地板 5.0, 连续 100 次 EMA ≤ d_block_H 则减半
H 每块更新 2 次; H 层越界罚 penalty_NN_H 系数 100/单位, 不封顶 (CheckAction)
```

## 3. 关键超参（同名差异项全部取 td3prrlrb 的取值）

| 参数 | 取值 | 说明 |
|---|---|---|
| `--rate` / `--prate` | 0.01 / 0.0001 | critic / actor 学习率 |
| `--discount` | **0.99** | 以块(5 min)为单位生效；半衰期 69 块≈5.8 h（td3per 为 0.98） |
| `--tau` | 0.001 | 目标网络软更新 |
| `--bsize` / `--rmsize` | 64 / 40000 | 批量 / 缓冲容量（≈28 天；同步放大以保持"遍历遍数"不变） |
| `--warmup` | 200 | 纯随机填充步数 |
| PER | `α=0.6, β=0.4→1, ε=0.01`, 退火增量 `3e-5` | 与 td3per 一致 |
| TD3 | `policy_delay=2, policy_noise=0.1, noise_clip=0.3, cost_critic_mode=max` | |
| `--epsilon` | **4000** | ε 1→0 的**决策次数**；H 每 5 步决策 ⇒ 20k 环境步（td3per 为 50000） |
| 探索噪声 | `--wt_noise 0.05`、`--sig_noise_max 0.2`、`--sig_noise_min 0.05` | 逐维独立抽样；信号维 = `max(0.2·ε, 0.05)`（td3per 为标量 ±0.05·ε 且 6 维共用一个值） |
| λ1 | `lam_max=100, lam_min=3`，初值 5.0 | td3per H 层为 `lam_max=50` |
| `d_block_H` | 50.0 / 地板 5.0 | td3per 为 2.5 / 1.25 |
| `CheckAction` 越界系数 | **100/单位，不封顶** | td3per（根 `myfunc.py`）为 500/单位 |
| `W_GAP` | **10.0** | td3per 为 50.0 |
| `--soc_tol` / `--tracker_soc_tol` | 0.1 / 0.0 | 0.1 豁免只属于上层 gap 罚；下层尽力追随 |
| `--pb_cap` | −1（不限制） | 仅实验用 |
| `--phase_feat` | 1（H 观测 4→6 维） | 0=关 / 1=sin·cos / 2=+分段剩余特征 |
| `--wt_bias_init` | 1 | Cp 高效区 bias 初始化 |
| `--rbb_fluctuation_limit` / `--rbb_rated_power` | 3.0 / 10.0 | RBB 窗口参数 |

## 4. 继承自规则下层的结构性不变量

| 不变量 | 结论 |
|---|---|
| 平抑波动 | `P_F ≤ 3 MW` 在所有**可行**步成立；仅当窗口为空（风速阶跃等物理不可行）才越限 |
| 可行性 | `clip_amount ≡ 0`（可行步）；不可行步 `clip_amount>0` 表示 RBB 兜底接管 |
| 规则最优性（块内） | SOC 为纯积分器、代价对 `|target−E|` 单调 ⇒ 一步填平 + 夹窗口 = 块内 best effort |

推论：块末残差 `|soc_target − SOC_end|` ≈ 该目标的不可行量；但风阶跃造成的**被迫吞吐**
也会改变 SOC，所以 `gap_pen` 中含一部分"H 只能部分预判的风险费"。

## 5. 记录文件（输出目录 `output/TD3PERRB-runN/`）

| 文件 | 内容 |
|---|---|
| `df_td3perrb.csv` | 每步全量：`r1/c1_step/gap_raw/gap_pen/soc_signal/soc_target/soc_delta`、RBB 窗口量、`clip_amount`、`d_cap_c/d_cap_d`、收益分解、`lam1`、`d_block_H`、相位列 |
| `df_episode_td3perrb.csv` | 每 episode：`episode_reward/cost/gap_pen/profit`、`deg_charge/deg_discharge` |
| `df_test9.csv` | 每 480 步的 9-case 测试：`r1/c1/gap_pen/cvm/pf_viol/profit/soc_track_err` |
| `update_stats_H.csv` | H 层每块 2 批更新的 TD/Q/target/batch 统计均值 |

CSV 采用**增量追加**落盘（`util.flush_csv_append_batch`：本表按"攒一批 → flush → 清空"写）。

## 6. 与 td3prrlrb 的差异（全部差异仅此一处）

| 项 | td3prrlrb | td3perrb |
|---|---|---|
| 奖励/成本系数 | 分段（`wT_u×wT_p`、`wB_p×wB_soc`、`C1_RISK[wind]`） | **非分段（三者恒 1.0）** |
| SOC 安全带（仅日志） | 按风况分档 | 单一档 `[1.5, 4.5]` |
| 其余（下层规则、观测、超参、对偶、课程、记录） | — | **逐字一致** |

`td3prrlrb/README.md` 的 R1–R4 标定记录（折扣 0.98→0.99、探索噪声逐维化、相位特征、
死区归属分离、CheckAction 不封顶）全部被 td3perrb 原样继承。

## 7. 运行

```bash
cd HSRL
python td3perrb/td3perrb_main.py          # 默认 train, 1000 episode (1440 步/episode)
```

依赖与其它 main 相同（`env.py`/WFSim 的 MATLAB engine、torch、tensorboard、pandas）。
`test_five_days.py --projects td3perrb` 可做五天随机风速的单文件推理测试
（只前向不更新网络，逐日统计越限次数/越限量/利润）。

## 8. 待观察

1. **λ1 / d_block_H 的标定**：c1 系数由分段的 0.8~1.2 变为恒定 1.0，量级变化 ±20%；
   沿用 td3prrlrb 的 `d_block_H=50/5` 与 `λ1∈[3,100]` 是"对齐实验"的选择，不是重新标定。
2. **收益权重的消失**：分段版对 LOW 风给 1.2、HIGH 风给 0.9 的风机收益权重，非分段版一律 1.0，
   因此高低风况下的收益尺度差异不再被显式补偿。
3. **目标带**：`[0.5, 5.5]` 的目标带是隐性软 SOC 下限；SOC 很低时结构性地罚 gap（沿用 td3prrlrb）。
