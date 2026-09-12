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
H 每块更新 2 次; H 层越界罚 penalty_NN_H 系数 100/上限 50 (CheckAction)
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

CSV 采用**增量追加**落盘（`_flush_csv`），避免高迭代数时整表重写的卡顿。

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
参数：`--warmup 200`、`--discount 0.98`、`--policy_delay 2`、`--soc_tol 0.1`、
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
