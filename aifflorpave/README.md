# aifflorpave —— PAVE 式 critic 几何正则（验证臂）

## 1. 它要修什么（本仓库实测）

| 指标（基线 16/100 @ ckpt 28000） | 实测 | 文献参考（Lee et al. 2026 的 Base） |
|---|---|---|
| 相邻决策 ∇_aQ 符号翻转率 | **33–35%**（ckpt 10000 处 41%） | 9.0–35.5% |
| 5 台机组 ∇_aQ 同号率 | **45.7%**（≈随机） | — |
| 全 6×6 ∇²_aaQ 负定性率 | **4.5%**（256/256 仅 0.5%） | 14–47% |
| `slopeSD` 随训练增长 | **2.7×**（2.93→7.84） | M_sup 增长约 4× |

理论：策略敏感度 `L ≤ M/μ`（M=‖∇²_sa Q‖，μ=−λ_max(∇²_aa Q)）。

## 2. 三项辅助损失（只作用于 critic）

```
L_MPR  = E_ε ‖∇_a Q(s+ε,a) − ∇_a Q(s,a)‖²       (M 的有限差分代理)
L_VFC  = E   ‖∇_a Q(s,a)   − ∇_a Q(s',a)‖²       (轨迹相邻状态的场一致)
L_Curv = E_v max(0, vᵀ∇²_aa Q(s,a) v + δ)        (保曲率，防过平滑塌陷)
```

## 3. 与论文的两处显式差异

1. **附加优化步**而非联合损失：`super().update_policy()` 走完基线后，再对 profit 双
   critic 走一次几何梯度（`--pave_every k` 可降频）。好处：基线路径逐位不变、可审计、
   单因子消融；代价：critic 优化步数 +1。
2. 只作用于 **profit 双 critic**（actor 梯度主通道）；cost 通道与 L 层完全不碰。

## 4. 口径与开关

| CLI | 默认 | 说明 |
|---|---|---|
| `--critic_act` | `relu` | `relu`=与基线逐位同构；`silu`=C²（Curv 项的前提） |
| `--pave_w_mpr` / `--pave_w_vfc` | 0.0 | 已按 ‖∇_aQ‖² 归一化（无量纲） |
| `--pave_w_curv` | 0.0 | 原始量纲，**仅 act=silu 有效**（ReLU 的 ∇²_aaQ≡0） |
| `--pave_sigma` / `--pave_delta` / `--pave_nv` | 0.05 / 0.0 / 1 | MPR 扰动 std / Curv 严格凹裕度 / Hutchinson 采样数 |
| `--pave_report_only 1` | 0 | **只测量不施加**，写 `pave_stats.csv`（标定权重用） |
| `--pave_log_every` | 50 | `pave_stats.csv` 记录间隔（0=关） |

**标定流程**：先跑 `--critic_act silu --pave_report_only 1`（或 relu 下只测 MPR/VFC），
从 `<output>/pave_stats.csv` 读三项量级与 `gscale`，再给权重。

## 5. 不变量

`python agentutil/test_aifflorpave_contract.py`：
① `CriticPAVE(act='relu')` 与 `model.Critic` **数值逐位相同**（同 seed）；
② 关掉 PAVE 时 `SP_TD3_PAVE` 与 `SP_TD3` 的行动作逐位相同；
③ 几何步的梯度只落在 profit 双 critic（actor / cost 的梯度为 None 或 0）；
④ `pave_terms` 三项有限且非负；ReLU 下 Curv ≡ 0。

## 6. 目录

* `aifflorpave_pave.py` —— CriticPAVE + MPR/VFC/Curv
* `aifflorpave_ddpg.py` —— `SP_TD3_PAVE(SP_TD3)`
* `aifflorpave_main.py` —— 由 `agentutil/port_aifflorpave.py` 从 `aifflor_main.py` 派生
  （`python agentutil/port_aifflorpave.py --check` 证明只差 7 处）
