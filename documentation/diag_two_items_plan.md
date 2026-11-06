# 诊断实验：两项能力核验（S1 / S2 / S3 / E2）

> 目的：用 FLORIS 把"网络学不会"拆成**可判定的单项能力**，超参数全部冻结为生产默认。
> 一键启动：`python agentutil/diag_launch.py`（后台常驻，四段顺序，各 70k 步后停）。

## 0. 冻结的超参数（只改任务，不改算法）

| 项 | 值 | 项 | 值 |
|---|---|---|---|
| hidden1 / hidden2 | 16 / 100 | γ | 0.95 |
| bsize | 256 | τ | 1e-3 |
| rmsize | 40000（H 块条目） | prate | 1e-4 |
| utd_h / utd_l | 4 / 10 | policy_delay | 2 |
| wt_noise | 0.05 | noise_clip | 0.3 |
| PER | α 0.6 / β 0.4→1 / ε 0.01 | warmup | 1440 步 |
| nb_turbine | 5 | seed | 0 |

## 1. 四段任务

| 段 | 风速 | 电价 | 奖励 | 动作 | 风速池 | 检验什么 |
|---|---|---|---|---|---|---|
| **S1** | `u≡10` | `pr≡580` | `pr·P_G·dt − 400·Σ越界`（**λ≡0 ⇒ 无波动成本**） | 5 维 a，**g≡0** | — | 能否把**常数动作**爬到出力最优（上下文 bandit） |
| **S2** | 真实 day 序列 | `pr≡580` | 同上 | 同上 | clean | 能否学 `a*(u)`（状态条件化） |
| **S3** | 真实 day 序列 | 300/580/980 日程 | 同上 | 同上 | clean | 价格 3.3× 变化是否破坏值尺度（**最优解与 S2 相同**） |
| **E2** | 真实 day 序列 | 日程 | 生产口径 `pr·P_G·dt − λ·c1`，**λ 固定 2.0（物理 20）** | actor 输出标量 `s`，`a=0.33·s` | **noise** | 一维问题上"收益 vs 波动"能否找到最优 s |

实现：`aifflor_main.py` 新增 `--diag_u / --diag_pr / --diag_scalar / --diag_freeze_g /
--diag_lambda_fixed / --pen_h`（默认全关 ⇒ 生产路径逐位不变）。

## 2. 参考最优（`agentutil/diag_make_refs.py` → `output/diag/refs_*.csv`）

* `P_max(u)`、`a*(u)`：生产物理下 5 维坐标上升（批量化，29 个 u 点）。**已算得**：
  `a*` 几乎与 u 无关，形如 `[0.34~0.36, 0.50, 0.22~0.26, 0.24~0.28, 0.3333]`
  （注意：**`a1=0.5` 顶到上界**，是尾流转向解；`[0.2×4,1/3]` 在生产物理下**不是**最优）。
  `u=10` 时 `P_max=3.7009 MW`（vs 全 1/3 的 3.559）。
* E2（λ=20、加噪池、day122）的**动态规划最优**：`J*=47020 元/天`，
  `s*` 分布 p5 0.14 / p50 0.48 / p95 1.00，最优解处违约率 5.98% ⇒ 是真正的内点权衡。
  （clean 池下 λ=20 的最优是 `s≡1`，权衡退化 ⇒ E2 必须用加噪池。）

## 3. 指标与判据（`agentutil/diag_report.py`）

| 指标 | 定义 | 通过 |
|---|---|---|
| **η** | `mean(P_G/P_max(u))`（末 20% 窗口，确定性动作） | ≥0.97 |
| pattern 误差 | `mean|a_j − a*_j(u)|` | <0.03 |
| 越界率 | `P_F>3` 的步占比 | <5% |
| 动作 std | `applied` 动作步间标准差（run90 = 0.028 是病征） | — |
| `r2_dua` | `R²(P_G~u+a) − R²(P_G~u)`（动作是否提供信息） | >0.05 |
| **regret**（E2） | `(J*−J_policy)/|J*|` | <5% |
| `s_mean` vs `s*_mean`（E2） | 学的 s 与 DP 最优 s* | 差 <0.1 |

## 4. 判定树

```
S1 过 ? ─ 否 → 底层失败：梯度/探索/容量/初始化（先看 supervise 容量自检）
   └ 是 → S2 过 ? ─ 否 → 状态条件化/信用分配（观测尺度、缺 Δu、探索覆盖）
              └ 是 → S3 过 ? ─ 否 → 值尺度/归一化（低电价段梯度太小）
                          └ 是 → 实验 1 通过（agent 有最大化出力能力）
E2：regret<5% ? ─ 否 → 一维权衡的表达/信用分配（c1 可预测性、λ 量级）
```

## 5. 运行与产物

```
python agentutil/diag_launch.py                # 四段顺序，各 70k 步
python agentutil/diag_launch.py --stages S1    # 单段
python agentutil/diag_launch.py --smoke        # 每段 400 步自检（产物进 *-SMOKE 目录）
```

* 每段：`output/DIAG-<段>/AIFFLOR-runN/{df_hsrl.csv, ...}`
* 日志：`output/diag/logs/<段>.log`；进度：`output/diag/status.json`
* 评分：每段结束自动跑 `diag_report.py`，追加 `output/diag/summary.csv`
* 机时：~0.22 s/步 ⇒ 70k 步 ≈ **4.3 h/段**，四段 ≈ **17 h**
