# 单步 PF critic：成本判别精度 + 行为梯度精度

## 任务

单步功率波动（cost）：

```text
PF = |P(u_cur, a) − P(u_prev, a)|
```

- 动作：1D `a0∈[0,0.5]`，广播到 5 台（五台同因子）；
- 输入：`state=(u_prev,u_cur)`、`action=a0`；**不含任何未来风速**；
- 关键性质：给定 `(u_prev,u_cur,a)` 时 PF 是**确定性函数**（aifflor 准稳态），
  所以这是一个干净的成本判别/梯度检验任务，不存在五步 F5 的不可约风速噪声。

## 数据

`make_dataset_pf.py`：从 FFC 取相邻两步 `(u_prev,u_cur)`，两步都在 `[3.5,25]`；
按**天**划分 train/val/test（136/29/29 天）；real 用 test 天 + 结构化动作。
共 **53,000** 条（40k/5k/5k + 3k）：

| 项 | 值 |
|---|---|
| PF 均值 / sd | 1.305 / 1.332 MW |
| PF 范围 | 0 – 14.18 MW |
| FLORIS 坏点 | 0 |
| 动作 | `a0` 在 `u_prev,u_cur` 两步相同（受控口径） |

## 网络

仍是 PER critic：

```text
state (u_prev/10, u_cur/10)  (2 维)
  → Linear(2→16) → ReLU → Linear(16→100)（无激活）
  → cat(a0) 101 → Linear(101→80) → ReLU → Linear(80→50) → ReLU → Linear(50→1)
```

- 参数量 **14,009**；target 同构、`τ=1e-3` 软更新（监督任务不参与 loss）；
- 目标 `PF/5`，输出层 bias 初始化为 train 均值；Adam `1e-3` 余弦到 `1e-4`，
  batch 256，40k 步，best step=39,000（val RMSE 0.0268 MW）。

## 结果

### 成本判别（PF，MW）

| split | n | R² | RMSE | MAE | MAPE |
|---|---:|---:|---:|---:|---:|
| train | 40,000 | 0.999593 | 0.0269 | 0.0172 | 3.82% |
| val | 5,000 | 0.999608 | 0.0268 | 0.0174 | 3.93% |
| **test** | 5,000 | **0.999572** | **0.0271** | **0.0176** | **4.08%** |
| **real** | 3,000 | **0.999693** | **0.0227** | 0.0148 | 3.06% |

条件均值基线（test）：

| 基线 | R² | RMSE |
|---|---:|---:|
| `E[PF|u_cur]` | 0.1997 | 1.1710 |
| `E[PF|u_prev,u_cur]` | 0.8212 | 0.5535 |

网络把 2D 分箱基线从 0.5535 降到 **0.0271 MW**，说明它学到了
`PF=|P(u_cur,a)-P(u_prev,a)|` 的连续非线性（含动作影响）。

### 成本检出 `PF > 3 MW`（test）

| 指标 | 值 |
|---|---:|
| AUC | **1.0000** |
| Accuracy | 0.9986 |
| Precision | 0.9917 |
| Recall | 0.9937 |
| F1 | **0.9927** |
| TP / FP / FN / TN | 476 / 4 / 3 / 4517 |

3 MW 是 c1 的硬折点；几乎完美的检出说明这个 critic 可以当“高波动事件”判别器。

### 行为梯度 `∂PF/∂a`（test，400 点，δ=0.01，PF≥0.5）

| 指标 | 值 |
|---|---:|
| Pearson r | **0.9549** |
| 过原点斜率 | **1.0448** |
| 中位相对误差 | 0.3604 |
| 符号一致率 | **0.9600** |
| MAE | 0.725 |
| 逐样本余弦 mean / median | **0.880 / 1.000** |
| cosine > 0.8 | **94.0%** |
| cosine < 0 | 6.0% |
| 范数比中位 | 1.077 |

⇒ 成本几乎完美，梯度也**基本正确**：r 0.955、斜率≈1、符号 96%、94% 的点
余弦>0.8。剩余误差主要来自 `|·|` 在 PF≈0 的折点、以及 PF 很小处的相对误差；
我们已过滤 PF<0.5，但边界附近仍非线性。

## 关键发现：`P_prev` 特征会让梯度塌掉

如果状态里加 `P_prev`（`u2p`：`u_prev,u_cur,P_prev`），成本精度会略好：

| state | test R² | test RMSE | 梯度 r | 梯度斜率 | 余弦 mean | 余弦<0 |
|---|---:|---:|---:|---:|---:|---:|
| **`u2`（u_prev,u_cur）** | 0.999572 | **0.0271** | **0.9549** | **1.0448** | **0.880** | 6% |
| `u2p`（+P_prev） | 0.999622 | 0.0254 | 0.0252 | 0.0023 | 0.195 | 40% |

原因：`P_prev=P(u_prev,a)` 是**动作相关**的；网络用 `P_prev` 作为 shortcut 拟合
PF，但 autograd 时 `P_prev` 被固定，漏掉了 `∂P_prev/∂a`，所以梯度几乎为零。
这不是网络 bug，而是**特征选择**问题：

- 只做成本判别：`u2p` 可以；
- 要用 `∂PF/∂a` 做控制：**必须用 `u2`**，不要把 `P_prev`/`u_eff` 这类
  动作相关量作为输入。

## 结论

- 单步 PF 在 `(u_prev,u_cur,a)` 下是确定性的，PER critic（14k 参数）可以做到
  **test R² 0.9996、RMSE 0.027 MW、PF>3 检出 AUC=1.0 / F1=0.993**。
- 行为梯度用 `u2` 时 **r 0.955、斜率 1.045、符号 0.96、94% 点余弦>0.8**，
  已经足够做方向性优化；剩余 6% 反向点集中在 PF 很小/折点附近。
- **成本精度高 ≠ 梯度正确**：`u2p` 成本更好但梯度塌成 ~0，说明动作相关特征
  会制造 shortcut；做 cost critic 时要按“是否要梯度”选状态。

## 复现

```powershell
$PY = 'C:\Users\yigul\anaconda3\envs\llmenhancedwsis\python.exe'

& $PY agentutil\critictest\make_dataset_pf.py
& $PY agentutil\critictest\train_pf.py `
    --data agentutil\critictest\data\aifflor_pf_scalar.npz `
    --out  agentutil\critictest\results\pf_u2 `
    --state u2 --target pf --steps 40000
& $PY agentutil\critictest\evaluate_pf.py `
    --ckpt agentutil\critictest\results\pf_u2\pf_best.pt `
    --data agentutil\critictest\data\aifflor_pf_scalar.npz `
    --out  agentutil\critictest\results\pf_u2
```

产物：`data/aifflor_pf_scalar.npz`、`results/pf_u2/`（`pf_best.pt` / `metrics.json` /
`summary.txt` / `history.csv` / `plots/`），均被 `.gitignore` 排除。
