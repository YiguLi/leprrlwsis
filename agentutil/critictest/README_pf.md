# 单步 PF critic：成本判别精度 + 行为梯度精度

## 任务

单步功率波动（cost）：

```text
PF = |P(u_cur, a)  −  P(u_prev, a_prev)|
```

- 当前动作：1D `a∈[0,0.5]`，广播到 5 台；
- `P_prev = P(u_prev, a_prev)` 是**上一步动作**产生的**过去量**，决策时已知且固定，
  对当前动作 `a` 的导数为 0；
- 输入：`state=(u_prev,u_cur,P_prev)`、`action=a`；**不含未来风速**；
- 关键性质：给定这三个量时 PF 是确定性的（aifflor 准稳态），
  所以这是干净的成本判别/梯度检验任务。

## 数据（两种口径）

| 文件 | `P_prev` 的动作 | 用途 |
|---|---|---|
| `aifflor_pf_sep.npz` | **独立 `a_prev`**（真实口径） | 主结果：`u2p=(u_prev,u_cur,P_prev)` |
| `aifflor_pf_scalar.npz` | 与当前 `a` 相同（受控口径） | 早期对照；说明 `P_prev` 口径的影响 |

两文件都是 FFC 相邻两步、按天划分 136/29/29 天、40k/5k/5k + 3k real。
真实口径下 PF 均值 **1.851**、sd **1.914**、范围 0–20.39 MW。

## 网络

PER critic：

```text
state (u_prev/10, u_cur/10, P_prev/5)  (3 维)
  → Linear(3→16) → ReLU → Linear(16→100)（无激活）
  → cat(a) 101 → Linear(101→80) → ReLU → Linear(80→50) → ReLU → Linear(50→1)
```

- 参数量 **14,025**；target 同构、`τ=1e-3` 软更新（监督任务不参与 loss）；
- 目标 `PF/5`；Adam `1e-3` 余弦到 `1e-4`，batch 256，40k 步；
- 真实口径 best step=??（见 `results/pf_sep_u2p/summary.txt`）。

## 主结果：真实口径（`a_prev` 独立）

### 成本判别（PF，MW）

| split | n | R² | RMSE | MAE | MAPE |
|---|---:|---:|---:|---:|---:|
| train | 40,000 | 0.998925 | 0.0628 | 0.0435 | 10.24% |
| val | 5,000 | 0.9989 | 0.0609 | 0.0433 | 10.39% |
| **test** | 5,000 | **0.998857** | **0.0609** | **0.0433** | **10.39%** |
| **real** | 3,000 | **0.998965** | **0.0545** | 0.0399 | 9.63% |

基线（test）：`E[PF|u_cur]` R²=0.2944/RMSE=1.5139；
`E[PF|u_prev,u_cur]` R²=0.4915/RMSE=1.2851。

`PF>3 MW` 检出（test）：**AUC=0.9998、Accuracy=0.9938、Precision=0.9861、
Recall=0.9808、F1=0.9834**（TP=920/FP=13/FN=18/TN=4049）。

### 行为梯度 `∂PF/∂a`（test，400 点，δ=0.01，PF≥0.5）

| 指标 | 值 |
|---|---:|
| Pearson r | **0.9635** |
| 过原点斜率 | **0.9643** |
| 中位相对误差 | 0.3279 |
| 符号一致率 | **0.9054** |
| MAE | 2.0157 |
| 逐样本余弦 mean / median | **0.795 / 1.000** |
| cosine > 0.8 | **89.8%** |
| cosine < 0 | 10.2% |
| 范数比中位 | 1.049 |

⇒ 真实口径下，**P_prev 作为固定过去量**，成本 R²≈0.999、梯度 r≈0.96、
斜率≈0.96、90% 点余弦>0.8；这是可用的 cost critic。

### 对照：去掉 `P_prev`（`u2=(u_prev,u_cur)`）

| state | test R² | test RMSE | 梯度 r | 梯度斜率 | 余弦 mean | 余弦<0 |
|---|---:|---:|---:|---:|---:|---:|
| **`u2p`（含 P_prev）** | **0.998857** | **0.0609** | **0.9635** | **0.9643** | **0.795** | 10.2% |
| `u2`（不含 P_prev） | 0.680688 | 1.0184 | 0.7827 | 0.6259 | 0.605 | 19.8% |

真实口径下 `a_prev` 与当前 `a` 独立，`P_prev` 不能由 `(u_prev,u_cur,a)` 推出；
去掉它，成本 R² 从 0.999 掉到 0.681，梯度也变差。**P_prev 是必要的因果状态，
不是 shortcut。**

## 为什么之前说 `P_prev` 会造成梯度塌陷（更正）

早期受控数据里让 `u_prev` 和 `u_cur` **共用同一个 `a`**，于是
`P_prev = P(u_prev, a)` 变成了**当前动作**的函数；网络用 `P_prev` 拟合 PF 后，
autograd 时又把它当固定输入，漏掉 `∂P_prev/∂a`，梯度才塌成 r≈0.025。
这是**数据集口径造成的假象**，不是 `P_prev` 本身的问题。

在真实口径（`a_prev` 独立）下，`P_prev` 对当前 `a` 的导数确实为 0，
`u2p` 的梯度正常（r=0.9635、斜率=0.9643、符号=0.905）。

**正确做法**：

- `P_prev` 用**上一步动作/实测**得到，决策时固定；
- 当前动作 `a` 只作用于 `P(u_cur,a)`；
- 不要把 `P_prev` 用当前 `a` 重算，也不要把 `u_eff(a)` 这类当前动作相关量
  混进状态（那才会产生真正的 shortcut）。

## 受控口径的参考结果（同一 `a`）

| state | test R² | test RMSE | 梯度 r | 梯度斜率 | 余弦 mean | 余弦<0 |
|---|---:|---:|---:|---:|---:|---:|
| `u2` | 0.999572 | 0.0271 | 0.9549 | 1.0448 | 0.880 | 6.0% |
| `u2p`（P_prev=P(u_prev,a)） | 0.999622 | 0.0254 | 0.0252 | 0.0023 | 0.195 | 40.2% |

这个表只说明“同一动作口径下把动作相关量当输入”的假象；真实口径看上面的主结果。

## 结论

- **真实口径**：`u2p`（含固定 `P_prev`）成本 **R²=0.9989、RMSE=0.061 MW**，
  `PF>3` 检出 **AUC=0.9998 / F1=0.9834**；
  `∂PF/∂a` **r=0.9635、斜率=0.9643、符号=0.905、89.8% 点余弦>0.8**。
- `P_prev` 是过去量，和当前动作无关，是合法且必要的因果特征；不要用当前动作
  重算它。
- 之前 `u2p` 梯度塌陷是受控数据集把 `a_prev` 和 `a` 设成同一个造成的；
  用户指出的问题成立，已按独立 `a_prev` 重新生成/训练/评价。
- 剩余 10% 反向余弦点集中在 PF≈0 的 `|·|` 折点附近；控制时过滤 PF 很小的点
  或做局部平滑。

## 复现

```powershell
$PY = 'C:\Users\yigul\anaconda3\envs\llmenhancedwsis\python.exe'

# 真实口径：a_prev 独立
& $PY agentutil\critictest\make_dataset_pf.py --sep_prev_action `
    --out agentutil\critictest\data\aifflor_pf_sep.npz
& $PY agentutil\critictest\train_pf.py `
    --data agentutil\critictest\data\aifflor_pf_sep.npz `
    --out  agentutil\critictest\results\pf_sep_u2p `
    --state u2p --target pf --steps 40000
& $PY agentutil\critictest\evaluate_pf.py `
    --ckpt agentutil\critictest\results\pf_sep_u2p\pf_best.pt `
    --data agentutil\critictest\data\aifflor_pf_sep.npz `
    --out  agentutil\critictest\results\pf_sep_u2p
```
