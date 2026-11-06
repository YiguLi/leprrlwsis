# critictest —— 用 PER 结构的 critic-target 拟合风场出力 `P(u, a)`

> **扩展**：
> * 五步功率和任务（`window=5`，只给窗口首步 `u1`）见
>   [README_5step.md](README_5step.md)。
> * 五步功率**波动和**任务（`F5=Σ|ΔP|`，只给 `u1`）见
>   [README_fluct.md](README_fluct.md)。
> * **逐步奖励分解 critic**（FiLM + 共享逐步 reward 头 + 终端头 + QComposer，
>   按 `macro_critic_step_reward_decomposition_architecture.md`）见
>   [README_step_decomp.md](README_step_decomp.md)。
> * **单步 PF critic**（成本判别 + 行为梯度）见 [README_pf.md](README_pf.md)。
> * **五步块整体 C1 critic**（重尾成本 + 折点分组梯度）见 [README_c1.md](README_c1.md)。

## 一句话结论

**PER 的 critic 结构（`1→16→100 ∥ 5 → 80 → 50 → 1`，14 313 参数）可以很好地拟合
aifflor 的 `P(u,a)`**：主臂在独立 test 上 `R²=0.999916`、`RMSE=0.131 MW`、
`MAPE=0.548%`；在"实测风速 + 结构化动作"的 real 集上 `RMSE=0.062 MW`、
`MAPE=0.690%`。动作灵敏度也基本对：`∂P/∂a` 的 pooled Pearson `r=0.966`、
过原点斜率 `0.999`、符号一致率 `92.3%`。

但有一个**必须说清楚的前提**：如果让网络直接回归 `P`（消融臂 `--target p`），
整体 R² 仍然很高（0.9997），**低风速相对误差与动作响应却明显变差**
（`MAPE=2.95%`、按 `s=P/u³` 衡量的动作响应 `R²` 只有 `0.416`）。
主臂把 aifflor 准稳态关系里**精确的 `u³` 因子**提到网络外
（网络学 `s=1000·P/u³`，预测时 `P̂=ŝ·u³/1000`），同样的结构才同时拟合好
"风速趋势"和"五维动作响应"。这不是加输入特征，而是**输出参数化**。

---

## 1. 任务与口径

### 1.1 被拟合对象

* 风场仿真：**aifflor / FLORIS 4.6.6**，5 台 5D 单排、`D=100 m`、`ρ=1.2`、
  `shear=0.12`、`TI=0.06`、`γ≡0`。
* 生产配置（与 `aifflor/aifflor_main.py` 一致）：
  `AifflorPhysics(calibrated=False, power_scale=LEVEL_COEF)`，
  `LEVEL_COEF=1/1.6696≈0.598946`（把 FLORIS 电平抬到 WFSim）。
* 输入：风速 `u∈[3.5,25] m/s`；行为 `a∈[0,0.5]^5`（5 台轴向因子）。
  物理层把 `a<0.025` 抬到 `0.025`（高斯尾流 `a=0` 奇点，见
  `aifflor_turbine.A_PHYS_LO`）；数据集同时保存原始 `a`（网络输入）与 `a_eff`。
* 输出：风场总有功 `P` [MW]。
* 锚点：`u=10, a=1/3` ⇒ **3.559146 MW**（`aifflor/README.md` 的 3.294 是
  `calibrated=True` 的另一条口径，不要混）。

### 1.2 为什么主目标用 `s = 1000·P/u³`

aifflor 是准稳态 FLORIS：单机 `P=0.95·½ρA·Cp(a)·u_eff³`，尾流只改变
`u_eff/u` 的常数 ⇒ **`P` 的主导项 ∝ `u³`**。实测同一动作下 `s=1000·P/u³`：

| u (m/s) | 3.5 | 5 | 8 | 10 | 15 | 20 | 25 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `1000·P/u³`（a≡1/3） | 3.8805 | 3.6564 | 3.5858 | 3.5591 | 3.5584 | 3.5511 | 3.5501 |

`3.5→25` 只变化 **0.330（9.0%）**，且随 u 平滑。把 `u³` 提走后，网络只需学
5 维动作响应 `s(u,a)`（其中残余的 u 依赖很弱）；预测时乘回 `u³`。
`--target p` 消融臂直接回归 `P/10`，`--target log` 回归 `log P`，用于对照。

### 1.3 数据

`make_dataset.py` 一次生成 **53 000** 点，FLORIS 层 **0 坏点 / 0 兜底**：

| split | n | 采样 |
|---|---:|---|
| train | 40 000 | `u~U(3.5,25)`，`a~U(0,0.5)^5` |
| val | 5 000 | 同上（独立） |
| test | 5 000 | 同上（独立） |
| real | 3 000 | `u` 从 `FFC_speed_noise_1.csv` 实测序列抽；`a` 取 5 种动作族 |

real 的 5 种动作族：`level`（整场同水平）、`ramp`（沿程斜坡）、`cp_opt`
（单机 Cp 最优 1/3 附近）、`farm_opt`（场级 0.20 附近）、`hetero`（逐台独立）。
它们不是策略分布，但覆盖了工程上常见的动作形状。

覆盖集 `P∈[0.089, 56.803] MW`，`s∈[1.266, 3.931]`；real `P∈[0.174, 41.761] MW`。
数据与训练全部固定 seed；`data/`、`results/` 由 `.gitignore` 排除，可随时重建。

---

## 2. 网络：与 PER 逐层相同

```text
state x = [u/10]                     (ns = 1)
   │
   ├─ Linear(1→16) ─ ReLU
   ├─ Linear(16→100)                    ← PER 的 fc4 本来就没有激活，这里保持原样
   │
action a (5) ────────────────────────── cat → 105
                                      ├─ Linear(105→80) ─ ReLU
                                      ├─ Linear(80→50)  ─ ReLU
                                      └─ Linear(50→1)
```

* 直接复用 `model.Critic(arch='v1', hidden1=16, hidden2=100)`；参数量 **14 313**
  （online 与 target 同构）。
* target：`hard_update` 同步初值，训练中每步 `soft_update(τ=1e-3)`（PER 口径）。
* **单步确定性标签 ⇒ 没有 bootstrap**。PER 里 target 网络用于算
  `r+γQ'(s',a')`；这里标签就是 `s`（或 `P/10`），等价于 `γ=0`，target 不参与
  loss。它仍按 PER 的 `τ` 维护，作为 EMA 副本做一致性检查
  （test 上 online-target `mean|ΔP|=0.0073 MW`）。
* 输出层 bias 初值 = 训练目标均值（PER 生产路径用 `BiasInitTracker` 设价值基线；
  这里等价处理，避免前几千步都在搬量级）。

训练：Adam `lr=1e-3` 余弦退火到 `1e-4`、batch 256、40 000 步、梯度裁剪 2.0
（PER 的 `clip_grad_norm_` 口径）、按 val RMSE 选最优。CPU 约 **169 s**
（4.22 ms/step）。

---

## 3. 结果

### 3.1 出力拟合（`P`，单位 MW）

| split | n | R² | RMSE | MAE | MAPE | max\|e\| |
|---|---:|---:|---:|---:|---:|---:|
| train | 40 000 | 0.999924 | 0.1251 | 0.0686 | 0.528% | 2.721 |
| val | 5 000 | 0.999914 | 0.1327 | 0.0712 | 0.540% | 1.413 |
| **test** | 5 000 | **0.999916** | **0.1310** | 0.0712 | **0.548%** | 1.998 |
| **real** | 3 000 | **0.999734** | **0.0617** | 0.0267 | **0.690%** | 0.857 |

动作响应本身（`s=1000·P/u³`，由预测 P 反算）：test `R²=0.994384`、
`RMSE=0.02274`；real `R²=0.992347`、`RMSE=0.03398`。P 的 R² 之所以更高，是因为
`u³` 的动态范围（0.09→56.8 MW）占掉了绝大部分方差；**s 的 R² 才是"五维动作
学得好不好"的指标**。

### 3.2 test 按风速分箱

| u 区间 (m/s) | n | R² | RMSE (MW) | MAPE |
|---|---:|---:|---:|---:|
| [3.5, 5.0) | 331 | 0.99910 | 0.0024 | 0.730% |
| [5.0, 7.0) | 495 | 0.99934 | 0.0056 | 0.604% |
| [7.0, 9.0) | 441 | 0.99906 | 0.0120 | 0.574% |
| [9.0, 11.0) | 454 | 0.99859 | 0.0234 | 0.541% |
| [11.0, 13.0) | 481 | 0.99871 | 0.0349 | 0.512% |
| [13.0, 16.0) | 676 | 0.99864 | 0.0741 | 0.558% |
| [16.0, 20.0) | 963 | 0.99907 | 0.1237 | 0.522% |
| [20.0, 25.0) | 1159 | 0.99906 | 0.2396 | 0.498% |

低风速段（3.5–5 m/s，P 只有 0.1–0.4 MW）RMSE 仅 **0.0024 MW**、MAPE 0.73%，
没有出现"直接拟合 P"时低风速相对误差崩掉的问题（消融臂该段 R² 只有
0.536、MAPE 18.4%）。

### 3.3 real 按动作族

| family | n | R² | RMSE (MW) | MAPE | mean P |
|---|---:|---:|---:|---:|---:|
| level | 600 | 0.999944 | 0.0271 | 0.480% | 4.91 |
| ramp | 600 | 0.998672 | 0.1219 | 1.733% | 4.59 |
| cp_opt | 600 | 0.999948 | 0.0309 | 0.362% | 5.71 |
| farm_opt | 600 | 0.999909 | 0.0339 | 0.435% | 5.12 |
| hetero | 600 | 0.999914 | 0.0365 | 0.442% | 5.23 |

`ramp` 是最难的族（沿程存在强尾流耦合），RMSE 0.122 MW / MAPE 1.73%，
约 **2.7% 的相对误差**（0.122/4.59），其余族在 0.4–0.5% 量级。

### 3.4 动作梯度 `∂P/∂a`（test，400 点，δ=0.01）

真值 = aifflor 逐维中心差分；预测 = autograd。
自检：autograd vs 网络自身差分（δ=1e-3）`median|Δ|=9.9e-5`、`p90=4.6e-4`
MW/unit（ReLU 分段线性，δ=1e-2 的割线跨折点，不能逐点相等）。

| dim | n | Pearson r | 过原点斜率 | 中位相对误差 | 符号一致率 | MAE |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 400 | 0.9788 | 1.0044 | 0.215 | 0.963 | 1.549 |
| 1 | 400 | 0.8701 | 0.9727 | 0.401 | 0.884 | 1.047 |
| 2 | 400 | 0.8400 | 0.9275 | 0.464 | 0.862 | 0.896 |
| 3 | 400 | 0.9366 | 0.9525 | 0.387 | 0.891 | 0.932 |
| 4 | 400 | 0.9620 | 1.0039 | 0.267 | 0.997 | 1.789 |
| **pooled** | 2000 | **0.9657** | **0.9987** | 0.348 | 0.923 | 1.243 |

`|g|` 中位：真值 1.396 vs 网络 1.416 MW/unit；p90：11.489 vs 11.800。
⇒ **量级与方向都对**；dim1/dim2 的 r（0.84–0.87）略低，是尾流耦合最强、
最难由 16/100 的容量精确刻画的两维。

### 3.5 消融：直接拟合 `P`（`--target p`）

| 臂 | 目标 | test R² | test RMSE | test MAPE | real R² | real RMSE | s R²(test) | grad r | grad slope |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **sp（主）** | `1000·P/u³` | 0.999916 | 0.131 | 0.548% | 0.999734 | 0.062 | 0.9944 | 0.966 | 0.999 |
| p（消融） | `P/10` | 0.999673 | 0.259 | 2.947% | 0.998454 | 0.149 | 0.4156 | 0.930 | 0.927 |

直接拟合 P 的 R² 看起来仍很高，但它主要是在拟合 `u³` 趋势；相对误差
（MAPE 2.95% vs 0.55%）与动作响应（s R² 0.416 vs 0.994）都明显更差。
⇒ 同样约 14k 参数、同样层数，**输出参数化比加宽网络更关键**。

### 3.6 解析代理对照（不是网络）

`P≈k·u³·Σ4a(1−a)²`，k 在 train 上最小二乘（k=0.001364，忽略尾流）：

| split | R² | RMSE (MW) | MAPE |
|---|---:|---:|---:|
| train | 0.98305 | 1.8632 | 7.76% |
| test | 0.98412 | 1.8067 | 7.58% |
| real | 0.96812 | 0.6759 | 9.68% |

网络相对这个"单机 Cp·u³ 可加代理"把 test RMSE 从 1.81 MW 降到 0.13 MW。
注意这个代理**不含尾流**，只作量级参照，不是同容量基线。

### 3.7 图

`results/sp/plots/`：

| 图 | 内容 |
|---|---|
| `fit_test.png` | test 真值-预测散点（线性 + log-log） |
| `per_u_bin.png` | 分风速段 `1−R²`（对数）与 RMSE |
| `residual_vs_u.png` | 残差 vs 风速 + 分箱均值±sd |
| `train_curve.png` | online/target val RMSE 与 val R² |
| `gradient_parity.png` | 逐维 `∂P/∂a` 真值 vs 网络 |
| `real_families.png` | real 各动作族的 `1−R²` 与 RMSE |
| `online_target.png` | online-target 预测差直方图 |

---

## 4. 运行

环境：`C:\Users\yigul\anaconda3\envs\llmenhancedwsis\python.exe`
（Python 3.12 / torch 2.12 CPU / numpy 2.4 / floris 4.6.6）。**不需要 MATLAB。**

```powershell
$PY = 'C:\Users\yigul\anaconda3\envs\llmenhancedwsis\python.exe'

# 1) 生成数据集（约 36 s；53k 点）
& $PY agentutil\critictest\make_dataset.py

# 2) 主臂：s = 1000·P/u³（约 3 min，训练完自动评价 + 出图）
& $PY agentutil\critictest\train_critic.py --target sp `
    --data agentutil\critictest\data\aifflor_ua_p.npz `
    --out  agentutil\critictest\results\sp

# 3) 消融：直接拟合 P/10
& $PY agentutil\critictest\train_critic.py --target p `
    --data agentutil\critictest\data\aifflor_ua_p.npz `
    --out  agentutil\critictest\results\p

# 4) 只评价已有 checkpoint（可重复跑；不会重训）
& $PY agentutil\critictest\evaluate.py `
    --ckpt agentutil\critictest\results\sp\critic_best.pt `
    --data agentutil\critictest\data\aifflor_ua_p.npz `
    --out  agentutil\critictest\results\sp
```

常用开关：`make_dataset.py --n_train/--n_val/--n_test/--n_real_per_family/--seed/
--calibrated/--power_scale`；`train_critic.py --steps/--bsize/--lr/--tau/--target/
--state_feat/--hidden1/--hidden2/--seed/--threads`。

产物（都被 `.gitignore` 排除）：

```text
agentutil/critictest/data/aifflor_ua_p.npz     # u,a,a_eff,p,s,split,family,meta
agentutil/critictest/data/aifflor_ua_p.json    # 数据口径快照
agentutil/critictest/results/sp/critic_best.pt # online/target state_dict + meta
agentutil/critictest/results/sp/history.csv
agentutil/critictest/results/sp/metrics.json
agentutil/critictest/results/sp/summary.txt
agentutil/critictest/results/sp/per_u_bin.csv
agentutil/critictest/results/sp/per_family.csv
agentutil/critictest/results/sp/gradient.csv
agentutil/critictest/results/sp/plots/*.png
```

---

## 5. 文件

| 文件 | 内容 |
|---|---|
| `common.py` | 路径 bootstrap、aifflor 物理、采样、`s` 目标变换、PER critic 构造、指标 |
| `make_dataset.py` | 生成 `(u,a)→P` 数据集（覆盖集 + 真实动作族） |
| `train_critic.py` | online + target 监督训练、验证/早停/checkpoint |
| `evaluate.py` | 四个 split 的 R²/RMSE/MAPE、分风速段、动作族、梯度体检、图 |

本目录遵守 `AGENTS.md` 的约定：**只 import 项目模块，反向不 import**；所有生成物
都在 `data/` / `results/` 下（已 ignore），脚本本身入库。

---

## 6. 口径与局限（引用结果时请一并说明）

1. **准稳态 FLORIS**：每个控制周期内假定轴向因子已到位、尾流已平衡；不建模
   变桨/转矩执行器动力学与尾流输运延迟（与 aifflor 的声明相同）。
2. **绝对电平**：`LEVEL_COEF` 把 FLORIS 对齐到 WFSim 的**平均**电平；逐点比值
   仍有 1.31–2.15 的范围（强降载/异质动作区偏差最大，见
   `aifflor/README.md` §3.7.1）。本实验拟合的是"这条 aifflor 曲线"，不是 WFSim 真值。
3. **采样分布**：train/val/test 是 `u` 均匀、`a` 5 维均匀的**函数覆盖**，不是策略
   分布；real 用实测风速 + 5 种动作族，但也只覆盖了工程上常见的形状。
4. **`s` 目标不是"额外信息"**：`s=1000P/u³` 是 `P` 的确定性重参数化，网络输入里
   没有 `u³` 这个特征；`--target p` 是同一任务的直接拟合版本。两者用同一结构、
   同一数据、同一训练预算对比。
5. **梯度指标是局部量**：autograd 给的是 ReLU 网络的局部导数；真值是 δ=0.01 的
   中心差分（物理光滑）。dim1/dim2 的 r 0.84–0.87、pooled 中位相对误差 0.35
   说明**尾部/强尾流耦合点**仍有偏差；若要用于逐台寻优，建议在目标工况附近
   再做一次局部校准，或加宽 `--hidden1/--hidden2`。
6. **target 网络的角色**：单步监督下它不参与 loss，只是 PER `τ` 软更新的 EMA 副本；
   若要把这个 critic 放回 RL 的 TD 更新，必须用真实 `(s,a,r,s',a')` 转移重新定义
   bootstrap，不能直接复用本实验的"标签=输出"口径。
7. **MAPE 分母**：`max(|P|, 0.05 MW)`；低风速近零点的相对误差会被放大，故同时报
   RMSE/MAE 与分风速段。
