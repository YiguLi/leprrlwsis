# WFSim 为什么"更接近稳态求解"而不是"动态求解"

> **一句话**：本项目经 `WFSim_port` 调用 WFSim 时，**一次调用的本质是"1 次收敛的稳态解 + 4 次单步（1 s）确认"**：
> 第 1 步用 `dt = Inf` 直接把非定常项清零（[Make_Ax_b.m:7-11](../bin/core/Make_Ax_b.m#L7-L11)），
> 之后 4 步每步只迭代 1 次、且**输入完全不变**，实测每步只让流场移动 ~1e-5 m/s（相当于 `u_Inf` 的 1e-6）、
> 总功率变化 ≤ 6e-5 MW。它**不是**"5 秒动态推进"。
>
> 证据分三层：**结构**（源码强制 dt=∞）、**尺度**（4 s ≪ 50 s 机间对流时间）、
> **运行期实测**（探针逐步骤读数）。三层结论一致。

核查日期：2026-10-05 ｜ 引擎：MATLAB R2025a + `F:\anaconda\envs\wsis` ｜ 场景：`windTunnel`（5 台 5D 单排）

---

## 1. 决胜性结构证据：`k = 1` 就是 `dt = ∞` 的稳态解

### 1.1 非定常项的开关

```matlab
% bin/core/Make_Ax_b.m:6-11
if sol.k == 1 && Wp.sim.startUniform == 0
    dt = Inf;
else
    dt = Wp.sim.h;
end
```

而 `windTunnel` 场景**把 `startUniform` 写死为 `false`**（[meshing.m:207](../bin/core/meshing.m#L207)）
⇒ **每一次 `WFSim_port` 调用都必然走 `dt = Inf` 分支**，没有例外。

`dt` 只进入一处——非定常（时间导数）项：

```matlab
% bin/core/Dynamical.m:14-19
StrucDiscretization.ax.aP = ... + Rho.*dxx.*dyy2./dt;   % a_P^0 = ρ·Δx·Δy2/Δt
StrucDynamical.ccx        = vec(Rho.*dxx(...)'.*dyy2(...)'./dt);
```

`dt = Inf ⇒ 1/Inf = 0`（IEEE）⇒ 第 1 步的矩阵里**没有任何 ∂/∂t 的贡献**，
它就是经典的定常二维不可压动量方程 + 连续方程（`sys.A * sol.x = sys.b`）。
**第 1 步不是"时间推进一小步"，而是"求解稳态"。**

### 1.2 第 1 步是"迭代到收敛"的固定点求解，不是时间积分

- `windTunnel` 的 `startUniform = false` ⇒ [WFSim.m:103-107](../WFSim.m#L103-L107) 给 `max_it = 50`；
  收敛判据 `conv_eps = 1e-6`（[WFSim.m:80](../WFSim.m#L80)）。
- 每轮迭代重建矩阵并做**耦合稀疏线性解** `sys.A(pRCM,pRCM) \ sys.b(pRCM)`
  （[Computesol.m:29-31](../bin/core/Computesol.m#L29-L31)）；
- 收敛量取"相邻两次迭代的流场差"`sol.u − sol.uu`（[MapSolution.m:37-39](../bin/core/MapSolution.m#L37-L39)）；
- 第 1 步还带欠松弛 `alpha = min(1 − 0.9^it, 1)`（[MapSolution.m:41](../bin/core/MapSolution.m#L41)）。

⇒ 这是**稳态不动点的迭代求解**（类似 Picard/SIMPLE 式耦合迭代），不是显式/隐式时间步进。

### 1.3 第 2–5 步只是"每步 1 次迭代"的确认

- `scriptOptions.max_it_dyn = 1`（[WFSim.m:81](../WFSim.m#L81)）⇒ `k ≥ 2` 时**每步只解一次**；
- `max_it_dyn` 的语义是"每个时间步内的最大迭代次数"，即"每步不解到收敛，只走一步"；
- 时间轴 `time = 0:h:L = 0,1,2,3,4,5 s`，`NN = 5`（[meshing.m:317-318](../bin/core/meshing.m#L317-L318)）；
- 返回给 Python 的是**第 5 步**的解：`power_out.out = sum(sol_array{1,5}.turbine.power)`
  （[WFSim_port.m:55](../WFSim_port.m#L55)）。

### 1.4 5 步之间**没有任何驱动量变化**

```matlab
% bin/core/meshing.m:193-198
for i = 1:L+1
    turbInput(i) = struct('t', h*i, ...
                          'phi',      zeros(NT,1)+Control.phi, ...
                          'CT_prime', zeros(NT,1)+Control.c_prime, ...
                          'a',        zeros(NT,1)+Control.a );
end
```

`Control.phi / c_prime / a` 在全部 `i` 上取**同一组常数**，因此 `dCT_prime ≡ 0`
（[meshing.m:321-324](../bin/core/meshing.m#L321-L324)）。而 `WFSim_port` 每次调用只传一对 `(u, a)`
标量/向量（[WFSim_port.m:25-46](../WFSim_port.m#L25-L46)），风与动作在**单次调用内是常量**。
⇒ 即便把 `dt = h` 的项留着，也**没有随时间变化的驱动力**去推动流场，
`k ≥ 2` 只能是对已收敛场的"再确认"。

---

## 2. 运行期实测：4 步只让流场动 1e-5 m/s

探针：[`agentutil/probe_wfsim_steady_steps.m`](../agentutil/probe_wfsim_steady_steps.m)
（用 `WFSim_port(..., want_full=true)` 把 5 步完整解全部回传，逐步骤读 `time / turbine.power / u`）。

条件：`u_Inf = 10 m/s`、`a ≡ 1/3`、`phi = 0`、`forcescale = 1.5`、`powerscale = 0.95`。

| 步 `k` | `time` [s] | **P_total [MW]** | `mean(u)` [m/s] | 与上一步 `max\|Δu\|` [m/s] | `rms\|Δu\|` [m/s] |
|---|---|---|---|---|---|
| **1（dt=∞）** | 1.0 | **3.446264614** | 9.075921 | 相对**均匀初场**：**4.8458**（48.5 % `u_Inf`） | 1.6819 |
| 2（dt=1 s） | 2.0 | 3.446202465 | 9.075921 | 1.913e-05 | 8.003e-07 |
| 3（dt=1 s） | 3.0 | 3.446206259 | 9.075921 | 1.534e-05 | 7.198e-07 |
| 4（dt=1 s） | 4.0 | 3.446209469 | 9.075921 | 1.222e-05 | 6.552e-07 |
| 5（dt=1 s） | 5.0 | **3.446212146** | 9.075921 | 1.112e-05 | 6.025e-07 |

**读法（三个关键量）**

1. **第 1 步完成了全部工作**：相对"还没有尾流的均匀场"，`max|Δu| = 4.85 m/s`——
   尾流亏损场、上游机组的遮挡、整个稳态流场都在这一步解出来。
2. **第 2–5 步是数值噪声级**：每步 `max|Δu| ≈ 1–2e-5 m/s`（`u_Inf` 的 ~1e-6），
   `rms|Δu| ≈ 6–8e-7 m/s`。比第 1 步小 **5–6 个数量级**。
3. **返回的功率 ≈ 第 1 步的功率**：`P(5)/P(1) − 1 = −1.52e-05`
   （3.446264614 → 3.446212146 MW，差 ≈ 52 W / 3.45 MW）；
   `k ≥ 2` 每步功率变化 ≤ **6.2e-05 MW**（62 W）。`mean(u)` 五步在 6 位有效数字上完全一致。

⇒ 工程读数上，`sol_array{1,5}` 与"第 1 步的稳态解"**不可区分**。

---

## 3. 为什么"5 步"不构成物理时间窗口（三条独立论据）

### 3.1 时间尺度：4 s 连一台机间距的 8 % 都不到

| 时间尺度（`u = 10 m/s`） | 值 | 与调用内推进量（`k ≥ 2` 共 4 s）之比 |
|---|---|---|
| 域内流通时间 `Lx/u = 70D/u` | 700 s | **0.57 %** |
| 一台间距对流时间 `5D/u` | 50 s | **8.0 %** |
| 调用内"物理推进" `(NN−1)·h = 4 s` | 4 s | — |
| 该 4 s 内尾流最多输运 | 40 m = **0.40 D** | 远小于 5D 机间距 |

也就是说：**即便初场没收敛，这 4 s 也搬不动尾流**（连一台机间距都走不到）。

### 3.2 系数尺度：非定常项把解"锚"在上一场上

离散系数（`Dynamical.m:14-19`、`SpatialDiscr_Hybrid.m`、`Turbulence.m`，取 `dx = 25.09 m`、`dy2 = 57.14 m`、`ρ = 1.2`）：

| 项 | 表达式 | 量级 [kg/s] |
|---|---|---|
| 非定常/惯性 | `ρ·dx·dy2/dt`（`dt = 1 s`） | **1720.4** |
| 对流 | `ρ·u·dy2`（`u = 10 m/s`） | **685.7** |
| 湍流附加扩散（首台附近 `lm ≈ 2.5 m`） | `ρ·lm²·dy2/(dy·dx2)` | 0.30 |
| 湍流附加扩散（`x = 500 m`，`lm ≈ 22.5 m`） | 同上 | 24.2 |
| 湍流附加扩散（`x = 2000 m`，5 台混合长度**叠加**后） | 同上 | ~2e3 |

- 惯性项比对流项大 **2.5 倍** ⇒ 每个 `dt = 1 s` 步都强烈"锚定"上一场的值；
- 近场格 Peclet `|F|/D ≫ 2` ⇒ hybrid 格式退化为**纯上风**（数值扩散主导）；
- 下游混合长度逐台叠加（`Turbulence.m` 里 `lm = lm + Lmu_2D_WF(...)` 对每台机组求和）使远场涡扩散增强——
  这让远场更接近"扩散平衡的稳态"，同样不产生快速动态。

### 3.3 迭代预算：第 1 步已收敛，后 4 步各只 1 次迭代

第 1 步把非线性耦合系统迭代到 `eps ≤ 1e-6`（至多 50 次）；后 4 步 `max_it_dyn = 1`。
**"已收敛的场" + "恒定输入" + "每步 1 次迭代" = 后 4 步没有可做的事**，实测正是 1e-5 量级。

---

## 4. 与"真正的动态求解"逐项对照

| 维度 | 真实动态求解应有 | 本项目经 `WFSim_port` 的实际状态 |
|---|---|---|
| 时间推进 | 覆盖尾流对流/恢复的时间尺度 | 4 s vs 50–700 s（**0.6 %–8 %**） |
| 驱动力时变 | 风速/动作在一段仿真内变化 | 单次调用内 `(u, a, phi)` **恒定**（`dCT_prime ≡ 0`） |
| 调用间状态 | 上一时刻流场作为下一步初值 | 每次调用都 `InitWFSim` 重新**从均匀场起**（[InitWFSim.m:14-17](../bin/core/InitWFSim.m#L14-L17)）⇒ **无记忆** |
| 转子/发电机动态 | 惯量、变桨速率、转矩控制 | **无**：功率是瞬时代数式 `P = 0.95·½ρA·Cp·mean(Ue³)`（[Actuator.m:69](../bin/core/Actuator.m#L69)） |
| 额定/切出/限幅 | 机型功率曲线、变桨限功率 | **无**（限幅只在 Python 侧：动作带 `[0,0.5]`、`3.5 ≤ u ≤ 25`） |
| 尾流输运延迟 | 上游动作经对流时间后才影响下游 | 同一调用内一次性解出全流场（**零延迟**） |
| 求解方式 | 时间积分器 | **耦合线性解 `A\b` + 稳态不动点迭代**（`Computesol.m:31`） |

---

## 5. 对 RL 任务与跨引擎比较的含义

1. **env 的被控对象是"准稳态尾流 + 零输运延迟"**：`|ΔP_G|`（波动罚 `P_F`）与奖励中的成本动态
   **只能**来自"风速序列本身 + 储能动作"，**不可能**来自尾流输运。
   解释 `P_F_LIMIT = 3 MW`、`W_GAP`、`L_CLIP_CAP` 这类口径时必须带这个前提。
2. **跨引擎比较是自洽的**：`aifflor` 侧 `declared_scope()` 同样声明准稳态；
   `yawflor` 的 `YAW_TRANSPORT_DELAY` 默认 `off`（开启即给 env 引入记忆、并破坏确定性判据）。
   两边都"无输运延迟"⇒ WFSim↔FLORIS 的差异只剩**尾流形状/电平**，没有"动态 vs 准稳态"的混淆项。
3. **"块首取值 = 5 分钟分辨率单次调用"成立**：正因为逐次调用无记忆，
   曲线探针里 `curve/*_5min_first`（块内第 1 分钟的引擎返回值）确实等价于
   "把引擎按 5 min 分辨率调一次（动作保持整块）"，`--block 1` 与 `--block 5` 在同一分钟取值相同
   （已在 `output/engine_curve_random_day{,_b5}` 实测核对）。
4. **不能主张的东西**：本链路里没有尾流传播延迟、没有转子惯性、没有变桨/转矩动态。
   任何"由于尾流输运导致功率滞后/超调"的解释在这条路径上**不成立**（必须换模型才能谈）。

---

## 6. 边界与反例：什么情况下它**确实**在解动态

- **把 `h` 放大 / 让输入随时间变化**：原版 WFSim 的 `windTunnel` 用法是"`turbInput` 逐时间步变化"，
  此时 `k ≥ 2` 的 `dt = h` 项才真正承担时间推进。本项目 `WFSim_port` 传常数输入，故该路径被"架空"。
- **`startUniform = true`**：`k = 1` 只迭代 1 次（[WFSim.m:103-104](../WFSim.m#L103-L104)），
  整条链路退化为"多次单步松弛"。本项目用的是 `false`。
- **下游远场**：混合长度逐台叠加使涡扩散在该处可与惯性项比拟，出流段更"稳态扩散平衡"——
  但这只让结论更强，不产生快速动态。
- **适用范围**：本报告针对 `windTunnel` 场景（5 台 5D 单排）与 `WFSim_port` 的恒定输入调用方式。
  `windTunnel_10turb` 等其他场景的 `h = 1, L = 5` 相同，结论不变
  （`clwindcon_*` 大场景的 `h/L` 需另查）。

---

## 7. 复现

**方式 A（推荐）——在 MATLAB 命令行里：**

```matlab
cd('<repo root>');  addpath(genpath('agentutil'));
out = probe_wfsim_steady_steps(10.0, [1/3 1/3 1/3 1/3 1/3]);
```

**方式 B——经 MATLAB 引擎（本报告的数据就是这么得到的）：**

```python
# ⚠ MATLAB 启动需要能写工作区之外的偏好/临时目录（受限沙箱下会 CreateFile failed: Access is denied）
import os, matlab.engine as me
eng = me.start_matlab()
eng.cd(os.getcwd(), nargout=0)
eng.eval("addpath(genpath('agentutil'));", nargout=0)
eng.eval("out = probe_wfsim_steady_steps(10.0, [1/3 1/3 1/3 1/3 1/3]);", nargout=0)
eng.quit()
```

- 产物：`output/wfsim_steady_steps.mat`（含 `P / T / MU / MX / MN / DU / DRMS / D0`）
- 探针：[`agentutil/probe_wfsim_steady_steps.m`](../agentutil/probe_wfsim_steady_steps.m)（只读引擎，不改任何项目文件）
- 两次独立运行的数值**逐位一致**（确定性引擎）

---

## 8. 关联材料

- 同结论的首次登记：提交 `903561f`（更正"WFSim_port 不是只推 5 s 的准稳态，而是 `dt = Inf` 直接解稳态"）
- 引擎"无记忆"的独立核查：`agentutil/probe_wfsim_memoryless.py`
- 准稳态声明的另一侧：`aifflor_turbine.declared_scope()`、`aifflor/README.md` §2.4
- `want_full` 开关的逐位 A/B（只在"是否回传 test 字段"上有差异，数值不变）：`WFSim_port.m` 头注释、`output/_wfsim_ab/`
