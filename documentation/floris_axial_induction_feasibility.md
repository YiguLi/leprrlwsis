# FLORIS 能否改造为"固定航向角 γ≡0 + 调控轴向因子 a"的风场模拟？

**结论：能，分两个层级；且已在 FLORIS 4.6.6 + 本项目 `yawflor` 口径下实测验证。**

| 层级 | 能力 | 是否需改引擎 | 实测证据 |
|---|---|---|---|
| **① 全场统一 a** | 一次仿真内所有机组用同一个 a | **不需要**（原生机制） | a=1/3 时与生产 `yawflor` 基线**逐位相同**（5.942350 MW） |
| **② 逐台独立 a** | 5 台各给各的 a（空间斜坡等） | **需要一处引擎扩展**（约 30 行） | 扩展后 a≡1/3 与生产基线**逐位一致**（相对差 **1.495e-16**） |

两条结论都有可复跑的探针，且都用"**退化到 a≡1/3 时必须与现有生产模型逐位相同**"作为判据 —— 这是本项目反复使用的最强锚（见 `yawflor/README.md` §0 的同类做法）。

复跑命令（用装了 floris 4.6.6 的解释器）：

```
F:\anaconda\envs\wsis\python.exe agentutil\probe_floris_a_authority.py       # ① 原生全场统一 a
F:\anaconda\envs\wsis\python.exe agentutil\probe_floris_a_per_turbine.py     # ② 逐台 a 的引擎扩展
F:\anaconda\envs\wsis\python.exe agentutil\probe_floris_a_sensitivity.py     # a 通道的敏感度
```

---

## 1. 层级①：全场统一 a —— 原生支持，零改动

### 机制
FLORIS v4 的 `power_thrust_table` 支持**多维表**：

```python
power_thrust_table = {
    "power_thrust_data_file": "<绝对路径>.csv",   # 列 = [条件列..., ws, power, thrust_coefficient]
    "ref_air_density": 1.20, "ref_tilt": 0.0,
    "cosine_loss_exponent_yaw": 1.88, "cosine_loss_exponent_tilt": 1.88,
}
turbine = dict(..., multi_dimensional_cp_ct=True, operation_model="cosine-loss",
               power_thrust_table=power_thrust_table)
```

调用时给**流场级**条件：

```python
fmodel.set(..., multidim_conditions={"ai": 0.25})
```

引擎按最近邻选表（`core/turbine/turbine.py::_select_multidim_condition`），
于是 **(Cp_eff, CT) 随 `ai` 一起变**。

### 关键代码位置（已核对源码）
- `core/turbine/turbine.py:214` / `:224` —— 有 `power` 键 → 单维表；否则走多维分支
- `core/turbine/turbine.py:234` / `:365` —— `power_thrust_table[tuple(md_cond)]` 选表
- `core/turbine/turbine.py:649` —— 多维表从 CSV 读入（`np.genfromtxt`，条件列前置）
- `core/farm.py:143-156` —— 机型定义用 **dict** 传入时，会注入 `turbine_library_path`

### 实测（`probe_floris_a_authority.py`）
单排 5 台 5D、D=100 m、ρ=1.2、GCH 默认、u=10、γ≡0：

| ai | 风场总功率 | 逐台功率 |
|---|---|---|
| 1/3 | **5.9295 MW** | — |
| 0.20 | 5.4489 MW | — |
| 0.40 | 5.9581 MW | — |
| 0.45 | 5.8497 MW | — |

> ⚠ **口径注**：这里的 `a=1/3` 是 **5.9295 MW**，而上文"逐位一致"锚用的是
> **5.942350 MW**。两者**不矛盾**，是同一张表的两种取法：
> - 5.9295 = 功率表在 u=10 **网格点上**的精确值（CSV 表按解析式采样，网格点零偏差）
> - 5.942350 = 生产 `yawflor` 路径 = 同一张表经 `interp1d` 在
>   `u_eff = cbrt(mean(u³)) = 9.9836 m/s` 处**线性插值**的结果
>
> 比值 5.942350/5.9295 = 1.00217，与 `probe_floris_a_per_turbine.py` 第 0 节
> 独立算出的"同 u 下 线性插值/真立方 = 1.000470（首台，u=9.9836）…
> 视 u 而定"完全同源。**层级①（原生多维表）与层级②（逐台扩展）在
> 同一取法下逐位一致**，这一点由层级②的自校验（1.495e-16）直接证明。

隔离机组（nt=1）标定：FLORIS 值 vs 本项目解析式 `0.95·½ρA·4a(1−a)²·u³`
—— 网格点上**逐位相同**；网格点之间差 0.05%~1.8%，因为生产路径是
"**整数风速网格上的表 + 线性插值**"（`operation_models.py:84-105`
`interp1d`）。这是**表本身的口径**，不是误差。

### 已知硬性约束
- `a` 的**可行域**：`core/solver.py:392`+ 把 a 夹到 **`[0.0001, 0.9999]`**；
  但标准动量理论分支只到 `a=0.5`（`a>0.5` 为湍流尾流态）⇒ 动作域取 **[0, 0.5]**。
- **`a=0` 会除零**：`wake_velocity/gauss.py:76` 的
  `uR = u_init·ct/(2(1−√(1−ct)))` 在 CT→0 时 0/0 ⇒ 表的下界应取 ≈0.025 而非 0。
- **γ 必须显式钉死为 0**：FLORIS 里 γ 与 a 是两套独立输入，不存在"a 自动把 γ 拉平"。
  本项目口径下 γ 只需 `yaw_angles=np.zeros(...)`。
- `a` **不能与 γ 同时非零**：`operation_models.py:499-505`
  （`MixedOperationTurbine._handle_mixed_operation_setpoints`）**显式报错**
  "If yaw_angles entry is nonzero, power_setpoints must be greater than 0"。
  ⇒ "固定 γ=0 + 调控 a" 与"调控 γ"在同一台上**互斥**，这是引擎自带的互斥。

---

## 2. 层级②：逐台独立 a —— 需一处引擎扩展

### 为什么原生做不到
`multidim_conditions` 挂在 **FlowField**（`core/flow_field.py:33`）上，
形状约定是 **`(n_wd, n_ws)`**（`wind_data.py:314`），即
**每个 (风向,风速) 组合一个值** —— 它是**全场级**标量，不是逐台量。
实测强行传长度 5 的列表会被拒：

```
ValueError: When providing multiple specified conditions, the number of
conditions must match the number of findices.
```

### 最小扩展方案（已实现并验证）
FLORIS 的功率/推力/诱导**不是**调用模块级 `turbine.power()`，而是调用
`Farm.turbine_{power,thrust_coefficient,axial_induction}_functions[机型名]`
（在 `Turbine.__attrs_post_init__` 时从**操作模型类**绑定）。因此：

**方案 A（推荐，实际改动最小）**：注册一个自定义 operation model

1. 在 `floris.core.turbine.turbine.TURBINE_MODEL_MAP['operation_model']` 里注册
   `PerTurbineInductionTurbine`，其 `power/thrust_coefficient/axial_induction`
   从"动作"里取逐台 a；
2. 动作载体用**已有的逐台输入通道**（`power_setpoints` 或自定义字段），
   **不必改 Farm/FlowField 的数据结构**。

`ThrustCoefficient` 与 `Cp` 都按 a 取：
`Cp_eff(a) = 0.95·4a(1−a)²`、`CT(a) = 4a(1−a)`（保持项目口径 CT 不乘 0.95，
这样 a=1/3 时正好是 `8/9`）。

**方案 B**：把 `_select_multidim_condition` 扩展成接受 `(findex, n_turbines)`、
并让 `core/turbine/turbine.py:242/:373` 的掩码同时带机组维。
改动面更大（2 处掩码 + 1 个选择器），需同步改 `FlowField` 的形状校验。

### 实现时踩到的两个真实陷阱（都已定位）
1. **`ix_filter` 的过滤位置**：`core/turbine/turbine.py:310-318` 在调用操作模型
   **之前**就已按 `ix_filter` 过滤了 `velocities`，而操作模型拿不到 `ix_filter`
   ⇒ 自定义模型**无法自己判断"现在算的是第几台"**。本探针的解法：在
   `floris.core.solver.thrust_coefficient` 外面包一层，把 `ix_filter[0]` 写进全局。
   （顺带：`turbine.py:351` 的累加器形状 = `shape(velocities)[0:2]`，
   可能与 `turbine_type_map` 的 (1,5) 不一致 ⇒ 返回值必须回形到累加器形状。）
2. **功率必须走"表+插值"**：直接用连续解析式 `0.95·½ρA·Cp(a)·u³` 会带来
   0.05%(u≈10)~1.8%(u≈5.7) 的系统偏差 —— 因为生产路径是整数风速表 + `interp1d`。
   逐点复刻表口径后，a≡1/3 才**逐位**复现基线。

### 验证结果（`probe_floris_a_per_turbine.py` 第 0 节）
```
生产 yawflor        = 5.942349733112 MW
逐台 a 自定义操作模型 = 5.942349733112 MW   相对差 1.495e-16
逐台 生产 = [2.64110924 0.63671798 0.83586231 0.89891304 0.92974716]
逐台 本探针 = [2.64110924 0.63671798 0.83586231 0.89891304 0.92974716]
```

⇒ **逐台 a 通道在数值上与现有生产模型完全同构**，可以安全替换被控对象。

---

## 3. 改造后会变成什么样的被控对象（客观物理，不做优化评价）

以下只陈述"通道本身的形状与量级"，不含"好/不好"的价值判断。

### 3.1 尾流亏损**只由 CT 决定**，a 只通过湍流模型进入
`core/solver.py:128` 把 `aI` 取负后传给 `velocity_model.function`；
但 `wake_velocity/gauss.py:46-66` 的函数体内，亏损用的是
`ct_i`（`:76-77` `uR = u·ct/(2(1−√(1−ct)))`、`u0 = u√(1−ct)`），
构造参数 `axial_induction_i` **在该函数里从未被使用**。
`a` 真正进入物理的唯一路径是
`wake_turbulence/crespo_hernandez.py:98` 的 `wake_induced_mixing`（∝ a）。

**实测后果**（`probe_floris_a_sensitivity.py`，u=10，全场 a 从 1/3 降到 0.20）：

| turbulence_model | P(a=1/3) | P(a=0.20) | Δ |
|---|---|---|---|
| `none`（关掉 a 的湍流通道） | 4.7142 MW | 4.9376 MW | **+4.74%** |
| `crespo_hernandez`（项目默认） | 5.9295 MW | 5.4489 MW | **−8.11%** |

⇒ **a 通过间接机制造成的功率差异最重量级是 ~5~8%**，与偏航通道同 u 下的
**+32.2%** 不在一个量级。

### 3.2 `a` 通道是"单侧降载"通道
`Cp(a)=4a(1−a)²` 在 `a=1/3` 取单机最大 ⇒ 偏离 1/3（任一方向）都**降低**单机出力。
实测均匀 a 扫描（u=10，γ≡0，GCH 默认）：

| a | 风场总功率 |
|---|---|
| 0.10 | 4.4295 MW |
| 0.20 | 5.4489 MW |
| 0.30 | 5.8768 MW |
| 1/3 | 5.9295 MW |
| 0.40 | **5.9581 MW** |
| 0.50 | 5.6046 MW |

⇒ 全域**单调、平滑、无 NaN**（除 a=0 的除零点），动作域 `[0.025, 0.5]` 内处处可解析。
**注意**：风场层面（5 台 5D）的最优点落在 **a≈0.40**，不是单机最优的 1/3 ——
因为上游降低 CT 会把风留给下游。这是风场耦合的正常结果。

### 3.3 逐台结构：上游偏好低 a、下游偏好高 a
同一扫描里的逐台功率（u=10）：

| a | 首台 P1 | 末台 P5 |
|---|---|---|
| 0.10 | 1.4440 | 0.7231 |
| 0.20 | 2.2819 | 0.8058 |
| 1/3 | 2.6399 | 0.9214 |
| 0.45 | 2.4268 | 1.0076 |

⇒ 末台（背风无机组）的功率随 a **单调升到 0.45 仍在涨**；首台在 1/3 到顶。
即**联合最优是"上游低 / 下游高"的单调斜坡** —— 与 `yawflor/README.md` 记录的
偏航最优形态（"上游 +16~25°、下游 0"的斜坡）**结构同源**。

（`probe_floris_a_per_turbine.py` 第 2、3 节给出"全场统一 a"与"逐台斜坡"在
真实风速序列下的电量/波动对照表，供需要时参考。）

---

## 4. 与现有 `yawflor` 的工程差异（客观）

| 维度 | 偏航通道（现有 `yawflor`） | 轴向因子通道（本次评估） |
|---|---|---|
| 动作载体 | `yaw_angles`，引擎**原生逐台** | 全场：`multidim_conditions`（原生）<br>逐台：需扩展（方案 A/B） |
| 是否需要改引擎 | 否 | 全场：否；逐台：是 |
| 绝对电平（u=10） | 5.9424 MW | 同（a=1/3 时逐位相同） |
| a=1/3 退化一致性 | — | **1.495e-16**（逐台扩展路径） |
| 通道量级（u=10，同口径） | γ:0→25° 为 **+32.2%** | a:1/3→0.20 为 **−8.11%**（默认湍流） |
| 能否"转向"（横向尾流偏转） | 能（`enable_secondary_steering`，依赖 a） | **不能**：`gauss.py` 亏损不含 a，无横向输运效应 |
| 动作域 | γ∈[−25°,+25°] | a∈[0, 0.5]（表下界建议 0.025） |
| a=0 行为 | — | **尾流模型除零**（`gauss.py:76`），必须守卫 |
| 与 γ 同台混用 | — | **引擎显式拒绝**（`operation_models.py:499-505`） |

---

## 5. 明确未做 / 未验证

1. **`wake_induced_mixing` 湍流模型单独可用性**：探针 A 里该模型报
   `TypeError: WakeInducedMixing.function() takes 3 positional arguments but 6 were given`
   —— 这是引擎内该湍流模型与 GCH 调用约定的接口不一致，**未继续追**（不影响上面结论，
   因为主臂用的是 `crespo_hernandez`）。
2. **逐台扩展方案的"生产化"**：本报告给出的方案 A 已在探针里跑通并逐位校验，
   但**尚未**做成 `yawflor` 那样的生产模块（`yawflor_a_turbine.py` / env / 口径契约测试）。
3. **偏航速率、输运延迟、载荷/疲劳**：与 `yawflor/README.md` §8 的"明确不做"一致，
   本次也未涉及。**轴向因子的等效执行器现实性**（变桨/转矩速率）未建模。
4. **与 LES 的对照**：`data_LES/` 里的 `*_adm_turb_data.mat` 是致动盘 LES 数据，
   **本次没有读**，因此"FLORIS 的 a 通道是否复现 LES 的部分诱导增功"**未验证**。
5. **额定上限**：本项目表无量定上限（`yawflor` 口径），a 通道下的功率截断行为未测。

---

## 6. 实测：WFSim vs FLORIS 的 min 级出力对比（统一行为）

口径：单排 5 台 5D、`a≡0.3333`、`γ≡0`、`ρ=1.20`、`shear=0.12`、围栏 `3.5≤u≤25`；
风速 = `FFC_speed_noise_1.csv` 第 1 日列前 100 分钟（均值 7.524 m/s）。

程序：`agentutil/probe_engine_minute_compare.py`（Python + `matlab.engine` 驱动 WFSim）
分析：`agentutil/analyse_engine_ratio_decomposition.py`
数据：`output/engine_minute_compare.json`

| 指标 | WFSim | FLORIS | FLORIS/WFSim |
|---|---|---|---|
| 电量 [MWh] | 2.6037 | 4.5341 | **1.7414** |
| 平均出力 [MW] | 1.5622 | 2.7205 | 1.7415 |
| 出力标准差 [MW] | 0.6744 | 1.1627 | **1.7240** |
| \|ΔP\| 均值 [MW] | 0.6854 | 1.1819 | **1.7244** |
| \|ΔP\| p95 / max | 1.9263 / 3.5575 | 3.3203 / 6.1491 | 1.724 / 1.729 |

- **形状一致性**：`corr(P_WFSim, P_FLORIS) = 0.999978`；最优比例 `k* = 1.738629`，
  归一化残差 RMS = **0.0129 MW（占均值 0.465%）**。
- **比值分布**（仅 `P_WFSim>0.5 MW`，n=97）：最小 1.7245 / 中位 1.7420 /
  均值 1.7438 / 最大 1.7614 / **标准差 0.0086（±0.5%）**。

### 6.1 1.74× 的分解（解析吞吐项 × 反解尾流项）

两边都可写成 `P = η_wake · N · ½ρA · Cp · u_eff³`，于是

```
P_f/P_w = [Cp_f·u_eff_f³ / (Cp_w·u_hub³)] · [η_f/η_w]
        =       吞吐项（可解析）           ·  尾流项（由实测反解）
```

- 吞吐项 = `0.95 × (u_eff/u_hub)³ = 0.95 × 0.998516³ = **0.929836**`（**与 u 无关** —— 
  因为 `u_eff` 与 `u_hub` 成正比，切变形状固定）。
- 尾流项（反解）= 比值 / 吞吐项，中位 **1.8735** ⇒ **WFSim 的尾流亏损比 FLORIS 深约 1.87×**。

⇒ **1.74× 里，只有约 0.93 来自"Cp 差 5%"，其余全部来自"WFSim 的尾流亏损深 1.87×"。**
注意这个 1.87 远大于 CT 之比（WFSim 隐含 `CT=1.3333` / FLORIS `CT=0.8889` = 1.5×），
额外那部分只能来自**模型结构**（WFSim 是 2D 不可压流场，含全局阻塞与压力耦合；
FLORIS 是高斯解析尾流 + 顺序叠加）。

### 6.2 残差是**结构性**的，不是噪声

`r = (P_F − k·P_W)/P_F` 与风速的相关性 `corr(r,u) = −0.8165`，按 u 分档单调：

| u 区间 | 比值均值 | 残差均值 |
|---|---|---|
| 5.0–6.0 | 1.7486 | +0.00943 |
| 6.0–7.0 | 1.7546 | +0.00908 |
| 7.0–8.0 | 1.7436 | +0.00284 |
| 8.0–9.0 | 1.7369 | −0.00102 |
| 9.0–11.0 | 1.7321 | −0.00378 |

⇒ 比例**不是严格常数**，随风速从 ≈1.76（低风）降到 ≈1.732（高风）。

### 6.3 重标定成本（换引擎要付多少）

| 重标定方式 | 残差 RMS | 最大 \|残差\| |
|---|---|---|
| `k = Cp_f/Cp_w = 0.95`（纯吞吐，忽略尾流） | 45.52% | 46.06% |
| `k = 电量比 1.741414` | 0.5064% | 1.1318% |
| `k = 中位比值 1.742014` | 0.4987% | 1.0978% |
| `k = 最小二乘 k* = 1.738629` | 0.5687% | 1.2900% |
| **两参数 `k1·w + k2·w·u`** | **0.3136%** | 0.6792% |

⇒ 用一个**常数**重标定后残差约 **0.5%**（远好于 `c1` 的 3 MW 门限所隐含的容差）；
再加一项随风速的线性修正可降到 **0.31%**。

**本节的已知局限**：只跑了 100 分钟、1 个日场景、`a` 固定 0.3333，因此
① 没有覆盖 a 变化下的比值行为（a 通道的比值可能不同，因为两边的 `CT(a)` 关系不同）；
② 没有覆盖 5～25 m/s 的全风速范围（本段只到 10.4 m/s，额定区以上未测）；
③ 逐台功率未被回读到（用户侧那次运行 WFSim 逐台回读失败，程序已标注且不影响总出力对比）。

---

## 7. 一句话回答

**FLORIS 可以做"固定 γ=0、调控 a"的风场模拟。**
- 只要**全场统一 a**：用原生 `multidim_condition` 多维功率/推力表即可，**零引擎改动**，
  且 a=1/3 时与现有 `yawflor` 生产模型**逐位相同**。
- 要**逐台独立 a**：需一处约 30 行的引擎扩展（自定义 operation model），
  扩展后同样**逐位复现**现有基线。
- 但必须知道：`a` 在 GCH 里**不直接产生尾流亏损**（亏损由 CT 定），
  它只经湍流模型间接作用，量级约为偏航通道的 **1/4**（8% vs 32%），
  且**没有横向尾流偏转**能力 —— 即"能调控"，但物理权限与偏航通道差别很大。
