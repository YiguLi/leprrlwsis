# AIFFLOR —— `td3per` 的 FLORIS 引擎版

> **一句话**：`aifflor` 基本上就是 `td3per` 把 **WFSim 风场仿真器换成 FLORIS**，
> 其余完全一致。

**基线与验收锚**：基线 = `td3per`（上层 TD3 + 下层 RL BESS，同动作空间）。
物理引擎 = FLORIS 4.6.6（**纯 Python，不启 MATLAB**），尾流参数**按 WFSim 标定**
（验证集相对误差 **66.9% → 1.1%**）。

---

## 0. 与 `td3per` 的差异清单（**只有引擎**）

| 项 | `td3per` | `aifflor` |
|---|---|---|
| 物理引擎 | WFSim（MATLAB，`WFSim_port`） | FLORIS（纯 Python） |
| env 类 | `env.WindBattery`（`physics=None` ⇒ MATLAB 路径） | `WindBatteryAIF`（`physics=AifflorPhysics`） |
| 尾流参数 | WFSim `meshing.m` 的 `forcescale` 等 | FLORIS + **标定到 WFSim 的参数** |
| **动作语义** | 轴向因子 `a∈[0,0.5]`（5 维）+ SOC 信号 | **完全相同** |
| **奖励 / 成本 / 对偶 / n 步 / 相位特征** | `td3per_myfunc` | **逐字复用（零改动）** |
| 算法本体（TD3+PER） | `td3per_ddpg` | **逐字复用（零改动）** |
| 输出目录 | `output/TD3PER-runNN` | **`output/AIFFLOR-runNN`**（`--env` 默认值已改） |
| TB scalar tag / CSV 列 | 与 `td3per` 逐字同名 | **逐字同名**（方法区分靠目录名 + CSV 名 `df_aifflor.csv`） |
| 训练预算 | 1000 episode = 1,440,000 步 | **相同**（未改，与 `yawflor` 不同） |

**为什么 `myfunc` 可以一处不改**（这是与 `yawflor` 最大的区别）：
`yawflor` 把动作带从轴向因子 `[0,0.5]` 换成无量纲偏航 `[−1,+1]`，所以它必须改
`CheckAction` / `PENALTY_NN_H_COEF`（400→100，保持"满带越界 = 200 元"）/
`violation_amount` 边界。`aifflor` 的**动作语义与 `td3per` 完全相同** ⇒ 带宽不变
⇒ 那些常量一个都不用动。

---

## 1. 两套仿真器的 I/O 映射（本环境的核心）

| 维度 | WFSim（`td3per` 原用） | FLORIS（`aifflor`） |
|---|---|---|
| **入口** | `WFSim_port(u, 0.0, a[5], 0.0)` | `eval_batch(m, u, a[5])`（向量化） |
| **动作 ∈ ℝ⁵** | **轴向因子 `a`**，`env.py` 夹到 `[0,0.5]` | **同一个 `a`**（逐台），夹带同上 |
| 动作 → 物理 | `CT_prime=4a(1−a)²`(=`Cp`)；力 `=1.5·½ρUe²·CT_prime/(1−a)` | 表 `CT=4a(1−a)`、`Cp_eff=0.95·4a(1−a)²` |
| **航向** | `Control.phi = 0`（恒，`env.py` 永远传 0） | `yaw_angles ≡ 0`（恒） |
| 端口维 | 无（2D 单排，`Cry=2D` 对全体相同） | 第五维（`y=0` 单排） |
| **第 6 维** | SOC 目标信号，**不进风机** | 同（`env.PowerOutput` 已截到前 5 维） |
| **输出** | 风场**总有功** [MW]（`out/1e6`） | 风场**总有功** [MW]（Σ 逐台） |
| **围栏** | `3.5 ≤ u ≤ 25` ⇒ 否则 0（`env.py`） | **同**（共用 `env.py` 的守卫） |
| 布局 / 密度 / 切变 | 单排 5 台 5D、D=100、ρ=1.20、shear=0.12 | **同** |

⇒ **动作语义零改动、输出语义零改动**，因此 `env.py` **一行都不用改**：
`PowerOutput(u0, act_wind)` 是唯一物理出口（`reset`/`TestReset`/`ScenarioReset`/
`step`/`TestStep`/`ScenarioStep` 共 6 处），`physics is not None` 时即分派给后端。

**动作→轴功率的换算**（`aifflor_turbine.py` 与 `yawflor_turbine.py` 同构）：
`P_turb = 0.95 · ½ρA · Cp(a) · u_eff³`，`Cp(a) = 4a(1−a)²`，以**整数风速网格表 +
`interp1d`** 求值（复刻 `SimpleTurbine.power` 口径）。

---

## 2. 引擎差异（声明，不试图抹平）

1. **尾流参数必须标定**：FLORIS 随包默认对 WFSim 的 MAPE = **66.9%**；用
   `agentutil/floris_calibrated_defaults.py` 的标定参数后误差降到下面的水平
   （`documentation/floris_calibration_to_wfsim.md`）。本包**默认施加标定参数**。

### ⚠ 精度必须分两个口径引用（2026 更正）

`calib_floris_to_wfsim.metrics()` 返回两个不同的量，**不能混用**：

* `mape` = **原始** MAPE（直接比较）⇒ **这才是实际误差，也是 RL 真正经历的偏差**
* `mape_k` = **用最优常数 `k*` 整体重标定后**的 MAPE，而
  `k* = ⟨PW,PF⟩/⟨PW,PW⟩` 是**由 WFSim 数据拟合出来的缩放因子**

实测（`agentutil/analyse_calibration_accuracy.py`）：

| 数据集 | n | **原始 MAPE** | 原始 MedAPE | `k*` | 重标后 MAPE | 比值 std |
|---|---|---|---|---|---|---|
| 标定 `cal` | 120 | **4.297%** | 3.828% | 1.0050 | 0.495% | 0.0520 |
| 验证 `val` | 60 | **4.447%** | 3.509% | 0.9890 | 1.103% | 0.0560 |
| **全域 350 点（3–20 m/s）** | 350 | **6.422%** | 4.858% | 1.0256 | 2.556% | 0.0785 |
| └ 其中 `u < 6 m/s` | 60 | **8.352%** | 6.363% | **1.0548** | 5.483% | 0.0791 |
| └ 其中 `u ≥ 6 m/s` | 290 | **6.022%** | 4.606% | 1.0256 | 2.556% | 0.0760 |

**⇒ 引用时一律写全口径**：

* **不缩放：全域 6.4%（中位 4.9%）/ 验证集 4.4%** —— 这是实际精度；
* **理想常数缩放后：2.6% / 1.1%** —— 只有在愿意接受一个全局标定系数时才成立。

**偏差是系统性的、非随机**：`k* > 1` 表示 FLORIS **系统性偏低**
（全域偏低 2.6%，低风速段偏低 **7.5%**）。`resid_k`（形状残差）为 4.2%~5.7%，
说明标定抓住了**形状**，残留的是一个**电平偏差**。

### `clip(u<0→0)` 的影响（方案 A 的定量依据）

`aifflor` 的功率路径**本来就** clip（`_power_of_u` 里 `u = np.clip(u, 0, None)`）。
全域 350 点上对比"clip vs 不 clip"：

| 量 | 结果 |
|---|---|
| 农场总功率最大改变 | **0.0136%** |
| 逐台最大改变 | 1.1483%（仅出现在该台本身出力≈0 时） |
| `u<0` 占全部转子位置 | 80 / 15750 = **0.51%** |
| `u<0` 的机组分布 | **T4 35% + T5 65%**（T1–T3 为 0） |
| `u<0` 的平面位置 | **100% 在转子中心那一格** |

⇒ **"`u<0` 只影响 T4/T5 的转子中心，最多使农场总功率改变 0.0136%"**，
这句话可直接写进论文。
（工具：`agentutil/probe_clip_gap_fullrange.py`、`agentutil/analyse_clip_impact.py`、
WFSim 侧脚本 `agentutil/run_clip_check_wfsim.m`。）
2. **适用域**：标定只针对 **5 台 5D 单排 + `a` 通道 + `γ≡0`** ⇒ `build_model` 会断言；
   换布局/机型/开偏航通道必须**重新标定**。
3. **绝对电平仍不同**：u=10、a=1/3 时 FLORIS **3.294 MW** vs WFSim **3.446 MW**
   ⇒ **跨引擎的 profit 不可直接比较**，只能同轨比较或归一化后比较。
   （2026 更正：上面 §2 的表格给出了完整口径，"1.1%"是**理想常数缩放后**的验证集数字。）
4. **准稳态**：每周期假定 `a` 已到位、尾流已平衡；不建模变桨/转矩执行器动力学与
   尾流输运延迟（`aifflor_turbine.declared_scope()`）。
5. **外部验证缺口**：`agentutil/probe_engine_newday_validate.py` 在 6 个随机日的
   6 个随机 100 分钟窗口上实测 **NaN/非有限值 0 条**；但抽样最低风速只到 6.05 m/s，
   **3.5–6 m/s 段尚未验证**。

---

## 3. 文件与"单一真源"

克隆的**唯一真源** = `agentutil/port_aifflor.py`（替换表 + `--check` 可逆性校验）：

| 文件 | 关系 |
|---|---|
| `aifflor/aifflor_ddpg.py` | **逐字复制** `td3per/td3per_ddpg.py`（零改动） |
| `aifflor/aifflor_myfunc.py` | **逐字复制** `td3per/td3per_myfunc.py`（**零改动**） |
| `aifflor/aifflor_main.py` | `td3per/td3per_main.py` + 替换表（env 类 / 不启 MATLAB / **输出目录名 `--env`** / 文案 / RUN_TAG） |
| `aifflor/aifflor_turbine.py` | **新写**：I/O 映射表 / 口径 / 模型构造 / 向量化求值（单一真源） |
| `aifflor/aifflor_env.py` | **新写**：`AifflorPhysics`（NaN 守卫 + 预取）+ `WindBatteryAIF` |

```bash
python agentutil/port_aifflor.py --apply    # 落盘
python agentutil/port_aifflor.py --check    # 校验磁盘 == 源 + 替换表（双向）
python agentutil/test_aifflor_contract.py   # 环境契约（只读，不需 MATLAB）
```

---

## 3.5 **必须知道的一条物理限制**：`a = 0` 是引擎死点（已修，但改变了动作语义）

这是本环境**唯一**"动作 ≠ 物理"的地方，也是踩过的坑，单独一节说清。

### 现象
FLORIS 高斯尾流的初始尺度（`wake_velocity/gauss.py:76`）：

```
uR = u · ct / (2 · (1 − √(1 − ct)))          ct = 4a(1−a)
```

`a = 0 ⇒ ct = 0` 时这是 **0/0**。实测（`agentutil/probe_aifflor_nan_map.py`）：

| 动作 | 裸引擎结果 |
|---|---|
| **恰好 `a = 0`**（单台或全场） | **100% NaN** |
| `a = 0.025` 及以上 | **0%** |
| 随机均匀采样 `a ~ U(0, 0.5)` | 0%（几乎不可能命中单点） |

### 为什么会咬人
`td3per_myfunc.CheckAction` 把动作夹到 **`[0, 0.5]`** ⇒ 策略只要把某台推到 `a ≤ 0`
（"想停机/全力降载"），就被夹成**恰好 0** ⇒ **正好命中死点**。

`aifflor` 训练实测（4000 步）：**评估点 5748，坏点 251 = 4.37%**，
且**逐条重试救回 0**（单点也是 NaN）⇒ 全部走"退回标称动作 `a = 1/3`"。
NaN 随后一路变成 `ptur` → 下层观测 → TD 误差 → PER 优先级，
最终崩在 `PER_buffer.py:181` 的 `assert priority > 0`
（`new_priorities = |TD|/mean + ε`，`ε = 0.01 > 0` ⇒ **只有 NaN 能触发**）。

> ⚠ 当时的第一版守卫放在**直算分支**里，而 `prefetch` 把 NaN 存进了缓存、
> **缓存命中路径绕过了守卫** ⇒ 修了等于没修。这是那次调试的真正教训。

### 修法（三层，`probe_engine_random_actions.py` / `aifflor_turbine.py` / `aifflor_env.py`）
1. **`CT` 按 FLORIS 原厂口径夹带**【根本修复】：
   `CT = clip(4a(1−a), 0.0001, 0.9999)`，与 `operation_models.py:134/:250` 完全一致。
   夹到 0.0001 后 `uR/u = 0.999975`、`u0/u = 0.99995`、`σ0 → 0.3536D`，
   即"几乎无尾流"的**正确**极限（解析式已验证：`uR/u ≡ 1−a` 在 CT ≥ 0.0001 逐位成立）。
   同时 `axial_induction` 也改为由**夹过之后的 CT** 反解 `½(1−√(1−CT))`，保持自洽。
2. **出口统一守卫 + 缓存不存坏值**：`AifflorPhysics.power()` 在**出口**检查有限性
   （缓存命中亦然，坏值即丢弃重算）；`AT.eval_batch` 做"整批 → 逐条重试 → 退回标称"
   三级兜底，并只把有限值写进缓存。
3. **引擎安全下界** `A_PHYS_LO = 0.025`（第二道防线，见下）。

### 代价（必须写进论文/对比声明）
`AifflorPhysics.a_clip` 把 `a ∈ [0, 0.025)` 抬到 `0.025`。这是**第二道防线**——
CT 夹带修好之后 `a = 0` 本身已经是安全的（实测风场出力 0.000000 MW，
解析正确：`Cp(a=0) = 0`），保留它是为了避开 CT→0 的数值敏感带。
区间占学习带 5%，`a = 0.025` 时风场出力仅 **1.65e-6 MW**（≈0）
⇒ 绝对影响可忽略，但**这是"策略动作被改写"而不是"引擎算错"**，必须声明。

> **学习带仍是 `[0, 0.5]`**，与 `td3per` 逐字相同
> ⇒ `CheckAction` / 越界罚 400 / `violation_amount` 一处都不用改。
> 若要彻底消除这道改写，可把学习带改成 `[0.025, 0.5]`（需同步改 `CheckAction`
> 与越界罚系数以维持"满带越界 = 200 元"）—— 本项目**没有**这么做。

### 为什么 FLORIS 自己"不接受 0"（根因说明）
不是物理发散，是**五种写法都没给 `CT = 0` 留分支**：
1. `wake_velocity/gauss.py:76` `uR = u·CT/(2(1−√(1−CT)))` 是 **0/0**（**可去奇点**：
   洛必达 ⇒ `uR → u`；同理 `u0 → u`、`σ0 → 0.3536D`）；
2. `wake_deflection/gauss.py:143` 挠曲模型同款 0/0；
3. `gauss.py:90-93` 的 `x0` 用 `1−√(1−ct)` 作分母，CT=0 时退化；
4. 近尾流 ramp `(x−xR)/(x0−xR)` 在 CT=0 时区间塌缩；
5. FLORIS 原厂**靠 `clip(ct, 0.0001, 0.9999)` 兜住了** ⇒ 表路径永远碰不到 CT=0。

**而 `aifflor` 为了做逐台 `a` 自己算 CT，第一版没有夹带 ⇒ 把原厂的安全网拆掉了。**
这是本项目引入的问题，不是 FLORIS 的缺陷。

### 相关工具
* `agentutil/probe_aifflor_nan_map.py` —— 裸引擎的 NaN 危险区地图（不需 MATLAB）* `agentutil/diag_aifflor_nan.py` —— 训练期 NaN 插桩定位器
  （首个非有限值即抛异常中止，并打印 q/target/TD/权重轨迹）
* `agentutil/test_aifflor_contract.py` §F —— 6 项 NaN 回归断言
  （含"缓存被投毒为 NaN 时出口仍有限"）

---

## 3.6 标定不可改善性：**参数层面已穷举，无两全解**

目标函数只用"农场总功率"这一个标量，所以它的实现方式不唯一。要判断是否存在
"功率对得上 **且** 不非物理（无 `u<0`、局地亏损 <0.9）"的替代解，我把**所有可加
自由度的位置**都扫了一遍：

| 轮次 | 落点 | 自由度 | 结果 |
|---|---|---|---|
| 1 | 单尾流**形状** | CT 倍率 `c_ct`（解耦功率/动量） | 抬 CT 补功率 ⇒ 尾流更深 ⇒ 更非物理 |
| 2 | 单尾流**宽度** | 增长率倍率 `k_mult` | 放宽后 MAPE 1.10% → 4.54% → 13.8% |
| 3 | **叠加规则** | 幂平均指数 `n`（`√(Σd²)` → `Σd`） | 降 `n` ⇒ 亏损涨到 **4.35**（更糟） |
| 4 | **叠加规则** | "持久性"权重 `q` | MAPE 跳到 **114%~207%** |

脚本：`probe_power_match_alternatives.py`（1,2）、`probe_custom_combination.py`（3）、
`probe_persistent_combination.py`（4）。**没有一轮找到两全解。**

### 一个被 trace 推翻的假设（登记以免重蹈）

第 4 轮我先假设"标定后的尾流亏损**随下游衰减**，所以要在叠加规则里加持久性"。
**trace 直接否证了它**（`_solver_hook` 里逐机组打印组合函数输入）：

```
T1: max(累积w)= 0.000  max(本台增量v)=6.931
T2: max(w)= 6.931  max(v)=6.858
T3: max(w)= 8.774  max(v)=6.853
T4: max(w)= 9.741  max(v)=6.849
```

**单尾流的增量亏损几乎恒定（6.93→6.85）——本来就不衰减**（因为 `d ∝ u_initial/σ`
里 `u_initial` 随剪切在增大，抵消了 σ 的增长）。标定后的参数已经把尾流调成"不衰减"了。
⇒ "加持久性"是**在已经满足的维度上再加约束**，所以只会破坏拟合。

另外第一版我在外面套了 `max(d_old, d_std, d_persist)`，实测是 **no-op**
（`d_persist < d_std` 恒成立 ⇒ 24 行输出完全相同）——这是"加了 `max` 就等于没加"的陷阱。

### 结论：缺的不是系数，是**域物理**

WFSim 是 `Ly = 4D` 的**封闭风洞**（Neumann 边界），而 FLORIS 的高斯尾流是开放域。
三轮参数扫描 + 两次叠加规则改造都无法弥合 ⇒ **在"FLORIS 高斯尾流 + 任何自洽叠加"
这个框架内无解**。可行方向只剩：
1. 接受并声明（影响限于 T5，占农场出力 1.7%）；
2. 改 WFSim 侧（`forcescale` / `Ly` / 边界条件）⇒ 基线失去原有出处，需重标定；
3. 换成**结构不同**的模型（如 FLORIDYN 这类动态尾流模型，见对话正文的评估）。

---

## 4. 运行与注意
```powershell
cd "C:\Users\SCP Site19\Desktop\llm enhanced wsis\leprrlwsis"
F:\anaconda\envs\wsis\python.exe aifflor\aifflor_main.py
```

* **必须用 `F:\anaconda\envs\wsis\python.exe`**（FLORIS 4.6.6 装在那里；
  3.12.x 之间的扩展模块不兼容，用 base 解释器会报 `pyexpat` DLL 错误）。
* **不启 MATLAB**（`aifflor_main` 已把 `import matlab` / `start_matlab()` 整段移除）。

### TensorBoard 标签：**scalar tag 刻意不改**，run 名改了

这是一处**故意的**设计，必须说清（否则会以为漏改）：

| TB 里的东西 | `td3per` | `aifflor` | 说明 |
|---|---|---|---|
| **scalar tag**（曲线名） | `r1` `c1` `lam1` `P_F` `test9/overall/r1` … | **完全相同**（95 个 tag 全同） | **刻意逐字同名** ⇒ 便于在 TB 里把两个方法的同名曲线叠在一起对齐比较 |
| **run 名**（左侧那一栏） | `TD3PER-runNN` | **`AIFFLOR-runNN`** | 方法身份**只**靠这里区分 |
| CSV 文件名 | `df_td3per.csv` | `df_aifflor.csv` | 第二重区分 |

原因：`td3per_main.py` 里**没有任何** tag 含 `TD3PER` 字样（已全量核对：95 个 tag 都是
`r1`/`c1`/`P_F`/`test9/*` 这类通用名）。所以"方法名"在 TB 里**只能**由 run 名承担 ——
这正是 `--env` 必须从 `TD3PER` 改成 `AIFFLOR` 的原因；若不改，aifflor 的 run 在 TB
左侧会显示成 `TD3PER-runN`，与真 `td3per` 无法分辨。

⚠ 因此：**在 TB 里比较 aifflor 与 td3per 时，同名曲线（如 `r1`）是可以直接叠加的**
（这正是 tag 不改的目的）；但**如果两个 run 的目录名相同就无法分辨**，所以别让
aifflor 用回 `--env TD3PER`。
* **输出目录 = `output/AIFFLOR-runNN`**（`--env` 默认值由 `TD3PER` 改为 `AIFFLOR`，
  与 `yawflor` 用 `YAWFLOR` 同一套命名惯例）⇒ 与 `td3per` 的 `output/TD3PER-runNN`
  **完全分开**。
  * 产物 CSV = `df_aifflor.csv`（`RUN_TAG='aifflor'`）。
  * TB 的 scalar tag 与 `td3per` **逐字同名**（这是刻意的，便于跨 run 对齐比较）；
    方法区分靠目录名 + CSV 名。
  * ⚠ **run 编号是全局递增的**（`util.get_output_folder` 扫 `output/` 下所有
    `-runN` 取最大 +1）⇒ 首个 aifflor run 会是 `AIFFLOR-run26` 之类的数，
    不是从 0 开始。这不是 bug，是既有约定。
  * ⚠ 下游分析脚本（`compare_*` / `test_five_days.py` / `analyse_lam1.py`）若按
    `output/TD3PER-*` 匹配，需要**显式传目录名**才能找到 aifflor 的 run。

### 已实测（2026）

| 项 | 结果 |
|---|---|
| 契约测试 `test_aifflor_contract.py` | **全部通过**（myfunc 逐位相同 / 动作语义 / 标定生效 / 不引入 matlab / 观测与守卫 / 预取确定性） |
| 端到端训练 60 步 | 跑通，**10.4 s**；预取 25 次（批量 25 / 单条 25 / **命中 48**），NaN 守卫 **0** 次 |
| 输出目录 | **`output/AIFFLOR-runNN`** ✅（产物 `df_aifflor.csv` + tfevents） |
| 外部验证（6 随机日 × 100 分钟，每步随机动作） | 非有限值 **0** 条 |
| `u=10, a=1/3` 单点 | **3.294116 MW**（标定后现值，作为回归锚） |
