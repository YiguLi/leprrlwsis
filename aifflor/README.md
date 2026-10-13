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
| 尾流参数 | WFSim `meshing.m` 的 `forcescale` 等 | **FLORIS 随包默认**（`calibrated=False`）＋ **出口电平系数** `1/1.6696` |
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

1. **物理用 FLORIS 随包默认，精度靠一个出口电平系数**（**当前默认配置**，见 §3.7.2）：

   | 口径 | 全域 350 点原始 MAPE | `k*` | 重标后 MAPE |
   |---|---|---|---|
   | FLORIS 随包默认（无系数） | 68.476% | 1.6696 | 66.962% |
   | **随包默认 × 电平系数（当前默认）** | **7.189%** | **1.0000** | **0.001%** |
   | 标定到 WFSim 的 6 参数（可选） | 6.422% | 1.0256 | 2.556% |

   ⇒ 两种配置精度相当（7.19% vs 6.42%），但**电平系数方案的 `k*=1.0000`**：
   那个 67% 的电平偏差被**完全**吸收，剩下的 7.19% 是**纯形状**差异。
   `agentutil/floris_calibrated_defaults.py` 的 6 参数标定仍保留可用
   （`calibrated=True`），但**不再是默认**。

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
| `aifflor/aifflor_hsrl.py` | **新写**：`--hsrl` **方法分支**的全部口径（常量 + 纯函数，单一真源；见 §3.8） |

```bash
python agentutil/port_aifflor.py --apply    # 落盘
python agentutil/port_aifflor.py --check    # 校验磁盘 == 源 + 替换表（双向）
python agentutil/test_aifflor_contract.py   # 环境契约（只读，不需 MATLAB）
python agentutil/test_aifflor_hsrl_contract.py   # --hsrl 分支契约（纯算术 + 桩环境，见 §3.8）
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

## 3.7 RL 超参能否跨引擎平移？（`nstep` / `batchsize` / `rmsize` / update 次数）

结论：**能平移，而且理由比"优化复杂度相同"更直接、更强。**
（工具：`agentutil/analyse_hyperparam_transfer.py`）

### 直接证据：两条功率轨迹几乎完全相关

100 步（1 分钟）真实风速序列、标称动作下：

| | mean | std | **std/mean** | **acf(1)** | acf(5) |
|---|---|---|---|---|---|
| WFSim | 1.5622 | 0.6744 | **0.4317** | **+0.0436** | +0.1636 |
| FLORIS 标定后 | 1.5098 | 0.6447 | **0.4270** | **+0.0444** | +0.1646 |
| FLORIS 库存（对照） | 2.7205 | 1.1627 | 0.4274 | +0.0439 | +0.1645 |

```
corr(WFSim, FLORIS 标定后) = 0.999959        ← 轨迹几乎一致
比值 (标定后/WFSim): 中位 0.9683, std 0.0077  ← 近似"常数倍率"
```

**机制**：两个引擎吃**同一组风-价序列**，而功率几乎成比例。所以引擎差异等效于
**在同一个信号上乘一个近似常数的倍率** —— 不是两个不同的任务。

### 四个超参各自的判据（三条全部满足）

| 超参 | 依赖的量 | WFSim | FLORIS 标定 | 判定 |
|---|---|---|---|---|
| `nstep` | 奖励**相关长度** | acf1 0.0436 | acf1 0.0444 | ✅ 差 <1% |
| `batchsize` | 信噪比（std/mean） | 0.4317 | 0.4270 | ✅ 差 <1% |
| `rmsize` | **状态占据**范围 | 同一风-价场景决定 | 同 | ✅ 共享 |
| update 次数 | 相关长度 × 信噪比 | — | — | ✅ 由上面两项决定 |

**⚠ 但"奖励尺度相关"的量会有 ~4% 的偏移**（不是超参，是归一化常数）：

* `obs = [pr, u, E, P_G_prev]`，`Normalize` 用**固定常数除法**
  （`pr/1000, u/10, E/10, P_G_prev/1000`）⇒ **第 4 维 `P_G_prev/1000` 跨引擎差 ~3.7%**。
  它的动态范围是 0–0.03，3.7% 的相对偏移 ≪ 动态范围，不会改变网络输入分布的量级。
* 若要与 WFSim 精确对齐：把观测按 FLORIS 口径归一化，或训练时把 reward/cost
  统一乘 `1/1.037`（这正是早先实测的 `COST_SCALE` 等效值 1.816→0.972 的同一件事）。

### ⚠ 两个**不能**平移的东西（别混淆）

1. **`P_F` 平滑阈值（`P_F_LIMIT = 3 MW`）、`W_GAP = 10 MW`、`L_CLIP_CAP = 20 MW`
   等绝对 MW 门限** —— 这些由**动作**主导而非引擎（`clip_amount` 主要来自命令与窗口之差），
   但严格说仍有 ~4% 的引擎电平成分。若要严格一致，需按引擎的电平比同步缩放。
2. **profit 的绝对值** —— 3.7% 的电平差 + 7% 的**状态相关**分量
   （见下）⇒ **跨引擎只可同轨/归一化比较**。

### 状态相关分量：两个引擎在哪儿分歧最大（重要）

把全域 350 点的比值 `r = P_F/P_W` 做方差分解：

```
总方差 0.00616 = 档内散布 0.00542 (88%) + 档间漂移 0.00074 (12%)
corr(r, min(a))  = −0.658      ← 与**动作强度**强相关
corr(r, mean(a)) = −0.546
corr(r, u)       = −0.104      ← 与风速几乎无关
```

⇒ **两引擎的分歧由"控制强度/异质性"驱动，不是由风速驱动。**
在强降载、动作差异大的区域分歧最大（`r` 可到 1.10–1.11），在温和动作区最小。
这对 RL 有实际含义：**策略越激进，用它学到的结论往 WFSim 迁移时的偏差越大。**

---

### 3.7.1 用 FLORIS **默认参数**是否更好？—— 两件事必须分开

用户直觉："默认参数肯定比自定义参数不容易出错"。这个直觉要把两件事拆开：

**① 「改了 FLORIS 吗」——没有。**（已取证）

```
floris 包最新修改时间 = 2026-09-30 05:06:11   （只有 __pycache__ 被重新编译）
aifflor/ 创建时间     = 2026-10-01 03:13:21
=> floris 包在 aifflor 之后被修改过吗: False
=> 疑似我方新增的模块: 无（157 个文件，无 custom_superposition / probe / aifflor）
```

标定是**纯数据配置**，经 FLORIS **公开 API**（`apply_calibrated_wake_params` / `set_param`）
施加，来源单一且可复现（`agentutil/floris_calibrated_defaults.py`）。
⇒ **stock 与标定在"代码出错风险"上没有区别**；标定只是"多了一组 6 个参数"。

**② 可平移性（`nstep`/`batchsize`/`rmsize`/update）——与参数选择无关。**

| 100 步真实风速序列 | `std/mean` | `acf(1)` | `corr` vs WFSim | 比值 std |
|---|---|---|---|---|
| WFSim | 0.4317 | +0.0436 | 1.0 | — |
| **FLORIS stock** | 0.4274 | +0.0439 | **0.999980** | **0.0103** |
| FLORIS 标定 | 0.4270 | +0.0444 | 0.999959 | 0.0077 |

⇒ 两者**同样成立**，stock 的比值甚至更紧。
**因为可平移性只取决于"引擎差异是否近似常数倍率"，而两者都是。**

**③ 但训练可用性差别巨大（这才是关键）。**

| 全域 350 点 | 原始 MAPE | `k*` | 重标后 MAPE | 比值 std |
|---|---|---|---|---|
| **FLORIS stock** | **68.476%** | 1.6696 | 66.962% | 0.1453 |
| FLORIS 标定 | **6.422%** | 1.0256 | 2.556% | 0.0785 |

stock 的 FLORIS 比 WFSim **低 67%** ⇒ 所有**绝对 MW 口径的常数**都要跟着改：

| 常数 | 现值 | **stock 下应为** | 标定下应为 |
|---|---|---|---|
| `COST_SCALE_TD3` | 100 | 59.89 | 97.51 |
| `VALUE_SCALE` | 1000 | 598.94 | 975.07 |
| `OBS_POWER_SCALE` | 10 | 5.99 | 9.75 |
| `W_GAP` | 10 | 5.99 | 9.75 |
| `P_F_LIMIT` | 3 | 1.80 | 2.93 |

⇒ **用 stock 参数反而"更容易出错"**：不改这些常数，`W_GAP = 10 MW` 在 stock 下
相当于 WFSim 口径的 **16.7 MW**，罚项/门限的相对位置全错；改了就是**十来个常数的手工调参**，
比"一组标定参数"更容易出错、更难追溯。

**结论**：stock 在"可审计性"上更干净，但它把 67% 的电平偏差转嫁成
**十来个归一化常数的重调 + 一个 74% 的 profit 尺度错位**；
标定只引入**一处、有单一真源、可复现**的数据配置。**就本任务而言标定更不易出错。**

### 3.7.2 实际采用的配置：**stock 物理 + 单一电平系数**

按"只用电平系数"的决定，`aifflor` 的**默认物理配置**是：

```python
AifflorPhysics(nt=5, calibrated=False, power_scale=AT.LEVEL_COEF)
#   calibrated=False           ⇒ FLORIS **随包默认**尾流参数（不改任何参数，可审计）
#   AT.LEVEL_COEF = 1/1.6696   ⇒ 出口电平系数 ≈ 0.5989459
```

`LEVEL_COEF` 的由来（`aifflor/aifflor_turbine.py`）：`1/k*`，其中
`k* = ⟨P_W,P_F⟩/⟨P_W,P_W⟩` 由**全域 350 点**（3–20 m/s、0.5 m/s 粒度、每档 10 个
随机行为）拟合 ⇒ `k* = 1.6696`。

#### 四种口径的完整对照（`agentutil/analyse_level_coef_modes.py`）

**① 全域 350 点（3–20 m/s，随机动作）**

| 口径 | 原始 MAPE | MedAPE | `k*` | 重标后 MAPE | 比值范围 |
|---|---|---|---|---|---|
| stock（原厂默认，无系数） | 68.476% | 67.548% | 1.6696 | 66.962% | 1.311–2.149 |
| **stock × 电平系数** | **7.189%** | 6.478% | **1.0000** | **0.001%** | 0.785–1.287 |
| 标定（6 参数，无系数） | **6.422%** | 4.858% | 1.0256 | 2.556% | 0.898–1.388 |
| 标定 × 电平系数 | 37.873% | 38.586% | 0.6143 | 38.574% | 0.538–0.831 |

> **「stock × 系数」的 `k* = 1.0000`、重标后 MAPE = 0.001%** —— 意思是
> 那个电平偏差被**完全**吸收，FLORIS 与 WFSim 之间只剩**纯形状**差异（7.189%）。
> 这是"单一系数"能达到的最好结果。

**② 100 步真实风速序列上的波动（标称动作，`P_F_LIMIT = 3 MW`）**

| 口径 | mean | std | **std/mean** | \|ΔP\| 均值 | 限幅后均值 | vs WFSim |
|---|---|---|---|---|---|---|
| **WFSim（基准）** | 1.5622 | 0.6744 | **0.4317** | 0.6854 | **0.0056** | — |
| stock（无系数） | 2.7204 | 1.1627 | 0.4274 | 1.1818 | 0.0737 | +1209.54% |
| **stock × 电平系数** | **1.6294** | **0.6964** | 0.4274 | **0.7079** | **0.0069** | **+22.50%** |
| 标定（无系数） | 1.5098 | 0.6447 | 0.4270 | 0.6548 | 0.0040 | −28.11% |
| 标定 × 电平系数 | 0.9043 | 0.3861 | 0.4270 | 0.3922 | 0.0000 | −100.00% |

⇒ **均值差 4.3%、std 差 3.3%、`|ΔP|` 均值差 3.3%**；
`std/mean = 0.4274` vs WFSim 的 `0.4317`（差 **0.99%**）。
**相对波动是尺度不变量，本来就几乎相同** —— 这正是"超参可平移"（§3.7）的根据。

**③ 锚点（`u=10, a=1/3`）**

| 口径 | 值 [MW] | 相对 WFSim（3.4462） |
|---|---|---|
| stock | 5.942350 | +72.43% |
| **stock × 电平系数** | **3.559146** | **+3.28%** |
| 标定 | 3.294116 | −4.41% |
| 标定 × 电平系数 | 1.972997 | −42.75% |

⇒ 新默认的锚点是 **3.559146 MW**（旧默认 3.294116 只在 `power_scale=1.0` 时成立）。

#### 适用域声明（必须写进论文）

`LEVEL_COEF` 在**全域平均**意义下最优，但逐点比值范围 **0.785–1.287**，且分歧
**由动作强度驱动**（`corr(r, min(a)) = −0.658`，与风速几乎无关）
⇒ **强降载 / 动作异质区的偏差最大**。样本内（领域随机策略）均值意义上可用；
**外推到极端降载策略时需重新核对**。

---

## 3.8 `--hsrl` 方法分支：**"块总放电量"语义**（2026-10 新增）

`aifflor` 现在有**两条方法分支**，靠命令行开关区分（**不加开关 = 旧口径，逐位不变**）：

```powershell
python aifflor\aifflor_main.py            # 旧口径: SOC 目标信号 + 规则残余动作 + 平抑窗口
python aifflor\aifflor_main.py --hsrl     # HSRL:  块总放电量 + 绝对 pb + 指数式追踪成本
```

* **`aifflor/aifflor_hsrl.py` 是 HSRL 口径的唯一真源**（常量 + 纯函数 + warmup 摊还执行器 +
  20 步测试函数）。共享模块 `hsrl_soctarget.py` / `hsrl_warmup.py` / `aifflor_myfunc.py`
  **一行都没改** ⇒ 另外 7 个项目不受影响。
* `aifflor_main.py` 里 `train_hsrl()` 与 `train()` **并列实现**（不共用循环）：
  未加 `--hsrl` 时 `train_hsrl` 与 `aifflor_hsrl` 的函数**一行都不会被调用**，
  因此旧路径可以断言"逐位不变"（见本文末的回归证据）。

### 口径对照表

| 项 | 旧口径（`train()`） | `--hsrl`（`train_hsrl()`） |
|---|---|---|
| **H 第 6 维含义** | SOC 目标信号 `a6∈[−0.5,0.5]` → `soc_target=0.25+0.41667·a6` | **块总放电量 `g` [MW·min]，放电为正** |
| H 第 6 维边界 | 人为信号带 `[−0.5, 0.5]` | **可行域** `[−C_max, +D_max]`（SOC + 额定功率，**不含波动**） |
| H 第 6 维越界罚 | 400 元/单位（不封顶） | 400 元/MW·min（不封顶） |
| H 第 6 维探索噪声 | `sig_noise_max 0.2 → min 0.01`（信号量纲） | `--hsrl_sig_noise_max 0.5 → 0.025` [MW·min]（= 满量程 2.5 的 20% → 1%） |
| **H r1** | `pr·P_G·dt − 490·max(−pbrule,0)·dt − pen` | `pr·P_G·dt − pen5 − pen6`（**去掉 bessdeg**） |
| H 块末 gap 罚 | `10·max(0,\|soc_target−SOC_末\|−tol)` | **删除**（执行偏差由 L 的 c_L 承接） |
| H c1 | 波动当量 + `W_GAP·unfeasible_gap` | 同公式，但 `unfeasible_gap ≡ 0`（平抑窗口已关） |
| **L 动作** | 残余 `pb = rule_action(基线) + Δ` | **绝对 `pb`**（正 = 充电，与 env 同号） |
| L 动作裁剪 | `CheckAction` ±0.5 + RBB 平抑窗口 | **只按 SOC + 额定功率**（RBB 以 `fluctuation_limit=inf` 构造） |
| **L r2** | `−1·min(clip_amount, 20)`（含波动裁剪那一段） | `−200·越界[MW] − 20·放电[MW·min]` |
| **L c_L（图上公式）** | `+1·trackerr`（到"Soctrack 范围 ∩ 平抑窗口"的距离） | `1 − exp(−\|(g−Σ已放电)/(δ−t_L) − a_L\|)` |
| L 观测 | `[pg(t−1)/10, ptur/10, SOC/0.5, target/0.5, n_rem/5]` | `[u/10, pr/1000, SOC/0.5, P_G(t−1)/5, g/2.5]` |
| λ2 | `[0.5, 2]`，初值 1（MW↔MW 兑换率） | `[2, 50]`，初值 10（c_L 无量纲 vs r2 元量级） |
| `d_block_L` | 1.2（地板 0.15） | 0.5（地板 0.05） |
| warmup 下层 | 规则追 `soc_target`（`Δ≡0` 基线） | **摊还执行器** `pb = clip(−(g−Σ)/n_rem, 物理域)` |
| 训练中测试 | 9-case（45 步，`df_test9.csv`） | **20 步协议**（见下，`df_test20_hsrl.csv`） |
| 产物 CSV | `df_aifflor.csv` / `df_episode_aifflor.csv` | `df_hsrl.csv` / `df_episode_hsrl.csv` / `df_test20_hsrl.csv` |

公式出处：`c_L` 就是 `未命名/PER_main0.py:116-118` 的原式
（`c2 = -np.exp(-abs((signal - sum(pb_l[:-1]))/(5-tt) - pb)) + 1`；那里 `signal` = H 第六维、
`tt = step%5`、`pb_l` = 本块已执行的实际功率）——`agentutil/test_aifflor_hsrl_contract.py`
第 C 节把它逐位对照。

### 三个必须写清的推导 / 坑

**① 块级可行域：单位是 MW·min，且 SOC 约束**不乘 δ**

```
D_max = min(P_rated·δ , (E − E_min)·η_d·60)          [MW·min]  放电上限（g 的正侧）
C_max = min(P_rated·δ , (E_max − E)·60/η_c)          [MW·min]  充电上限（g 的负侧）
g ∈ [−C_max, +D_max]
```

两条约束都**不含波动**（用户口径）：

1. **额定功率**：每步 `|pb| ≤ P_rated`，块内 δ 步 ⇒ 总量 ≤ `P_rated·δ`；
2. **SOC 能量**：放电量 `q` [MW·min] 让电池能量下降 `q/η_d` [MW·min]
   （因 `ΔE = P_D·dt/η_d`，且 `1 MWh = 60 MW·min`）⇒ 整块 ≤ `(E−E_min)·60`
   ⇒ `Σq ≤ (E−E_min)·η_d·60`（充电同理，效率在分母）。

⚠ **陷阱**：写成 `(E−E_min)·η_d/dt` 只是**单步**功率上限（= 58.8·E MW），
整块要换算成**能量**（×60，**不是** ×δ）。本项目容量 0.5 MWh、额定 0.5 MW，
两者在中段都由额定功率决定（±2.5 MW·min 相同），**只有在 SOC 贴边时才分叉**
（如 `E=0.01 MWh`：正确 0.588 vs 错写 2.94，差 5 倍）。契约测试第 B 节用
"把 D_max 恰好花完 ⇒ SOC 恰好收在下界"这一物理判据把它钉住。

**② `Σ` 与 `a_L` 都用"实际施加值"**：`a_L = −pb_applied`（`pb_applied` 是 env 回传的
`P_C − P_D`），`Σ` 是同一口径的块内累计（**不含当前步**）。这与原式一致（那份代码是把
env 回传的 `action[-1]` 事后 append 进 `pb_l` 的）。

**③ "零成本动作 = 摊还计划"**：`g` 与 `Σ` 都在命令之前确定 ⇒ 若 L 命令
`pb = −(g−Σ)/n_rem`（且不被物理域裁掉），则 `a_L = (g−Σ)/n_rem` ⇒ `c_L ≡ 0`（精确 0）。
warmup 期的下层规则就是这个"摊还执行器"，启动时用 400 组随机 `(E,g,Σ,n_rem)` 断言
（`hsrl.check_amortized_baseline`，被物理域裁剪时 `c_L>0` 只可能来自"物理上做不到"）。

### 已知局限（**刻意保留**的原方法形态）

1. **L 观测不含 `Σ` / `n_rem`**（用户口径：回到"H 观测 + 上层目标"这 5 维）⇒ `c_L` 的
   参照量 `(g−Σ)/n_rem` 对策略**不可观测**：单步转移对 `c_L` 非 Markov，PER 按单步采样
   会带偏 critic。契约测试第 G 节把这条局限写成可执行记录（同一观测对应不同块内进度）。
   若要修，最小改动是追加两维 → `[u/10, pr/1000, SOC/0.5, P_G/5, g/2.5, Σ/2.5, n_rem/5]`
   （需同步 `agent_L` 的 `nb_states=7`）。
2. **H 对"块内波动"没有直接控制**：`g` 只约束块**总量**，块内怎么分配由 L 决定，而 L 的
   r2/c_L 都只看沉淀量与计划偏差 ⇒ 平抑压力全部落在 H 的 c1（事后计费）上。
   在容量 0.5 MWh / 额定 0.5 MW 的口径下，BESS 对 3 MW 限幅的平抑能力本就有限
   （见 §2 的波动表），但这仍是语义上一个应当自知的口径。
3. **`d_block_L = 0.5` 会让 λ2 容易贴顶**：实测 1000 步烟测里 `EMA_c_L ≈ 1.6 > 0.5`
   ⇒ λ2 很快升到上限 50（此时跟踪项对 L 的权重 = 50 倍）。若希望 λ2 在中段工作，
   把预算放宽（`--hsrl_d_block_l 2.0` 之类）或等 L 学会兑现计划后由 curriculum 自动
   减半（减半条件是"连续 50 块 EMA ≤ 预算"，贴顶时不会触发）。这是**调参旋钮**，不是 bug。

### 训练中的测试：20 步协议（替换 9-case）

* 电价**恒 300**（低谷）；风速 = 从 `FFC_speed_noise_1.csv` 的"天"列里**随机取一列 + 随机起点**
  截 20 步（= 4 个块）；开局 **SOC = 0.25 MWh**（注意 `env.ScenarioReset` 里硬编码的
  `E = 5.0` 会被显式覆写 —— 旧 9-case 其实一直是在 `E=5 > max_E` 的病态起点上跑的）。
* **触发频率不变**：`step > warmup` 且 `episode_steps % 480 == 0`（与旧 9-case 同一处）。
* 采样用**独立** `RandomState`（默认每次触发重新随机，`--hsrl_test_seed >0` 可固定复现）
  ⇒ 不吃训练流、不改变训练轨迹。
* 指标：逐块 + `OVERALL` 两档写 `df_test20_hsrl.csv`；TB 走 `test20/overall/*` 与
  `test20/block{i}/*`（旧 9-case 的 `test9/*` 在 `--hsrl` 下不再产生）。

### 回归证据（本次改动）

| 项 | 结果 |
|---|---|
| **旧口径逐位不变** | `--max_steps 300 --seed 1`（无 `--hsrl`）改前/改后 `df_aifflor.csv` **SHA256 完全相同**（`0091C06A…F8A2`） |
| `agentutil/test_aifflor_hsrl_contract.py` | **全部通过**（A–J 共 60+ 条断言：可行域⇔RBB 逐位、c_L⇔原式逐位、符号/裁剪/罚、H 奖励差 = bessdeg、观测、λ2、摊还自检、20 步协议手算逐位） |
| `--hsrl --warmup 300 --max_steps 1000 --seed 1` 端到端 | 跑通；摊还自检 400 例全过；20 步测试在 step 480/960 各触发一次；FLORIS 坏点 **0**；无 NaN（除 `ema_c1_H`/`ema_cL_L` 在 EMA 建立前的预期 NaN） |
| 烟测自洽量 | `pbrule ≡ pb_applied`、`clip_amount ≡ viol_L`（逐位）；`g` 在 warmup 期按价格游走饱和到 ∓2.5、策略期由策略自行决定；策略期 `corr(g_raw, 块Σr1) = +0.163`（**奖励梯度方向正确**：放电为正 ⇒ 收益为正） |
| `agentutil/test_aifflor_contract.py` | **1 项既有失败**（与本改动无关）：`build_L_obs_distilled 逐位相同` —— `td3per_myfunc` 的版本在 `nolow` 提交里多了 `soc_signal=None` 形参，`aifflor_myfunc` 的副本没跟上（`git log` 可查：该漂移早于本次改动）；aifflor 的调用方式不受影响 |

---

## 4. 运行与注意

```powershell
cd "C:\Users\SCP Site19\Desktop\llm enhanced wsis\leprrlwsis"
F:\anaconda\envs\wsis\python.exe aifflor\aifflor_main.py            # 旧口径
F:\anaconda\envs\wsis\python.exe aifflor\aifflor_main.py --hsrl      # HSRL 分支（见 §3.8）
```

* **解释器必须是装了 FLORIS 的那个环境**（原先唯一可用的路径是
  `F:\anaconda\envs\wsis\python.exe`；3.12.x 之间的扩展模块 ABI 不兼容，
  用 base 解释器会报 `pyexpat` DLL 错误）。2026-09-30 起
  `agentutil/floris_import_guard.py` 给了换机器的退路：
  ① `FLORIS_PYTHON=<解释器>` 显式指定；② 该路径在本机**不存在**时自动退回
  **功能判据**（`import pyexpat` 成功即放行，真正要挡的是 ABI 不匹配）。
  ⇒ 在**没有 `F:` 盘**的机器上直接用本机环境即可，例如
  `C:\Users\yigul\anaconda3\envs\llmenhancedwsis\python.exe`
  （FLORIS 4.6.6 + torch 2.12.0+cpu）。路径**存在但不等**时仍硬失败（判据未被削弱）。
* **不启 MATLAB**（`aifflor_main` 已把 `import matlab` / `start_matlab()` 整段移除）。
* ⚠ **把 stdout 重定向到文件/管道时**要 `set PYTHONIOENCODING=utf-8`：`hc.summary()`
  里的 `−`(U+2212) 在 GBK 代码页下会 `UnicodeEncodeError`（直接跑在终端里不会）。

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
| 契约测试 `test_aifflor_contract.py` | **全部通过**（myfunc 逐位相同 / 动作语义 / 标定生效 / 不引入 matlab / 观测与守卫 / 预取确定性）—— ⚠ 2026-10 起 `build_L_obs_distilled 逐位相同` 一条为**既有失败**（`nolow` 提交给 td3per 版加过 `soc_signal` 形参，见 §3.8 末表），与本分支无关 |
| 契约测试 `test_aifflor_hsrl_contract.py`（`--hsrl`） | **全部通过**（可行域⇔RBB 逐位 / c_L⇔原式逐位 / 符号与罚 / H 奖励差 / 观测 / λ2 / 摊还自检 / 20 步协议手算） |
| `--hsrl` 端到端 1000 步 | 跑通（FLORIS 坏点 0；20 步测试在 step 480/960 触发；产物 `df_hsrl.csv` + `df_test20_hsrl.csv` + `update_stats_H.csv`） |
| 旧口径改前/改后同 seed 300 步 | `df_aifflor.csv` **SHA256 完全相同**（`0091C06A…F8A2`）⇒ `--hsrl` 分支对旧路径零影响 |
| 端到端训练 60 步 | 跑通，**10.4 s**；预取 25 次（批量 25 / 单条 25 / **命中 48**），NaN 守卫 **0** 次 |
| 输出目录 | **`output/AIFFLOR-runNN`** ✅（产物 `df_aifflor.csv` + tfevents；`--hsrl` 时为 `df_hsrl.csv` 等，见 §3.8） |
| 外部验证（6 随机日 × 100 分钟，每步随机动作） | 非有限值 **0** 条 |
| `u=10, a=1/3` 单点 | **3.294116 MW**（标定后现值，作为回归锚） |
