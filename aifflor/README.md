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
   `agentutil/floris_calibrated_defaults.py` 的标定参数后 **1.1%**
   （`documentation/floris_calibration_to_wfsim.md`）。本包**默认施加标定参数**。
2. **适用域**：标定只针对 **5 台 5D 单排 + `a` 通道 + `γ≡0`** ⇒ `build_model` 会断言；
   换布局/机型/开偏航通道必须**重新标定**。
3. **绝对电平仍不同**：u=10、a=1/3 时 FLORIS **3.294 MW** vs WFSim **3.446 MW**
   ⇒ **跨引擎的 profit 不可直接比较**，只能同轨比较或归一化后比较。
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
