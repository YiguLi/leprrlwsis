# HSRL — LLM 增强的风电场-储能联合优化调度

基于深度强化学习（DRL）的**风电场 + 电池储能（BESS）联合优化调度**研究代码。
上层策略学习风机控制（桨距 / 轴向诱导因子），下层策略或规则控制器调度储能功率，
物理环境由 MATLAB 实现的 **WFSim**（2D URANS  CFD 风电场模拟器）通过 MATLAB Engine API 提供。

> 本仓库为**私有仓库**，尚未选定开源许可证（见文末 §14）。

---

## 1. 问题设定

| 要素 | 配置 |
|---|---|
| 风电场 | 5 台风机（风洞布局，转子直径 100 m），由 WFSim 仿真 |
| 储能 | 容量 **6 MWh**、额定功率 **±10 MW**，充放电效率 η = 0.98 |
| 时间尺度 | Δt = **1 min**；一个 episode = **1440 步 = 24 h**（1 天） |
| 电价 | 峰谷平三档，硬编码于 `env.py` 的 `CalPrice()`：**300 / 580 / 980 元/MWh** |
| 波动约束 | 并网功率平滑度 **\|ΔP_G\| ≤ 3 MW** |
| 退化成本 | 放电 **490 元/MWh**（`BESS_DEG_RMB_PER_MWH`） |
| 初始 SOC | 3.0 MWh（训练时在 `[0.5, 5.5]` 随机） |

**优化目标**（三者相互冲突，是本题的核心张力）：

1. **套利收益** —— 低价时段充电、高价时段放电，最大化 `Σ P_G·pr·Δt`
2. **平抑波动** —— 将并网功率变化率压在 3 MW 以内，抑制风电出力波动对电网的冲击
3. **退化成本** —— 每次放电按 490 元/MWh 计费，抑制无意义的频繁吞吐

**分层控制结构**：

| 层 | 频率 | 动作 |
|---|---|---|
| **H（上层）** | 每 **5 步 = 1 块（5 min）** 决策一次 | 5 台风机的控制量（`[0, 0.5]`）+ 1 个 SOC 目标信号 `a6 ∈ [-0.5, 0.5]` |
| **L（下层）** | 每步决策 | BESS 功率 `pb`（MW，正 = 充电） |

- 在 RLRB 系列（规则下层）中，SOC 目标映射为 `soc_target = 3 + 5·a6 ∈ [0.5, 5.5]`，
  进而在 RBB 可行窗口 `[pbrulemin, pbrulemax]` 内由规则一步填平。
- **RBB（Rule-Based BESS）** 负责硬保证波动约束与物理可行性：当学习到的动作不可行时，
  由规则裁剪兜底，因此"平抑波动"不依赖 RL 学习。

---

## 2. 方法一览

`test_five_days.py` 中注册了 9 个可对比方法（`PROJECTS` 注册表，权威定义见该文件 L201–L331）：

| 简称 | 目录 | 上层算法 | 奖励函数 | 下层 | 主脚本 |
|---|---|---|---|---|---|
| **PER** | 根目录 + `PER/` | DDPG（`SP_DDPG`） | 原始 | RL（基线 5 维观测） | `PER_main0.py` |
| **PRRBB** | `prrbb/` | DDPG | **分段奖励** | RL（基线 5 维） | `prrbb/prrbb_main.py` |
| **PRLE** | `prle/` | DDPG（`PRLE_DDPG`） | 分段 + **LLM 增强** | RL（基线 5 维） | `prle/prle_main.py` |
| **TD3PER** | `td3per/` | **TD3**（`SP_TD3`） | **非分段**（移植 td3perrb 上层） | RL（蒸馏 5 维，追踪 **SOC 目标**） | `td3per/td3per_main.py` |
| **TD3PRRBB** | `td3prrbb/` | TD3 | 分段 | RL（蒸馏 5 维） | `td3prrbb/td3prrbb_main.py` |
| **TD3PRLE** | `td3prle/` | TD3（`SP_TD3_PRLE`） | 分段 + LLM | RL（蒸馏 5 维） | `td3prle/td3prle_main.py` |
| **TD3PRRLRB** | `td3prrlrb/` | TD3 | 分段 | **规则 `SocTracker`** | `td3prrlrb/td3prrlrb_main.py` |
| **TD3PERRB** | `td3perrb/` | TD3 | **非分段** | 规则 `SocTracker` | `td3perrb/td3perrb_main.py` |
| **TD3LEPRRLRB** | `td3leprrlrb/` | TD3（`SP_TD3_LEPRRLRB`） | 分段 + **LLM 在线阈值微调** | 规则 `SocTracker` | `td3leprrlrb/td3leprrlrb_main.py` |

另有未注册进对比脚本的早期基线：

| 名称 | 位置 | 说明 |
|---|---|---|
| H-DDPG | `H_main.py` + `H_ddpg.py` | 双层 DDPG（上层功率分配 + 下层储能控制） |
| S-DDPG | `S_main.py` + `S_ddpg.py` | PDO-DDPG（策略解耦优化）单层架构 |
| RBB | `rbb/` | 上层 DDPG 学习 + **下层规则** BESS 的首个版本；其 `rbb_rule_bess.RuleBasedBESS` 被 `per` / `td3per` 等方法复用 |

### 命名约定（据代码与各目录 README 归纳，非官方规范）

| 片段 | 含义 |
|---|---|
| `td3` / 无前缀 | TD3 / DDPG 算法 |
| `per` | Prioritized Experience Replay（优先经验回放） |
| `pr` | Piecewise Reward（分段奖励） |
| `rb` / `lrb` | Rule-Based BESS（规则下层；`l` = lower） |
| `le` | LLM-Enhanced |
| `bb` | 上下两层均由 RL 学习 |

---

## 3. 方法演进关系

```text
DDPG 系                          TD3 系（算法替换）
────────                         ──────────────────
PER ────────────────────────────→ TD3PER
  │
PRRBB ──(加 LLM)──→ PRLE          TD3PRRBB ──(加 LLM)──→ TD3PRLE
                                    │
                                    │  下层：TD3 学习 → 规则 SocTracker
                                    ↓
                                TD3PRRLRB ──┬──(奖励：分段 → 非分段)──→ TD3PERRB
                                            │
                                            └──(加 LLM 分段阈值在线微调)──→ TD3LEPRRLRB
```

关键继承关系（均已在各目录 README 中明确）：

- `td3prrlrb` ← `td3prrbb`：**上层保留 TD3，下层由 TD3 改为规则控制器** `SocTracker`
- `td3perrb` ← `td3prrlrb`：**唯一差异是奖励不分段**（收益权重与风险系数恒取 1.0）
- `td3per`（**反向改造**：`td3perrb` 的上层 + `td3per` 自己的 RL 下层）：
  上层 6 维动作第 6 维语义由"影子电价"改为 **SOC 目标信号**（与 td3perrb 逐项一致：
  非分段奖励、块末 gap 罚、相位观测、逐维探索噪声、`lam1∈[3,100]`、每块 2 次更新）；
  下层仍是 TD3+PER 的 RL BESS，但信号与追踪目标同为 SOC 目标
  （观测第 3 维 `soc_target/10`、`r2 = −100·clip_amount`、`c2 = 1000·|soc_target − SOC'|`）。
  ⚠️ 改造前的 `output/TD3PER-run*` 权重（4 维 H 观测 + 影子电价下层）与新代码不兼容。
- `td3leprrlrb` ← `td3prrlrb`：增加 **LLM 在线微调风速分段阈值**（±1 m/s 硬约束）+ 回放池 relabel
- `td3prle` ← `td3prrbb`：增加 LLM 元控制器；`prle` ← `prrbb` 同理
- 各 TD3 版本与同名 DDPG 版本的差异集中在：双 critic、延迟策略更新、目标平滑、PER

---

## 4. 目录结构

```text
HSRL/
├── env.py                    风电场-储能环境（Gym 风格），动态电价 CalPrice()
├── myfunc.py                 奖励计算 / 动作检查 / 归一化（根目录基线版）
├── model.py                  Actor-Critic 网络定义（多项目共享）
├── H_main.py / H_ddpg.py     H-DDPG（双层 DDPG）
├── S_main.py / S_ddpg.py     S-DDPG（PDO-DDPG）
├── PER_main0.py / PER_ddpg.py / PER_buffer.py    PER-DDPG 与优先回放
├── memory.py / memory_reward.py / random_process.py / util.py    共享组件
│
├── PER/                      优先经验回放组件（buffer / utils / notebook）
├── rbb/                      上层学习 + 下层规则 BESS（rbb_rule_bess 被 per/td3per 复用）
├── prrbb/                    DDPG + 分段奖励 + 学习型下层
├── prle/                     PRRBB + LLM 增强
├── pernewact/                仅剩 .pyc 编译缓存，源文件已丢失，未入库
├── td3per/                   TD3 + PER（上层 td3perrb 口径：非分段 + SOC 目标信号；下层 RL BESS 追 SOC 目标）
├── td3prrbb/                 TD3 + 分段奖励 + 学习型下层
├── td3prle/                  TD3 + 分段奖励 + LLM
├── td3prrlrb/                TD3 + 分段奖励 + 规则下层
├── td3perrb/                 TD3 + 非分段奖励 + 规则下层
├── td3leprrlrb/              TD3 + LLM 分段微调 + 分段奖励 + 规则下层
├── data_piecewise/           分段边界工具（kmeans / dbscan / boundaries）
│
├── WFSim.m / WFSim_port.m    MATLAB 仿真入口与 Python 调用封装
├── WFSim_addpaths.m          MATLAB 路径初始化
├── solve_PARK.m / myfun.m    PARK 尾流模型基线（fmincon 最优推力系数）
├── bin/                      WFSim 核心（core/ + analysis/）
├── libraries/                第三方 MATLAB 库（见 §13）
│
├── test_five_days.py         ★ 五天随机风速统一评测（兼容 9 个方法，多组模型）
├── test_compare_projects.py  多项目统一风力测试对比
├── test_scenario.py          场景化测试（给定风速/电价序列）
├── test_simple.py            单模型随机风速测试
├── test.py                   早期综合测试（⚠️ 会因缺失模块报错，见 §12）
├── compare_wind_speed.py     风速数据对比（平滑 vs 加噪）
├── compare_episode_reward.py 多项目 Episode Reward 对比（MAPE）
├── extract_training_wind_days.py   训练风速日提取到 TensorBoard
├── generate_day_sequence.py  训练日随机序列生成器
│
├── DEPENDENCIES.md           ★ 完整依赖清单（Python / MATLAB / 第三方库）
├── CHANGELOG.md              分段奖励函数版本记录
├── AGENTS.md                 协作与 git 备份协议
└── README.md                 本文件
```

---

## 5. 环境依赖

### Python（推荐 3.8）

```bash
conda create -n hsrl python=3.8
conda activate hsrl

pip install numpy pandas matplotlib
pip install torch                      # 或按 CUDA 版本安装
pip install tensorboard

# MATLAB Engine API for Python
cd "C:\Program Files\MATLAB\R20xxx\extern\engines\python"
python setup.py install
```

| 包 | 用途 |
|---|---|
| `numpy` | 数值计算、随机过程 |
| `pandas` | CSV 读写、风速/电价序列 |
| `torch` | Actor-Critic 网络与训练 |
| `matlab.engine` | 调用 WFSim 仿真器 |
| `tensorboard`（`torch.utils.tensorboard`） | 训练与评测日志 |

> 项目**没有** `requirements.txt` / `pyproject.toml`；完整依赖明细见 [`DEPENDENCIES.md`](DEPENDENCIES.md)。

### MATLAB

- **版本**：R2017b 或更高（需支持 MATLAB Engine API for Python）
- **必需工具箱**：Optimization Toolbox（`solve_PARK.m`）、Symbolic Math Toolbox（`expvec.m` 网格加密）
- **可选工具箱**：Image Processing Toolbox（仅 `export_fig` 使用）
- **首次运行**：在 MATLAB 中执行 `run('WFSim_addpaths.m')` 添加全部子目录到路径

---

## 6. 数据准备（⚠️ 新克隆必读）

`env.py` 第 41 行硬依赖风速序列文件：

```python
self.wind_noise = pd.read_csv("FFC_speed_noise_1.csv")
```

**但该文件（约 4.93 MB）被 `.gitignore` 排除，不在仓库中** —— 直接克隆后运行会报
`FileNotFoundError`。使用前需自备该文件并放在项目根目录。

| 文件 | 是否入库 | 说明 |
|---|---|---|
| `FFC_speed_noise_1.csv` | ❌ **未入库** | 训练/测试风速列池（195 列 ≈ 195 天），**必须自备** |
| `FFC_speed_1.csv` | ❌ 未入库 | 平滑版风速序列（4.93 MB） |
| `test_scenarios.csv` | ✅ 已入库 | 测试场景配置 |
| `fixed_day_sequence.csv` | ✅ 已入库 | 训练用固定随机日序列（`generate_day_sequence.py` 生成） |
| `compare_per_prle_episode_profit.csv` | ✅ 已入库 | 对比实验结果 |
| `FFC_Mast_Wind_Data/` | ❌ 未入库 | 原始测风数据（44 MB，2007–2010） |
| `data_LES/*.mat` | ❌ 未入库 | TU Delft LES 参考数据 |
| 电价数据 | — | **不需要外部文件**，`env.py` 内硬编码峰谷平三档 |

---

## 7. 快速开始

所有 `*_main.py` 与 `env.py` 均依赖 **当前工作目录 = 项目根**（`test_five_days.py` 会自动 `chdir`，
其余脚本不会），因此务必先 `cd` 到根目录：

```bash
cd HSRL
python td3prrlrb/td3prrlrb_main.py
```

---

## 8. 训练

```bash
cd HSRL

# DDPG 系
python H_main.py                  # H-DDPG（双层）
python S_main.py                  # S-DDPG
python PER_main0.py               # PER-DDPG
python rbb/rbb_main.py            # 纯规则基线
python prrbb/prrbb_main.py        # PER + 分段奖励
python prle/prle_main.py          # PRRBB + LLM（需配置 LLM，见 §10）

# TD3 系
python td3per/td3per_main.py
python td3prrbb/td3prrbb_main.py
python td3prle/td3prle_main.py    # 需配置 LLM
python td3prrlrb/td3prrlrb_main.py
python td3perrb/td3perrb_main.py
python td3leprrlrb/td3leprrlrb_main.py    # 需配置 LLM
```

默认 **1000 episode × 1440 步**。各方法可用 `--help` 查看参数；以 `td3prrlrb` 为例的常用项：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--warmup` | 200 | 纯随机填充块数 |
| `--discount` | 0.99 | 折扣因子（以 5 min 的"块"为单位） |
| `--policy_delay` | 2 | TD3 延迟策略更新频率 |
| `--phase_feat` | 1 | H 观测加入 24h 相位特征（4→6 维） |
| `--wt_bias_init` | 1 | 风机输出层 bias 初始化到 Cp 高效区 |
| `--epsilon` | 4000 | 探索 ε 从 1 衰减到 0 的**决策次数**（× 5 = 环境步数） |
| `--rbb_fluctuation_limit` | 3.0 | RBB 波动限值（MW） |
| `--rbb_rated_power` | 10.0 | RBB 额定功率（MW） |

---

## 9. 评测与测试

### `test_five_days.py` —— 主评测脚本

用固定种子从训练风速池抽 5 天，**只前向不更新网络**，对比多方法多组模型：

```bash
cd HSRL

# 单方法
python test_five_days.py --td3prrlrb output/TD3PRRLRB-run164

# 每方法多组模型（依次跑 run1/run2/run3）
python test_five_days.py \
    --td3per     output/TD3PER-run137 output/TD3PER-run139 \
    --td3prrlrb  output/TD3PRRLRB-run164 output/TD3PRRLRB-run166 \
    --td3perrb   output/TD3PERRB-run170

# 只测部分方法 / 冒烟测试
python test_five_days.py --projects td3prrlrb td3perrb --episode-length 120

python test_five_days.py --list-projects        # 列出全部可测方法
```

输出（CSV 与 TensorBoard **双份**）：

| 文件 | 内容 |
|---|---|
| `five_day_days.csv` | 每天 × 每方法 × 每组模型（三项指标） |
| `five_day_summary.csv` | 每方法 × 每组模型的 5 天均值/标准差 |
| `five_day_method_summary.csv` | **★ 主表**：每方法跨「全部模型 × 全部天」的 mean ± std |
| `five_day_steps.csv` | 每步明细（`--no-save-steps` 可关） |

三项核心指标：`viol_count`（波动越限次数）、`viol_excess_mw`（越限总量 MW）、`profit_rmb`（利润，元）。

### 其它测试脚本

```bash
python test_compare_projects.py    # 多项目统一风力对比
python test_scenario.py            # 给定风速/电价序列的场景测试
python test_simple.py              # 单模型随机风速测试
python compare_wind_speed.py       # 风速数据对比
python compare_episode_reward.py   # Episode Reward 对比（MAPE）
```

---

## 10. LLM 配置

`prle`、`td3prle`、`td3leprrlrb` 三个方法需要调用 LLM 做元控制，配置由各自目录下的
`llm_config.json` 提供：

```jsonc
// <method>/llm_config.json
{
    "provider": "deepseek",     // deepseek / anthropic / openai
    "api_key": "sk-...",        // 留空则回退到环境变量
    "model": "deepseek-v4-flash"
}
```

`load_llm_config()` 的解析顺序为：**config 文件的 `api_key` → 对应环境变量**
（`deepseek` → `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`，`anthropic` → `ANTHROPIC_API_KEY`，
`openai` → `OPENAI_API_KEY`）。也可用 `--llm_config <path>` 指定其他配置文件。

若不想让密钥落盘，把 `api_key` 留空改用环境变量即可：

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
```

> 这三个 `llm_config.json` 当前**已入库**（私有仓库）。若将来转为公开仓库或增加协作者，
> 建议改为「`api_key` 留空 + 环境变量」，或将该文件移出版本控制。

---

## 11. 输出与日志

| 路径 | 内容 | 版本控制 |
|---|---|---|
| `output/<METHOD>-runN/` | 每步明细、每 episode 汇总、9-case 测试、更新统计等 CSV | ❌ 已忽略（本地可达数 GB） |
| `output/test_five_days/` | 五天评测的 CSV 与 TensorBoard | ❌ 已忽略 |
| `runs/` | TensorBoard 训练日志（`events.out.tfevents.*`） | ⚠️ 当前已入库（3 个事件文件） |

训练产生的记录文件以 `td3prrlrb` 为例：

| 文件 | 内容 |
|---|---|
| `df_td3prrlrb.csv` | 每步全量：`r1/c1_step/gap_raw/gap_pen/soc_signal/soc_target/soc_delta`、RBB 窗口量、`clip_amount`、`d_cap_c/d_cap_d`、收益分解、`lam1` |
| `df_episode_td3prrlrb.csv` | 每 episode：`episode_reward/cost/gap_pen/profit`、充放电退化量 |
| `df_test9.csv` | 每 480 步的 9-case 测试 |
| `update_stats_H.csv` | H 层每块 2 批更新的 TD/Q/target/batch 统计 |

---

## 12. 已知问题

1. **风速数据缺失**：`FFC_speed_noise_1.csv` 未入库，新克隆无法直接运行（见 §6）。
2. **缺失源文件的编译缓存**：`sac*.py`、`ppo.py`、`ae.py`、`rule.py`、`model2/3.py`、`O_*`、`W_*`
   等源文件已丢失，仅存 `.pyc`。`test.py` 会因导入这些模块抛 `ModuleNotFoundError`。
3. **旧版 Python 缓存**：`bin/__pycache__/` 部分文件由 Python 3.6/3.7 编译，项目曾迁移版本。
4. **硬编码绝对路径**：`prrbb/sample_wind_percentiles.py` 中写有 `c:\Users\yigul\...`，
   迁移到其他机器需修改。
5. **MATLAB 引擎多实例**：`myfunc.py` 与多个 `*_main.py` 各自 `matlab.engine.start_matlab()`，
   可能启动多个 MATLAB 实例，注意内存占用。
6. **`from pickle import FALSE`**：项目多处使用该非常规导入，运行时会触发 `ImportWarning`。
7. **必须 `cd` 到项目根运行**：除 `test_five_days.py` 外，脚本均不做 `chdir`，
   相对路径（`FFC_speed_noise_1.csv`、`test_scenarios.csv`）依赖 CWD。

---

## 13. 文档索引与第三方组件

### 项目文档

| 文档 | 内容 |
|---|---|
| [`DEPENDENCIES.md`](DEPENDENCIES.md) | 完整依赖清单：Python 包、MATLAB 组件、WFSim 模块、工具箱 |
| [`CHANGELOG.md`](CHANGELOG.md) | 分段奖励函数两版差异记录（`SEG_COEFFS` 的 HIGH 风速行） |
| [`AGENTS.md`](AGENTS.md) | 协作约定与 git 备份/回退协议 |
| `td3prrlrb/README.md` | 规则下层架构、奖励设计、R1–R4 四轮标定记录 |
| `td3perrb/README.md` | 非分段奖励版与分段版的逐项对照 |
| `td3leprrlrb/README.md` | LLM 分段阈值在线微调的机制与验证 |

### 第三方组件（版权归原作者，使用前请确认许可）

| 组件 | 来源 | 用途 |
|---|---|---|
| **WFSim** | TU Delft（Boersma & Doekemeijer, 2017）<br><https://github.com/Bartdoekemeijer/WFSim> | 2D URANS 风电场模拟器（本项目已定制修改） |
| **export_fig** | Oliver Woodford / Yair Altman | 高质量图形导出（需 Ghostscript） |
| **mexnc** / **snctools** | NetCDF MEX / 高层工具箱 | NetCDF 读写 |
| **sparse_null** | Pawel Kowal (2006) | 稀疏矩阵零空间计算（连续性方程投影） |
| **natsort** | Stephen Cobeldick | 自然顺序字符串排序 |
| **numSubplots** | Rob Campbell (2010) | 最优子图布局 |
| PALM / Jensen-PARK 模型 | 外部参考模型 | WFSim 精度验证与 DRL 对比基线 |

---

## 14. 许可与致谢

本项目**当前未选定开源许可证**。在补充 `LICENSE` 前请勿对外分发。

特别注意：仓库内含第三方代码（`libraries/`、`bin/`、WFSim 定制版，见 §13），
若将来转为公开仓库，需先逐一确认这些组件的许可条款并可保留其署名。
