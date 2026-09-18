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
  非分段奖励、块末 gap 罚、相位观测、逐维探索噪声、`lam1∈[0.967,4.833]`、每块 2 次更新）；
  下层仍是 TD3+PER 的 RL BESS，但信号与追踪目标同为 SOC 目标
  （观测第 3 维 `soc_target/10`、`r2 = −1·clip_amount`、
  `c2 = +1·trackerr [MW]`（到**参考集合** = Soctrack范围 ∩ 平抑窗口 的距离）：
  `F ≠ ∅` 时零成本集合是该区间；`F = ∅` 时**不存在范围, 只有纯粹一个竭力值 `r*`**
  （窗口里离 Soctrack 范围最近的端点）, 顶到它才不罚；
  块首够不着的目标跳变由 r1 块末 gap 罚承接）。
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
├── pernewact/                仅剩 .pyc 编译缓存（源文件已丢失）
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
├── test_scenario.py          场景化测试（给定风速/电价序列；被 7 个 *_myfunc 的 nine_case_test import）
├── test_simple.py            单模型随机风速测试
├── test.py                   早期综合测试（⚠️ 会因缺失模块报错，见 §12）
├── compare_wind_speed.py     风速数据对比（平滑 vs 加噪）
├── compare_episode_reward.py 多项目 Episode Reward 对比（MAPE）
├── extract_training_wind_days.py   训练风速日提取到 TensorBoard
├── generate_day_sequence.py  训练日随机序列生成器
│
├── agentutil/                ★ agent 自建的**项目无关**工具与回归测试（约定目录，见下）
│   ├── README.md                     目录约定 + 脚本清单 + bootstrap 模板
│   ├── test_c1_uniform.py            H 层 c1 口径不变量
│   ├── test_critic_bias.py           critic 输出层 bias（价值基线）初始化
│   ├── test_warmup_schedule.py       warmup 调度 + bias 触发点
│   ├── test_value_scale.py           H 层 --value_scale 归一化
│   ├── test_l_lower_unit.py          RL-BESS 下层口径 ÷100
│   ├── audit_coverage.py             九方法覆盖审计 + H 层 η1 分析
│   ├── analyse_q_scale.py            Q 尺度瞬态 / 相对 TD 误差
│   ├── ab_critic_bias.py             同 seed 短程 A/B 驱动
│   ├── probe_action_symmetry.py      ∂Q/∂a 动作对称性探针
│   └── probe_value_scale.py          归一化两臂学习曲线诊断
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

## 6. 数据文件

`env.py` 第 41 行硬依赖风速序列文件：

```python
self.wind_noise = pd.read_csv("FFC_speed_noise_1.csv")
```

这些数据文件**均已随仓库入库**（本仓库的上传策略是：除 `output/` 与压缩包外，
所有文件都进版本库），克隆后即可直接运行。

| 文件 | 是否入库 | 说明 |
|---|---|---|
| `FFC_speed_noise_1.csv` | ✅ 已入库（4.93 MB） | 训练/测试风速列池（195 列 ≈ 195 天） |
| `FFC_speed_1.csv` | ✅ 已入库（4.93 MB） | 平滑版风速序列 |
| `test_scenarios.csv` | ✅ 已入库 | 测试场景配置 |
| `fixed_day_sequence.csv` | ✅ 已入库 | 训练用固定随机日序列（`generate_day_sequence.py` 生成） |
| `compare_per_prle_episode_profit.csv` | ✅ 已入库 | 对比实验结果 |
| `FFC_Mast_Wind_Data/` | ✅ 已入库（44 MB，1018 文件） | 原始测风数据（2007–2010） |
| `data_LES/*.mat` | ✅ 已入库（49 MB） | TU Delft LES 参考数据（2 涡轮 / 6 涡轮场景） |
| 电价数据 | — | **不需要外部文件**，`env.py` 内硬编码峰谷平三档 |

> 唯一未入库的是 `output/`（训练产物，本地 7 GB+）与压缩包
> （`HSRL.zip` 137 MB、`td3prrlrb.zip`；`documentation/documentation.7z` 为例外，保留入库）。

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
| `--warmup` | 200 | warmup 块数（**已不再纯随机**，见文末附注） |
| `--warmup_sched` | `fixed` | warmup 上层动作调度：`fixed` = 0.33 起每块 ±0.05、clip[0.20,0.45]、信号维恒 0；`random` = 旧 `random_action()`（对照臂） |
| `--warmup_bess` | `rule` | BESS-RL 项目（per/prrbb/prle/td3per/td3prrbb/td3prle）warmup 下层行为：`rule` = RBB 窗口 + SOC 目标跟踪（确定性，无随机）；`random` = 旧 `agent_L.random_action()`（对照臂） |
| `--warmup_soc_target` | 3.0 | warmup 期规则下层跟踪的 SOC 目标（MWh） |
| `--discount` | 0.99 | 折扣因子（以 5 min 的"块"为单位） |
| `--policy_delay` | 2 | TD3 延迟策略更新频率 |
| `--phase_feat` | 1 | H 观测加入 24h 相位特征（4→6 维） |
| `--wt_bias_init` | 1 | 风机输出层 bias 初始化到 Cp 高效区 |
| `--critic_bias_init` | `estimate` | critic 输出层 bias（价值基线）初始化：`estimate` = 用 warmup 块均值估计 `r̄/(1−γ)`；`none` = 对照臂 |
| `--critic_bias_p/_c` | −1 | 显式覆盖利润/成本基线（>0 生效，**须保持 `--critic_bias_init estimate`**） |
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

### `agentutil/` —— agent 自建的项目无关工具与回归测试

约定：**所有项目无关的临时/自建文件一律放 `agentutil/`**（详见 `AGENTS.md` 与
`agentutil/README.md`）。目录内脚本自带路径 bootstrap，可从任意 CWD 调用：

```bash
# 纯 Python 不变量测试（不需要 MATLAB）
python agentutil/test_c1_uniform.py        # H 层 c1 口径统一
python agentutil/test_critic_bias.py       # critic bias 初始化
python agentutil/test_warmup_schedule.py   # warmup 调度 + bias 触发点
python agentutil/test_value_scale.py       # H 层 --value_scale 归一化
python agentutil/test_l_lower_unit.py      # RL-BESS 下层口径 ÷100 / SOC 目标接线
python agentutil/test_soctarget_uniform.py # 下层 SOC 目标三项目逐位等价
python agentutil/audit_coverage.py         # 九方法覆盖审计 (归一化/bias/η1)

# 分析 / 诊断 / A-B 驱动
python agentutil/analyse_q_scale.py                       # 扫 output/*/update_stats_H.csv
python agentutil/ab_critic_bias.py --project td3perrb --dry-run
python agentutil/probe_action_symmetry.py output/<run1> [run2 ...]
python agentutil/probe_value_scale.py
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
| `output/<METHOD>-runN/` | 每步明细、每 episode 汇总、9-case 测试、更新统计等 CSV | ❌ 已排除（本地 7 GB+） |
| `output/test_five_days/` | 五天评测的 CSV 与 TensorBoard | ❌ 已排除 |
| `runs/` | TensorBoard 训练日志（`events.out.tfevents.*`） | ✅ 已入库 |

> 本仓库的排除范围只有两类：`output/` 与压缩包。其余文件（含编译缓存、
> 测风数据、LES 数据、文档 PDF、`未命名/` 副本）全部入库。

训练产生的记录文件以 `td3prrlrb` 为例：

| 文件 | 内容 |
|---|---|
| `df_td3prrlrb.csv` | 每步全量：`r1/c1_step/gap_raw/gap_pen/soc_signal/soc_target/soc_delta`、RBB 窗口量、`clip_amount`、`d_cap_c/d_cap_d`、收益分解、`lam1` |
| `df_episode_td3prrlrb.csv` | 每 episode：`episode_reward/cost/gap_pen/profit`、充放电退化量 |
| `df_test9.csv` | 每 480 步的 9-case 测试 |
| `update_stats_H.csv` | H 层每块 2 批更新的 TD/Q/target/batch 统计 |

---

## 12. 已知问题

1. **仓库体积约 130 MB**：测风数据（44 MB）、LES 参考数据（49 MB）、`未命名/` 副本（21 MB）
   与两个 `documentation.7z` 均随仓库入库；`output/`（本地 7 GB+）与其余压缩包已排除。
2. **缺失源文件的编译缓存**：`sac*.py`、`ppo.py`、`ae.py`、`rule.py`、`model2/3.py`、`O_*`、`W_*`
   等源文件已丢失，仅存 `.pyc`。`test.py` 会因导入这些模块抛 `ModuleNotFoundError`。
3. **旧版 Python 缓存**：`bin/__pycache__/` 部分文件由 Python 3.6/3.7 编译，项目曾迁移版本。
4. **硬编码绝对路径**：`prrbb/sample_wind_percentiles.py` 中写有 `c:\Users\yigul\...`，
   迁移到其他机器需修改。
5. **MATLAB 引擎多实例**：`myfunc.py` 与多个 `*_main.py` 各自 `matlab.engine.start_matlab()`，
   可能启动多个 MATLAB 实例，注意内存占用。
6. **`from pickle import FALSE`**：项目多处使用该非常规导入，运行时会触发 `ImportWarning`。
7. **必须 `cd` 到项目根运行**：除 `test_five_days.py` 与 `agentutil/` 下的脚本外，
   其余脚本均不做 `chdir`，相对路径（`FFC_speed_noise_1.csv`、`test_scenarios.csv`）依赖 CWD。
   `agentutil/` 下的脚本自带 bootstrap（`chdir` 到仓库根 + 把仓库根与 `agentutil/`
   都放进 `sys.path`），因此可以用任意 CWD、任意路径调用。
8. **新文件放哪里**：项目无关的临时/自建脚本（探针、分析、回归测试）一律放
   `agentutil/`，不要散落在根目录或各方法目录；生产模块（`hsrl_*.py`、`myfunc.py`、
   `*_main.py` 等，被 9 个 main 直接 import）必须留在根目录。详见 `AGENTS.md` 与
   `agentutil/README.md`。

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

## 附：H 层约束口径统一（`--c1-mode`，9 个方法共用单一真源 `hsrl_cost.py`）

此前各项目的 `c1`/`λ`/`d_block` 是各自标定的历史遗留，**互不可比**（gap 系数 50 vs 10、
`d_block_H` 1.2 / 12 / 50、λ1 区间 20/3、50/3、100/3、500/50 —— 同一物理事件被定了不同的价）。
现统一为**波动当量**口径，常量集中在根模块 `hsrl_cost.py`：

| 项 | 统一取值 | 含义 / 理由 |
|---|---|---|
| `c1` | `P_F + K_V·(P_F−3)⁺` [MW·step]，`K_V = 20` | 平滑项斜率 1（λ_min 才能真正承载平抑偏好）；越限额外 20 倍（由"单次典型事件 0.85 MW 吃掉整块额度"反推）；`unfeasible_gap` 不再单列——它与越限是**同一事件**（实测触发率逐位相同 0.056%） |
| `d_block_H` | **15**（= 3 MW × 5 步），地板 = 初值 | 语义 = 硬约束允许的最大块消耗（"贴着 3 MW 上限跑满整块"）；curriculum 因此失效（有意为之：对事件型约束收紧预算只会与硬约束冲突） |
| `λ` | `λ_min = 0.10·pr_ref·dt ≈ 0.967`，`λ_max = 5λ_min ≈ 4.83`，初值 = λ_min | λ_min = **平抑支付意愿**（边际收益的 10%）；越限有效价 = `21·λ_min ≈ 20.3 元/(MW·step)` = 边际收益的 2.1×，λ 升至上限时 ≈ 10× |
| 对偶更新 | 乘性 `λ ← λ·exp(η·(EMA/d − 1))`，η = 0.02 | 对尺度免疫，避免旧加性式在 c 换量纲后顶格（λ2 顶到 500 即此类事故） |
| 块末 SOC 追踪罚 | **移出 c1 → 进 r1**（`sum_r1 -= gap_pen`），固定价 10 元/MWh | 偏好用固定价、约束才配对偶；避免 λ 同时定价追踪。5 个项目原系数均为 10，已核对一致 |

* `--c1-mode uniform`（默认）走统一口径；`--c1-mode legacy` 保留各项目原公式作为**对照臂**；启动时打印当前口径与有效价格。
* **不变量测试**：`python agentutil/test_c1_uniform.py` —— 断言 8 个项目的 `Cal_Rew_*` 在 uniform 下返回**逐位相同**的 c1（`P_F∈{0,0.5,2.9,3,3.5,5,8}` 偏差 0），legacy 分支与旧公式逐项相符，且 `Σ₅ fluct_cost(3.0) = 15 = d_block`（硬约束当量自洽）。
* 有效价参照（pr = 580）：边际收益 `pr·dt = 9.67 元/(MW·step)`；平滑 0.967（10%）；越限 20.3（2.1×）。
* 各 `*_myfunc.py` 的文档字符串描述的仍是 **legacy** 公式（`--c1-mode legacy` 可达）；uniform 下 c1 由 `hsrl_cost.c1_apply()` 替换。

---

## 附：H 层值函数回归空间归一化（`--value_scale`，默认 `s = 1000`）

**问题**：`model.py` 的 `Critic` 输出层 `fc3(50→1)` 只把 weight 初始化到 ±3e-3，bias 保持
PyTorch 默认 ±1/√50（实测初值 `Q ≈ 0.0804`），而 H 层块收益使回归目标达到 **O(10⁴)**
（块收益均值 ~2.6e2 元 × `1/(1−γ)`，γ=0.99）——**初值与目标差 ~3e5 倍**。

**做法**：让 H 层 critic 在**归一化空间**回归。`r1`/`c1` 只在**唯一一处**换算：各 agent
`update_policy()` 采样之后 `profit_batch / s`、`cost_batch / s`；其余一切保持原始口径。

| 对象 | 是否缩放 | 说明 |
|---|---|---|
| H 层 replay memory（warmup 填入 / 策略期写入 / `relabel_all()` 重写） | **不缩** | 三条路径都存原始口径 ⇒ 天然同口径，不存在混装；换算只发生在回归时 |
| `critic_p/p2/c/c2` 四张网及其 target | **缩** | 共用同一个 `target_q_batch_{p,c}`，只有一处除法 |
| actor loss `−(Q_p − λ·Q_c)` | 自动 | 两通道同除 s ⇒ 正因子，Adam 下等价，**λ 语义不变** |
| `hsrl_agent_init.BiasInitTracker` | **缩** | 它读的是 main 里的**原始**块收益累加器 ⇒ 写入 `fc3.bias` 前 `/s`（含 `--critic_bias_p/_c` 覆盖值，按原始口径给出）。估计量对样本线性，故 SE/块数/触发点/γ 都不用重调 |
| λ 区间、EMA 对偶（`ema_cost` vs `d_block_H`）、curriculum | **不缩** | 乘性对偶对 r1/c1 的换算一无所知 |
| `update_stats_H.csv` / TensorBoard（`q_*` / `td_*` / `target_*`） | 还原 | agent 内部 `×s` 还原成原始口径 ⇒ 与历史 run、`agentutil/analyse_q_scale.py` 直接可比 |
| PER 优先级 `|TD_p|/mean_p + |TD_c|/mean_c + ε` | 不缩 | 自归一化比值，对 s 免疫（`ε` 加在 O(1) 量上，占比不变） |
| L 层 `r2`/`c2` | **不缩** | `value_scale` 默认 1.0；L 值函数尺度本就 O(1)（`d_block_L` = 20 / 2.5） |

**实测（`agentutil/test_value_scale.py` B 段，真实 `SP_TD3` + 同一批数据，合成常数回归 R=5000）**

* **语义等价（守住）**：`reward/cost/target` 四个纯收益字段两臂**还原后逐位相等**；
  `q_*`/`min_p_next` 类字段两臂**精确差 s 倍**（证明还原生效）；两臂最终收敛到同一目标
  （误差 2.1% vs 0.00%）。
* **速度并未变快（预测被否证）**：原以为"初值↔目标距离 ∝ 1/s ⇒ 瞬态 ∝ 1/s"，
  但 Adam 下 s=1 臂会把**隐层末级激活从 0.8 放大到 ~200**（实测 `|h5|` mean 202.6）
  来把输出顶到 O(10⁴)，因此在 ~50 次更新内两臂都已贴近目标。
  **归一化买到的是"内部尺度/条件数"（s=1000 臂用 O(1) 激活拟合同一目标），
  不是步数**。真实收益需在 MATLAB 环境跑 A/B 才能定量。
* warmup 路径：`fill_bess_transition` 原样透传 `r2/c2`（存储侧无缩放）；
  `BiasInitTracker` 用原始块样本估计、写 `fc3.bias` 前 `/s`（`s=1` 时与旧行为逐位一致）。

**用法与测试**

* `--value_scale 1000`（默认）；`--value_scale 1`（或 0）关闭归一化，用作对照臂。
* `python agentutil/test_value_scale.py` —— 覆盖性（8 个 agent + 9 个 main 的换算点齐全，且
  `sum_r1`/`sum_c1`/对偶/curriculum/relabel/warmup 填入**均未被缩放**）+ 上述数值不变量
  + `Σ₅ fluct_cost(3) = 15` 与 λ 语义正交。
* `python agentutil/probe_value_scale.py` —— 学习曲线/隐层激活量级的诊断（复现上面"速度未变快"
  的观测）。

---

## 附：RL-BESS 下层数值口径统一 ÷100（6 个 RL 下层的项目）

**动机**：下层的设计目的是 `r2 → 0`（规则动作不被 RBB 裁剪）与 `c2 → 0`（追踪到位），
而原系数把两个通道都放在 10²~10³ 量级（`r2 = 10·shadow_profit − 100·clip_amount`、
`c2 = 1000·soc_deviation`），使 L 层值函数目标远离 0、且与网络初值差 2~3 个数量级。
所以 L 层**不做 H 层的 `--value_scale` 归一化**，而是直接把**数值单位**降两个数量级
（`agent_L.value_scale` 保持 1.0）：

| 项 | per / td3per | prrbb / prle / td3prrbb | td3prle |
|---|---|---|---|
| `r2` 的 clip 系数 | 100 → **1** | 100 → **1** | `min(100·(clip+over), 1000)` → **`min(1·(clip+over), 10)`** |
| `r2` 的 shadow 系数 | 10 → **0.1**（td3per 无此项） | 10 → **0.1**（td3prrbb/prle 改为 SOC 目标后亦无此项） | 无 |
| `c2` 系数 | 1000 → **10** | 700 → **7** | `10·(SOC−target)²` → **`0.1·(SOC−target)²`** |
| `d_block_L` / 地板 | 20 / 2.5 → **0.2 / 0.025** | 14 / 1.75 → **0.14 / 0.0175** | 40 / 40 → **0.4 / 0.4** |
| `agent_L.cost_limit` | 10 → **0.1** | 10 → **0.1** | 10 → **0.1** |
| L 加性对偶步长 η2 | 0.01 → **1.0** | 0.01 → **1.0** | 0.01 → **1.0** |
| λ2 区间 / 初值 | **不变** | **不变** | **不变** |

**为什么 η2 要 ×100（关键）**：`d_block_L` 必须与它约束的 c2 同口径 —— 这是
"`d_block` 始终代表**同一物理容忍度**"的直接后果（与 H 层 `d_block_H = 15` 恒等于
"3 MW × 5 步的容忍度"是同一条原则）。c2 缩 100 倍后若 η2 不动，`λ2 += η2·(ema_c2 − d)`
的增量会小 100 倍、对偶慢 100 倍；同乘 100 后 `Δλ2` 逐位不变，且因为 r2/c2 **两个通道
同除 100**，给定 λ 下 actor 目标只差一个正因子 ⇒ **λ2 轨迹与约束语义完全不变**，
只是 L 层值函数的数值单位变了。

**不受影响 / 需要留意的**

* **H 层**一切不变：`d_block_H = 15`、`--value_scale 1000`、`λ1/curriculum` 均未动。
* **η1 不需要改（与 η2 的处境不同）**：H 层是**乘性**对偶 `λ ← λ·exp(η1·(EMA/d − 1))`，只看 `EMA/d` 的**比值**；而 `sum_c1` / EMA / `d_block_H` 全在原始口径，归一化只作用于 critic 回归空间 ⇒ 比值不变 ⇒ η1 = 0.02 原样保留。即便将来把 H 对偶也搬进归一化口径（EMA/s、d/s），比值仍不变 ⇒ 依旧不用改。反过来 L 层是**加性**对偶 `λ += η2·(EMA − d)`，又真的把 c2 与 `d_block_L` 同除了 100（差值也缩 100 倍）⇒ 必须 η2×100。四种组合的实算对照见 `python agentutil/audit_coverage.py`：
  乘性（原始口径 vs 归一化口径）|A−B| = 8e-15（逐位免疫）；加性补 η×k |E−D| = 4e-14（与旧口径一致）；加性不补 η |E−C| = 3.86（λ 几乎不动 —— η2 的困境）。
* **日志口径变化**：`r2`/`c2` 相关列（`sum_c2`、`df_episode_*.csv` 的 L 成本列、
  TensorBoard 的 `reward_lower/*`）全部 ÷100 —— 跨版本对比绝对数值时要乘回 100。
* `Cal_r2_PER_baseline()`（prrbb/prle/td3prrbb）是 **LEGACY 基线**，其系数（10 / 700）
  **刻意保持原值**，用于对照旧 PER 设计；`agentutil/test_l_lower_unit.py` 对此有断言。
* 测试夹具（`myfunc.py` 的 `S_test`/`nine_case_test`、`test_simple.py`、
  `test_compare_projects.py`、`test_five_days.py` 的注释）已同步为同一口径。

**不变量测试**：`python agentutil/test_l_lower_unit.py`

* 六项的系数/预算/`cost_limit`/η2 与目标值一致；
* **物理量守恒**：`d_block_L / c2系数` 与 `η2 × c2系数` 与 ÷100 之前逐位一致
  （0.02 MWh、4 MWh² 与 10 / 7 / 0.1）；
* 动态实算：`r2(0)=0`、`r2(1 MW)=−1 元`；规则动作 `c2=0`、差 0.1 MWh 可达量 `c2=1.0`；
  `td3prle` 的 `C2(±2 MWh)=0.4=d_block_L`、裁剪惩罚封顶 10 元 ⇔ 合计 10 MW（与 ÷100 前同）。

---

---

## 附：下层信号语义统一为 SOC 目标（`td3prrbb` / `prle`，单一真源 `hsrl_soctarget.py`）

**背景**：9 个方法的下层信号有**两种**语义。本次把 `td3prrbb` 与 `prle` 从"影子电价"改成
"**SOC 目标**"（与 `td3per` 对齐）；两者上层的 PRRBB 风价分段奖励（`wT`/`wB`）**保持不动**。

| 项目 | 第 6 维语义 | 下层 | L 观测的目标 | `r2` | `c2` |
|---|---|---|---|---|---|
| td3per / td3prrbb | SOC 目标 | RL | 蒸馏 5 维第 3 维 = `soc_target/10` | `−1·clip_amount` | `+1·trackerr` |
| prle | SOC 目标 | RL（+ LLM relabel） | `Normalize(obs)` + 末维 = `soc_target/10` | `−1·clip_amount` | `+1·trackerr` |
| td3perrb / td3prrlrb / td3leprrlrb | SOC 目标 | 规则 SocTracker | — | — | — |
| td3prle | SOC 目标（二次势中心） | RL | 蒸馏 5 维 | `−min(1·(clip+over), 10)` | `0.1·(SOC−target)²` |
| **per / prrbb** | **影子电价** | RL | `Normalize(obs)` + a6 原值 | `0.1·shadow_profit − 1·clip` | `10 / 7·soc_deviation` |

**下层口径**（集中在 `hsrl_soctarget.py`，三个 SOC 目标 + RL 下层的项目共用）

* `a6 → soc_target = clip(3 + 5·a6, 0.5, 5.5)`（块级 5 步固定）
* `r2 = −1·clip_amount`（`clip_amount = |pb_cmd − pbrule|`；影子套利项随语义删除）
* 块末追踪偏好进 **r1**：`10·max(0, |soc_target − SOC_end| − 0.1)`（uniform；legacy 进 c1）
* `c2 = +1·trackerr` [MW] —— **两区间分治**：Soctrack 范围 `[δ/(g·n_rem), δ/g]`（摊还端…一步填平端，
  `n_rem = 5 − step%5`）交平抑窗口 `[pbrulemin, pbrulemax]` 得参考集合；`trackerr` = 实际施加的
  `pb`（由 SOC 变化反解）到该集合的距离。**零成本集合 = 参考集合**：交集非空时是**区间**
  （L 可任选努力水平）；**空交集（目标在窗口内够不着）时不存在范围，只有纯粹的一个"竭力值"**
  `r* = clip(δ/g, W)`（W 里离 B 最近的端点，`ref_is_point=True`）——只有恰好顶到 `r*` 才不罚，
  两侧偏离都按 `|pbrule − r*|` 线性计费（缺口本身不可达 ⇒ 由上层块末 gap 罚承接）；
  `n_rem = 1` 时退化为"一步填平"点（= 旧 `slack` 口径）；窗口不可行时 `trackerr = 0`
* `r2`/`c2` **同为 MW 口径、系数同为 1** ⇒ 可直接相加/比较；L 层数值口径随之重标：
  `d_block_L = 1.2`（⇔ 允许搁置 0.02 MWh/块）、地板 `0.15`、`cost_limit = 0.6`、`η2 = 0.16`、
  `λ2 ∈ [1, 20]`（现在读作"MW 追踪缺口 : MW 越窗"的兑换率）

**记录列**（与 td3per 对齐）

* CSV：`soc_signal` / `soc_target` / `soc_slack` / `soc_track_err`
* TensorBoard：`target/soc_target`、`target/soc_signal`、`action/soc_target`、`soc_track_err`、
  `gap_penalty`、`sum_gap_pen`、`reward_lower/soc_slack`
* 原 `shadow_price` / `shadow_profit` 列随语义删除；`bess_profit_net` 改用**真实** `bess_profit`

**不变量测试**

* `python agentutil/test_soctarget_uniform.py` —— 三个项目在 60 组随机输入（含不可行窗口）上
  `(r2, c2, soc_target, slack)` **逐位相同**；`td3per` 的本地实现与单一真源在 343 组网格上
  逐位一致；零成本动作 = 规则动作；目标映射与观测目标维一致
* `python agentutil/test_l_lower_unit.py` —— SOC 支的接线（都指向真源）+ 其它支的系数与
  ÷100 物理量守恒

**遗留**：`Cal_r2_PER_baseline()`（prrbb / prle / td3prrbb）是 **LEGACY 基线**，
系数保持 ÷100 之前的原值，用于对照旧 PER 设计。

**评测夹具同步**：`test_compare_projects.py` 的 PRRBB(影子)/PRLE(SOC) 不再共用同一公式；
`test_simple.py` 的 TD3PRRBB / TD3PER 分支改为直接调用各自项目 myfunc（此前手写的影子公式已 stale）。

## 附：warmup 重构 + critic bias 触发修复（`--warmup_sched`，9 个方法共用单一真源 `hsrl_warmup.py`）

**问题（`output/TD3PRRLRB-run3` day0 实测）**

1. **bias 初始化从不触发**：触发条件写成"块末分支内 `step == args.warmup`"，而 episode 长度
   1440 是 5 的倍数 ⇒ 块末步恒为 `step ≡ 4 (mod 5)`，默认 `--warmup 200 ≡ 0` **永不相等**。
   结果 critic 起点 = `fc3.bias` 的 PyTorch 默认值（实测 `q_p_mean = 0.0497`）。
2. **warmup 自身是唯一"BESS 疯狂 churn"的窗口**：上层全随机 ⇒ `soc_target` 每块跳 ±2.5 MWh，
   而 RBB 窗口内每块只能移动 0.36~0.52 MWh（85% 的块目标不可达）⇒ `SocTracker` 全程顶格往返；
   退化只按放电毛量计（`490 元/MWh`），实测块收益 `r̄ = −82.0 元/块`（其中退化 −151.0）。
   该 `r̄` 经 `r̄/(1−γ)` 得到的基线是 **−8196**，与 day0 真实价值量级
   （全状态平均 `Σγᵏr = +9983`）**反号**。

**改动**

| 层 | 旧 | 新 |
|---|---|---|
| 上层动作 | `random_action()`（wt ~ U(0,0.5)、信号 ~ U(−0.5,0.5)） | `WarmupSchedule`：wt 各维 0.33 起、每块 ±0.05、clip **[0.20,0.45]**（Cp 保持最优的 86%~100%，E[ΣCp]=2.82 ≈ 95% 最优）；信号维**恒 0**（SOC 目标语义 ⇒ `soc_target ≡ 3`；影子电价语义 ⇒ 中性 0）——两者都等于 actor 的 `init_wt_action_bias(..., sig_value=0.0)`，即 warmup 与策略起步**同分布** |
| 下层 BESS（BESS-RL 的 6 个项目） | `agent_L.random_action()`（每步随机 pb） | `WarmupBessRule`：RBB 窗口内"一步填平"跟踪 SOC 目标（语义与各项目 `SocTracker.act` 逐位相同，有对照断言）；该步**真实的** (s_L, a_L=规则命令, r2, c2, s_L′) 经 `fill_bess_transition()` 入池（"事后填入"，a_L 不再取 agent 抽的随机值） |
| bias 触发 | `step == args.warmup`（不可达） | `hsrl_agent_init.BiasInitTracker`：warmup 期逐块收集，在 **warmup 之后第一个块末**一次性应用（`first_trigger_step(warmup)`），且排在 `step > warmup` 的首次更新之前 |

**效果（同 day0 序列，离线干跑 5 seed）**：`r̄ = +70.4 元/块 ⇒ b_p ≈ +7042`（与 +9983 差 1.4×、同号），
退化降到 **17.5 元/块**（其中 712 元是"把 SOC 从 episode 随机初值拉到 3"的一次性对齐放电，
之后稳态只剩 0.2 元/块），c1 = 5.7 元/块。自动估计偏保守（成本通道 c1 只有策略段的 ~42%），
需要更贴近策略量级时可显式给 `--critic_bias_p 8000 --critic_bias_c 1000`。

**开关**：`--warmup_sched {fixed,random}`、`--warmup_bess {rule,random}`、
`--warmup_wt_start/_step/_lo/_hi`、`--warmup_soc_target`。
**不变量测试**：`python agentutil/test_warmup_schedule.py` —— 触发点（warmup∈{199,200,201,204,500} 各恰好
触发一次且在首次更新前）、旧条件反例、调度边界/可复现/不吃全局 RNG、规则行为体与 `SocTracker.act`
逐位一致（20000 组随机含 3080 组不可行窗口，max|Δ|=0）、transition 填入计数。

**注（"事后填入"能做什么、不能做什么）**：BESS-RL 项目的 warmup 期 r2/c2 在这些奖励定义下
**天然接近 0** —— SOC 目标支（`td3per`/`td3prrbb`/`prle`）的零成本动作**就是**规则动作
`clip(δ/g, 平抑窗口)`（参考集合由它定义 ⇒ `trackerr ≡ 0`），且 `r2 = −1·clip_amount ≡ 0`（规则动作不越窗）；
`per/prrbb`（影子电价支）的 `r2 = 0.1·shadow_profit − 1·clip_amount` 而影子价恒 0（`shadow_profit ≡ 0`）；
`td3prle` 的二次势 c2 以目标为中心、warmup 期目标恒 3.0 ⇒ 仅在 SOC≠3 时非零。
所以本机制保证的是**环境侧行为正常 + H 层 warmup 基线可用 + L buffer 内是真实的 (s,a,r,c,s′) 行**，
**不等价于对 L 的行为克隆**；若要 L 真正 warm start 到规则，需要另加监督项或塑形奖励（见 `hsrl_warmup.py` 文档）。

---

## 14. 许可与致谢

本项目**当前未选定开源许可证**。在补充 `LICENSE` 前请勿对外分发。

特别注意：仓库内含第三方代码（`libraries/`、`bin/`、WFSim 定制版，见 §13），
若将来转为公开仓库，需先逐一确认这些组件的许可条款并可保留其署名。
