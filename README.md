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
| **TD3PER** | `td3per/` | **TD3**（`SP_TD3`） | **非分段**（移植 td3perrb 上层） | RL（5 维规则原配料，追踪 **SOC 目标**） | `td3per/td3per_main.py` |
| **TD3PRRBB** | `td3prrbb/` | TD3 | 分段 | RL（5 维规则原配料） | `td3prrbb/td3prrbb_main.py` |
| **TD3PRLE** | `td3prle/` | TD3（`SP_TD3_PRLE`） | 分段 + LLM | RL（5 维规则原配料） | `td3prle/td3prle_main.py` |
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
  （5 维观测 `[pg(t−1), ptur, SOC, soc_target, n_rem]`、`r2 = −1·min(clip_amount, 20)`、
  `c2 = +1·trackerr + 0.15·max(0, −pbrule) [MW]`（到**参考集合** = Soctrack范围 ∩ 平抑窗口 的距离；
  第二项是**放电退化成本罚** `L_DEG_COEF = 0.15`，设计上比跟踪信号**小 6.67 倍**（每 MW·min，
  要求 5~10 倍），与跟踪项同处 c2 通道 ⇒ 比值与 λ2 无关；只对放电计费，已接线
  td3per / td3prrbb / td3prle / prle 四个有 RL 下层的项目；`hsrl_soctarget` 的常量表有完整说明）：
  `F ≠ ∅` 时零成本集合是该区间；`F = ∅` 时**不存在范围, 只有纯粹一个竭力值 `r*`**
  （窗口里离 Soctrack 范围最近的端点）, 顶到它才不罚；
  块首够不着的目标跳变由 r1 块末 gap 罚承接）。
  H 层动作越界罚（`penalty_NN_H`）现为 **400 元/单位**（`PENALTY_NN_H_COEF`，本轮由 200 加倍；
  下层 `PENALTY_NN_L_COEF = 200` 不变；⚠ `s_p=1000` ⇒ critic 空间 0.4/单位，与经济斜率仍同量级）。
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
| `--warmup_sched` | `fixed` | warmup 上层动作调度：`fixed` = 1/3 起每块 ±0.05、clip[0.05,0.45]、信号维恒 0；`random` = 旧 `random_action()`（对照臂） |
| `--warmup_bess` | `rule` | BESS-RL 项目（per/prrbb/prle/td3per/td3prrbb/td3prle）warmup 下层行为：`rule` = RBB 窗口 + SOC 目标跟踪（确定性，无随机）；`random` = 旧 `agent_L.random_action()`（对照臂） |
| `--warmup_soc_target` | 3.0 | warmup 期规则下层跟踪的 SOC 目标（MWh） |
| `--discount` | 0.99 | 折扣因子（以 5 min 的"块"为单位） |
| `--policy_delay` | 2 | TD3 延迟策略更新频率 |
| `--nstep` | **5** | H 层 **n 步回报**（滑窗 stride-1：每个块都是某条转移的起点，条数 ≈ 块数、状态不丢；`reward=Σγ^k r_k`、自带 bootstrap 系数 `γ^n`）。把需要自助传播的深度从 `1/(1−γ)=50` 降到 `50/n` ⇒ 冻结批实测 n=5 自洽比 0.79 / n=10+mean ≈1.0（1-step 仅 0.41）。**默认 5 = 生产口径**（依据 run12/run13 的 2v2 观测：`\|dQp/da\|` 5~8 倍且不衰减、早期块奖励 160~181 vs 55~91 元、自洽比 0.59/0.60 vs 0.34/0.47；⚠ 非受控 A/B）。**代码默认值 2026-10 由 10 统一为 5**（11 个 `*_main.py`；见本文末「`--nstep` 默认值 10 → 5」附录），此前 `--nstep 10` 那批 run 请显式传 10 复现。**复现 run1~run10 请显式 `--nstep 1`**（那条路径逐位不变）。注意条数与 `rmsize` 口径随 n 变（1 条 = n 块 = 5n 环境步）。**六个 TD3 方法已统一**（td3per / td3perrb / td3prrlrb / td3prrbb / td3prle / td3leprrlrb；`agentutil/test_nstep_uniform.py` 验证滑窗条数/r_和/`γ^n`/done 截断/context 拼接六者同源）；四个 PRRBB 系项目共用自缩放奖励，其 `target_agg` 的 `mean` 分支同样启用 |
| `--rmsize` | **20000** | H 层 memory **条数**（覆盖环境步 = `rmsize × 5 × nstep`）。默认 20000 = 生产口径（与 run12/run13 一致）：n=5 时 ≈ **500k 环境步 ≈ 1.8 epoch**，即"最近约 1.8 个 epoch 的滚动窗口"；≤60k 步的 run 里仍填不满（12,000 条）。**复现 run1~run10 请传 `--rmsize 80000`**（1-step 口径 ≈400k 步）。**六个 TD3 方法已统一为 20000**（此前只有 td3per 下调、"其余五方法维持 80000"的例外表已在批次 2 取消） |
| `--target_agg` | `min` | H 层 twin **profit**-critic 的 target 聚合：`min` = TD3 原口径（悲观、防过估计）；`mean` = 去偏（未收敛期实测 +18 %，但真实网两网相关 0.99+ 时仅 ~2 %）。actor 的梯度聚合始终为 `min`（`_pair_min`），此开关只改回归目标 |
| `--viol_derate` | **0（关闭）** | ⚠ **⚠ 默认已回退为 0（关闭）：c=5.0 试行过一次（run14）并实测有害，详见本行末尾。** **越界乘性惩罚系数 c**：`r1` 的**风机收益**项乘 `f = exp(−c·x)`，`x = Σ` 前 5 维越界量（机组限功率 `[0,0.5]`；**dim5 signal 不计**）。动机（`agentutil/probe_q_profile_boundary.py` 一维 `Q(a_j)` 剖面实测，run13 ckpt40000）：线性罚 `200/单位` 在 critic 空间只有 `200/s_p = 0.2`，而框内经济斜率实测 0.27~4.5 ⇒ **跨界几乎不花钱**，`Q` 峰贴在约束边界上（dim4 峰 `a=0.05`；dim5 峰甚至在界外 `−0.60`）⇒ 平滑 actor 逼近角点解时一半状态落界外（dim4 25.7%、dim5 47.1%）；且典型越界（深度 ~0.09）的罚贡献 `0.018` 比 critic 自身 TD 误差 `0.64` 小 30+ 倍。乘性罚**与收益同尺度**（界面斜率 `= c·`该步风机收益）故随风速/电价自缩放；乘 `turbine_profit`（实测 100 % 的步 `≥0`）而非净利润，排除"亏损步上 `f<1` 反而奖励越界"的符号陷阱，并保证 **`r1` 单调不增**。**只动 `r1`**：env 物理量与 `revenue`/`turbine_profit`/`degradation`/`profit` 等**记录列全部不变**（`df_test9.csv` 另**新增**三列 `turbine_profit`/`derate_loss`/`penalty_NN_H` 供审计：`profit/45 − r1 == (罚 + 削减 + 未解释残差)/45`，正交于既有列，桩环境端到端验证过 `c=0` 与 `c=5` 的 `profit`/`profit_econ`/`dsoc` 逐位相同）。**复现历史 run1~run14 请传 `--viol_derate 0`**（该路径逐位不变：`f` 恰为 1.0；见 `agentutil/test_viol_derate.py`）。**六个 TD3 方法均已装此开关**（td3per / td3perrb / td3prrlrb / td3prrbb / td3prle / td3leprrlrb；helper 逐位同源，见 `agentutil/test_viol_derate_uniform.py`）。⚠ **四个 PRRBB 系项目**（td3prrlrb / td3prrbb / td3prle / td3leprrlrb）的 r1 对收益**重加权**（`wT`/`wB`≠1）⇒ 界面斜率 = `c·wT·风机收益`，同一个 `c` 的实际强度与 td3per/td3perrb 不同，且 `test9_compare [6]` 会显示非零"未解释残差"（正常，与 c 无关）；要给它们复标 c，先用 `agentutil/probe_q_profile_boundary.py` 量各自的框内斜率 `s_in`。**<br>⚠ 失败模式（run14，c=5.0，同 step 0–12k 对照 run13；两者只差此开关）**：块奖励 116.8 → **29.4（−5×）**；越界率 24.2% → **38.2%（并在 18%~78% 间振荡）**；罚/步 5.07 → 10.27、削减/步 0 → 9.46、净 r1/步 23.35 → **5.88（坏窗口 −14.6）**；**`Qbar` 3.8 未跟随（锚 4~9.9 → 2.1~3.0）⇒ 自洽比 0.4~0.8 → 1.25~1.78（高估约 2×）**；`\|dQp/da\|/Qbar` 0.85 → **2.08**；策略抖动 `rms(1)` 0.122 → **0.360**；test9 `profit_econ` −15 %、sd +43 %。**机制**：乘性罚让**奖励电平依赖策略自身的越界率** ⇒ 电平非平稳 ⇒ 与固定的 `s_p=1000` 和 warmup 期定的 bias 基线失配 ⇒ critic 相对可实现回报系统性高估 ⇒ actor 追虚高 `Q` ⇒ 越界更多 ⇒ 削得更狠（正反馈）。**教训**：奖励侧惩罚若其幅值随策略自身行为变化，必须先保证**奖励电平平稳**（例如对削减做运行均值归一化，或让 `s_p`/bias 基线与新电平一起重标），否则值层校准会失效。c=5.0 的备选值（13.5 翻倍 / 28 接近上限）**未经实验**，启用前请先解决上述电平问题 |
| `--phase_feat` | **0** | H 观测的 24h 相位特征：0=无(默认, 4 维) / 1=sin·cos(6 维) / 2=再加分段特征(9 维)。默认由 1 改为 0：mode1 实测是"时钟查找表"（反事实 clock_share 97–100%），去掉后风速解释度 1.9–5% → 9.0–14.5% |
| `--wt_bias_init` | 1 | 风机输出层 bias 初始化到**中性宽带** `U(0.05,0.45)`（与 warmup 带同区间；**只讲单机物理，不预设风场最优的形状**，见下「附：5 维轴向因子的逐台接线」§先验两轮修正） |
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

> **网络维度按 checkpoint 自动探测**（`_detect_hidden`）：同一次评测里可以混跑网络宽度不同的
> 权重（历史上出现过 16/100 与 64/128；改造前的 `per/prrbb/prle` 是 16/100），无需手填
> `--hidden1/--hidden2`。观测维数同理（`_detect_nb_states` → 相位模式）。
> 反例：`test_compare_projects.py` / `test_simple.py` 只跑改造前口径，其 args 里写死 16/100；
> 评测六个 TD3 方法请一律用本脚本。

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
python agentutil/test_csv_incremental.py   # 日志表增量落盘 (50 整数倍 step 卡顿回归)
python agentutil/test_test9_settlement.py  # test9 存货结算 (profit vs profit_econ, 580)
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
9. **历史 run 与当前环境不可比**：2026-09-28 修掉了"5 维风机动作只有 dim0 生效"的接线
   bug（dim1~4 曾是死维），修正改变了动作空间的物理含义 ⇒ 此前所有 run / checkpoint
   须在新环境重训后再比较。机制、修法与复验见
   [附：5 维轴向因子的逐台接线](#附5-维轴向因子的逐台接线wfsim_portm-入口修正2026-09-28)。

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

## 附：5 维轴向因子的**逐台**接线（`WFSim_port.m` 入口修正，2026-09-28）

**症状。** 上层动作的前 5 维（5 台风机控制量）此前**只有 dim0 真正进入 WFSim**：
`env.py::PowerOutput` 把动作交给 `matlab.double([list(act_wind)])`（**1×K 行向量**），
`WFSim_port.m` 原样写入 `Control.a` / `Control.c_prime`，而 `bin/core/meshing.m:196-197`
    `'CT_prime', zeros(NT,1) + Control.c_prime,   'a', zeros(NT,1) + Control.a`
在 MATLAB 隐式扩展下把 `NT×1 + 1×K` 变成 **NT×K 矩阵**；`bin/core/Actuator.m:14` 取的
`input = Wp.turbine.input(sol.k)` 是标量 struct（字段是矩阵），`:55 CT(kk) = input.CT_prime(kk)`
与 `:61 a(kk) = input.a(kk)` 都是**线性索引**，`kk=1..NT` 全部落在**第一列** ⇒ 5 台机组
一律拿到 `c_prime(a₀)`。**dim0 退化为"全场共用的集体诱导因子"，dim1~4 是死维**，
第 6 维（SOC 信号）落在第 6 列同样被忽略。

**修法（单点入口，覆盖所有读 `Control` 的分支）。** 不改 `meshing.m`（它有 4 处同样模式、
另有 2 处硬写 `2*ones(NT,1)`），只改入口两处：
1. `WFSim_port.m`：动作**要求恰好 NT=5 维并转成列向量**（`Control.a = a(:)`），
   使 `zeros(NT,1)+Control.a` 成为逐元素运算。维度守卫做成**对称**的（`numel(a) ~= NT` ⇒ `error`）：
   实测**少传**（4 维）与**多传**（6 维 = 5 台+信号、10 维）都曾出问题——少传本就报错，
   而多传是**静默只取前 5 个**（`[.2]*5+[.5]*5` 与 `[.2]*5` 的总功率逐位相同 3.717941 MW），
   与"只有 dim0 生效"是同一类静默失效，故一并封死。"策略 6 维 → 风场 5 维"的翻译
   由 `env.py::PowerOutput` 独家负责（截到 `num_turbine` 后再下传）。
2. `env.py::PowerOutput`：`clip` 前先截到 `self.num_turbine`，防多传的维再次触发隐式扩展。
一次性补丁脚本：`agentutil/patch_a_induction_wiring.py`（已执行；锚点要求恰好命中 1 次，
重复执行安全中止）。

**复验（`agentutil/probe_a_per_turbine.py`，u = 10 m/s，三种动作组）。** 修前/修后对照：

| 动作（5 维） | 修前逐台 `CT_prime` | 修后逐台 `CT_prime` | 修后 Δtotal |
|---|---|---|---|
| `all 0.20` | `0.512 ×5` | `0.512 ×5` | 基线 3.717941 MW |
| `a2=0.50` 其余 0.20 | `0.512 ×5`（**与基线逐位相同**） | `[.512 .512 **.5** .512 .512]` | −0.177970 MW |
| `a4=0.50` 其余 0.20 | `0.512 ×5`（**与基线逐位相同**） | `[.512 .512 .512 .512 **.5**]` | −0.091684 MW |
| `ramp 0.1..0.5` | `0.324 ×5`（只取 `c_p(0.1)`） | `[.324 .512 .588 .576 .5]`（各取 `c_p(a_d)`） | −0.307925 MW |

复验含两类**断言**：① `port` 臂（生产入口）中实测变化的机组集合必须恰为
`{d : c_p(a_d) ≠ c_p(0.20)}`（故 `ramp` 的第 1 台 `a=0.20` 本就应"不变"），且 Δtotal 非零；
② `env` 臂中 `PowerOutput(u, 6 维动作)` 与 `WFSim_port(u, 5 维前缀)` **逐位相同**（6 组 Δ = 0.00e+00），
且第 6 维取 `+0.25` 与 `−0.40` 的风功率逐位相同 ⇒ 信号维未泄漏。脚本用 `raw` 臂（`meshing.m` 直连）
保留原始故障态的复现，便于日后核对。

**影响（务必注意）。** 修正改变了动作空间的**物理含义**：dim0 从"全场集体因子"变为"第 1 台机组"，
dim1~4 从死维变为有效维。因此**在此之前产出的全部 run、checkpoint 与基于它们的结论
（含 `agentutil/probe_a_induction_power.py` 里"dim2 斜率 +2.4 MW/单位"之类）都不再与新环境可比**，
需在新环境重训后再做跨 run 比较。

### 续：动作先验的两轮修正 —— 最终落到**中性宽带**（2026-09-28）

接线修好之后，动作侧的**先验**还停留在旧判据上：`WT_START=0.33 / WT_LO,WT_HI=0.20,0.45`、
`init_wt_action_bias(U(0.25,0.40))`、`env.*Reset()` 的标称动作 `0.3333`，理由统一是
"该带内**单机** `C_p = 4a(1−a)²` 保持 86%~100% 最优"。旧带的病**不在"窄"，而在"位置"**：
它把 `a<0.20` 整段排除在外。

`agentutil/probe_a_farm_structure.py` 扫 WFSim（u = 6/8/10/12/14，MATLAB R2025a）
得到的事实（**只对"均匀动作族"成立**）：

| u (m/s) | 6 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|
| 均匀族峰位 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 |
| 峰功率 MW | 0.8031 | 1.9036 | 3.7179 | 6.4246 | 10.2020 |
| 全体 `a=1/3` | 0.7444 | 1.7645 | **3.4462** | 5.9551 | 9.4564 |
| ±2% 平台 | [0.20,0.25] | 同 | 同 | 同 | 同 |
| ±5% 平台 | [0.15,0.30] | 同 | 同 | 同 | 同 |

由此**唯一站得住**的结论是：**单机 Cp 最优 ≠ 风场总功率最优** —— 全体 `a=1/3` 比均匀
`a=0.20` 低 **7.3%**（u=10：3.4462 vs 3.7179 MW），把单机 Cp 拉满反而亏全场。

> ⚠ **第一轮改错了（已纠正，记录以免重犯）。** 第一轮据此把先验对准"均匀族峰位"
> （`0.20 / [0.15,0.25] / U(0.17,0.23)`，脚本 `port_wt_prior.py`）——两个错：
> ① 那个 0.20 只是**均匀族**上的峰，逐维结论又来自**坐标轮询**（每次只动一维、其余钉在
> 0.20），**不是联合最优**；用来支持"维间无竞争"的"拼接值与单独最优一致"更是**循环论证**
> （拼接的那组配置本就不变）。② 更根本的是**方法**：**风场最优是随尾流状态变化的单调
> 上升斜坡** —— 最下游机组背风侧无机组，其功率 ∝ `CT(a₅)·Ue³`，上游动作固定时 `Ue` 与
> `a₅` 无关 ⇒ 它应被推向单机最优 `≈1/3`；需要降载留风的是**上游**。这个**形状取决于
> 尾流效应，是策略要学的函数**，不是可以用几个仿真点测定并写进先验的常数；而第一轮的
> 窄带还会从**上方掐住末台**，把"偏高"的偏差换成了"偏低"。

**最终口径（中性宽带，只有一条带、一处真源；脚本 `agentutil/relax_wt_prior.py`）**：

| 项 | 旧（第二轮错值） | 最终 | 理由 |
|---|---|---|---|
| `hsrl_warmup.WT_START` | 0.20 | **1/3** | 单机 `C_p` 最优点 —— **只讲单机物理**，不对"风场最优在哪"表态 |
| `hsrl_warmup.WT_LO, WT_HI` | 0.15, 0.25 | **0.05, 0.45** | 只排除 a<0.05 的**停机悬崖**（`C_p`<18%）与 a>0.45 的低效高诱导区 |
| `init_wt_action_bias(lo, hi)`（6 个 main） | 0.17, 0.23 | **0.05, 0.45** | 与上面**同一条带** ⇒ "warmup 与策略起步同分布"这条不变量精确保留 |
| `env.{reset,TestReset,ScenarioReset}` 的 `a_temp` | 0.20 | **0.3333** | `P_G_prev` 的波动基准是约定，取单机物理点，不带风场主张 |

原则：**先验只编码单机物理，不编码风场最优的形状。** 后者是策略要学的函数。
`test_warmup_schedule.py` 的判据也相应退回**纯契约**（起点 = 1/3、界内、步长 ≤0.05、
截断只发生在贴墙处、中性宽带被真正走到两端：seed=7 的 41 块实测 `min=0.05 / max=0.45 /
E[a]=0.3288`；2001 块平稳均值 = 0.2501 = 带中点），**不再**断言任何"效率/最优"类判据。

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

## 附：H 层值函数回归空间归一化（**双尺度** `--value_scale` / `--cost_scale`）

**问题**：`model.py` 的 `Critic` 输出层 `fc3(50→1)` 只把 weight 初始化到 ±3e-3，bias 保持
PyTorch 默认 ±1/√50（实测初值 `Q ≈ 0.0804`），而 H 层块收益使回归目标达到 **O(10⁴)**
（块收益均值 ~2.6e2 元 × `1/(1−γ)`，γ=0.99）——**初值与目标差 ~3e5 倍**。

**做法**：让 H 层 critic 在**归一化空间**回归。`r1`/`c1` 只在**唯一一处**换算：各 agent
`update_policy()` 采样之后 `profit_batch / s_p`、`cost_batch / s_c`；其余一切保持原始口径。

**为什么拆成两个 scale（profit/cost 分开）**：`r1` 与 `c1` 的典型量级差 ~20 倍
（r̄≈200 元/块 vs c̄≈10 MW·step/块），共用 `s=1000` 时 cost critic 的回归目标只有 O(0.4)、
**可学变化**更只有 O(0.004) —— 恰好落在 `model.Critic` 输出层 `init_w=3e-3` 的尺度上，
结果 **cost critic 塌成常数**（实测 run9：`Q_c ≡ 360`、`∂Q_c/∂a ≡ 0`，四张网全同；后果是
`λ·Q_c` 对 actor 梯度恒 0 ⇒ λ 无论怎么调都不影响策略）。故拆为：

| 项目组 | `s_p`（r1） | `s_c`（c1） | 块奖励 r1/s_p vs c1/s_c | 值 Q_p/s_p vs Q_c/s_c | λ_actor |
|---|---|---|---|---|---|
| **TD3 全系列（6 个）** | 1000 | **100**（`hc.COST_SCALE_TD3`） | 0.19 vs 0.11（同量级） | 5.0 vs 3.6（平衡） | **λ_raw×(s_c/s_p) = λ_raw/10** |
| 其余 3 个（per/prrbb/prle） | 1000 | 1000（= s_p，旧行为） | 0.19 vs 0.011 | 5.0 vs **0.36（14× 失衡）** | λ_raw（不变） |

**λ 随 scale 自动重标（不手改，杜绝"改了 scale 忘改 λ"的方向错误）**：actor 目标
`−(q_p − λ·q_c)` 里 `q_p = Q_p/s_p`、`q_c = Q_c/s_c`，乘回 `s_p` 得
`Q_p_raw − λ·(s_p/s_c)·Q_c_raw` ⇒ 有效物理价 = `λ·(s_p/s_c)`。要保持"λ 是元/MW·step 的
物理价"（与 raw 对偶、`d_block_H = 15` 同口径）必须 `λ_actor = λ_raw·(s_c/s_p)` ——
`hc.h_settings()` 用 `lam_scale()` 自动完成 ⇒ **物理价恒为 λ_raw**。
实测 `s_c` 1000→100 时 λ 区间 `0.967~4.833 → 0.0967~0.4833`（即 ÷10）。

| 对象 | 是否缩放 | 说明 |
|---|---|---|
| H 层 replay memory（warmup 填入 / 策略期写入 / `relabel_all()` 重写） | **不缩** | 三条路径都存原始口径 ⇒ 天然同口径，不存在混装；换算只发生在回归时 |
| `critic_p/p2/c/c2` 四张网及其 target | **缩** | 共用同一个 `target_q_batch_{p,c}`，只有一处除法 |
| actor loss `−(Q_p − λ·Q_c)` | 自动 | 两通道各除自己的 s ⇒ 正因子（Adam 下等价）；**λ 必须按 `lam_scale()=s_c/s_p` 重标**才保持物理价（`h_settings()` 自动做） |
| `hsrl_agent_init.BiasInitTracker` | **缩** | 读的是 main 里的**原始**块收益累加器 ⇒ 写 `fc3.bias` 前 `b_p /s_p`、`b_c /s_c`（含 `--critic_bias_p/_c` 覆盖值，按原始口径给出）。估计量对样本线性，故 SE/块数/触发点/γ 都不用重调 |
| λ 区间、EMA 对偶（`ema_cost` vs `d_block_H`）、curriculum | **不缩** | 乘性对偶对 r1/c1 的换算一无所知 |
| `update_stats_H.csv` / TensorBoard（`q_*` / `td_*` / `target_*`） | 还原 | agent 内部**按通道**还原（profit `×s_p`、cost `×s_c`）⇒ 与历史 run、`agentutil/analyse_q_scale.py` 直接可比 |
| PER 优先级 `|TD_p|/mean_p + |TD_c|/mean_c + ε` | 不缩 | 自归一化比值，对 s 免疫（`ε` 加在 O(1) 量上，占比不变） |
| L 层 `r2`/`c2` | **不缩** | `agent_L.value_scale = 1.0`（`cost_scale=None` ⇒ 跟随 `s_p=1.0`）；L 值函数尺度本就 O(1)（`d_block_L` = 20 / 2.5） |

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

**实测（B2/D2 段）**：`s_c` 1000→100（λ ÷10）时 —— 首次更新两臂日志**恰好差 s_c 之比（10×）**
（网络同一初值 ⇒ 证明按通道还原生效）；**收敛后原始口径 `Q_c` 收敛到同一目标**
（19.998 vs 19.995，目标 20）⇒ **改 `s_c` 只改数值条件、不改物理量**；
且有效物理权衡 `λ·Q_c/s_c` 两臂一致（0.01933 vs 0.01933）。
⚠ 归一化把 loss/梯度同时缩小 ⇒ 固定的 `clip_grad_norm_(max_norm=2.0)` 语义随 scale 变
（实测 s=1000 时范数只有门槛的 0.10×⇒**从不触发**；s=100 时约 1.0×⇒开始介入）——
这是已知副作用，**动 scale 前先决定裁剪阈值策略**。

* `--value_scale 1000`（默认）；`--value_scale 1`（或 0）关闭归一化，用作对照臂。
* `--cost_scale 100`（TD3 六个 main 的默认）/ `1000`（= 关闭对 cost 的单独缩放）。
  语义：`<=0` ⇒ 与 `s_p` 相同（旧行为）。λ 由 `hsrl_cost.lam_scale()` 自动跟随。
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

| 项 | per / td3per | prrbb / prle / td3prrbb |
|---|---|---|
| `r2` 的 clip 系数 | 100 → **1** | 100 → **1** |
| `r2` 的 shadow 系数 | 10 → **0.1**（td3per 无此项） | 10 → **0.1**（td3prrbb/prle 改为 SOC 目标后亦无此项） |
| `c2` 系数 | 1000 → **10** | 700 → **7** |
| `d_block_L` / 地板 | 20 / 2.5 → **0.2 / 0.025** | 14 / 1.75 → **0.14 / 0.0175** |
| `agent_L.cost_limit` | 10 → **0.1** | 10 → **0.1** |
| L 加性对偶步长 η2 | 0.01 → **1.0** | 0.01 → **1.0** |
| λ2 区间 / 初值 | **不变** | **不变** |

> 注：`td3prrbb` / `prle` / `td3prle` 此后已并入 `hsrl_soctarget` 的**两区间口径**
> （`r2`/`c2` 同为 MW、系数同为 1、`d_block_L = 1.2`、`λ2 ∈ [0.5, 2]`（无量纲兑换率）、`η2 = 0.02`（乘性），见下节）；
> 上表的 ÷100 只对 `per` / `prrbb`（仍在影子电价口径）与历史对照有效。

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
  `td3prle` 与 `prle`/`td3prrbb`/`td3per` 的下层 `(r2, c2, target)` **逐位相同**
  （`agentutil/test_soctarget_uniform.py` 四人等价断言）。

---

---

## 附：下层信号语义统一为 SOC 目标（`td3prrbb` / `prle` / `td3prle`，单一真源 `hsrl_soctarget.py`）

**背景**：9 个方法的下层信号有**两种**语义。本次把 `td3prrbb` 与 `prle` 从"影子电价"改成
"**SOC 目标**"（与 `td3per` 对齐）；两者上层的 PRRBB 风价分段奖励（`wT`/`wB`）**保持不动**。

| 项目 | 第 6 维语义 | 下层 | L 观测 (5 维) | `r2` | `c2` |
|---|---|---|---|---|---|
| td3per / td3prrbb / td3prle | SOC 目标 | RL（**残余动作** `pb_cmd = 规则 + Δ`） | 统一 5 维 **规则原配料**：`[pg(t−1), ptur, SOC, soc_target, n_rem]`（各 ÷10，`n_rem`÷5） | `−1·clip_amount` | `+1·trackerr` |
| prle | SOC 目标 | RL（+ LLM relabel；**仍是绝对动作**, 未同步残余） | 同左（统一 5 维，不再用 `Normalize(obs)` 基线） | `−1·clip_amount` | `+1·trackerr` |
| td3perrb / td3prrlrb / td3leprrlrb | SOC 目标 | 规则 SocTracker | — | — | — |
| **per / prrbb** | **影子电价** | RL | `Normalize(obs)` + a6 原值 | `0.1·shadow_profit − 1·clip` | `10 / 7·soc_deviation` |

**下层口径**（集中在 `hsrl_soctarget.py`，四个 SOC 目标 + RL 下层的项目共用：`td3per` / `td3prrbb` / `prle` / `td3prle`）

* `a6 → soc_target = clip(3 + 5·a6, 0.5, 5.5)`（块级 5 步固定）
* `r2 = −1·min(clip_amount, 20)`（`clip_amount = |pb_cmd − pbrule|` = **两段 clip 合计**：±10 执行器 + 平抑窗口；各 main 传命令值，r2 封顶 20 MW；影子套利项随语义删除）
  * ⚠ **已知口径洞（保留不改，2026-03 决定）**：**窗口不可行**（`pbrulemin > pbrulemax`）时 RBB
    **完全忽略命令**、改施加物理边界 `pbphysicsmin/max`（可与命令**反号**），而 `c2` 已按设计豁免
    （`trackerr ≡ 0`）**但 `r2` 没有** ⇒ 该步 `r2` 可打到封顶 −20，规则动作（Δ=0）也躲不掉，
    且 `Δ` 对它的杠杆 ≤ 1 MW/MW（不可控）。实测（起点+阵风工况 387 组）：54.8% 命令 ≠ 施加值、
    48.1% 的 r2 打到 −20。影响只落在 L 的 reward/critic（λ2 只吃 `c2`，对偶不受影响）；
    故 warmup 的"零成本行"在**这 ~1% 的步上不成立**。见 `agentutil/test_l_residual.py` 第 B4 节
* 块末追踪偏好进 **r1**：`10·max(0, |soc_target − SOC_end| − 0.1)`（uniform；legacy 进 c1）
* `c2 = +1·trackerr` [MW] —— **两区间分治**：Soctrack 范围 `[δ/(g·n_rem), δ/g]`（摊还端…一步填平端，
  `n_rem = 5 − step%5`）交平抑窗口 `[pbrulemin, pbrulemax]` 得参考集合；`trackerr` = 实际施加的
  `pb`（由 SOC 变化反解）到该集合的距离。**零成本集合 = 参考集合**：交集非空时是**区间**
  （L 可任选努力水平）；**空交集（目标在窗口内够不着）时不存在范围，只有纯粹的一个"竭力值"**
  `r* = clip(δ/g, W)`（W 里离 B 最近的端点，`ref_is_point=True`）——只有恰好顶到 `r*` 才不罚，
  两侧偏离都按 `|pbrule − r*|` 线性计费（缺口本身不可达 ⇒ 由上层块末 gap 罚承接）；
  `n_rem = 1` 时退化为"一步填平"点（= 旧 `slack` 口径）；窗口不可行时 `trackerr = 0`
* **L 动作 = 残余动作**（TD3 全系列 `td3per` / `td3prrbb` / `td3prle`）：
  `pb_cmd = hst.rule_action(soc_target, SOC, W, n_rem) + Δ`，`Δ` 由 actor 输出。
  基线 `hst.rule_action` 与各项目 `SocTracker.act` **逐位一致**（含不可行窗口与死区，
  4000 组断言），它承担窗口/额定限幅/被迫吞吐等全部非线性；网络只学"相对规则偏多少 [MW]"。
  * **起点 = 规则，是结构保证而非初始化偶然**：`hinit.zero_actor_output(agent_L)` 把 actor 输出层
    权重+bias 置 0 ⇒ t=0 时 `Δ≡0` ⇒ 命令逐位等于规则动作 ⇒ `r2 ≡ 0` **且** `c2 ≡ 0`
    （规则动作正是 L 奖励的唯一最优/零成本动作；见下"参考集合"）。
  * **入 buffer 的动作是 Δ**（= actor 的动作空间），交给 RBB 的是**命令值** `pb_cmd`
    ⇒ `clip_amount = |pb_cmd − pbrule|` 仍是**两段 clip 合计**（±10 执行器 + 平抑窗口），
    不会出现"|cmd|>10 时该方向梯度恒 0"的死区；事后填入用 `cmd_pb`(=命令) 与
    `rule_pb`(=基线) 核对 `|rule_pb + a_L − cmd_pb| == 0`（抓接线 bug），
    而"命令被窗口夹掉"单独计 `n_clipped`（**正常工况**, 不是失配）。
  * **残余自由度 = 参考集合的宽度**（实测）：`F = ∅`（≈82% 的步，只有"竭力值"免费）时
    **任何 Δ≠0 都在两个通道之一被计费**；`F` 是区间（≈9%）时免费残余区间**恰好 = F − pb_rule**；
    窗口不可行（≈9%，整步不可归责）时至少"减吞吐"方向完全平坦。
    ⇒ 旧参数化下"规则的绝对功率依赖观测里没有的窗口"这一死结被基线吃掉，残余只学小修正。
  * 探索半宽 `--l_noise_amp` 默认 **1.0 MW**（旧式 ±0.05 MW 比所需动作小 ~50 倍）。
  * **旧 checkpoint 不兼容**（L 观测与动作语义同时改变）。

* **L 观测 = 规则的原配料**（单一真源 `hsrl_soctarget.build_L_obs`，四项目逐位相同）：
  `[pg(t−1)/10, ptur/10, SOC/10, soc_target/10, n_rem/5]`
  —— `pg(t−1)` 与 `ptur` 给出平抑窗口中心（± 已知限幅）；SOC 与目标同刻度 ⇒
  `δ = 10·(obs[3] − obs[2])`、`need = clip(δ/(g·n_rem), W)` 都能直接导出，
  规则动作 `clip(δ/g, W)` 的全部输入都在观测里；`n_rem/5` 让"还剩几步"可知。
  （旧观测是 `[pbrulemin/10, pbrulemax/10, soc_target/10, SOC/10, u/10]`：
  δ 只能从两个近似相等的 /10 数里做差，分辨率 ~0.015，且没有 n_rem。）
* `r2`/`c2` **同为 MW 口径、系数同为 1** ⇒ 可直接相加/比较；L 层数值口径随之重标：
  `d_block_L = 1.2`（⇔ 允许搁置 0.02 MWh/块）、地板 `0.15`、`cost_limit = 0.6`
* `λ2` 是**无量纲兑换率**（"1 MW c2 值多少 MW r2"）：**上限 = r2 的 2 倍、下限 = r2 的 1/2**
  （`λ2 ∈ [0.5, 2]`，关于中性 1.0 对数对称 ⇒ `MAX·MIN = 1`，初值 `1.0` = 中性兑换率）；
  对偶为**乘性** `λ2 ← clip(λ2·exp(0.02·(EMA_c2/d_block_L − 1)), 0.5, 2)` —— 只看无量纲比值，
  换单位不会静默改对偶速度（实测失配比值 ≈11 时约 22%/块 ⇒ 4 块到上限，非 bang-bang）；
  curriculum 以"连续满足 **50 块**"计量（≈ EMA 时间常数 1/α）

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
* `python agentutil/test_l_residual.py` —— 残余动作的 6 组不变量：基线 ≡ `SocTracker.act`（逐位）/
  `Δ≡0 ⇒ r2≡0 ∧ c2≡0` 与"残余自由度 = 参考集合宽度"/零初始化输出层 ⇒ 输出**精确** 0/
  三个 main 的接线（命令入 RBB、Δ 入 buffer、日志列、探索半宽）/ ddpg 侧噪声/事后填入核对语义

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
| 上层动作 | `random_action()`（wt ~ U(0,0.5)、信号 ~ U(−0.5,0.5)） | `WarmupSchedule`：wt 各维 **1/3** 起、每块 ±0.05、clip **[0.05,0.45]**（**中性宽带**：只排除 a<0.05 的停机悬崖与 a>0.45 的低效高诱导区，**不预设**风场最优的形状——它是随尾流状态变化的**单调上升斜坡**，是策略要学的函数）；信号维**恒 0**（SOC 目标语义 ⇒ `soc_target ≡ 3`；影子电价语义 ⇒ 中性 0）——两者都等于 actor 的 `init_wt_action_bias(..., sig_value=0.0)`，即 warmup 与策略起步**同分布**（带区间与 init 区间完全相同）。⚠ 2026-09-28 的两轮修正：旧值 `0.33 / [0.20,0.45]` 的判据"**单机** `C_p` 86%~100% 最优"用错了对象（全体 `a=1/3` 比均匀 `a=0.20` 低 **7.3%**）；中间那轮改成"对准测出的均匀族峰位 `0.20 / [0.15,0.25]`"**同样是错的**（那是坐标轮询而非联合最优，且窄带会从上方掐住最下游机组） |
| 下层 BESS（TD3 全系列 3 个项目，**残余动作**） | `agent_L.random_action()`（每步随机 pb） | **规则臂 = 策略期基线本身**：`pb_rule = hst.rule_action(H 信号给出的目标, SOC, W, n_rem)`（与策略期用的是同一个 `rbb_system` 参数），写进 buffer 的动作是 **Δ ≡ 0**。启动自检 `hwarm.check_residual_baseline()` 逐位核对"warmup 规则行为体 = 策略期基线"（eta/time/rated_power/soc_tol 任一漂移即报错）；默认调度下与改造前的 `WarmupBessRule`(固定追 3.0) **逐位相同**（4400 组 max\|Δ\|=0，因固定调度信号维恒 0 ⇒ `soc_target_from_signal(0)=3.0=--warmup_soc_target`）。对照臂 `--warmup_bess random` 的环境行为**未变**（仍是绝对随机功率 `U(-10,10)` MW），只是 buffer 内改记为 Δ = cmd − rule |
| 下层 BESS（其余 3 个 BESS-RL 项目: per / prrbb / prle） | `agent_L.random_action()`（每步随机 pb） | `WarmupBessRule`：RBB 窗口内"一步填平"跟踪 SOC 目标（语义与各项目 `SocTracker.act` 逐位相同，有对照断言）；该步**真实的** (s_L, a_L=规则的**绝对功率**命令, r2, c2, s_L′) 经 `fill_bess_transition()` 入池（"事后填入"，a_L 不再取 agent 抽的随机值） |
| bias 触发 | `step == args.warmup`（不可达） | `hsrl_agent_init.BiasInitTracker`：warmup 期逐块收集，在 **warmup 之后第一个块末**一次性应用（`first_trigger_step(warmup)`），且排在 `step > warmup` 的首次更新之前 |

**效果（同 day0 序列，离线干跑 5 seed）**：`r̄ = +70.4 元/块 ⇒ b_p ≈ +7042`（与 +9983 差 1.4×、同号），
退化降到 **17.5 元/块**（其中 712 元是"把 SOC 从 episode 随机初值拉到 3"的一次性对齐放电，
之后稳态只剩 0.2 元/块），c1 = 5.7 元/块。自动估计偏保守（成本通道 c1 只有策略段的 ~42%），
需要更贴近策略量级时可显式给 `--critic_bias_p 8000 --critic_bias_c 1000`。

**开关**：`--warmup_sched {fixed,random}`、`--warmup_bess {rule,random}`、
`--warmup_wt_start/_step/_lo/_hi`、`--warmup_soc_target`（**残余动作的 3 个 TD3 项目里不生效**：
规则臂跟随 H 信号目标，固定调度下即 3.0；给了别的值会打印提示）。
`--warmup_sched random` 在残余动作项目里是**行为变化**：信号维不再恒 0 ⇒ 规则臂跟踪每块随机目标
（改造前固定追 3.0），启动时也会提示。
**不变量测试**：`python agentutil/test_l_residual.py` 第 H 节 —— 规则行为体 ≡ `hst.rule_action`
（同 target 逐位）、"改造前固定追 3.0" ≡ "改造后信号目标"（默认调度 4400 组逐位）、
启动自检能抓住 eta_c/rated_power/soc_tol/time 漂移、对照臂随机动作口径未变。
`python agentutil/test_warmup_schedule.py` —— 触发点（warmup∈{199,200,201,204,500} 各恰好
触发一次且在首次更新前）、旧条件反例、调度边界/可复现/不吃全局 RNG、规则行为体与 `SocTracker.act`
逐位一致（20000 组随机含 3080 组不可行窗口，max|Δ|=0）、transition 填入计数。

**注（"事后填入"能做什么、不能做什么）**：BESS-RL 项目的 warmup 期 r2/c2 在这些奖励定义下
**天然接近 0** —— SOC 目标支（`td3per`/`td3prrbb`/`prle`/`td3prle`）的零成本动作**就是**规则动作
`clip(δ/g, 平抑窗口)`（参考集合由它定义 ⇒ `trackerr ≡ 0`），且 `r2 = −1·min(clip_amount, 20) ≡ 0`（规则动作不越窗）；
`per/prrbb`（影子电价支）的 `r2 = 0.1·shadow_profit − 1·clip_amount` 而影子价恒 0（`shadow_profit ≡ 0`）。
所以本机制保证的是**环境侧行为正常 + H 层 warmup 基线可用 + L buffer 内是真实的 (s,a,r,c,s′) 行**，
**不等价于对 L 的行为克隆**。
（TD3 全系列改用**残余动作**后，这条不再是缺口：`Δ≡0` 就**逐位**等于规则下层，
warmup 期的 (s_L, Δ=0, 0, 0, s_L′) 行本身就在教 actor"保持 Δ=0 即最优"，
即"warm start 到规则"由参数化结构直接给出，不需要监督项或塑形奖励。）

---

## 附：`--nstep` 默认值 10 → 5（全方法 CLI 统一，2026-10）

**做了什么**：把**所有带 `--nstep` 开关的训练入口**（`aifflor` / `aifflorpave` / `singleaif` /
`yawflor` / `td3per` / `td3perrb` / `td3perseg` / `td3prrlrb` / `td3prrbb` / `td3prle` /
`td3leprrlrb`，共 11 个 `*_main.py`）的 CLI 默认值由 `10` 改为 `5`，并同步其 help 文案与
随附注释头（`---- 默认值 = N (生产口径) ----`）；`agentutil/test_phase_uniform.py` 里
「`--nstep` 默认 10」的静态断言同步改为 5。`prle` / `prrbb` / `rbb` / `未命名/*` 本来就没有
这个开关，按"无锚点即跳过"处理，未改动。

**为什么**：代码默认（10）与本仓库文档口径（`README` 默认值表、"n 步回报"章节、
`agentutil/test_nstep_h.py` 说明文字都按 **5 = 生产口径** 写）长期不一致 —— 文档侧的
依据是**冻结批离线验证**（1-step + min-target 在 40k 次更新后只到 MC 真值的 34 %；
n=10 + mean 在 20k 次到 90 %，n=5 自洽比 0.79）与 run12/run13 的 2v2 观测（⚠ 非受控 A/B）。
本次把代码对齐到文档口径。

**影响面（重要）**

* `nstep` 只经由 **CLI 默认值**生效：`*_ddpg.py` 的构造签名默认仍是 `nstep=1`
  ⇒ 显式传 `--nstep N` 的运行**逐位不变**（要复现 `--nstep 10` 那一批 run，照样显式传 10 即可；
  复现 run1~run10 的 1-step 口径仍传 `--nstep 1`）。
* **不传** `--nstep` 的新 run 与旧 run 可比性按 `nstep` 变化处理（回报视界与 `γ^n` bootstrap 系数都变）。
* `--rmsize` 的口径说明与 `nstep` 无关（`rmsize` 就是条数 ≈ 覆盖环境步；旧注释里
  "`rmsize × 5 × nstep`" 的乘数写法已作废，见 `td3per_main.py` 的 `--rmsize` 注释）。

**迁移与复核**：迁移脚本 `agentutil/port_nstep_default5.py`（默认 dry-run，
`--apply` 写盘，`--check` 校验已达标，`--revert` 反向还原）—— 每个文件要求在
`--nstep` 代码块作用域内**恰好命中 1 次**，任一文件不符则**整体不写盘**；另对每次替换做
长度变化断言（只允许个位数字符变化）。回归：`agentutil/test_nstep_h.py`（TEST OK）、
`agentutil/test_nstep_uniform.py`（六个 TD3 方法全通过）、`agentutil/test_phase_uniform.py`
（TEST PASSED）。另外用**抽出 argparse 块真跑**的方式核对了 7 个入口的运行时默认值
（`parse_args([]).nstep == 5`、显式 `--nstep 7` 仍为 7、help 文案同步）；其余 4 个入口
（`aifflorpave` / `td3prle` / `td3prrbb` / `yawflor`）因抽出片段缺少其模块级符号无法这样执行，
以静态断言 + `ast.parse` 覆盖。

---

## 14. 许可与致谢

本项目**当前未选定开源许可证**。在补充 `LICENSE` 前请勿对外分发。

特别注意：仓库内含第三方代码（`libraries/`、`bin/`、WFSim 定制版，见 §13），
若将来转为公开仓库，需先逐一确认这些组件的许可条款并可保留其署名。
