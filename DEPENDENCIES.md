# HSRL 项目依赖文档

> 生成日期：2026-08-11  
> 项目路径：`c:\Users\yigul\Desktop\llm enhanced wsis\code\HSRL`

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. Python 依赖](#2-python-依赖)
  - [2.1 核心第三方包](#21-核心第三方包)
  - [2.2 Python 标准库](#22-python-标准库)
  - [2.3 各模块依赖详情](#23-各模块依赖详情)
  - [2.4 缺失源文件的编译缓存](#24-缺失源文件的编译缓存)
- [3. MATLAB 组件](#3-matlab-组件)
  - [3.1 WFSim 风电场模拟器](#31-wfsim-风电场模拟器)
  - [3.2 外部库](#32-外部库)
  - [3.3 MATLAB 工具箱依赖](#33-matlab-工具箱依赖)
- [4. 跨语言接口](#4-跨语言接口)
- [5. 数据文件依赖](#5-数据文件依赖)
- [6. 环境配置建议](#6-环境配置建议)

---

## 1. 项目概述

本项目是一个基于深度强化学习（DRL）的风电场-储能联合优化调度系统。项目使用 **Python** 实现强化学习算法（DDPG、SAC、PPO、PER-DDPG），通过 **MATLAB Engine API** 调用 **WFSim（Wind Farm Simulator）** 风电场模拟器进行物理仿真。

**主要子项目：**

| 子项目 | 目录 | 说明 |
|--------|------|------|
| H-DDPG | 根目录 `H_*` | 双层 DDPG 架构（上层功率分配 + 下层储能控制） |
| S-DDPG | 根目录 `S_*` | PDO-DDPG（Policy-Decoupled Optimization）单层架构 |
| PER | 根目录 `PER_*` | 带优先经验回放（Prioritized Experience Replay）的 DDPG |
| RBB | [rbb/](rbb/) | 基于规则的 BESS（Rule-Based BESS） |
| PRRBB | [prrbb/](prrbb/) | PER + 分段奖励函数（Segmented Reward） |
| PRLE | [prle/](prle/) | PRRBB + LLM 增强（LLM-Enhanced Meta-Controller） |
| 未命名 | [未命名/](未命名/) | 旧版备份代码（与根目录结构基本一致） |

---

## 2. Python 依赖

### 2.1 核心第三方包

| 包名 | 版本参考 | 用途 | 使用模块 |
|------|---------|------|---------|
| **numpy** | ≥1.19 (Python 3.8) | 数值计算、数组操作、随机过程 | 所有 `.py` 文件 |
| **pandas** | ≥1.2 | CSV 数据读写、风速/电价数据处理 | `env.py`, `myfunc.py`, `S_ddpg.py`, `H_main.py`, `S_main.py`, `PER_main0.py`, `rbb_main.py`, `prrbb_main.py`, `prle_main.py`, 及分析脚本 |
| **torch** (PyTorch) | ≥1.7 (CPU/CUDA) | 深度神经网络（Actor-Critic 网络） | `model.py`, `H_ddpg.py`, `S_ddpg.py`, `PER_ddpg.py`, `prle_ddpg.py`, `util.py` |
| **matlab** + **matlab.engine** | MATLAB R2017b+ | MATLAB Engine API for Python — 调用 WFSim 风电场模拟器 | `myfunc.py`, `env.py`, `H_main.py`, `S_main.py`, `PER_main0.py`, `rbb_main.py`, `prrbb_main.py`, `prle_main.py`, `test.py` |
| **tensorboard** (torch.utils.tensorboard) | 随 PyTorch 安装 | 训练过程可视化与日志记录 | `H_main.py`, `S_main.py`, `PER_main0.py`, `rbb_main.py`, `prrbb_main.py`, `compare_wind_speed.py`, `extract_training_wind_days.py`, `compare_episode_reward.py` |

> **注意：** 项目中没有 `requirements.txt`、`setup.py`、`pyproject.toml` 或 `environment.yml`。以上依赖是从所有 `.py` 文件的 `import` 语句中提取的。

### 2.2 Python 标准库

| 模块 | 用途 |
|------|------|
| `os`, `sys`, `pathlib` | 文件系统操作、路径管理 |
| `argparse` | 命令行参数解析 |
| `copy` (deepcopy) | 模型深拷贝（target network 更新） |
| `pickle` (FALSE) | 仅用于 `from pickle import FALSE`（非标准用法） |
| `warnings` | 抑制警告输出 |
| `random` | 随机数生成、经验回放采样 |
| `collections` (deque, namedtuple) | 经验回放缓冲区数据结构 |
| `operator` | PER 优先级比较 |
| `bisect` | 电价分段函数查找 |
| `json` | LLM 阈值调整数据（PRLE 子项目） |
| `csv`, `io` | CSV 文件读写 |
| `math` | 数学运算 |

### 2.3 各模块依赖详情

#### 根目录核心模块

| 文件 | 说明 | 关键依赖 |
|------|------|----------|
| [H_main.py](H_main.py) | H-DDPG 训练主循环 | numpy, pandas, torch, matlab.engine, tensorboard |
| [S_main.py](S_main.py) | S-DDPG 训练主循环 | 同上 |
| [PER_main0.py](PER_main0.py) | PER-DDPG 训练主循环 | 同上 |
| [test.py](test.py) | 多算法综合测试脚本 | 同上（引用 sac, ppo 等缺失模块） |
| [model.py](model.py) | Actor-Critic 神经网络定义 | numpy, torch, torch.nn |
| [H_ddpg.py](H_ddpg.py) | 双层 DDPG Agent | numpy, torch, torch.nn, torch.optim |
| [S_ddpg.py](S_ddpg.py) | PDO-DDPG Agent | 同上 |
| [PER_ddpg.py](PER_ddpg.py) | PER-DDPG Agent（含 Lamarckian 层） | numpy, torch, torch.nn, torch.optim |
| [PER_buffer.py](PER_buffer.py) | 优先经验回放缓冲区 | numpy, PER_utils |
| [memory.py](memory.py) | RingBuffer + SequentialMemory 经验回放 | numpy, collections |
| [memory_reward.py](memory_reward.py) | 带奖励的经验回放 | numpy, collections |
| [random_process.py](random_process.py) | Ornstein-Uhlenbeck 随机过程（探索噪声） | numpy |
| [util.py](util.py) | 工具函数（软/硬更新、CUDA、输出目录） | torch, torch.autograd |
| [env.py](env.py) | 风电场-储能环境（Gym-like） | numpy, pandas, matlab, bisect |
| [myfunc.py](myfunc.py) | 奖励计算、动作检查、归一化 | numpy, pandas, matlab.engine |

#### 子项目模块

| 文件 | 说明 | 关键依赖 |
|------|------|----------|
| [rbb/rbb_main.py](rbb/rbb_main.py) | RBB 训练主循环 | numpy, pandas, torch, matlab.engine |
| [rbb/rbb_rule_bess.py](rbb/rbb_rule_bess.py) | 基于规则的 BESS 功率裁剪 | numpy |
| [rbb/rbb_myfunc.py](rbb/rbb_myfunc.py) | RBB 奖励函数（SOC 跟踪） | numpy, pandas |
| [rbb/rbb_test.py](rbb/rbb_test.py) | RBB 测试脚本（详细指标记录） | numpy, pandas, torch, matlab.engine |
| [prrbb/prrbb_main.py](prrbb/prrbb_main.py) | PRRBB 训练主循环 | numpy, pandas, torch, matlab.engine |
| [prrbb/prrbb_rule_bess.py](prrbb/prrbb_rule_bess.py) | PRRBB BESS 裁剪（同 RBB） | numpy |
| [prrbb/prrbb_myfunc.py](prrbb/prrbb_myfunc.py) | 分段奖励函数（风速×电价 9 区段） | numpy, pandas |
| [prrbb/sample_wind_percentiles.py](prrbb/sample_wind_percentiles.py) | 风速百分位采样工具 | csv, random, math, pathlib |
| [prle/prle_main.py](prle/prle_main.py) | PRLE 训练主循环（LLM 增强） | numpy, pandas, torch, matlab.engine |
| [prle/prle_ddpg.py](prle/prle_ddpg.py) | PRLE-DDPG Agent（扩展 SP-DDPG） | numpy, torch, torch.nn |
| [prle/prle_buffer.py](prle/prle_buffer.py) | PRLE 缓冲区（奖励重标记） | numpy, PER_utils |
| [prle/prle_llm.py](prle/prle_llm.py) | LLM 元控制器（窗口统计 + 阈值调整） | json, os |
| [prle/prle_myfunc.py](prle/prle_myfunc.py) | PRLE 动态风速阈值奖励函数 | numpy, pandas |

#### 分析/工具脚本

| 文件 | 说明 | 关键依赖 |
|------|------|----------|
| [compare_wind_speed.py](compare_wind_speed.py) | 风速数据对比（平滑 vs 加噪） | pandas, tensorboard |
| [compare_episode_reward.py](compare_episode_reward.py) | 多项目 Episode Reward 对比 (MAPE) | pandas, numpy, tensorboard |
| [test_compare_projects.py](test_compare_projects.py) | 多项目统一风力测试对比 | numpy, pandas, torch, matlab.engine |
| [test_simple.py](test_simple.py) | 单模型随机风速测试 | numpy, pandas, torch, matlab.engine |
| [extract_training_wind_days.py](extract_training_wind_days.py) | 训练风速日提取到 TensorBoard | pandas, numpy, tensorboard |
| [generate_day_sequence.py](generate_day_sequence.py) | 训练日随机序列生成器 | argparse, random |
| [prrbb/testcsvrecordpy.py](prrbb/testcsvrecordpy.py) | PRRBB 越限分析（P_F 按风速档） | pandas, numpy |

### 2.4 缺失源文件的编译缓存

以下 `.pyc` 文件存在于 `__pycache__/` 目录中，但对应的 `.py` 源文件已丢失（可能已被删除或移动）：

| 编译缓存 | 说明 | Python 版本 |
|----------|------|-------------|
| `sac.cpython-38.pyc` | SAC (Soft Actor-Critic) 算法 | 3.8 |
| `sac_S.cpython-38.pyc` | SAC-S 变体 | 3.8 |
| `sac_SP.cpython-38.pyc` | SAC-SP 变体 | 3.8 |
| `sac_SP2.cpython-38.pyc` | SAC-SP 第二版变体 | 3.8 |
| `ppo.cpython-38.pyc` | PPO (Proximal Policy Optimization) 算法 | 3.8 |
| `ae.cpython-38.pyc` | Autoencoder 模块 | 3.8 |
| `rule.cpython-38.pyc` | 基于规则的控制模块 | 3.8 |
| `model2.cpython-38.pyc` / `model3.cpython-38.pyc` | 备选网络架构 | 3.8 |
| `O_ddpg.cpython-38.pyc` / `O_env.cpython-38.pyc` | "O" 变体 DDPG/环境 | 3.8 |
| `W_ddpg.cpython-38.pyc` / `W_env.cpython-38.pyc` / `W_memory.cpython-38.pyc` | "W" 变体 DDPG/环境/记忆 | 3.8 |
| `S_ddpg2.cpython-38.pyc` / `S_ddpg2H.cpython-38.pyc` / `S_ddpg2L.cpython-38.pyc` | S-DDPG 第二版变体 | 3.8 |
| `S_ddpg3.cpython-38.pyc` / `S_ddpg_CL.cpython-38.pyc` / `S_ddpg_pan.cpython-38.pyc` | S-DDPG 第三版/课程学习/全景变体 | 3.8 |
| `S_env.cpython-38.pyc` | S 环境模块 | 3.8 |
| `H_ddpg_PER3.cpython-38.pyc` | H-DDPG PER 第三版 | 3.8 |
| `H_memory.cpython-38.pyc` | H-DDPG 记忆模块 | 3.8 |
| `LLM_myfunc.cpython-38.pyc` | LLM 相关函数模块 | 3.8 |
| `bin/__pycache__/ActorNetwork.cpython-36.pyc` | Actor 网络（旧版） | **3.6** |
| `bin/__pycache__/ActorNetwork.cpython-37.pyc` | Actor 网络（旧版） | **3.7** |
| `bin/__pycache__/array_add.cpython-37.pyc` | 数组操作辅助 | **3.7** |

> ⚠️ **警告：** `test.py` 和 `未命名/test.py` 中引用了 `sac`, `sac_S`, `sac_SP`, `ppo` 等模块，由于源文件缺失，这些导入将导致 `ModuleNotFoundError`。

---

## 3. MATLAB 组件

**MATLAB 文件总数：464 个 `.m` 文件**（含 `未命名/` 旧版备份），其中活跃使用约 **272 个**。

### 3.1 WFSim 风电场模拟器

**来源：** TU Delft (S. Boersma & B. Doekemeijer, 2017)  
**仓库：** <https://github.com/Bartdoekemeijer/WFSim>  
**版本：** 基于原始 WFSim，已针对本项目进行定制修改

WFSim 是一个基于 **2D URANS (Unsteady Reynolds-Averaged Navier-Stokes)** 方程的中保真度风电场模拟器。采用有限体积法（FVM）在交错网格上进行混合空间离散化，使用全隐式 Euler 格式进行时间推进。

**模拟流程：** 初始化网格 → 每个时间步内迭代至收敛（空间离散 → 湍流模型 → 时间离散 → 致动盘模型 → 边界条件 → 系统组装 → 稀疏求解 → 解映射）

**支持的模拟场景（定义于 `meshing.m`）：**

| 场景名称 | 涡轮数 | 转子直径 | 说明 |
|----------|--------|----------|------|
| `windTunnel` | 5 | 100 m | 风洞配置，5 台涡轮机一列排开 |
| `windTunnel_3turb` | 3 | 100 m | 3 涡轮机风洞变体 |
| `windTunnel_10turb` | 10 | 100 m | 10 涡轮机风洞变体 |
| `clwindcon_3turb` | 3 | 178.3 m | CL-Windcon 项目场景 |
| `clwindcon_9turb` | 9 | 178.3 m | CL-Windcon 项目场景 |
| `clwindcon_80turb` | 80 | 178.3 m | 工业级大规模风电场 |

**模型变体：**
- **非线性模型**（默认）：完整的非线性 URANS 方程
- **线性化模型**（`Linearversion=1`）：在当前运行点解析线性化，用于模型预测控制
- **投影方法**（`Projection=1`）：通过投影消除连续性方程以降低状态维度

#### 核心模块 (`bin/core/`)

| 文件 | 功能 |
|------|------|
| [WFSim.m](WFSim.m) | **主入口函数** — 初始化 + 时间步进循环 |
| [WFSim_timestepping.m](bin/core/WFSim_timestepping.m) | 时间步进主循环 |
| [InitWFSim.m](bin/core/InitWFSim.m) | 风电场初始化（网格、涡轮机、边界条件） |
| [meshing.m](bin/core/meshing.m) | 计算网格生成（支持多种风电场布局） |
| [Actuator.m](bin/core/Actuator.m) | 致动盘模型（Actuator Disk）— 涡轮机推力/功率计算 |
| [Actuator_Joeri.m](bin/core/Actuator_Joeri.m) | 致动盘模型变体（Joeri 版本） |
| [Dynamical.m](bin/core/Dynamical.m) | 动态系统矩阵组装 |
| [SpatialDiscr_Hybrid.m](bin/core/SpatialDiscr_Hybrid.m) | 混合空间离散化 |
| [BoundaryConditions.m](bin/core/BoundaryConditions.m) | 边界条件设置 |
| [Turbulence.m](bin/core/Turbulence.m) | 湍流模型 — 基于 Prandtl 混合长度理论，含 5 种变体（WFSim1-WFSim5） |
| [Make_Ax_b.m](bin/core/Make_Ax_b.m) | 线性系统 `Ax = b` 构建 |
| [Computesol.m](bin/core/Computesol.m) | 线性系统求解 |
| [ComputeDerivatives.m](bin/core/ComputeDerivatives.m) | 导数计算 |
| [ComputeDerivatives_Joeri.m](bin/core/ComputeDerivatives_Joeri.m) | 导数计算变体 |
| [Compute_B1_B2_bc.m](bin/core/Compute_B1_B2_bc.m) | B1/B2 矩阵 + 边界条件计算 |
| [Solution_space.m](bin/core/Solution_space.m) | 解空间投影 |
| [ConstructMu.m](bin/core/ConstructMu.m) | 构造 μ 向量 |
| [ConstructLmu.m](bin/core/ConstructLmu.m) | 构造 Lμ 向量 |
| [MapSolution.m](bin/core/MapSolution.m) | 解映射 |
| [Updateboundaries.m](bin/core/Updateboundaries.m) | 边界更新 |
| [MakingSparseMatrix.m](bin/core/MakingSparseMatrix.m) | 稀疏矩阵构建 |
| [MakingSparseMatrixl.m](bin/core/MakingSparseMatrixl.m) | 稀疏矩阵构建（L 型） |
| [MakingSparseMatrixlo.m](bin/core/MakingSparseMatrixlo.m) | 稀疏矩阵构建（LO 型） |
| [vec.m](bin/core/vec.m) / [expvec.m](bin/core/expvec.m) | 向量操作工具 |
| [downloadLESdata.m](bin/core/downloadLESdata.m) | LES 参考数据下载 |

#### 分析模块 (`bin/analysis/`)

| 文件 | 功能 |
|------|------|
| [WFSim_animation.m](bin/analysis/WFSim_animation.m) | 流场动画生成 |
| [WFSim_linear_animation.m](bin/analysis/WFSim_linear_animation.m) | 线性化模型动画 |
| [Animation.m](bin/analysis/Animation.m) | 通用动画工具 |
| [CompareLinearWithNonlinear.m](bin/analysis/CompareLinearWithNonlinear.m) | 线性 vs 非线性模型对比 |
| [CompareNonlinearWithPALM.m](bin/analysis/CompareNonlinearWithPALM.m) | 非线性 vs PALM 对比 |
| [CheckLinearmodelNumerical.m](bin/analysis/CheckLinearmodelNumerical.m) | 线性模型数值检查 |
| [SimLinearWithProjectionModel.m](bin/analysis/SimLinearWithProjectionModel.m) | 带投影的线性仿真 |
| [SimLinearWithoutProjectionModel.m](bin/analysis/SimLinearWithoutProjectionModel.m) | 无投影的线性仿真 |
| [vaf.m](bin/analysis/vaf.m) | 方差占比（Variance Accounted For） |
| [hline.m](bin/analysis/hline.m) / [vline.m](bin/analysis/vline.m) | 水平/垂直线绘制 |

#### 入口/封装文件

| 文件 | 功能 |
|------|------|
| [WFSim.m](WFSim.m) | MATLAB 直接调用入口 `sol_array = WFSim(Wind, Control)` |
| [WFSim_port.m](WFSim_port.m) | Python 调用封装 `power_out = WFSim_port(w_s, w_a, a, p)` |
| [WFSim_addpaths.m](WFSim_addpaths.m) | 路径初始化（添加所有子目录到 MATLAB path） |
| [solve_PARK.m](solve_PARK.m) | 基于 `fmincon` 的最优推力系数求解（PARK 模型） |
| [myfun.m](myfun.m) | `solve_PARK.m` 的优化目标函数 |

### 3.2 详细信息

各 MATLAB 组件及第三方库的详细信息请参见 [§3.1 WFSim 核心模块](#31-wfsim-风电场模拟器)、[§3.4 引用的外部模型](#34-引用的外部模型) 和 [§3.5 第三方 MATLAB 库清单](#35-第三方-matlab-库清单)。

### 3.3 MATLAB 工具箱依赖

| MATLAB 工具箱 | 依赖函数/功能 | 使用位置 | 必要性 |
|--------------|-------------|---------|--------|
| **Optimization Toolbox** | `fmincon` (约束非线性优化) | [solve_PARK.m](solve_PARK.m) | **必需**（PARK 模型优化） |
| **Symbolic Math Toolbox** | `syms`, `solve` (符号计算) | [bin/core/expvec.m](bin/core/expvec.m) | **必需**（涡轮机周围指数网格加密） |
| **Image Processing Toolbox** | `imresize` | `libraries/export_fig/` | 可选（图形导出时可回退到手动缩放） |
| **MATLAB Engine API for Python** | `matlab.engine` Python 包 | Python 侧调用 WFSim | **必需**（跨语言接口） |

> **最低 MATLAB 版本要求：** R2017b 或更高（支持 MATLAB Engine API for Python）

### 3.4 引用的外部模型

| 模型/工具 | 类型 | 说明 |
|-----------|------|------|
| **PALM** (Parallelized LES Model) | 大涡模拟代码 | `CompareNonlinearWithPALM.m` 加载 PALM 生成的 LES 参考数据，使用 RMSE 和 VAF 指标验证 WFSim 精度 |
| **Jensen/PARK 尾流模型** | 工程尾流模型 | `solve_PARK.m` 和 `myfun.m` 中使用，k=0.08 尾流衰减常数，用于 DRL 对比基线 |
| **LES 参考数据** (TU Delft) | 预计算仿真数据 | `downloadLESdata.m` 通过 `websave` 从 TU Delft 服务器下载 `.mat` 格式 LES 数据，存储于 `data_LES/` |

### 3.5 第三方 MATLAB 库清单

| 库名 | 来源 | 文件数 | 用途 |
|------|------|--------|------|
| **export_fig** | Oliver Woodford / Yair Altman (File Exchange) | 16 `.m` | 高质量图形导出（PDF/EPS/PNG），需 Ghostscript |
| **mexnc** | NetCDF MEX 接口 | ~94 `.m` | 底层 NetCDF C 库的 MATLAB 绑定，含常量定义和 MEX 构建脚本 |
| **snctools** | NetCDF 高层工具箱 | ~166 `.m` | 高层 NetCDF 读写函数（`nc_varget`, `nc_varput`, `nc_info`, `nc_dump` 等） |
| **sparse_null** (LREM_SOLVE) | Pawel Kowal (2006) | 3 `.m` | 稀疏矩阵零空间计算（`spspaces.m`, `luq.m`），用于连续性方程投影 |
| **natsort** | Stephen Cobeldick (File Exchange #47433) | 1 `.m` | 自然顺序字符串排序 |
| **numSubplots** | Rob Campbell (2010) | 1 `.m` | 最优子图布局计算 |

---

## 4. 跨语言接口

```
┌──────────────────────────────────────────────────────────┐
│                      Python 层                            │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐               │
│  │ RL Agent │  │   env.py  │  │ myfunc.py │               │
│  │ (PyTorch)│  │ WindBattery│  │ Cal_Rew() │               │
│  └────┬─────┘  └─────┬─────┘  └─────┬─────┘               │
│       │              │              │                      │
│       │     ┌────────▼───────┬──────▼─────────┐           │
│       │     │  matlab.engine │  matlab.engine │           │
│       │     │  .start_matlab │  .start_matlab │           │
│       │     └────────┬───────┴──────┬─────────┘           │
├───────┼──────────────┼──────────────┼─────────────────────┤
│       │     ┌────────▼──────────────▼──────┐              │
│       │     │      MATLAB 层               │              │
│       │     │  ┌────────────────────────┐  │              │
│       │     │  │  WFSim_port.m          │  │              │
│       │     │  │  wind_speed, wind_angle │  │              │
│       │     │  │  c_prime, phi           │  │              │
│       │     │  └───────────┬────────────┘  │              │
│       │     │              ▼               │              │
│       │     │  ┌────────────────────────┐  │              │
│       │     │  │  WFSim.m               │  │              │
│       │     │  │  2D Navier-Stokes CFD   │  │              │
│       │     │  │  Actuator Disk Model    │  │              │
│       │     │  └───────────┬────────────┘  │              │
│       │     │              ▼               │              │
│       │     │  ┌────────────────────────┐  │              │
│       │     │  │  turbine.power (输出)   │  │              │
│       │     │  └────────────────────────┘  │              │
│       │     └──────────────────────────────┘              │
└──────────────────────────────────────────────────────────┘
```

**调用流程：**
1. Python RL Agent 输出控制动作（偏航角 φ, 推力系数 C'T）
2. [env.py](env.py) 通过 `matlab.engine` 调用 [WFSim_port.m](WFSim_port.m)
3. [WFSim_port.m](WFSim_port.m) 添加路径后调用 `WFSim(Wind, Control)`
4. WFSim 运行 CFD 仿真，返回各涡轮机功率
5. Python 侧读取 `sol_array` 中的 `turbine.power` 计算奖励

---

## 5. 数据文件依赖

| 数据 | 路径 | 格式 | 说明 |
|------|------|------|------|
| LES 参考数据 | [data_LES/](data_LES/) | `.mat` | 大涡模拟湍流入流数据（2 涡轮/6 涡轮场景） |
| 风速数据 | `FFC_Mast_Wind_Data/` | `.csv` | 风速测量数据（`FFC_speed_*.csv`, `FFC_speed_noise_*.csv`） |
| 电价数据 | `price_FFC_2019.csv` | `.csv` | 2019 年电价数据 |
| 固定日序列 | `fixed_day_sequence.csv` | `.csv` | 训练用固定随机日序列（由 `generate_day_sequence.py` 生成） |
| 测试场景 | [test_scenarios.csv](test_scenarios.csv) | `.csv` | 测试场景配置 |

---

## 6. 环境配置建议

### Python 环境

```bash
# 推荐使用 Python 3.8（与 __pycache__ 编译版本一致）

# 创建虚拟环境
conda create -n hsrl python=3.8
conda activate hsrl

# 安装核心依赖
pip install numpy pandas matplotlib

# 安装 PyTorch（CPU 版本）
pip install torch torchvision torchaudio

# 或安装 PyTorch（CUDA 版本，如 CUDA 11.1）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu111

# 安装 MATLAB Engine API
cd "C:\Program Files\MATLAB\R20xxx\extern\engines\python"
python setup.py install
```

### MATLAB 环境

- **版本：** MATLAB R2017b 或更高
- **必需工具箱：** Optimization Toolbox、Symbolic Math Toolbox
- **可选工具箱：** Image Processing Toolbox（仅 `export_fig` 使用，无此工具箱也可正常运行）
- **工作目录设置：** 首次运行时需执行 `run('WFSim_addpaths.m')` 以添加所有子目录到 MATLAB 路径

### 已知问题

1. **缺失源文件：** `sac.py`, `sac_S.py`, `sac_SP.py`, `sac_SP2.py`, `ppo.py`, `ae.py`, `rule.py`, `model2.py`, `model3.py` 等文件的源代码已丢失，仅保留 `.pyc` 编译缓存。`test.py` 会因导入这些模块而报错。
2. **旧版 Python：** `bin/__pycache__/` 中部分文件使用 Python 3.6/3.7 编译，说明项目曾迁移过 Python 版本。
3. **硬编码路径：** `prrbb/sample_wind_percentiles.py` 中使用绝对路径 `c:\Users\yigul\Desktop\...` 指向 FFC 风速数据，迁移到其他机器需修改。
4. **MATLAB 引擎多实例：** `myfunc.py` 和多个 `*_main.py` 文件中各自调用 `matlab.engine.start_matlab()`，可能创建多个 MATLAB 实例，需注意资源共享和内存管理。
5. **`from pickle import FALSE`：** 项目中大量使用此非常规导入方式获取布尔值，实际效果等同于 `False = False`（pickle 模块无 `FALSE` 属性），运行时会产生 `ImportWarning`。
