# 本项目的协作约定（HSRL）

本文件会被 DeepSeek Harness 在每次新会话开始时自动注入到对话中。
涉及代码修改时必须遵守以下协议。

## 强制规则：改动代码前先确认备份策略

当用户请求涉及代码修改（改 bug、加功能、重构、调参、改配置、批量改动文件等）时，
在执行任何 edit / write 之前，必须先：

1. 检查工作区状态：`git status --short`
2. 若本次会话尚未确认备份策略，用 `ask_user_question` 工具询问用户，选项为：
   - 每次修改前自动 git 快照（推荐）
   - 只在我明确要求时备份
   - 本次会话不备份
   用户回答后，本次会话全程遵守该选择。
   用户在对话开头已明确说明备份偏好的，直接遵守，不再重复询问。

纯问答、只读任务（解释代码、查 git 历史、检索资料、看文件）不触发询问。

## 快照与提交协议

- **修改前快照**（策略为"每次修改前自动快照"时）：
  若 `git status --short` 非空，先执行
  `git add -A` 然后 `git commit -m "snapshot: <接下来要做的改动>"`；
  工作区干净则跳过。
- **改动完成即提交**：完成一次连贯改动后，立即执行
  `git add -A` 然后 `git commit -m "<type>: <改动简述>"`，
  type 取 feat / fix / refactor / docs / chore。
  不要等用户催促——每个完成态都必须成为可回退的版本。
- 提交信息一行写清"改了什么、为什么"；同一目标的多个文件改动合并为一次提交。

## 回退约定

- 单文件回退：`git restore -- <file>`
- 撤销整次提交（保留历史）：`git revert <commit>`
- 查看版本：`git log --oneline`；查看改动：`git diff`
- 回退前若工作区有未提交改动，先按上文做一次快照。
- 除非用户明确要求，禁止 `git reset --hard`、强制推送、删除 `.git` 或改写历史。

## 目录约定：`agentutil/` 放所有"项目无关"的自建文件

**设定（每次会话都必须遵守）**：

- **所有项目无关的临时/自建文件一律放进 `agentutil/`**，包括：一次性脚本、
  探针（probe）、分析工具、回归测试（test_*）、临时草稿、调试用的对比脚本。
  **不要**散落在仓库根目录或各项目子目录（`td3*/`、`per*/`、`prrbb/`、`prle/`、`rbb/` …）。
- **必须留在根目录**的是"生产依赖"：`hsrl_cost.py` / `hsrl_agent_init.py` /
  `hsrl_warmup.py`（9 个 main 直接 import）、`test_scenario.py`（7 个 `*_myfunc.py`
  的 `nine_case_test` import）、`myfunc.py` / `util.py` / `env.py` / `model.py` /
  `PER_buffer.py` / `*_ddpg.py` / `*_main.py`，以及根目录的评测主脚本
  `test_five_days.py`、`test_simple.py`、`test_compare_projects.py`、
  `compare_*.py`、`extract_training_wind_days.py`、`generate_day_sequence.py`。
- **方向性约束**：`agentutil/` 可以 import 项目模块；**项目代码不得 import
  `agentutil/`**（否则训练路径会被工具目录绑架）。
- **新脚本必须自带路径 bootstrap**（可被任意 CWD、任意路径调用）：

  ```python
  import os
  import sys

  _HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
  ROOT = os.path.dirname(_HERE)                          # 仓库根
  os.chdir(ROOT)                                         # 相对数据/输出路径以仓库根为基准
  sys.path.insert(0, ROOT)                               # 项目模块
  sys.path.insert(0, _HERE)                              # agentutil 内互相 import
  ```

  禁止 `ROOT = dirname(__file__)`（那是 agentutil 自己）与硬编码
  `C:\Users\...` 绝对路径。
- 目录内脚本清单与运行前置见 `agentutil/README.md`。

## 已忽略的内容

`output/`、`未命名/`、`FFC_Mast_Wind_Data/`、大体积 CSV、`__pycache__/` 等
由 `.gitignore` 排除，不参与版本控制。若用户要求纳入，先修改 `.gitignore` 再提交。
`agentutil/` **纳入版本控制**（其中的回归测试是正式资产）；只有其运行产物
（`*.csv`/`*.png`/日志）会被 `.gitignore` 排除。

## 关联 skill

操作细节与命令速查见 skill `git-backup`；用户输入 `/git-backup` 可随时手动触发一次快照。
