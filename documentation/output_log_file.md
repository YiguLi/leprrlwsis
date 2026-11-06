# 训练输出简化（`--log_file` / `--progress`）—— 12 个方法

> 实施日期：2026-11-04 ｜ 实现：`run_io.py` ｜ 契约：`python agentutil/test_run_io.py`（32 项）
> 迁移脚本（已执行，保留作审计）：`python agentutil/port_run_io.py [--apply]`

## 1. 为什么

`aifflor` / `testpg` 早就有 `--log_file`：逐行的 `--H-- / r1 / c1 / price …` 写进
`<output>/train.log`，控制台只留一行进度（步/秒 + ETA）。其余用 **MATLAB/WFSim 环境**的方法
没有这套东西，一次几十万步的训练会把终端刷出几十万行，既拖慢终端，也看不清进度。

这次把那段逻辑抽成 `run_io.py`，给 12 个原先没有该开关的 main 接上：

| 项目 | main |
|---|---|
| `td3per` | `td3per/td3per_main.py` |
| `td3prrbb` | `td3prrbb/td3prrbb_main.py` |
| `td3prle` | `td3prle/td3prle_main.py` |
| `td3prrlrb` | `td3prrlrb/td3prrlrb_main.py` |
| `td3perrb` | `td3perrb/td3perrb_main.py` |
| `td3leprrlrb` | `td3leprrlrb/td3leprrlrb_main.py` |
| `td3perseg` | `td3perseg/td3perseg_main.py` |
| `prle` | `prle/prle_main.py` |
| `prrbb` | `prrbb/prrbb_main.py` |
| `singleaif` | `singleaif/singleaif_main.py` |
| `yawflor` | `yawflor/yawflor_main.py` |
| `aifflorpave` | `aifflorpave/aifflorpave_main.py` |

（`aifflor` / `testpg` 本来就有 `--log_file`，未改动。）

## 2. 用法

```powershell
# 默认：与改造前**逐位相同**（详细输出照样打控制台）
& $PY td3prrbb\td3prrbb_main.py ...

# 简化：详细输出 → output/<env>-runN/train.log，控制台只留进度行
& $PY td3prrbb\td3prrbb_main.py --log_file 1 ...

# 简化 + 每 1000 步一行进度（默认 500）
& $PY td3prrbb\td3prrbb_main.py --log_file 1 --progress_every 1000 ...

# 只看日志、连进度行也不要
& $PY td3prrbb\td3prrbb_main.py --log_file 1 --progress 0 ...
```

控制台效果（与 `aifflor` 完全同款）：

```text
[log] 详细输出 → output\TD3PRRBB-run20\train.log（控制台只显示进度与汇总）
  [TD3PRRBB-run20]    1500/200000 (  0.8%)   14.6 步/秒  已用    2 min  剩余约  227 min
  [TD3PRRBB-run20]    2000/200000 (  1.0%)   14.6 步/秒  已用    2 min  剩余约  225 min
```

## 3. 语义（三个 flag）

| flag | 默认 | 作用 |
|---|---|---|
| `--log_file` | **0** | 0 = 不动输出（历史行为）；1 = `sys.stdout` **与 fd 1** 双重定向到 `<output>/train.log` |
| `--progress` | 1 | 1 = 每 `--progress_every` 步打一行进度（**只在 `--log_file 1` 时生效**） |
| `--progress_every` | 500 | 进度行间隔（环境步） |

**为什么要 fd 级重定向**：`WFSim.m:110/133` 的
`disp('Performing N forward simulations..')` 是 **MATLAB 侧**的输出，MATLAB engine 把它
直接写到进程的 **fd 1**，不经过 Python 的 `sys.stdout`；只重绑 `sys.stdout` 抓不到，
所以会"仍然每步刷屏"。`run_io` 现在会：先 `os.dup(1)` 留一个控制台副本 → `os.dup2(log_fd, 1)`
→ 所有写 fd 1 的东西（MATLAB、C 扩展）都进 `train.log`；进度行则写到那个控制台副本
（用控制台自己的编码，避免 cp936 下中文乱码）。`close()` 时 `os.dup2` 恢复 fd 1。

- 进度行写到真实控制台副本，不受重定向影响；
- 总步数在 `train(...)` 调用前由 `run_io.set_total(args.train_iter_H)` 设定
  （`args.train_iter_H` 在 `get_output_folder()` 之后才会被 `1000*max_episode_length`
  或 `--max_steps` 定稿，所以不能在建目录时就取）；
- 重定向失败（权限/磁盘）时自动退回原行为并打印原因（与 aifflor 一致）。

**默认值为什么取 0**：这批 main 已有大量历史 run 与外部脚本在解析控制台输出；
新增开关默认关闭 ⇒ 改造对既有流程"零行为变化"，需要时显式 `--log_file 1`。
若希望像 aifflor 一样默认开启，把三个 main 里的 `default=0` 改成 `1` 即可
（`agentutil/test_run_io.py` 会跟着失败，提醒你同步断言）。

## 4. 契约与回归

`python agentutil/test_run_io.py`（32 项）：

- **[A]** `log_file=0` 不重定向、不打进度；`log_file=1` 重定向、每 N 步一行、
  行内包含 `step/total`、run 名；`close()` 恢复 stdout；
- **[A2]** 重定向期间的 `print` 确实落进 `train.log`；`progress=0` 时不打进度行；
- **[B]** 12 个 main 静态检查：三个 flag 都在、`run_io.setup/tick/set_total` 都接上、
  `--log_file` 默认 0。
