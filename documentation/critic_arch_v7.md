# v7 critic：动作与状态首层拼接（input → 400 → 300 → 400 → 1）

> 实施日期：2026-11-02（critic）｜ 2026-11-03 追加：**v7 的 actor 也改为 400-300-400**（见 §1.2）
> 契约测试：`python agentutil/test_critic_arch_v7.py`（80 项）
> 离线性能对照：`agentutil/critictest`（C1 五步块基准），见 §4
> 结构定义：`model.Critic(arch='v7')`，宽度常量 `model.V7_DIMS = (400, 300, 400)`；
> actor 宽度 `model.V7_ACTOR_DIMS = (400, 300)`

## 1. 口径

按用户指定，**action 与 state 在最开始直接拼接**，再接 400 → 300 → 400 → 1：

```text
input = cat(state, action)          # ns + na'（na' = na，或 na+1 开 --critic_feat_power）
      → 400 → 300 → 400 → 1
```

与既有结构的差别：

| | 动作入口 | 宽度 | 参数量（ns=4, na=6） |
|---|---|---|---|
| `v1` | fc2（第 3 层）拼接 | 4→400→300→[306]→80→50→1 | 14 441（hidden 16/100） |
| `v1af` | **首层**拼接 | [10]→4·hidden1→hidden2→80→50→1 | 19 385（16/100）；152 881（100/300） |
| `v6` | 双流，末段合并 | s→32→32 ∥ a→64→64 → 96→128→256→256→128→1 | 150 081 |
| **`v7`** | **首层**拼接 | **[10]→400→300→400→1** | **245 501** |

- `v7` 与 `v1af` 的动机相同（动作一开始就参与非线性，避免"先压状态再拼动作"的信息瓶颈），
  但 `v1af` 的默认宽度是 `4·hidden1`（默认 64），调到 400 宽首层时整网是
  400→300→**80→50**，多两层窄尾；v7 是显式的 400/300/400 规格。
- 宽度**写死在 `V7_DIMS`**（与 v5/v6 同口径）：`--hidden1/--hidden2` 对 v7 不生效。

### 逐层形状（ns=4, na=6）

| 层 | 形状 | 参数 | 占比 |
|---|---|---|---|
| h1 | 10→400 | 4 400 | 1.8% |
| h2 | 400→300 | 120 300 | 49.0% |
| h3 | 300→400 | 120 400 | 49.0% |
| out | 400→1 | 401 | 0.2% |
| **合计** | | **245 501** | |

其它输入维（契约测试实测一致）：

| 场景 | 首层输入 | 参数量 |
|---|---|---|
| 生产 H（`obs` 4 维 + 6 维动作） | 10 | 245 501 |
| `--critic_wind5 1`（8 维 state） | 14 | 247 101 |
| 下层 L（5 维 state + 1 维动作） | 6 | 243 901 |
| `--critic_feat_power 1`（动作 +1） | 11 | 245 901 |

### 1.2 actor 也改为 400-300-400（2026-11-03）

`model.Actor` 的形状恒为 `state → hidden1 → hidden2 → hidden1 → action`
（输出层 `fc4` 不做 squash），所以 v7 下把 `hidden1/hidden2` 设成
`model.V7_ACTOR_DIMS = (400, 300)` 就得到 **state→400→300→400→action**：

| 层 | 形状（ns=4, na=6） | 参数 |
|---|---|---|
| fc1 | 4→400 | 2 000 |
| fc2 | 400→300 | 120 300 |
| fc3 | 300→400 | 120 400 |
| fc4（输出，±3e-3 初始化） | 400→6 | 2 406 |
| **合计** | | **245 106**（旧 16/100 只有 3 445） |

接线的三条边界：

1. **只对 H 层生效**：`aifflor_main` 只给 `agent_H` 传 `actor_arch='v7'`
   （`SP_TD3(..., actor_arch='v7')`）；`agent_L` 不传 ⇒ L 的 1 维 BESS actor
   仍是 `--hidden1/--hidden2`（16/100，3 445 参数），避免 1 维动作网被放大 100 倍。
2. `--hidden1/--hidden2` 在 v7 下**只作用于 L 层**；H actor 宽度写死在 `V7_ACTOR_DIMS`。
3. ⚠ **checkpoint 形状变了**：2026-11-03 之前训练的 v7 run（如 `AIFFLOR-V7-run20`）
   actor 是 16/100，**不能**直接加载进新 actor（`load_state_dict` 会报形状不符）；
   续训/评测旧 run 需要按旧宽度重建 actor。

## 2. 接线与可恢复性

- 层名 `h1/h2/h3/out`：输出层叫 **`out`** ⇒ `hsrl_agent_init._out_head` 能认到它
  （bias 初始化不会静默 no-op，契约测试 E 段）。
- `model.infer_arch` 用 `h1.weight` 唯一识别 v7（v7 没有 `s1/fc1`）⇒
  `build_critic_from_sd`、`td3per.SP_TD3.load_weights` 都能从 checkpoint 自动重建；
  `feat_from_sd` 按 `h1.weight.shape[1] == ns + na (+1)` 判 `--critic_feat_power`。
- `aifflor_main.py --critic_arch v7`：choices 已含 v7，**默认仍是 v6**（不改生产口径）；
  `--critic_action_first 1` 仍优先映射到 `v1af`。
- `aifflor_ddpg.SP_TD3` 把 `args.critic_arch` 透传给 4 个 critic（+4 个 target）；
  v7 时 H 层的 `agent_H = SP_TD3(..., actor_arch='v7')` 会把 actor/actor_target
  建成 400-300-400，并在启动日志打印
  `[actor] v7 ⇒ actor 与 critic 同宽: state→400→300→400→action`。

## 3. 静态性能（契约测试实测）

初始化（ns=4, na=6；state~U(0,1)，action~U(0,0.5)，2000 样本，`out_init=3e3`）：

| 结构 | 参数 | 初始 Q sd | 逐层激活 sd / 死单元% |
|---|---|---|---|
| v6 | 150 081 | ~0 | m1 0.046/45% · m2 0.034/48% · m3 0.025/47% · m4 0.026/46% |
| **v7** | **245 501** | **~1e-4** | h1 0.108/38% · h2 0.057/37% · h3 0.031/43% |

- v7 各层激活**不塌陷、无整层死单元**；初始 Q sd 与 v1/v6 同量级（都由 `out_init=3e3` 主导）。
- 动作**只在首层进入**：把 `h1` 的动作列置零后 `Q` 与 `a` **严格无关**（差值 0.0，
  不是"很小"）；状态列同理 ✓。
- 反向：全部参数梯度存在、有限、非恒零。

前向耗时（CPU，batch=256，200 次 `no_grad`；同机多次测量有 ±40% 抖动，取决于后台负载）：

| 结构 | ms/次 |
|---|---|
| v1 (16/100) | 0.33–0.43 |
| v1af (16/100) | 0.38–0.55 |
| v6 | 1.6–3.1 |
| v7 | 2.6–3.0 |

⇒ 与 v6 同量级：参数量多 63%，但每层都是 ≤400 的窄矩阵，
30k 步离线训练实测 **454.5 s**（v6 同机同协议 **545.2 s**）。

## 4. 离线性能：C1 五步块基准（v6 vs v7）

**协议**（与 `agentutil/critictest/README_c1.md` 完全一致，只换 `--arch`）：

```powershell
$PY = 'C:\Users\yigul\anaconda3\envs\llmenhancedwsis\python.exe'
& $PY agentutil\critictest\train_c1.py `
    --data agentutil\critictest\data\aifflor_ua_f5_scalar.npz `
    --out  agentutil\critictest\results\c1_obs_v6ctl_log `
    --state obs --arch v6 --target logc1 --steps 30000
& $PY agentutil\critictest\train_c1.py `
    --data agentutil\critictest\data\aifflor_ua_f5_scalar.npz `
    --out  agentutil\critictest\results\c1_obs_v7_log `
    --state obs --arch v7 --target logc1 --steps 30000
```

- 数据：40k/5k/5k 训练/验证/测试 + 3k real；state=6 维 `obs`（`P_G_prev,u1..u5`）；
  动作 1 维；目标 `log1p(C1)`；Adam 1e-3 余弦到 1e-4；bsize 256；seed 0。
- v6 对照是**本机重跑**的，与历史 `c1_obs_v6_log` 的数字**逐位一致**
  （R² 0.986637 / RMSE 4.6145 / 梯度 r 0.9595）⇒ 改动没有动到 v6 的数值路径。

### 4.1 成本精度（5 000 test / 3 000 real）

| 指标 | v6 | **v7** | 变化 |
|---|---:|---:|---|
| test R² | 0.9866 | **0.9920** | +0.0054 |
| test RMSE [MW] | 4.615 | **3.566** | −23% |
| test MAE [MW] | 1.723 | **0.986** | −43% |
| test MAPE | 8.26% | **3.95%** | 减半 |
| test log1p R² | 0.9932 | **0.9981** | +0.0049 |
| real R² | 0.9692 | **0.9887** | +0.0195 |
| real RMSE [MW] | 7.341 | **4.445** | −39% |

折点检出（`C1>15`）：AUC 0.9998 vs 0.9986，**F1 0.9874 vs 0.9586**（Prec 0.990 / Rec 0.985）。

### 4.2 行为梯度 `∂C1/∂a`（test 400 点，δ=0.01，autograd vs 物理中心差分）

| 指标 | v6 | **v7** |
|---|---:|---:|
| pooled pearson r | 0.9595 | **0.9815** |
| slope（拟合斜率，理想 1） | 0.8923 | **0.9784** |
| sign agree | 0.8869 | 0.8895 |
| 余弦 mean | 0.7450 | **0.7550** |
| 余弦 > 0.8 比例 | 87.25% | **87.75%** |
| 余弦 < 0（反向）比例 | 12.75% | **12.25%** |

分组（smooth = 未触发 `PF>3`，hinge = 触发）：

| 组 | 指标 | v6 | v7 |
|---|---|---:|---:|
| smooth (n=252) | pearson r | **0.9174** | 0.8752 |
| | slope | **1.0134** | 1.1440 |
| | sign agree | 0.8382 | **0.8672** |
| | 余弦 mean | 0.6349 | **0.6984** |
| | 余弦 < 0 | 18.25% | **15.08%** |
| hinge (n=148) | pearson r | 0.9554 | **0.9805** |
| | slope | 0.8919 | **0.9778** |
| | sign agree | **0.9662** | 0.9257 |
| | 余弦 mean | **0.9324** | 0.8514 |
| | 余弦 < 0 | **3.38%** | 7.43% |

### 4.3 读法（结论）

1. **成本拟合全面更好**：test/real 的 R²、RMSE、MAE、MAPE 全部改善，
   real 上 RMSE 从 7.34 降到 4.44（−39%），折点 F1 0.959→0.987。
   这与"动作一开始就和状态一起进非线性"的预期一致：`v6` 的动作要先被压成 64 维
   `ha` 再合并，`u·g(a)` 这类乘性结构要跨支路合成；v7 的首层就能直接做乘性组合。
2. **整体梯度相关性更好、标定更准**：pooled r 0.960→0.982，slope 0.892→0.978
   （v6 系统性低估斜率约 11%，v7 基本无偏），余弦均值略升。
3. **但 smooth 组与 hinge 组的取舍不同**：
   - smooth（低波动区）：v7 的余弦 0.635→0.698、反向比例 18.3%→15.1%、符号一致率升，
     但**相关 r 下降、斜率过冲**（1.01→1.14）——v7 在平滑区方向更一致、幅度偏大。
   - hinge（`PF>3` 折点区）：v7 的 r/斜率更好，但**余弦与符号一致率下降**
     （0.932→0.851、0.966→0.926，反向 3.4%→7.4%）——折点附近的**方向**略不如 v6。
   ⇒ 如果控制最关心"折点区的方向一致性"，v6 仍有优势；若关心成本估计精度与整体梯度标定，
   v7 更好。二者不是全面碾压关系。
4. **没测的东西（重要）**：上面是**离线监督**基准，动作只有 1 维（标量 farm-level），
   与生产 H critic 的 **6 维动作**（5 台风机 + g）不同；也没有跑 FLORIS/WFSim 的闭环。
   v7 在 6 维动作下的表现、以及闭环收益/成本，必须用一次完整训练确认
   （`python aifflor/aifflor_main.py --hsrl --critic_arch v7 ...`）。

## 5. 复现与回归

```powershell
$PY = 'C:\Users\yigul\anaconda3\envs\llmenhancedwsis\python.exe'

# 1) 结构契约（119 项：参数量/前向/反向/动作入口/infer_arch/bias 接线/CLI/
#    SP_TD3 端到端/9 个方法的默认 v7 接线 + 评测侧 arch 识别）
& $PY agentutil\test_critic_arch_v7.py

# 2) 两个 run 并排对比（读 metrics.json）
& $PY agentutil\critictest\compare_c1_arch.py `
    v6=agentutil\critictest\results\c1_obs_v6ctl_log `
    v7=agentutil\critictest\results\c1_obs_v7_log
```

产物（`c1_obs_v7_log/`、`c1_obs_v6ctl_log/`）含 `c1_best.pt` / `metrics.json` /
`summary.txt` / `history.csv` / `plots/`，均被 `.gitignore` 排除。

## 6. 2026-11-04：v7 成为**所有方法**的默认结构

用户口径："v7 是极为成功的结构，将所有方法的神经网络结构改为默认 v7"。

| 方法 | 改动 | 说明 |
|---|---|---|
| `aifflor` | CLI 默认 `v6 → v7` | critic + actor 都变（actor 自 2026-11-03 起已接） |
| `td3per` | CLI 默认 `v6 → v7`，`--l_critic_arch` choices + `v7` | `SP_TD3` 新增 `actor_arch`（只给 H 传） |
| `testpg` | CLI 默认 `v2 → v7` | 继承 `aifflor.SP_TD3`，H 传 `actor_arch='v7'` |
| `prle` / `singleaif` / `td3leprrlrb` / `td3perrb` / `td3perseg` / `td3prle` / `td3prrbb` / `td3prrlrb` / `yawflor` | **新增** `--critic_arch`（默认 `v7`）；`<pkg>_ddpg.py` 按 `arch` 分 `actor_cfg`/`critic_cfg` | 迁移脚本 `agentutil/port_v7_default.py`（dry-run/`--apply`，幂等） |

**刻意保留的两条"安全边界"**（否则历史模型与探针会静默失效）：

1. **构造函数的回落仍是 `v1`**：不传 `args.critic_arch` 的脚本（`agentutil/` 探针、
   `test_five_days.py`、`test_simple.py` 等手工构造 args 的地方）拿到的还是历史网络。
   只有真正走 CLI 起训练时才默认 v7。
2. **评测按 checkpoint 识别结构**：`test_five_days._detect_critic_arch()` 用
   `model.infer_arch(critic_p_<na>.pkl)` 反推 v1/v2/v5/v6/v7 并写进 `args.critic_arch`
   ⇒ 旧 v1/v6 模型与新的 v7 模型可以在同一次评测里共存；
   actor 宽度本来就由 `_detect_hidden()` 从 `actor_6.pkl` 反推（v7 = 400/300）。

另外，**actor 的 400/300 只给 H 层（`nb_actions > 1`）**：L 层是 1 维 BESS 动作，
仍用 `--hidden1/--hidden2`（16/100）——与 aifflor/td3per 的口径一致，避免 1 维动作网
被放大 100 倍。L 的 **critic** 仍是 v7（`--l_critic_arch v1` 可降级）。

验证：`agentutil/test_critic_arch_v7.py` **119 项全过**（含 9 个方法的"构造即 v7：
critic 245 501 参数、actor 400-300-400"）；`test_td3per_branch_v6_contract.py` 1628 条、
`test_phase_uniform.py` 全过。

## 7. 一句话

> **v7 = 把动作和状态在最开始就拼起来，再走 400→300→400。**
> 静态接线全绿（119 项契约），离线 C1 基准上成本拟合与整体梯度标定明显优于 v6
> （real RMSE −39%、MAE −43%、折点 F1 0.959→0.987、slope 0.89→0.98），
> 代价是参数量 +63%（245k vs 150k）与 hinge 组方向一致性略降；
> 2026-11-04 起 **aifflor / td3per / testpg 与 9 个老方法的默认结构都是 v7**，
> 旧口径用 `--critic_arch v1/v6` 复现；
> **闭环（FLORIS 训练 + WFSim 评测）仅在 aifflor 上验证到 130k 步（`AIFFLOR-V7-run20`）**。
