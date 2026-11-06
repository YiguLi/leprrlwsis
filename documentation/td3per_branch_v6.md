# `td3per` 的两项移植：branch（同块首宽动作覆盖）+ critic v6

> 实施日期：2026-10-30 ｜ 契约测试：`python agentutil/test_td3per_branch_v6_contract.py`
> （1628 条断言，纯静态/纯算术，**不跑仿真、不需要 MATLAB**）
>
> ⚠ **2026-11-04 起 `--critic_arch` 的 CLI 默认由 `v6` 改为 `v7`**
> （`cat(s,a)→400→300→400→1` + actor `s→400→300→400→a`，
> 见 `documentation/critic_arch_v7.md`）。本文的 v6 口径仍完整保留：
> `--critic_arch v6` 可逐位复现；`SP_TD3` 在拿不到 `args.critic_arch` 时仍回落 **v1**。

本文记录把 `aifflor` 的两项方法移植进 `td3per` 的口径。两项互不依赖，可单独开关：

| 方法 | 开关 | 默认 | 关闭时 |
|---|---|---|---|
| **critic v6**（`model.Critic(arch='v6')` 双流网络） | `--critic_arch` | **`v6`** | `--critic_arch v1` ⇒ 与改造前**逐位相同** |
| **branch**（同块首宽动作覆盖） | `--branch_every K` / `--branch_pilot_blocks N` | **关闭**（0/0） | 全部新代码走"关闭"分支，逐位不变 |

---

## 1. critic v6（新神经网络）

### 1.1 结构（`model.V6_DIMS`，单一真源）

```
state : ns → 32 → 32          （ns = 4 + 相位维 0/2/5）
action: na → 64 → 64          （na = 5（--nolow）或 6）
merge : cat(32+64)=96 → 128 → 256 → 256 → 128 → 1     ← 原始 a **不**再拼入
```

* `ns=4, na=6` 时单个 critic **150,081** 参数（v1 只有 14,441）。相位特征（`--phase_feat`）
  只改 `ns`，不改 `merge` 宽度。
* `--hidden1/--hidden2` 在 v6 下**不生效**（宽度写死在 `V6_DIMS`）；要复现历史宽度请
  `--critic_arch v1`。
* 输出层名字是 `out`（不是 `fc3`）⇒ `hsrl_agent_init._out_head` 已认 `out`，
  `--critic_bias_init estimate` 的价值基线初始化**不会静默 no-op**。

### 1.2 接线

| 位置 | 改动 |
|---|---|
| `td3per/td3per_ddpg.py` | `SP_TD3(..., critic_arch=None, critic_act='relu', critic_out_init='3e3')`；四个 critic + 四个 target 全部按同一 `critic_cfg` 构造 |
| `td3per/td3per_main.py` | `--critic_arch`（默认 `v6`）/`--critic_act`/`--critic_out_init`/`--l_critic_arch` |
| `SP_TD3.load_weights` | 按 `critic_p_<na>.pkl` 的 `state_dict` **自动推断 arch 并重建**（v1↔v6 双向往返；`test_five_days.py` / `nine_case_test` 无需知道训练时的 arch） |

⚠ **两个默认值不同，是有意的**：

* `td3per_main.py` 的 CLI 默认 = **`v6`**（生产口径，与 `aifflor` 一致）；
* `SP_TD3` 构造函数在拿不到 `args.critic_arch` 时的回落 = **`v1`**。

理由：`agentutil/` 里大量探针/契约测试直接构造 `SP_TD3(..., args=桩)` 并断言"逐位不变"，
若回落值改成 v6，那些断言会静默失效。**生产路径由 CLI 决定**，脚本构造保持历史网络。

### 1.3 下层 L 的 arch

默认 `--l_critic_arch inherit`（= 与 H 相同）。L 每块更新 10 次（H 是 4 次），v6 在 L 上的
算力占比比 H 高；想省算力可 `--l_critic_arch v1`（H 仍是 v6）。

---

## 2. branch（同块首宽动作覆盖）

计划书与 aifflor 侧口径：`documentation/aifflor_branch_wide_coverage_plan.md`。
**td3per 侧的单一真源 = `td3per/td3per_branch.py`**。

一句话：**每个块（5 步）从同一个「完整块首 state」上再跑若干个测试动作**，把它们各自的
块级 `(s, a, Σr1, Σc1, s', done)` 当成正常 TD 数据写进同一个 replay；但
actor / lower / λ / EMA / curriculum / CSV / TB / test **只用 actor 分支**。

### 2.1 与 aifflor 的语义差异（都是 td3per 自己的口径）

| 项 | aifflor（HSRL 分支） | td3per |
|---|---|---|
| 第 6 维 | 块总放电量 `g` [MW·min] | **SOC 目标信号** `a6 ∈ [-0.5,0.5]` → `soc_target`（`soc_target_from_signal`） |
| 块成本 | `hsrl.cal_rew_rb_hsrl` | **`Cal_Rew_RB`**（同一函数，含 `hc.c1_apply` 的 uniform/legacy 口径） |
| 块末 gap | HSRL 无 gap 罚 | **与主循环同式**：`10·max(0, |soc_target−SOC_末|−soc_tol)`；uniform 进 `r1`，legacy 进 `c1` |
| 下层 | `amortized_rule_action(g, …)` | `--branch_lower freeze`（默认）/`rule`/`none`（见下） |

### 2.2 第 6 维怎么取（`--branch_signal`）

| 值 | 含义 |
|---|---|
| **`soc`（默认）** | 目标 = **本块块首 SOC** ⇒ 下层冻结时块末 gap ≡ 0 ⇒ **`c1` 只由 5 维风机动作决定**（最干净的反事实） |
| `target` | 沿用 actor 本块的第 6 维（branch 与 actor 目标相同 ⇒ gap 是同一常数） |
| `zero` | `a6 = 0` ⇒ 目标 = SOC 带中点 |
| `pm1` | 随机 `±0.5` ⇒ 目标 = 可行带两端（对应 aifflor 的 `--branch_goal_pm1`） |
| `random` | `U(-0.5, 0.5)` |

### 2.3 下层怎么冻结（`--branch_lower`）

| 值 | 语义 |
|---|---|
| **`freeze`（默认）** | `pb` 的**命令**恒 0，但 RBB 仍把它夹进"平抑窗口 ∩ 物理窗口"；窗口不含 0 时电池做**最小幅度被迫吞吐**（否则 `|P_G(t)−P_G(t−1)|` 更大）。⚠ 不要读成 `pb≡0`/`E 不变` —— `df_branch_*.csv` 的 `soc_start/soc_end` 让"冻结得多干净"可审计 |
| `rule` | `pb = hst.rule_action(soc_target, …)`（零成本基线，与 `SocTracker.act` 逐位一致） |
| `none` | **严格** `pb≡0`：绕开 RBB（`P_G = P_tur`，SOC 一字不变），最纯的风机动作反事实 |

### 2.4 数据口径（计划 §4.3/§4.4 的硬约束）

| 通道 | 用哪些数据 |
|---|---|
| critic TD（4 个 critic） | **全部**（actor + branch） |
| 可选 1-step 辅助监督 `--branch_sup_beta` | **只 branch 行**（β 从 β0 线性降到 0.2·β0） |
| actor loss | **只用 actor 行**（`actor_mask` 切片；无 branch 时退化为 `[:]`） |
| λ / EMA / curriculum / CSV / TB / test | **只用 actor**（`update_stats['q_c_mean']` 也只算 actor 行） |
| 下层 L 的 replay | branch **从不写入**（`agent_L` 有独立 memory） |
| n 步 | branch **不进** `_pending` 链；`n_boot = γ`（actor 是 `γ^n`） |

采样：`--branch_critic_frac`（默认 0.45，夹到 ≤0.5）⇒ critic 批里 branch 行数
`= round(0.45·bsize)`，两池**池内**各按 `priority^α` 抽，branch 池内部按 `mode` 均衡。
batch 大小恒等于 `bsize`（池不够时自动回退，绝不静默缩批）。

### 2.5 动作集（`--branch_actions`，默认 `prior,low,high,uniform`）

| mode | 生成式 | 覆盖 |
|---|---|---|
| `prior` | `[0.2,0.2,0.2,0.2,1/3] + 逐维 U(−0.05,0.05)` | 标称工作点 |
| `low` | 逐维 `U(0.00,0.10)` | 强降载边界（`a<0.05` 占一半） |
| `high` | 逐维 `U(0.40,0.50)` | 满发边界（`a>0.45` 占一半） |
| `uniform` | 逐维 `U(0,0.5)` | 全域无偏 |
| `ramp` | 逐台线性 `U(0.02,0.12)→U(0.38,0.48)` | 上游重降载斜坡 |
| `alt` | 奇偶机组交替 lo/hi | 异向动作 |
| `single` | 其余 = prior，随机一台夹到低/高带 | 单台扰动 |

所有动作**按构造就在 `[0,0.5]` 内**。

### 2.6 成本

一个 branch 块 = `len(modes) × 5` 个额外环境步（默认 4 个测试动作 ⇒ +20 步）。
`--branch_every 10` ⇒ 每 10 块 +20 步 ⇒ **1.4×**（50+20 = 70）；`4` ⇒ **2.0×**；`1` ⇒ 5×。
另加每批 45% 的 branch 行（同一次反向传播，算力增量远小于仿真增量）。

---

## 3. 用法

```bash
# 生产口径：v6 critic，不 branch（与 aifflor 的默认组合一致）
python td3per/td3per_main.py --critic_arch v6

# v6 + branch（每 10 块一次，4 个测试动作）
python td3per/td3per_main.py --critic_arch v6 \
    --branch_actions prior,low,high,uniform --branch_every 10

# 阶段 A pilot：前 2000 块每块 branch（= 10k 环境步）
python td3per/td3per_main.py --critic_arch v6 \
    --branch_actions prior,low,high,uniform --branch_pilot_blocks 2000 \
    --branch_every 0 --max_steps 10000 --seed 1

# 最纯的风机反事实：严格 pb≡0 + 信号维 = 当前 SOC（gap 恒 0）
python td3per/td3per_main.py --branch_every 4 --branch_lower none --branch_signal soc

# 复现改造前的 td3per（v1 critic + 无 branch）
python td3per/td3per_main.py --critic_arch v1
```

产出：

* 训练日志：`[critic] arch=... 参数量 ...`、`[branch] ...` 汇总；
* `df_branch_td3per.csv`（`--nolow` 时 `df_branch_td3pernolow.csv`）：
  逐 branch transition —— `step/block_id/mode/a1..a5/soc_signal/soc_target/r1/c1/gap_pen/
  P_F_mean/P_F_max/n_fold/soc_start/soc_end/pbrule_sum/lam1/lower_ver`；
* TensorBoard `branch/*`：`c1_mean`、`r1_mean`、`n_trans`、`cover_lo`/`cover_hi`/`cover_fold`、
  `buffer_n_branch`/`buffer_frac`/`batch_frac`。

---

## 4. 契约测试覆盖（`agentutil/test_td3per_branch_v6_contract.py`）

| 段 | 断言 |
|---|---|
| A | 常量与 `td3per_ddpg`/`PER_buffer` 的元数据字段一致 |
| B | v6 参数量（14,441 / 150,081）、前向形状（4/6/9/5 维 state）、`infer_arch` 往返、`_out_head=out`、`V6_DIMS` |
| C | `SP_TD3` 回落 v1、显式 v6、`load_weights` 自动重建（v1↔v6 双向往返、权重逐位还原） |
| D | 7 种动作生成器（形状/边界/覆盖 ≥40%/均值序/单台/单调/可复现）、`parse_modes`、信号维反解 |
| E | 块首快照-恢复（6 标量逐位）+ 缺字段报错 |
| F | `run_branch_block` 的 `Σr1/Σc1/s2` 与**独立重算**逐位一致（4 类块首 × 3 种 lower × nolow；`none` 时 SOC 一字不变；PowerOutput 调用 = 1+BLOCK） |
| G | `trigger` 的 pilot/every、`signal_value` 5 种模式、β 线性衰减 1→0.2、`coverage`、**branch 不消耗全局 RNG** |
| H | PER 分池（计数/池正确/by_mode 均衡/空池/环形覆盖计数） |
| I | `push_branch_transition`（`n_boot=γ` vs actor `γ^n`、不进 `_pending`、4 个标签落盘、done⇒boot=0） |
| J | `update_policy`：`branch_mix=0` 走历史 `memory.sample`；`=0.45` 时批内 branch 占比 14/32、**actor 每批只吃 actor 行**（forward 钩子）、`update_stats['q_c_mean']` 只算 actor 行、池里无 branch 时自动回退 |
| K | `td3per_main.py` 的**静态接线**（不 import：import 期会启 MATLAB）：导入 `tbr`、15 个 `--branch_*` 全部注册、`--critic_arch` CLI 默认 v6、branch 调用点在块首且在主步推进之前、`tbr.*` 的每个成员都存在、`soc_tol/lower/nolow` 逐项传入、`lower_ver`/`β`/CSV/TB 都接上 |

---

## 5. 注意事项

1. **`freeze` 不是 `pb≡0`**：见 §2.3。要严格的 `pb≡0` 用 `--branch_lower none`。
2. **默认 `--branch_signal soc`** 让冻结下层时的块末 gap ≡ 0（`c1` 只由风机动作决定）；
   若想研究"目标不可达时的 gap 罚"，用 `--branch_signal target/pm1`。
3. **branch 不加速收敛本身**，它只补"同一状态上的动作边际"；成本 1.4×。
   计划书 §7 的验收（覆盖率/折点比例）已写进 TB 与 CSV，按它们决定是否加密。
4. **v6 的宽度写死**：要扫宽度请改 `model.V6_DIMS`（不是 `--hidden1`）。
5. `--critic_arch v1` + 不传 `--branch_*` = **改造前的 td3per 逐位复现**；老 run 的
   checkpoint 由 `load_weights` 自动识别结构，无需手填 arch。
