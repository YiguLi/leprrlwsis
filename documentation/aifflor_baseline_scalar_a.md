# aifflor baseline：PER 结构 + 非平滑 c1 + 单自由度风机动作（`--scalar_wt`）

> 一条命令跑：
> ```
> python agentutil/launch_baseline_scalar_a.py               # 跑满
> python agentutil/launch_baseline_scalar_a.py --max-steps 300   # 冒烟
> ```
> 契约测试：`python agentutil/test_aifflor_scalar_wt.py`（22 项，全过）

## 1. 与 aifflor 默认臂的**全部**差异（只有三条）

| # | 差异 | 开关 | 默认臂 | baseline |
|---|---|---|---|---|
| 1 | c1 口径 | `--c1_smooth 0` | 平滑（β=100, ε=0.05） | **非平滑**：`c1 = \|ΔP\| + 20·(\|ΔP\|−3)⁺` |
| 2 | 神经网络结构 | `--critic_arch v1` + 16/100 | v2 双流（60 617 参数） | **PER 的 v1**（`fc1→fc4→cat(a)→fc2→fc5→fc3`，14 441 参数）；actor 同为 3 498 参数 |
| 3 | 动作空间 | `--scalar_wt 1` | 5 个独立的轴向因子 | **五台全部 = a[0]**（1 个自由度；第 1..4 维存在但无用） |

其余全部沿用 aifflor 当前默认：HSRL 双层、`nstep=5`、γ=0.95（块级）、`utd_h=4`、
`bsize=256`、`rmsize=40000`、PER α=0.6/β=0.4、warmup=7200、`wt_noise 0.05→0`、
`s_p=1000 / s_c=100`、λ∈[2,50]、`d_block=15`、FLORIS 环境、加噪风速池
`FFC_speed_noise_1.csv`、第 6 维 g（块总放电量）仍是独立自由度。

> 说明："**与 PER 相同**"指的是**网结构**（层拓扑 + 宽度 16/100 + init_w=3e-3），
> 不是把训练器换成 PER（仍然是 SP_TD3：twin critic / 延迟策略 / target smoothing / PER 缓冲）。

## 2. 动作空间到底变成了什么（刻意设计）

实现：`model.proj_scalar_wt(a)` / `proj_scalar_wt_np(a)`，即线性投影

```
a[:5] ← [1, 0, 0, 0, 0] · a          (g 不变)
```

```python
def proj_scalar_wt(a, nt=5):
    out = a.clone(); out[..., :nt] = a[..., :1]; return out
```

所以：

* 有效自由度 **1**（第 0 维），第 1..4 维**严格无用**：`∂Q/∂a_j ≡ 0`（j=1..4，
  实测 `0.0e+00`，不是"很小"），env 出力也与它们无关（实测逐位相同）；
* 五台风机的出力因子恒为 `g(a_0) = 5·4a_0(1−a_0)²` ⇒ 这是"**整场统一降载**"的
  最粗糙版本：连逐台形状都不要（区别于 `--farm_level` 的 `a_i = s·c_i`）。

落地位置（三处，缺一处就会被绕过）：

| 位置 | 作用 |
|---|---|
| `aifflor_ddpg._project` → `_act` | actor 的**确定性输出**（critic 的演员评估点） |
| `aifflor_ddpg.select_action`（加完探索噪声之后） | 实际执行的动作 —— 否则噪声会把 5 维打散 |
| `aifflor_main._apply_scalar_wt`（`_diag_fix_action` / `explore_mix` 之后、`check_action_H_hsrl` 之前） | warmup 的 LHS 调度、diag 改写、早期覆盖混合之后强制恢复相等 |

## 3. 三种"压成 1 个自由度"的开关的区别（别混用）

| 开关 | 流形 | 每台的动作 | 用途 |
|---|---|---|---|
| `--scalar_wt 1` | `a_i = a_0` | 五台**完全相等** | 本 baseline |
| `--farm_level 1` | `a_i = s·c_i`，c=`FARM_CENTRE` | 保持逐台**形状**，只缩放 | 场级降载射线（输出/FARM1） |
| `--diag_scalar 1`（诊断） | `a_i = 0.33·sigmoid(a_0)` | 五台相等，但被 sigmoid/0.33 夹住 | DIAG 实验 2 专用 |

`--scalar_wt` 与 `--farm_level` **互斥**（main 里 assert 拒绝同时开）。

## 4. 必须知道的五个副作用（都是"刻意设计"的直接后果）

1. **warmup 覆盖率塌到 1 维**：LHS 仍在 5 维上采样，但投影后只剩 `a_0` 的分布；
   缓冲里因此**没有** off-manifold 动作。这是正确的（critic 不该看到不可能执行的动作），
   但也意味着 critic 只能学到 `a_0` 那一条曲线。实测 300 步冒烟：`a_0 ∈ [0.020, 0.495]`、
   std 0.119（覆盖满）。
2. **探索只在第 0 维有效**：第 1..4 维的噪声被投影丢弃 ⇒ 只有 `a_0` 上 ±(0.05·ε) 的探索。
3. **越界罚是 5 倍**：`a_0` 一旦离开 `[0,0.5]`，五台同时越界 ⇒ `pen5` 是"单台越界"的 5 倍
   （投影不 clip，越界仍交给 `check_action_H_hsrl`）。这是"一个动作驱动五台"的必然结果，
   不是 bug；若要改成"只罚一次"得动 reward，这次没做。
4. **critic 的动作分支输入是 5 个相同的数**：v1 的 `fc2` 收到 `cat(状态, a)`，前 5 维冗余；
   等价于 1 个输入 + 4 个副本（权重会被合并）。这正是 baseline 想测的"冗余动作维"。
5. **与 run3 的对比要分清变量**：run3 = v2 + 平滑 c1 + 5 自由度；本臂 = v1 + 非平滑 c1 + 1 自由度。
   三条同时变，**不能单独归因**；要归因就再跑两个单变量臂（v1+平滑+5维、v2+平滑+1维）。

## 5. 顺带修的一个真 bug（影响已有 FARM1 臂的可比性）

`update_policy` 里演员评估点原来**两条通道不一致**：

```python
# 旧：利润塔用未投影的 raw actor 输出
min_p_cur = self._pair_min(self.critic_p([s, self.actor(s)]),
                           self.critic_p2([s, self.actor(s)]))
#    成本塔用 _act（已投影）
_a_cur = self._act(self.actor, s)
```

⇒ 开 `--farm_level` / `--scalar_wt` 时，利润梯度跑到**动作流形之外**。
现在统一为 `_a_cur = self._act(...)` 后再喂两条通道。未开投影时 `_act` 是严格恒等
（契约测试 G 段断言 `_project(x) is x` 逐位相等），所以**默认臂逐位不变**；
但 `output/FARM1`（`--farm_level 1`）是在旧的不一致下跑的 —— 若要和新代码比，
FARM1 需要重跑。

## 6. 验证记录（2026-10-28）

* `agentutil/test_aifflor_scalar_wt.py`：**22 项全过**
  （投影语义 / 噪声后仍相等 / 梯度严格 0 / FLORIS 出力不变 / 硬折点 c1 / PER 结构 / 默认臂恒等）；
* 300 步冒烟（`output/SMOKE-BASELINE-SCALAR`）：
  * `applied_action_H` 与 `original_action_H` 的前 5 维**逐行严格相等**（max\|a_i−a_0\| = 0）；
  * `c1 − (P_F + 20·max(0,P_F−3))` 的 max = **3.6e-15**（硬折点；对比 run3 平滑版是常数 −0.1386）；
  * 启动日志出现 `[baseline] 动作空间 = 1 个自由度 …` 与 `[critic] arch=v1`；
* `test_c1_uniform.py` / `test_critic_bias.py` 仍通过。

## 7. 跑满之后建议看什么

与 `output/FULL-R2/AIFFLOR-run3`（v2/平滑/5 自由度）并排比：

| 指标 | 口径 |
|---|---|
| `η` / 越限率 / `Σc1` / 收益 | `hsrl_metrics.summarize(run_dir)` |
| `a_std_step` | 1 自由度下应**明显低于** 5 维臂（动作无法逐台微调） |
| `r2_dua` | 动作对出力的增量解释力：本臂应只剩"整体缩放"这一支 |
| critic 动作梯度 | `agentutil/probe_critic_blockgrad.py`（块级真值）——v1 的动作只有 1 维有效，看 `∂Q_c/∂a_0` 的 mean/sd |
| 折线敏感度 | 非平滑 c1 下真值折线比 ≈ 29×，看 v1 critic 是否比 v2 更看不到折点 |
