# v7.1 / v7.2 / v7.3：aifflor 的三个新 critic-actor 结构

> 实施日期：2026-11-06
> 结构定义：`model.Critic(arch='v7.1' / 'v7.2' / 'v7.3')`
> actor 定义：`model.Actor(..., pre_hidden=None/128)`
> 契约测试：`python agentutil/test_critic_arch_v7x.py`（162 项，纯静态）
> CLI：`python aifflor/aifflor_main.py --critic_arch v7.1 ...`（也接受 `7.1` 写法）

## 1. 口径

三个版本都建立在 v7 的「aifflor 上层 H 的 critic/actor」接线上，**默认结构仍是 v7**，
新结构只通过 `--critic_arch` 显式选择。

```text
v7.1
  critic : cat(state, action) → 500 → 400 → 500 → 1
  actor  : state → 500 → 400 → 500 → action

v7.2
  critic : s: ns → 128
           a: na' → 128
           cat(128 + 128 = 256) → 400 → 300 → 400 → 1
  actor  : state → 128 → 400 → 300 → 400 → action

v7.3
  critic : s: ns → 128
           a: na' → 128
           cat(256) → 500 → 400 → 500 → 1
  actor  : state → 128 → 500 → 400 → 500 → action
```

- `na' = na`；开 `--critic_feat_power 1` 时 `na' = na + 1`（动作输入追加 `P(a,u)/5`）。
- v7.2/v7.3 的 `256` **是拼接宽度**（128+128），不是额外隐藏层。
- actor 没有动作输入：v7.2/v7.3 的 actor 用**与 critic 相同的 128 维 state 编码器**，
  然后接 critic 的合并主干；编码层在 `model.Actor` 里叫 `fc0`。
- v7.1 的 actor 复用 `model.Actor` 的既有形状 `state→hidden1→hidden2→hidden1→action`，
  `hidden1=500, hidden2=400` 恰好是 `500→400→500`，没有 `fc0`。

## 2. 逐层形状与参数量（ns=4, na=6，含 bias）

### v7.1

| 层 | 形状 | 参数 |
|---|---|---:|
| h1 | 10 → 500 | 5 500 |
| h2 | 500 → 400 | 200 400 |
| h3 | 400 → 500 | 200 500 |
| out | 500 → 1 | 501 |
| **critic 合计** | | **406 901** |

| actor 层 | 形状 | 参数 |
|---|---|---:|
| fc1 | 4 → 500 | 2 500 |
| fc2 | 500 → 400 | 200 400 |
| fc3 | 400 → 500 | 200 500 |
| fc4 | 500 → 6 | 3 006 |
| **actor 合计** | | **406 406** |

### v7.2

| 层 | 形状 | 参数 |
|---|---|---:|
| s1 | 4 → 128 | 640 |
| a1 | 6 → 128 | 896 |
| m1 | 256 → 400 | 102 800 |
| m2 | 400 → 300 | 120 300 |
| m3 | 300 → 400 | 120 400 |
| out | 400 → 1 | 401 |
| **critic 合计** | | **345 437** |

| actor 层 | 形状 | 参数 |
|---|---|---:|
| fc0 | 4 → 128 | 640 |
| fc1 | 128 → 400 | 51 600 |
| fc2 | 400 → 300 | 120 300 |
| fc3 | 300 → 400 | 120 400 |
| fc4 | 400 → 6 | 2 406 |
| **actor 合计** | | **295 346** |

### v7.3

| 层 | 形状 | 参数 |
|---|---|---:|
| s1 | 4 → 128 | 640 |
| a1 | 6 → 128 | 896 |
| m1 | 256 → 500 | 128 500 |
| m2 | 500 → 400 | 200 400 |
| m3 | 400 → 500 | 200 500 |
| out | 500 → 1 | 501 |
| **critic 合计** | | **531 437** |

| actor 层 | 形状 | 参数 |
|---|---|---:|
| fc0 | 4 → 128 | 640 |
| fc1 | 128 → 500 | 64 500 |
| fc2 | 500 → 400 | 200 400 |
| fc3 | 400 → 500 | 200 500 |
| fc4 | 500 → 6 | 3 006 |
| **actor 合计** | | **469 046** |

其它输入维（契约测试实测）：

| 结构 | H critic（ns=4, na=6） | wind5（ns=8） | L critic（ns=6, na=1） |
|---|---:|---:|---:|
| v7.1 | 406 901 | 408 901 | 405 401 |
| v7.2 | 345 437 | 345 949 | 345 053 |
| v7.3 | 531 437 | 531 949 | 531 053 |

> L 层的 **actor** 不跟着放大：`aifflor_main` 只给 H 层传 `actor_arch`，
> L 的 1 维 BESS actor 仍是 `--hidden1/--hidden2`（默认 16/100，3 445 参数）。

## 3. 设计动机

- **v7.1**：v7 的「动作首层进网」保持不变，只把 `400-300-400` 加宽到 `500-400-500`。
  适合先验证「同一归纳偏置、更大容量」是否带来收益。
- **v7.2 / v7.3**：把状态和动作**分开编码**再拼接，避免首层直接把 10 维异质输入
  混在一起；128 维编码器很窄，参数量主要仍在合并主干。
  - v7.2 的主干与 v7 相同（`400-300-400`），可与 v7 做「只改编码方式」的对照；
  - v7.3 的主干是 `500-400-500`，可与 v7.1 做「编码方式 × 主干宽度」的对照。
- 三个版本的 actor 都与 critic 的**状态通路/主干同宽**，保持 v7 的「actor 与 critic
  同宽」口径。

## 4. 接线与可恢复性

- `model.Critic` 新增分支：
  - `v7.1` 复用 `h1/h2/h3/out` 层名；
  - `v7.2/v7.3` 使用 `s1/a1/m1/m2/m3/out`（与 v6 同一套命名习惯）。
- `model.Actor` 新增可选参数 `pre_hidden`：非空时先建 `fc0 = Linear(nb_states, pre_hidden)`，
  其余 `fc1..fc4` 不变。`pre_hidden=None` 时历史路径**逐位不变**。
- `model.infer_arch` 能唯一识别三个新结构：
  - `v7.1`：`h1` 输出 500 且 `h2` 形状 `(400,500)`（v7 是 400 / (300,400)）；
  - `v7.2/v7.3`：`s1/a1` 输出都是 128 且 `m1` 输入 256，再按 `m1` 输出 400/500 区分。
  - 不会被 `v6old`（`m1=(256,70/65)`）、`v4`（方阵）、`v5`（`(100,48)`）误判。
- `feat_from_sd` 对 v7.1 走 `h1`，对 v7.2/v7.3 走 `a1`，`--critic_feat_power` 可往返。
- 训练 checkpoint：`actor_6.pkl` / `critic_p_6.pkl` 与结构绑定；续训时必须传相同的
  `--critic_arch`。
- 评测 `test_five_days.py`：
  - critic 结构由 `_detect_critic_arch()` 调 `model.infer_arch` 自动识别；
  - v7.2/v7.3 的 actor 多一层 `fc0`，由 `_detect_actor_pre()` 从 `actor_6.pkl` 读出，
    经 `args.actor_pre` 传给 `aifflor_ddpg.SP_TD3`。
  - 这样旧 v7.1/v7.2/v7.3 模型可以在不改评测命令的情况下加载。

### CLI 别名

`--critic_arch` 的 canonical 值是 `v7.1/v7.2/v7.3`；argparse 的 `type=model.normalize_arch`
同时接受 `7.1/7.2/7.3`，归一后再做 choices 校验。

## 5. 复现与回归

```powershell
$PY = 'C:\Users\yigul\anaconda3\envs\llmenhancedwsis\python.exe'

# 新结构契约（162 项：参数量/层形状/前向反向/通路独立性/infer_arch/actor/SP_TD3/CLI）
& $PY agentutil\test_critic_arch_v7x.py

# 既有 v7 契约（不得回归）
& $PY agentutil\test_critic_arch_v7.py
```

> 以上都是**纯静态**检查（随机张量前向/反向，不构建环境、不推进 FLORIS、不训练）。

## 6. 一句话

> **v7.1 = cat(s,a) 500-400-500；v7.2 = 双 128 编码器 + cat256 + 400-300-400；
> v7.3 = 双 128 编码器 + cat256 + 500-400-500。** actor 对应
> `500-400-500` / `128-400-300-400` / `128-500-400-500`，默认仍为 v7。
