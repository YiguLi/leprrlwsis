"""
HSRL 智能体初始化助手（多项目共享）
==================================
当前只有一个功能：**critic 输出层 bias 的价值基线初始化**。

动机（实测）
------------
`model.py` 的 `Critic` 结构为 fc1→fc4→(cat a)→fc2→fc5→**fc3(50→1, 输出层)**，
其中只有 fc1/fc2 被 `fanin_init` 覆写，**fc3 保持 PyTorch 默认 U(±1/√50)=±0.1414**。
于是 critic 初始化时输出 Q≈O(0.1)，而回归目标是 O(10³)（如 TD3LEPRRLRB 实测
|Q_p|≈1.0e3~2.2e3，γ=0.98）——**差了约 4 个数量级**。

在 Adam 下（以及 `clip_grad_norm_(2.0)` 饱和时）每个参数每步只移动 ~lr，与梯度大小无关，
所以"把输出搬到正确量级"纯粹是一段**参数空间距离**：实测 |Q_p| 达到终值 50% 用掉
**42~46% 的训练预算**、达到 90% 用掉 **63~69%**（见 `analyse_q_scale.py`）。

修法：价值函数 = 「常数基线」+「随状态变化的部分」，而常数基线本来就该由**输出层 bias**
承载（`Q = Σ wⱼhⱼ + b`）。把 `fc3.bias` 初始化到 `r̄_block/(1−γ)`，就直接删掉这段
"搬基线"的旅程，网络只需学量级小得多的变化量。

注意
----
* **不改 `model.py`**（多项目共享）：一律在 agent 构造后覆写 —— 与项目既有惯例
  `init_wt_action_bias`（actor 的同类修复）一致。
* 覆写会同时作用到 4 个 critic（p/p2/c/c2）与其 **target 网络**；DDPG 系（无 p2/c2）
  自动跳过缺失属性。
* 估值必须用**当前策略轨迹上的块收益均值**（warmup 期观测），不要用 PER 采样 batch 的
  `reward_mean`（被高误差样本加权，偏高；实测 ~206 vs 轨迹均值几十）。
* 有限 episode 的正确缩放是 `(1−γ^n)/(1−γ)`（n = 每 episode 的块数）；n 较大时与
  `1/(1−γ)` 相差 <5%，这里用后者。**宁略低勿高估**：它只是初值。

触发条件（修正）
----------------
**旧写法（bug）**：各 main 把触发写在块末分支里

    if episode_steps % 5 == 4:                       # 块末: step ≡ 4 (mod 5)
        if (... != 'none') and step <= args.warmup:
            _warm_blocks.append((sum_r1, sum_c1))
            if step == args.warmup:                  # ← 要求 warmup ≡ 4 (mod 5)
                hinit.init_critic_bias(...)

episode 长度 1440 是 5 的倍数 ⇒ `episode_steps ≡ step (mod 5)`，块末步恒为
4, 9, …, 199, 204, …；而 `--warmup` 默认 **200 ≡ 0 (mod 5)** ⇒ `step == 200` 永远
不是块末 ⇒ **整个 bias 分支从不执行**（`output/TD3PRRLRB-run3` 实测 Q 起点
0.0497 = `fc3.bias` 的 PyTorch 默认初值）。只有 `warmup ≡ 4 (mod 5)`（如 199/204）
才可能触发。

**现写法**：用 `BiasInitTracker.on_block_end()`，在 **warmup 之后的第一个块末**
（`step >= warmup`，即 `first_trigger_step(warmup)`）一次性应用；该块本身仍是 warmup
块（它的块首决策发生在 `step <= warmup`），且主循环里块末结算排在
`if step > args.warmup:` 的更新段之前 ⇒ **首次策略更新之前生效**。
显式覆盖 `--critic_bias_p/_c` 同样只在该触发点落地（必须保持
`--critic_bias_init != 'none'`）。
"""
import numpy as np
import torch

try:
    from util import prCyan
except Exception:                                    # pragma: no cover
    def prCyan(s):
        print(s)


def bias_from_block_samples(samples, discount):
    """从 warmup 的块样本估计价值基线。

    Parameters
    ----------
    samples : list[(float, float)]
        每个 5 步块的 (sum_r1, sum_c1)（来自实际观测，不用 PER batch 均值）。
    discount : float
        块级折扣（H 层 γ）。

    Returns
    -------
    (b_p, b_c) : (float, float) 或 (None, None)
        利润/成本 critic 的基线估计 = 块均值 × 1/(1−γ)。
    """
    if not samples:
        return None, None
    r = float(np.mean([float(s[0]) for s in samples]))
    c = float(np.mean([float(s[1]) for s in samples]))
    g = 1.0 / max(1e-9, (1.0 - float(discount)))
    return r * g, c * g


# (online, target, 用哪个基线)
_PAIRS = (('critic_p', 'critic_target_p', 'p'),
          ('critic_p2', 'critic_target_p2', 'p'),
          ('critic_c', 'critic_target_c', 'c'),
          ('critic_c2', 'critic_target_c2', 'c'))


def init_critic_bias(agent, b_p=None, b_c=None, verbose=True):
    """把 critic 输出层 (fc3, 50→1) 的 bias 设为价值基线；weights 与其它层不动。

    Returns: 被设置的网络名列表（便于断言/日志）。
    """
    done = []
    for online, target, which in _PAIRS:
        b = b_p if which == 'p' else b_c
        if b is None:
            continue
        net = getattr(agent, online, None)
        if net is None or not hasattr(net, 'fc3'):
            continue                                   # DDPG 系无 p2/c2
        with torch.no_grad():
            net.fc3.bias.data.fill_(float(b))
        done.append(online)
        tgt = getattr(agent, target, None)
        if tgt is not None and hasattr(tgt, 'fc3'):
            with torch.no_grad():
                tgt.fc3.bias.data.fill_(float(b))
            done.append(target)
    if verbose:
        prCyan('init_critic_bias: 输出层 bias 置为 b_p={} b_c={} → 已设置 {} 个网络 {}'
               .format(None if b_p is None else round(float(b_p), 2),
                       None if b_c is None else round(float(b_c), 3), len(done), done))
    return done


def q_probe(agent, states, actions, critic='critic_p'):
    """诊断用：给定状态/动作，返回该 critic 的输出均值（验证基线是否生效）。"""
    net = getattr(agent, critic, None)
    if net is None:
        return None
    with torch.no_grad():
        return float(net([states, actions]).mean())


# =============================================================================
# 触发点（修正 step == warmup 的不可达 bug）
# =============================================================================
def first_trigger_step(warmup):
    """warmup 之后第一个块末步 = min{s ≡ 4 (mod 5) : s >= warmup}。

    主循环的块末判定是 `episode_steps % 5 == 4`，而 episode 长度（1440）是 5 的
    倍数 ⇒ 块末步恒为 step ≡ 4 (mod 5)。旧写法直接用 `step == warmup`，因此只有
    warmup ≡ 4 (mod 5) 时才可能触发（默认 200 永不触发）。
    """
    w = int(warmup)
    return w + ((4 - w) % 5)


class BiasInitTracker:
    """warmup 块样本收集 + critic 输出层 bias 的**一次性**应用。

    用法（主循环块末，`episode_steps % 5 == 4` 分支内、更新段之前）::

        info = bias_init.on_block_end(step, sum_r1, sum_c1, agent_H)
        if info:                      # 本次完成初始化（可打印 info）
            ...

    * `mode == 'none'` ⇒ 不收样本、永不应用（对照臂，与旧行为一致）。
    * 触发点 = `first_trigger_step(warmup)`；该块**计入**估计样本
      （它的块首动作仍是 warmup 动作），因此样本数 = 触发点之前的全部块末数。
    * `b_p/b_c > 0` 时覆盖自动估计值（`--critic_bias_p/_c`）。
    """

    def __init__(self, mode='estimate', warmup=200, discount=0.99,
                 b_p=-1.0, b_c=-1.0, verbose=True):
        self.mode = str(mode)
        self.warmup = int(warmup)
        self.discount = float(discount)
        self.override_p = float(b_p)
        self.override_c = float(b_c)
        self.verbose = bool(verbose)
        self.blocks = []
        self.applied = False
        self.info = None

    @classmethod
    def from_args(cls, args):
        return cls(mode=getattr(args, 'critic_bias_init', 'estimate'),
                   warmup=int(getattr(args, 'warmup', 0)),
                   discount=float(getattr(args, 'discount', 0.99)),
                   b_p=float(getattr(args, 'critic_bias_p', -1.0)),
                   b_c=float(getattr(args, 'critic_bias_c', -1.0)))

    @property
    def trigger_step(self):
        return first_trigger_step(self.warmup)

    def on_block_end(self, step, sum_r1, sum_c1, agent):
        """收集一个 warmup 块；到达触发点时应用 bias。

        Returns
        -------
        None 或 dict(b_p, b_c, n_blocks, step, trigger_step)
        """
        if self.mode == 'none' or self.applied:
            return None
        self.blocks.append((float(sum_r1), float(sum_c1)))
        if int(step) < self.trigger_step:
            return None
        b_p, b_c = bias_from_block_samples(self.blocks, self.discount)
        if self.override_p > 0:
            b_p = self.override_p
        if self.override_c > 0:
            b_c = self.override_c
        init_critic_bias(agent, b_p, b_c, verbose=self.verbose)
        self.applied = True
        self.info = dict(b_p=b_p, b_c=b_c, n_blocks=len(self.blocks),
                         step=int(step), trigger_step=self.trigger_step,
                         mean_r1=self.blocks and float(np.mean([b[0] for b in self.blocks])),
                         mean_c1=self.blocks and float(np.mean([b[1] for b in self.blocks])))
        if self.verbose:
            prCyan('BiasInitTracker: 触发于 step={} (warmup={}, 首块末={})，'
                   '用 {} 个 warmup 块估计 r̄={:.2f} c̄={:.2f} → b_p={:.1f} b_c={:.3f}'
                   .format(self.info['step'], self.warmup, self.trigger_step,
                           len(self.blocks), self.info['mean_r1'],
                           self.info['mean_c1'],
                           -1.0 if b_p is None else float(b_p),
                           -1.0 if b_c is None else float(b_c)))
        return self.info
