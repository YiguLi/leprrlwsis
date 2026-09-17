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
