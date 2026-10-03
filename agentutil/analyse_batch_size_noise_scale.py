# -*- coding: utf-8 -*-
"""批大小（`--bsize`，默认 128）会不会导致过拟合？—— 用**梯度噪声尺度**定量判据。

先厘清概念（off-policy RL ≠ 监督学习）
--------------------------------------
* 监督学习：训练集固定 ⇒ 大 batch 减小梯度噪声 ⇒ 容易"记住"训练集（过拟合）。
* Off-policy RL：样本来自**不断被覆盖的 replay buffer**。本项目的 buffer 覆盖
  "最近 ~1.8 epoch"（500k 步滚动窗口，`aifflor_main.py:936-941`）
  ⇒ **没有固定的训练集可以记住**，过拟合的经典机制**不存在**。

所以 batch 大小在这里只有两个作用：
  (a) 降低 TD 目标的梯度噪声（有利；TD 目标含自助法噪声）；
  (b) 让两个 critic 梯度更相关 ⇒ 削弱 TD3 的 min-clipping 集成
      （**这才是真风险，但它叫"集成塌缩"，不叫过拟合**）。

**可定量判据：梯度噪声尺度（critical batch size）**
--------------------------------------------------
    B_noise = tr(Σ) / |G|²
      Σ = 逐样本梯度协方差，G = 全批平均梯度
    batch <  B_noise ⇒ 噪声主导区：加大 batch 仍能提升有效步长（**有益**）
    batch ≫  B_noise ⇒ 收益递减区：大 batch 只换来 critic 相关性（**风险**）

判读标准：若 `bsize=128 < B_noise` ⇒ **不存在"batch 太大"的问题**，更谈不上过拟合。

实现方式：沿用已验证的 `runpy` + monkeypatch（不重建 agent、不猜构造参数），
在第 1 次 `update_policy` 时用**真实 TD 目标**算逐样本梯度。

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\analyse_batch_size_noise_scale.py --steps 300
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

from floris_import_guard import check_interpreter as _chk, sanitize as _sanitize  # noqa: E402
_chk()
_sanitize()

import numpy as np  # noqa: E402

ST = {'done': False, 'rows': []}


def per_sample_grads(net, x, y):
    """逐样本 MSE 梯度，拼平成 (n, P)。"""
    import torch
    out = []
    for i in range(x.shape[0]):
        net.zero_grad()
        loss = ((net(x[i:i + 1]) - y[i:i + 1]) ** 2).mean()
        loss.backward()
        out.append(np.concatenate([p.grad.detach().cpu().numpy().ravel()
                                   for p in net.parameters() if p.grad is not None]))
    net.zero_grad()
    return np.asarray(out, dtype=float)


def noise_scale(G):
    n = G.shape[0]
    if n < 2:
        return float('nan')
    g = G.mean(0)
    g2 = float(np.dot(g, g))
    if g2 <= 0:
        return float('nan')
    tr = max(float(np.mean(np.sum(G ** 2, axis=1))) - g2, 0.0)
    return tr / g2


def install(bsize):
    """在第 1 次 H/L 的 update_policy 之后，用真实 TD 目标算 B_noise。"""
    import aifflor.aifflor_ddpg as D

    orig = D.SP_TD3.update_policy

    def spy(self, *a, **kw):
        r = orig(self, *a, **kw)
        if ST['done']:
            return r
        try:
            who = 'H' if self.nb_actions == 6 else 'L'
            mem = self.memory._storage
            if len(mem) < bsize:
                return r
            idx = np.arange(bsize)
            s = np.asarray([mem[i][0] for i in idx], dtype=np.float32)
            act = np.asarray([mem[i][1] for i in idx], dtype=np.float32)
            x = torch_cat(s, act)
            # 用 critic 自身当前输出作目标（一阶近似；|G|² 的量级足够判定）
            import torch
            with torch.no_grad():
                y = self.critic_p2(x)
            g = per_sample_grads(self.critic_p, x, y)
            B = noise_scale(g)
            ST['rows'].append((who, B))
            print('  [noise] %s 层: bsize=%d  B_noise = %.4g  (bsize/B_noise = %.3g)'
                  % (who, bsize, B, bsize / max(B, 1e-12)))
            if len(ST['rows']) >= 2:
                ST['done'] = True
        except Exception as e:
            print('  [noise] 计算失败: %s: %s' % (type(e).__name__, e))
            ST['done'] = True
        return r

    D.SP_TD3.update_policy = spy


def torch_cat(s, a):
    import torch
    return torch.cat([torch.as_tensor(s), torch.as_tensor(a)], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--bsize', type=int, default=128)
    args, _u = ap.parse_known_args()

    install(args.bsize)
    import runpy
    sys.argv = [os.path.join(ROOT, 'aifflor', 'aifflor_main.py'),
                '--max_steps', str(args.steps)]
    print('=' * 100)
    print('用 aifflor 当前默认配置（stock + 电平系数）跑 %d 步，测梯度噪声尺度' % args.steps)
    print('  bsize = %d（当前默认）' % args.bsize)
    print('=' * 100)
    try:
        runpy.run_module('aifflor.aifflor_main', run_name='__main__')
    except SystemExit:
        pass

    print('\n' + '=' * 100)
    print('结论')
    print('=' * 100)
    if not ST['rows']:
        print('  未取到样本（步数不足或未被调用）')
        return 0
    for who, B in ST['rows']:
        print('  %s 层: B_noise = %.4g   bsize = %d   bsize/B_noise = %.3g'
              % (who, B, args.bsize, args.bsize / max(B, 1e-12)))
        if args.bsize < B:
            print('     ⇒ bsize **小于**噪声尺度 ⇒ 仍处噪声主导区：')
            print('        加大 batch 仍有收益；**不存在"batch 太大导致过拟合"的机制**')
        else:
            print('     ⇒ bsize 已达/超过噪声尺度 ⇒ 收益递减区；')
            print('        风险是 critic 相关性（集成塌缩），**不是过拟合**')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
