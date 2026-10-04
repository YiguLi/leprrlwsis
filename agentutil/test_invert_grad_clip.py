# -*- coding: utf-8 -*-
"""`InvertGradClip` / `action_bounds` 契约测试（纯 torch，无 MATLAB）。

验证三件事：
  1. ``action_bounds`` 对 H(6 维轴向/偏航)、H(5 维)、L(1 维) 返回正确逐维 lo/hi；
  2. ``InvertGradClip.forward`` ≡ ``clamp``；
  3. ``InvertGradClip.backward`` 的"边界反射"语义：
       - 界内/往界内推 → 梯度原样保留；
       - 上界外且往外推(g>0) → 取反；
       - 下界外且往外推(g<0) → 取反；
    以及关键性质：**界外梯度永不消失**（区别于普通 clamp 的死区）。

用法: python agentutil/test_invert_grad_clip.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
import torch

from model import InvertGradClip, action_bounds


def eq(a, b, tol=1e-6):
    return bool(np.allclose(np.asarray(a), np.asarray(b), atol=tol))


def test_action_bounds():
    lo, hi = action_bounds(6, 5)                       # H: 5 轴向 [0,0.5] + 信号 [-0.5,0.5]
    assert eq(lo, [0.0, 0.0, 0.0, 0.0, 0.0, -0.5]), lo
    assert eq(hi, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), hi

    lo, hi = action_bounds(6, 5, wt_lo=-1.0, wt_hi=1.0)   # yawflor: 偏航 [-1,1]
    assert eq(lo, [-1.0, -1.0, -1.0, -1.0, -1.0, -0.5]), lo
    assert eq(hi, [1.0, 1.0, 1.0, 1.0, 1.0, 0.5]), hi

    lo, hi = action_bounds(5, 5)                       # H(5 维, 无信号)
    assert eq(lo, [0.0] * 5) and eq(hi, [0.5] * 5)

    lo, hi = action_bounds(1, 5)                       # L(1 维): [-charge, charge]
    assert eq(lo, [-10.0]) and eq(hi, [10.0])
    print('[1] action_bounds: 6/6 维(轴向|偏航|5维|L) 逐维边界正确 ✓')


def test_forward_is_clamp():
    x = torch.tensor([[-0.2, 0.3, 0.7], [0.5, -0.1, 0.2]], requires_grad=True)
    lo = torch.tensor([0.0, 0.0, 0.0])
    hi = torch.tensor([0.5, 0.5, 0.5])
    y = InvertGradClip.apply(x, lo, hi)
    expected = x.detach().clamp(lo, hi)
    assert torch.equal(y, expected), (y, expected)
    print('[2] InvertGradClip.forward ≡ clamp ✓')


def test_backward_reflection():
    lo = torch.tensor([0.0])
    hi = torch.tensor([0.5])
    # 4 个样本（1 维动作）: [界内 0.3, 上界外 0.7, 下界外 -0.2, 上界外但梯度往内 0.6]
    x = torch.tensor([[0.3], [0.7], [-0.2], [0.6]], requires_grad=True)   # (4,1)
    y = InvertGradClip.apply(x, lo, hi)                                    # lo/hi 广播到 (4,1)
    # w = "critic 梯度" g 的方向:
    #   样本0 界内 g=+1 / 样本1 上界外往外 g=+1 / 样本2 下界外往外 g=-1 / 样本3 上界外往内 g=-1
    w = torch.tensor([[1.0], [1.0], [-1.0], [-1.0]])
    loss = (y * w).sum()
    g = torch.autograd.grad(loss, x)[0]
    # 期望:
    #   样本0 界内:  原样 → +1
    #   样本1 上界外 g>0: 取反 → -1
    #   样本2 下界外 g<0: 取反 → +1
    #   样本3 上界外 g<0(往内): 原样 → -1
    expected = torch.tensor([[1.0], [-1.0], [1.0], [-1.0]])
    assert torch.equal(g, expected), (g, expected)
    print('[3] InvertGradClip.backward 边界反射: 界内原样/界外往外推取反/往内推原样 ✓')


def test_grad_never_vanishes():
    """关键性质：界外梯度不为 0（普通 clamp 这里会是 0）。"""
    lo = torch.tensor([0.0])
    hi = torch.tensor([0.5])
    x = torch.tensor([0.8], requires_grad=True)   # 上界外
    y = InvertGradClip.apply(x, lo, hi)
    loss = y.sum()                                # g = +1（往外推）
    g = torch.autograd.grad(loss, x)[0]
    assert float(g[0]) == -1.0, g                 # 取反 → -1，非 0
    print('[4] 界外梯度永不消失 (普通 clamp 为 0, 此处 = %+.1f) ✓' % float(g[0]))


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    test_action_bounds()
    test_forward_is_clamp()
    test_backward_reflection()
    test_grad_never_vanishes()
    print('\n全部通过：InvertGradClip 前向=clip、反向=边界反射、界外梯度非零。')
