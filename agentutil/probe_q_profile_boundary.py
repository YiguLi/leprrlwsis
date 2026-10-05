# -*- coding: utf-8 -*-
"""逐维动作边界剖面探针: Q(a_j) 一维切片 + 界内外签名斜率 + 峰位。

为什么需要它
------------
"越界"这个现象有**两种完全不同的成因**，而修法完全不同：

  (A) **边界最优**: 该维的最优动作正好落在约束边界上（角点解）。Q(a_j) 在界内
      上升、在界外下降，峰值贴着边界。此时越界是"平滑策略逼近角点解"的必然
      副产品；修法是**把界外惩罚做陡**（让跨界明确亏损），或把动作参数化到有界。
  (B) **critic 外推**: critic 在界外把界内的经济斜率延续出去（符号或幅度错），
      actor 被"错误地"推出界。此时修法是修 critic（输入 clip / 有界参数化）。

只看 `M` 越界率或 `penalty_NN_H` 无法区分这两者。一维切片可以直接看出来：
  * 峰值位置（在界内 / 正好在界上 / 在界外）
  * 界外斜率 vs 理论惩罚斜率（线性罚 = penalty_coef / s_p）
  * "反差" = 典型越界深度 x |界外斜率|，与 critic 自身 TD 误差比较：
    反差 << TD 误差 ⇒ 边界位置落在 critic 噪声里, actor 定位不到边界。

口径
----
* H 状态 = Normalize_H(obs, minute, mode=1) = [obs0/1000, obs1/10, obs2/10, obs3/10,
  sin(2pi m/1440), cos(...)]（与 td3per_myfunc 逐位一致，**不 import 该模块以免启 MATLAB**）。
* 动作边界（CheckAction）: dim0-4 = [0, 0.5]（机组限功率/诱导因子）, dim5 = [-0.5, 0.5]（SOC 信号）。
* 一律在 **critic 回归空间**报数: r1/c1 进 critic 前会除以 value_scale(s_p=1000)，
  所以 raw 惩罚系数 200 对应 0.2 critic 单位/单位动作。

用法
----
    python agentutil/probe_q_profile_boundary.py run13 --steps 40000
    python agentutil/probe_q_profile_boundary.py run13 --steps 34000 40000 --td-error 0.64
    python agentutil/probe_q_profile_boundary.py run12 run13 --dims 4 5
    python agentutil/probe_q_profile_boundary.py run13 --grid -1.0 1.0 0.05
"""

import argparse
import csv
import glob
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np
import torch

from model import Actor, Critic

DEFAULT_BOUNDS = [(0.0, 0.5)] * 5 + [(-0.5, 0.5)]


def run_dir(name):
    """解析 run（单一真源见 `agentutil/runpaths.py`）：支持任意前缀，如 AIFFLOR-run62。"""
    from runpaths import resolve_run
    return resolve_run(name)


def _log_csv(d):
    """该 run 的逐步日志表（`df_td3per.csv` 或 `df_<tag>.csv`，按列识别）。"""
    from runpaths import find_log_csv
    p = find_log_csv(d)
    if not p:
        raise SystemExit('%s 里找不到逐步日志表（需含 original_action_H/turbine_profit）' % d)
    return p


def ckpt_steps(d):
    out = []
    for p in glob.glob(os.path.join(d, 'checkpoint_*')):
        m = re.search(r'(\d+)$', p)
        if m and os.path.exists(os.path.join(p, 'actor_6.pkl')):
            out.append(int(m.group(1)))
    return sorted(out)


def h_state(obs, minute, mode=1):
    """与 td3per_myfunc.Normalize_H(mode=1) 逐位一致（不 import 以免启 MATLAB）。"""
    base = [obs[0] / 1000.0, obs[1] / 10.0, obs[2] / 10.0, obs[3] / 10.0]
    if mode <= 0:
        return np.array(base)
    ang = 2.0 * np.pi * (int(minute) % 1440) / 1440.0
    return np.array(base + [np.sin(ang), np.cos(ang)])


def build_panel(d, n_panel, every, price='ANY'):
    """抽面板状态（H 层 6 维）。

    `price` ∈ {ANY, LOW, MEDIUM, HIGH}：按日志 `pr` 列**筛时段**。
    为什么要这个：Q 剖面是在面板上平均的，而 `r1 = pr·ptur·t` 把电价乘进奖励 ⇒
    **峰段与谷段的 Q 曲面本来就不同**。不筛时段时，峰段的"低 a"偏好会被谷段平均掉，
    于是看不到"策略为什么在峰段把 a 压到 0.02"。
    """
    levels = {'LOW': (300.0,), 'MEDIUM': (580.0,), 'HIGH': (980.0,)}
    want = levels.get(str(price).upper())
    rows, phase_err, n_seen = [], 0.0, 0
    with open(_log_csv(d), newline='') as f:
        rd = csv.reader(f)
        head = next(rd)
        ix = {c.strip(): i for i, c in enumerate(head)}
        for i, r in enumerate(rd):
            if i % every:
                continue
            n_seen += 1
            try:
                if want is not None:
                    if abs(float(r[ix['pr']]) - want[0]) > 1.0:
                        continue
                obs = np.array([float(x) for x in r[ix['observation']].strip().strip('[]').split()])[:4]
                pm = float(r[ix['phase_minute']])
            except (ValueError, IndexError, KeyError):
                continue
            st = h_state(obs, pm, 1)
            try:
                phase_err = max(phase_err, abs(st[4] - float(r[ix['phase_sin']])),
                                abs(st[5] - float(r[ix['phase_cos']])))
            except (KeyError, ValueError, IndexError):
                pass
            rows.append(st)
    if not rows:
        raise SystemExit('%s 里没抽到面板状态（price=%s）' % (d, price))
    idx = np.linspace(0, len(rows) - 1, min(n_panel, len(rows))).astype(int)
    return np.stack([rows[i] for i in idx]), phase_err, (len(rows), n_seen)


def load_nets(d, st, which='p'):
    sd = torch.load(os.path.join(d, 'checkpoint_%d' % st, 'actor_6.pkl'),
                    map_location='cpu', weights_only=False)
    act = Actor(sd['fc1.weight'].shape[1], sd['fc4.weight'].shape[0],
                hidden1=sd['fc1.weight'].shape[0], hidden2=sd['fc2.weight'].shape[0])
    act.load_state_dict(sd)
    act.eval()
    nm = 'critic_p_6.pkl' if which == 'p' else 'critic_c_6.pkl'
    sd = torch.load(os.path.join(d, 'checkpoint_%d' % st, nm),
                    map_location='cpu', weights_only=False)
    h2 = sd['fc4.weight'].shape[0]
    cri = Critic(sd['fc1.weight'].shape[1], sd['fc2.weight'].shape[1] - h2,
                 hidden1=sd['fc1.weight'].shape[0], hidden2=h2)
    cri.load_state_dict(sd)
    cri.eval()
    return act, cri


def profile(act, cri, X, dim, grid):
    """固定其它维在 actor 自身输出上, 只扫 dim 维 -> Q(a_j) 面板均值曲线。"""
    with torch.no_grad():
        A0 = act(X)
    qs = []
    for v in grid:
        A = A0.clone()
        A[:, dim] = float(v)
        with torch.no_grad():
            qs.append(float(cri([X, A]).mean()))
    return A0, np.asarray(qs)


def main():
    ap = argparse.ArgumentParser(description='逐维动作边界剖面: Q(a_j) 切片与界内外斜率',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='run 名（自动补 output/TD3PER-<name>）或目录')
    ap.add_argument('--steps', nargs='*', type=int, default=None, help='要看的 checkpoint（默认各 run 最后一个）')
    ap.add_argument('--dims', nargs='*', type=int, default=None, help='要剖的维（默认 0..5 全部）')
    ap.add_argument('--which', choices=['p', 'c'], default='p', help='剖哪个 critic（默认 profit 通道）')
    ap.add_argument('--panel', type=int, default=1500, help='面板状态数（默认 1500）')
    ap.add_argument('--every', type=int, default=20, help='读日志抽样步长（默认 20）')
    ap.add_argument('--grid', nargs=3, type=float, default=[-1.0, 1.0, 0.05],
                    metavar=('LO', 'HI', 'STEP'), help='扫描网格（默认 -1 1 0.05）')
    ap.add_argument('--s-p', type=float, default=1000.0, help='profit 通道回归尺度 s_p（默认 1000）')
    ap.add_argument('--penalty-coef', type=float, default=200.0, help='现有线性罚系数（默认 200/单位）')
    ap.add_argument('--exp-coef', type=float, default=5.0, help='拟议指数罚的指数系数 c（默认 5）')
    ap.add_argument('--td-error', type=float, default=None,
                    help='critic 自身 TD 误差（原始单位=块奖励；给了才算"反差 vs 噪声"）')
    ap.add_argument('--price', default='ANY', choices=['ANY', 'LOW', 'MEDIUM', 'HIGH'],
                    help='只在该电价档的状态上建面板（默认 ANY=不筛）。'
                         '峰段(980)与谷段(300)的 Q 曲面不同 ⇒ 判"峰段压 a"必须筛时段')
    args = ap.parse_args()

    grid = np.round(np.arange(args.grid[0], args.grid[1] + 1e-9, args.grid[2]), 6)
    dims = args.dims if args.dims else list(range(6))
    coef_lin = args.penalty_coef / args.s_p      # 现线性罚在 critic 空间的斜率
    td_c = (args.td_error / args.s_p) if args.td_error is not None else None

    print('线性罚理论斜率 = %.0f/%.0f = %.3f critic 单位/单位动作' % (args.penalty_coef, args.s_p, coef_lin))
    if td_c is not None:
        print('critic TD 误差  = %.1f 原始单位 = %.3f critic 单位  (PER 采样偏置, 是上界)' % (args.td_error, td_c))
    print('指数罚(拟议) 界外斜率 = c × 该步收益 × e^{-cx} / s_p —— 与收益同尺度, 需用实测 s_in 校准 c')

    for name in args.runs:
        d = run_dir(name)
        steps = args.steps if args.steps else [ckpt_steps(d)[-1]]
        X, perr, (n_pan_raw, n_seen) = build_panel(d, args.panel, args.every, args.price)
        X = torch.tensor(X, dtype=torch.float32)
        print('\n' + '=' * 108)
        print('=== %s: 面板 %d 状态 (H dim=%d; 相位口径自校验偏差 %.2e; 时段=%s: %d/%d 行入选) ==='
              % (os.path.basename(d.rstrip('/\\')), len(X), X.shape[1], perr,
                 args.price, n_pan_raw, n_seen))
        print('=' * 108)
        for st in steps:
            if not os.path.exists(os.path.join(d, 'checkpoint_%d' % st, 'actor_6.pkl')):
                print('  checkpoint_%d 不存在, 跳过' % st)
                continue
            act, cri = load_nets(d, st, args.which)
            with torch.no_grad():
                A0 = act(X)
                qbar = float(cri([X, A0]).mean())
            Av = A0.numpy()
            print('\n--- checkpoint %d | Qbar=%.3f (%s 通道) ---' % (st, qbar, args.which))

            # 逐维越界统计
            print('  %-5s %8s %10s %10s   %s' % ('dim', '越界占比', '平均深度', '最大深度', '边界'))
            stats = {}
            for j in dims:
                lo, hi = DEFAULT_BOUNDS[j]
                over = np.maximum(Av[:, j] - hi, lo - Av[:, j])
                nz = over > 1e-9
                stats[j] = (float(nz.mean()), float(over[nz].mean()) if nz.any() else 0.0, float(over.max()))
                print('  dim%-2d %8.3f %10.4f %10.4f   [%.1f, %.1f]' % (j, *stats[j], lo, hi))

            # 逐维剖面
            print('\n  %-5s %9s %11s %11s %11s %11s   %s'
                  % ('dim', '峰位', '界内(下)', '界外(下)', '界内(上)', '界外(上)', '判定'))
            for j in dims:
                lo, hi = DEFAULT_BOUNDS[j]
                A0j, qs = profile(act, cri, X, j, grid)

                def slope(x0, x1):
                    i0 = int(np.argmin(np.abs(grid - x0)))
                    i1 = int(np.argmin(np.abs(grid - x1)))
                    if i0 == i1:
                        return float('nan')
                    return float((qs[i1] - qs[i0]) / (grid[i1] - grid[i0]))

                peak = float(grid[int(np.argmax(qs))])
                peak_at_edge = int(np.argmax(qs)) in (0, len(grid) - 1)
                # 贴着下/上界的界内、界外斜率 (0.1 宽小窗 = actor 实际感受到的局部力)
                s_in_lo = slope(lo, lo + 0.1)
                s_out_lo = slope(lo, lo - 0.1)
                s_in_hi = slope(hi, hi - 0.1)
                s_out_hi = slope(hi, hi + 0.1)
                # 判定: 只看"实际越界那一侧"
                frac, depth, _ = stats[j]
                mean_a = float(Av[:, j].mean())
                upper = abs(mean_a - hi) < abs(mean_a - lo)
                if upper:
                    in_slope, out_slope, side = s_in_hi, s_out_hi, '上'
                    want = -coef_lin          # 上界: 界外正确斜率 = -penalty/s_p
                    peak_out = peak > hi + 1e-9
                else:
                    in_slope, out_slope, side = s_in_lo, s_out_lo, '下'
                    want = +coef_lin          # 下界: 界外正确斜率 = +penalty/s_p
                    peak_out = peak < lo - 1e-9
                contrast = abs(out_slope) * depth
                if frac < 1e-9:
                    verdict = '无越界'
                elif not np.isfinite(out_slope) or np.sign(out_slope) != np.sign(want):
                    verdict = ('%s界外斜率 %+.3f 符号错(应 %+.3f) => critic 偏好在界外 (B)'
                               % (side, out_slope, want))
                elif abs(out_slope) < abs(in_slope):
                    verdict = ('%s界外回正力 %.3f < 界内 %.3f => 惩罚太浅, 损失不对称指向界外 (A)'
                               % (side, abs(out_slope), abs(in_slope)))
                else:
                    verdict = ('%s界外回正力 %.3f >= 界内 %.3f => 主因是平滑策略逼近角点解'
                               % (side, abs(out_slope), abs(in_slope)))
                if peak_out:
                    verdict += ' | ⚠ Q 峰在界外 %+.2f' % peak
                if peak_at_edge:
                    verdict += ' | 峰在网格端点, 真峰可能更远'
                if td_c is not None and frac > 1e-9:
                    verdict += ' | 反差 %.3f vs 噪声 %.3f' % (contrast, td_c)
                print('  dim%-2d %+9.2f %11.3f %11.3f %11.3f %11.3f   %s'
                      % (j, peak, s_in_lo, s_out_lo, s_in_hi, s_out_hi, verdict))

            print('\n  读法: 界外"正确"斜率在下界为 %+.3f、上界为 %+.3f（越大越好 => 回到界内）。'
                  % (coef_lin, -coef_lin))
            print('        Q 峰落在界外 => critic 偏好在界外动作 = 惩罚没学到/太浅 (B);')
            print('        峰在界内但仍越界 => 平滑策略逼近角点解 + 界外回正力不足 (A)。')
            print('        "反差 = 平均深度 x |界外斜率|" 要 > critic 噪声, 边界位置才可辨识。')
            print('        注意: 局部斜率对面板抽样敏感, 换 --panel/--every 复核后再下结论。')


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    main()
