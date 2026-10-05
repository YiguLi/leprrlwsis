# -*- coding: utf-8 -*-
"""真值梯度 vs critic 梯度对比探针 —— 坐实 / 证伪 "critic 在 a 方向过拟合"。

立论
----
actor 的更新方向是 `∂Q_p/∂a`（通过 critic）。上一轮审计测到
`|∂Q_p/∂a|/Qbar ≈ 2.5`（step 92000 时 |∂Q_p/∂a|≈17.8，Qbar≈7.1），看起来"陡得不像
真的"。但单机 `C_p=4a(1−a)²` 在 `a=0.246`（实测动作均值）处 `dC_p/da≈0.79`，**不是
0**（0 只在峰点 a=1/3）。所以"critic 梯度是不是幻觉"**不能靠手算，只能靠对 env 做
有限差分**，把两者放进**同一个归一化 Q 空间**比。

本脚本
------
对同一批状态面板，做两件事并把它们放在同一单位：

  A) **critic 梯度**  g_crit[i] = ∂Q_p(s, a*)/∂a_i          （autograd；a* = clip(μ(s))）
  B) **真值梯度**    g_env[i]  = ∂(块奖励)/∂a_i 折算进同一空间（WFSim 中心差分）

折算（关键，写死在这里方便审计）
--------------------------------
H 层 critic 在**归一化空间**回归：Q_p_norm = E[Σγ^k r_block] / s_p，其中
`r_block` = 5 步 sum_r1（a 在一个块内固定 5 步）。风机项对 a 的依赖只有
`ptur = PowerOutput(u, a)`，故（把块内 5 步近似为面板同分布 i.i.d.）：

    ∂Q_norm/∂a_i ≈ 5 · pr · dt · ∂ptur/∂a_i / ((1−γ)·s_p)
                  = pr · ∂ptur/∂a_i / 240        (γ=0.98, s_p=1000, dt=1/60, 5 步)

    ∂ptur/∂a_i [MW/单位动作] 由 WFSim 中心差分给出（h=0.01，边界处退化为单侧）。

成本通道 ∂Q_c/∂a 单独不计：实测 ~0.07/单位 × λ~0.1 ≈ 0.007，可忽略。

判读
----
  ratio = g_crit / g_env（逐维 + 整体）:
    ≈1         → critic 忠实，慢收敛是噪声/符号冲突，不是过拟合；
    ≫1 (如 5×) → critic 在 a 方向被放大（过拟合），actor 追虚高梯度；
    ≪1         → critic 欠拟合（梯度没学出来）。

⚠ 需要 MATLAB（引擎启动与 main 相同：只 `start_matlab()`）。沙箱/无 MATLAB 环境用
`--no-matlab`：只算 critic 侧，并把面板落盘成 CSV 供另一台机器补算 env 侧。

用法
----
    python agentutil/probe_grad_true_vs_critic.py run48 --step 92000          # 完整对比
    python agentutil/probe_grad_true_vs_critic.py run48 --step 92000 --no-matlab   # 只 critic 侧
    python agentutil/probe_grad_true_vs_critic.py output/TD3PER-run48/checkpoint_92000
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

# ---- 折算口径（与 hsrl_cost / main 一致，单一真源处已在 hsrl_cost；此处镜像其默认值）----
DT = 1.0 / 60.0          # 步长 (h)
BLOCK = 5                # H 层一个决策块 = 5 步
GAMMA = 0.98             # 块级折扣
S_P = 1000.0             # H 层 r1 值函数归一化尺度
BOUNDS = [(0.0, 0.5)] * 5 + [(-0.5, 0.5)]   # CheckAction: WT x5 + signal x1


def conv_factor(pr, gamma=GAMMA, s_p=S_P):
    """把 ∂ptur/∂a [MW/单位] 折算成 ∂Q_norm/∂a 的系数 = 5·pr·dt/((1−γ)·s_p)。"""
    return BLOCK * float(pr) * DT / ((1.0 - float(gamma)) * float(s_p))


# ---------------------------------------------------------------------------
# 网络加载（与 probe_q_profile_boundary.py / audit_policy_stability.py 同口径）
# ---------------------------------------------------------------------------
def load_actor(ckpt_dir):
    sd = torch.load(os.path.join(ckpt_dir, 'actor_6.pkl'),
                    map_location='cpu', weights_only=False)
    m = Actor(sd['fc1.weight'].shape[1], sd['fc4.weight'].shape[0],
              hidden1=sd['fc1.weight'].shape[0], hidden2=sd['fc2.weight'].shape[0])
    m.load_state_dict(sd)
    m.eval()
    return m


def load_critic(ckpt_dir, which='p'):
    nm = 'critic_%s_6.pkl' % which
    sd = torch.load(os.path.join(ckpt_dir, nm), map_location='cpu', weights_only=False)
    h2 = sd['fc4.weight'].shape[0]
    m = Critic(sd['fc1.weight'].shape[1], sd['fc2.weight'].shape[1] - h2,
               hidden1=sd['fc1.weight'].shape[0], hidden2=h2)
    m.load_state_dict(sd)
    m.eval()
    return m


def resolve(run, step):
    """run 名（任意前缀，如 AIFFLOR-run62 / run15）或直接给 checkpoint 目录。"""
    if os.path.isdir(run) and os.path.exists(os.path.join(run, 'actor_6.pkl')):
        return run
    from runpaths import resolve_run
    d = resolve_run(run)
    if step is not None and os.path.isdir(os.path.join(d, 'checkpoint_%d' % step)):
        return os.path.join(d, 'checkpoint_%d' % step)
    cks = sorted(int(re.search(r'(\d+)$', p).group(1))
                 for p in glob.glob(os.path.join(d, 'checkpoint_*'))
                 if os.path.exists(os.path.join(p, 'actor_6.pkl')))
    if not cks:
        raise SystemExit('%s 没有可用 checkpoint' % d)
    return os.path.join(d, 'checkpoint_%d' % cks[-1])


def build_panel(n, seed=0):
    """确定性状态面板：u∈[4,14]（WFSim 有效区）、pr∈{300,580,980}、相位均匀。"""
    rng = np.random.RandomState(seed)
    u = rng.uniform(4.0, 14.0, n)
    pr = rng.choice([300.0, 580.0, 980.0], n)
    E = rng.uniform(0.5, 5.5, n)
    PG = rng.uniform(0.0, 8.0, n)
    ang = 2.0 * np.pi * rng.uniform(0, 1440, n) / 1440.0
    S = np.stack([pr / 1000.0, u / 10.0, E / 10.0, PG / 10.0,
                  np.sin(ang), np.cos(ang)], 1)
    return S.astype(np.float32), u, pr


def clip_action(A):
    """CheckAction 的界内裁剪（actor 无界输出 → env 实际看到的动作）。"""
    B = A.clone()
    for k, (lo, hi) in enumerate(BOUNDS):
        B[:, k].clamp_(lo, hi)
    return B


def critic_grad(actor, critic, S, A):
    """∂Q_p/∂a 在 (S, A) 处（A 为裁剪后的 6 维动作）。返回 (N, 6) 梯度。"""
    A = A.clone().detach().requires_grad_(True)
    q = critic([S, A]).sum()
    g = torch.autograd.grad(q, A)[0].detach().numpy()
    return g


# ---------------------------------------------------------------------------
# WFSim 侧（需要 MATLAB）
# ---------------------------------------------------------------------------
def _power(eng, u, a5):
    r = eng.WFSim_port(__import__('matlab').double([float(u)]),
                       __import__('matlab').double([0.0]),
                       __import__('matlab').double([list(map(float, a5))]),
                       __import__('matlab').double([0.0]))
    return float(r['out']) / 1e6


def env_grad_row(eng, u, a5, h=0.01):
    """∂P_farm/∂a_j 逐维中心差分 [MW/单位动作]；边界处退化为单侧。"""
    g = np.zeros(5)
    a5 = np.clip(np.asarray(a5, dtype=float), 0.0, 0.5)
    for j in range(5):
        ap, am = a5.copy(), a5.copy()
        ap[j] = min(0.5, a5[j] + h)
        am[j] = max(0.0, a5[j] - h)
        dp = ap[j] - am[j]
        if dp > 1e-9:
            g[j] = (_power(eng, u, ap) - _power(eng, u, am)) / dp
        elif a5[j] <= 0.0:                       # 贴下界：前向单侧
            ap2 = a5.copy(); ap2[j] = min(0.5, a5[j] + 2.0 * h)
            g[j] = (_power(eng, u, ap2) - _power(eng, u, a5)) / max(ap2[j] - a5[j], 1e-9)
        else:                                     # 贴上界：后向单侧
            am2 = a5.copy(); am2[j] = max(0.0, a5[j] - 2.0 * h)
            g[j] = (_power(eng, u, a5) - _power(eng, u, am2)) / max(a5[j] - am2[j], 1e-9)
    return g


# ---------------------------------------------------------------------------
def report(g_crit5, g_env5, u, pr):
    """g_*5: (N, 5)。打印逐维 + 整体对比。"""
    print('\n' + '=' * 96)
    print('逐维对比（g_crit = critic 梯度, g_env = WFSim 真值梯度, 同为 ∂Q_norm/∂a 单位）')
    print('=' * 96)
    print('  %-5s %12s %12s %10s %12s %12s %10s'
          % ('dim', 'g_crit均值', 'g_crit std', 'g_env均值', 'g_env std', 'ratio均值', '符号一致'))
    for j in range(5):
        gc = g_crit5[:, j]
        ge = g_env5[:, j]
        ratio = np.divide(gc, ge, out=np.full_like(gc, np.nan), where=np.abs(ge) > 1e-9)
        ratio = ratio[np.isfinite(ratio)]
        sign = np.mean(np.sign(gc) == np.sign(ge))
        print('  dim%-3d %+12.3f %12.3f %+12.3f %12.3f %10.2f %11.2f'
              % (j, gc.mean(), gc.std(), ge.mean(), ge.std(),
                 ratio.mean() if ratio.size else float('nan'), sign))

    gc_all = g_crit5.ravel()
    ge_all = g_env5.ravel()
    m_abs = lambda x: float(np.abs(x).mean())
    ratio_all = m_abs(gc_all) / max(m_abs(ge_all), 1e-9)
    # 5 维均值向量的皮尔逊相关（方向一致性）
    gcv, gev = g_crit5.mean(0), g_env5.mean(0)
    corr = float(np.corrcoef(gcv, gev)[0, 1]) if np.std(gcv) > 1e-9 and np.std(gev) > 1e-9 else float('nan')
    print('\n  整体: mean|g_crit| = %.3f   mean|g_env| = %.3f   ⇒  ratio = %.2f'
          % (m_abs(gc_all), m_abs(ge_all), ratio_all))
    print('        5 维均值向量相关 r = %+.2f' % corr)
    print('  u 范围 [%.1f, %.1f], pr 取值 %s' % (u.min(), u.max(), np.unique(pr).tolist()))
    print('  判读: ratio≈1 → critic 忠实(慢收敛是噪声/冲突); ratio≫1 → critic 过拟合放大; '
          'ratio≪1 → 欠拟合。')


def main():
    ap = argparse.ArgumentParser(description='真值梯度 vs critic 梯度（需 MATLAB；--no-matlab 只算 critic 侧）',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', help='run 名（自动补 output/TD3PER-<name>）或 checkpoint 目录')
    ap.add_argument('--step', type=int, default=None, help='checkpoint 步（默认该 run 最后一个）')
    ap.add_argument('--n', type=int, default=120, help='面板状态数（默认 120 ⇒ 1200 次 WFSim 调用）')
    ap.add_argument('--h', type=float, default=0.01, help='有限差分步长（默认 0.01）')
    ap.add_argument('--which', choices=['p', 'p2', 'both'], default='p', help='critic 通道（默认 p）')
    ap.add_argument('--no-matlab', action='store_true', help='只算 critic 侧，并把面板落盘')
    ap.add_argument('--out-panel', default=None, help='面板 CSV 输出路径（默认 agentutil/_grad_panel.csv）')
    ap.add_argument('--gamma', type=float, default=GAMMA, help='块级折扣（默认 0.98）')
    ap.add_argument('--s-p', type=float, default=S_P, help='值函数归一化尺度（默认 1000）')
    args = ap.parse_args()
    gamma, s_p = args.gamma, args.s_p

    ckpt = resolve(args.run, args.step)
    print('checkpoint: %s' % ckpt)
    S, u, pr = build_panel(args.n)
    print('面板: %d 个状态 (u∈[%.1f,%.1f], pr∈%s), 折算系数 conv(pr)=pr/%.1f'
          % (args.n, u.min(), u.max(), np.unique(pr).tolist(),
             ((1.0 - gamma) * s_p) / (BLOCK * DT)))

    actor = load_actor(ckpt)
    with torch.no_grad():
        A_raw = actor(torch.tensor(S))
    A_clip = clip_action(A_raw)
    print('actor 输出（裁剪后）a[:5] 均值 %s，越界(原始)占比 %.3f'
          % (np.round(A_clip.numpy()[:, :5].mean(0), 3).tolist(),
             float(((A_raw.numpy() < 0) | (A_raw.numpy() > 0.5))[:, :5].any(1).mean())))

    whichs = ['p', 'p2'] if args.which == 'both' else [args.which]
    for w in whichs:
        critic = load_critic(ckpt, w)
        g_crit = critic_grad(actor, critic, torch.tensor(S), A_clip)[:, :5]
        print('\n[critic %s] |∂Q/∂a| 逐维均值: %s   L2范数均值: %.3f'
              % (w, np.round(np.abs(g_crit).mean(0), 3).tolist(),
                 float(np.linalg.norm(g_crit, axis=1).mean())))

    # ---- env 侧 ----
    if args.no_matlab:
        out = args.out_panel or os.path.join(_HERE, '_grad_panel.csv')
        with open(out, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['idx', 'u', 'pr', 'a0', 'a1', 'a2', 'a3', 'a4'])
            for i in range(args.n):
                wr.writerow([i, '%.4f' % u[i], '%.1f' % pr[i]]
                            + ['%.4f' % v for v in A_clip.numpy()[i, :5]])
        print('\n[--no-matlab] 已落盘面板 → %s（%d 行）' % (out, args.n))
        print('在 MATLAB 机器上跑完整对比:  python agentutil/probe_grad_true_vs_critic.py %s --step %s'
              % (args.run, args.step))
        return

    import matlab
    import matlab.engine
    print('\n启动 MATLAB 引擎（与 main 相同：只 start_matlab()）...', flush=True)
    eng = matlab.engine.start_matlab()
    print('引擎就绪，开始 %d 次 WFSim 有限差分 (n=%d × 5 维 × 2)...' % (args.n * 5 * 2, args.n), flush=True)

    A5 = A_clip.numpy()[:, :5]
    g_env = np.zeros((args.n, 5))
    for i in range(args.n):
        g_env[i] = env_grad_row(eng, u[i], A5[i], args.h)
        if (i + 1) % 20 == 0:
            print('  ... %d/%d' % (i + 1, args.n), flush=True)

    # 折算进归一化 Q 空间：g_env_norm[i,j] = conv(pr[i]) · ∂P/∂a
    g_env_norm = g_env * np.array([conv_factor(p, gamma, s_p) for p in pr])[:, None]

    g_crit5 = critic_grad(actor, load_critic(ckpt, 'p'), torch.tensor(S), A_clip)[:, :5]
    report(g_crit5, g_env_norm, u, pr)
    eng.quit()


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    main()
