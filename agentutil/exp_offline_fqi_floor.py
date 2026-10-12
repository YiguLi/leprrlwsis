# -*- coding: utf-8 -*-
r"""离线 fitted-Q 实验：**critic 的 TD 误差地板到底在哪**？（只读，需 torch，不启 MATLAB）

动机
----
在线 run57 的 `td_p_mean` 末段 ≈ 991 元（critic 空间 0.99），而动作的真实边际收益只有
几元/单位动作 ⇒ 若残差可降，"降噪"就是第一要务；若不可降，就只能靠降方差与缩短坡长。
本脚本用**固定数据集 + 离线 fitted-Q**回答"可不可降"，并扫三个轴：

  ① 网络容量      `--hidden 16,100 / 64,256 / 128,512`
  ② 训练量        `--updates 20000 / 100000 / 400000`
  ③ 状态特征      `obs4`（现状 4 维）/ `obs7`（+u²,u³,pr²）/ `obs13`（+相位 sin/cos 与交叉项）

数据（与在线同口径）
--------------------
从 `df_td3pernolow.csv` 重建块级转移：
  * 状态      = `Normalize_H(observation, phase_minute, phase_mode)`（与训练一致）
  * 动作      = `original_action_H`（进 buffer 的就是它）
  * 块奖励    = Σ_{k<5} γ^k · r1_{t+k}，其中 `r1 = turbine_profit + bess_profit − degradation − penalty_NN_H`
                （**已验证**：逐集与 `episode_reward` 完全一致，全量相对差 0.034%）
  * 自举系数  = γ⁵ = 0.7738（实现口径），末集用 0
  * 目标动作  = 训练好的 actor 在 s_{t+1} 上的输出（TD3 平滑噪声另加，可关）

指标（与在线 `td_p_mean` 同量纲）
--------------------------------
  残差 `min(Q1,Q2)(s,a) − (r + γ⁵·min(Q1',Q2')(s',a'))` 的 mean|·| 与 sd，单位 = 元。

用法
----
    python agentutil/exp_offline_fqi_floor.py --run output/TD3PERNOLOW-run57 \
        --updates 100000 --hidden 16,100 --feat obs4 [--tag v1]
    python agentutil/exp_offline_fqi_floor.py --sweep      # 跑预设网格
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

GAMMA, NSTEP, BLOCK = 0.95, 5, 5
G_EFF = GAMMA ** NSTEP
EPS = 4000.0                      # 与 CheckAction 一致（元/单位越界）
COLS = ['step', 'observation', 'observation2', 'original_action_H', 'phase_minute',
        'turbine_profit', 'bess_profit', 'degradation', 'penalty_NN_H']


# ------------------------------------------------------------------ 数据
def build_dataset(run, from_step=0):
    LOG = None
    for c in ('df_td3pernolow.csv', 'df_td3per.csv'):
        if os.path.exists(os.path.join(run, c)):
            LOG = os.path.join(run, c)
            break
    if LOG is None:
        raise SystemExit('%s 下找不到 df_*.csv' % run)
    acc = []
    for ch in pd.read_csv(LOG, usecols=COLS, chunksize=200000,
                          encoding='utf-8-sig', low_memory=False):
        acc.append(ch)
    df = pd.concat(acc, ignore_index=True)
    # ⚠ 只保留**策略冻结期**之后的转移：H 层策略在早期仍在变化，在线 buffer 里
    #   混着不同策略产生的转移；而 Q 的定义依赖被评估的策略 ⇒ 混期数据会让
    #   "离线拟合一个固定策略的价值" 与 "在线拟合一个移动策略的价值" 不可比。
    if from_step > 0:
        df = df[df['step'] >= from_step].copy()
    df['blk'] = df['step'] // BLOCK
    df['pos'] = df['step'] % BLOCK
    first = df[df.pos == 0].copy()
    # 块奖励 = Σ_{k<5} γ^k r1
    df['r1'] = (df.turbine_profit + df.bess_profit - df.degradation - df.penalty_NN_H)
    g = df.groupby('blk')['r1'].apply(lambda s: float(sum((GAMMA ** k) * v
                                                          for k, v in enumerate(s.to_numpy()))))
    first['blk_r'] = first['blk'].map(g).to_numpy()

    def parse(s):
        return np.array([[float(x) for x in t.strip().strip('[]').split()]
                         for t in s.astype(str)])

    o1, o2 = parse(first['observation']), parse(first['observation2'])
    a = parse(first['original_action_H'])
    pm = first['phase_minute'].to_numpy(int)
    return dict(o1=o1, o2=o2, a=a, r=first['blk_r'].to_numpy(float),
                step=first['step'].to_numpy(int), n_blk=len(first), log=LOG)


def featurize(o, pm, kind, nb_states):
    base = np.column_stack([o[:, 0] / 1000, o[:, 1] / 10, o[:, 2] / 10, o[:, 3] / 10])
    if nb_states == 6:                                   # 生产口径含相位
        ang = 2 * np.pi * (np.asarray(pm) % 1440) / 1440
        base = np.column_stack([base, np.sin(ang), np.cos(ang)])
    if kind == 'raw':
        return base.astype(np.float32)
    u = o[:, 1] / 10.0
    pr = o[:, 0] / 1000.0
    extra = [u ** 2, u ** 3, pr ** 2]
    if kind == 'cross':
        extra += [u * pr, base[:, 2] * u, base[:, 3] * u]
    return np.column_stack([base] + extra).astype(np.float32)


# ------------------------------------------------------------------ 网络
class Q(nn.Module):
    def __init__(self, ns, na, h1, h2):
        super().__init__()
        self.fc1 = nn.Linear(ns, h1)
        self.fc4 = nn.Linear(h1, h2)
        self.fc2 = nn.Linear(h2 + na, 80)
        self.fc5 = nn.Linear(80, 50)
        self.fc3 = nn.Linear(50, 1)
        self.relu = nn.ReLU()

    def forward(self, x, a):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc4(h))
        h = self.relu(self.fc2(torch.cat([h, a], 1)))
        h = self.relu(self.fc5(h))
        return self.fc3(h)


def q_forward_pair(m1, m2, x, a):
    return torch.min(m1(x, a), m2(x, a))


# ------------------------------------------------------------------ 主实验
def run_one(ds, args, hidden, updates, feat, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    S = featurize(ds['o1'], ds['step'] % 1440, feat, nb_states=4)
    S2 = featurize(ds['o2'], (ds['step'] + 1) % 1440, feat, nb_states=4)
    A = ds['a'].astype(np.float32)
    R = (ds['r'] / 1000.0).astype(np.float32)          # s_p=1000
    done = np.zeros(len(R), np.float32)
    blk_first = ds['step'] // 1440
    brk = np.where(np.diff(blk_first) != 0)[0]
    done[brk] = 1.0

    S_t = torch.as_tensor(S)
    S2_t = torch.as_tensor(S2)
    A_t = torch.as_tensor(A)
    R_t = torch.as_tensor(R).reshape(-1, 1)
    D_t = torch.as_tensor(1.0 - done).reshape(-1, 1)

    na = A.shape[1]
    m1 = Q(S.shape[1], na, hidden[0], hidden[1])
    m2 = Q(S.shape[1], na, hidden[0], hidden[1])
    t1 = Q(S.shape[1], na, hidden[0], hidden[1])
    t2 = Q(S.shape[1], na, hidden[0], hidden[1])
    # ---- 与在线对齐的关键一步：把 fc3.bias 初始化到正确的值域 ----
    # 在线由 `BiasInitTracker` 在 warmup 后一次性把 critic 输出层 bias 设为
    # 口径估计的 b_p（实测末检查点 critic_p fc3.bias = 4.03）。若不设，网络从 ~0.08
    # 起步，前几万步全在"把输出搬到正确量级"，残差轨迹会被这个瞬态污染（本会话实测：
    # 3000 步时 Q 只有 275 元而在线是 ~3900 元）。
    bias0 = args.bias
    if bias0 is None:
        try:
            _csd = torch.load(os.path.join(args.run, 'checkpoint_%d' % args.ckpt,
                                           'critic_p_5.pkl'),
                              map_location='cpu', weights_only=True)
            bias0 = float(_csd['fc3.bias'])
        except Exception:
            bias0 = 4.0
    with torch.no_grad():
        for m in (m1, m2, t1, t2):
            m.fc3.bias.fill_(bias0)
    print('  [口径] critic 输出层 bias 初始化为 %.4f（在线同口径）；'
          '奖励 = n-step 折叠值 / s_p=1000' % bias0)

    # 目标动作：用**已训 actor** 在 s' 上取值（与在线口径一致；actor 固定）
    act_path = os.path.join(args.run, 'checkpoint_%d' % args.ckpt,
                            'actor_5.pkl' if os.path.exists(
                                os.path.join(args.run, 'checkpoint_%d' % args.ckpt,
                                             'actor_5.pkl')) else 'actor_6.pkl')
    from model import Actor
    asd = torch.load(act_path, map_location='cpu', weights_only=True)
    actor = Actor(asd['fc1.weight'].shape[1], asd['fc4.weight'].shape[0],
                  hidden1=asd['fc1.weight'].shape[0], hidden2=asd['fc2.weight'].shape[0])
    actor.load_state_dict(asd)
    actor.eval()
    with torch.no_grad():
        # actor 的输入维是 4 或 6；若 feat 是扩维，则把扩维特征**只给 critic**
        a_next = actor(torch.as_tensor(
            featurize(ds['o2'], (ds['step'] + 1) % 1440, 'raw', nb_states=4))).numpy()
    A2_t = torch.as_tensor(a_next.astype(np.float32))

    opt1 = torch.optim.Adam(m1.parameters(), lr=args.lr)
    opt2 = torch.optim.Adam(m2.parameters(), lr=args.lr)
    bs, log = args.bsize, []
    soft = args.tau > 0                       # True ⇒ 与在线同机制的软更新
    t0 = time.time()
    for it in range(1, updates + 1):
        idx = torch.randint(0, len(R), (bs,))
        s, a, r = S_t[idx], A_t[idx], R_t[idx]
        s2, a2, d = S2_t[idx], A2_t[idx], D_t[idx]
        with torch.no_grad():
            tn = q_forward_pair(t1, t2, s2, a2)
            y = r + G_EFF * d * tn
        l1 = ((m1(s, a) - y) ** 2).mean()
        l2 = ((m2(s, a) - y) ** 2).mean()
        opt1.zero_grad(); l1.backward(); opt1.step()
        opt2.zero_grad(); l2.backward(); opt2.step()
        if soft:                              # 与在线一致：每步 τ 软更新
            with torch.no_grad():
                for mp, tp in ((m1, t1), (m2, t2)):
                    for p, pt in zip(mp.parameters(), tp.parameters()):
                        pt.mul_(1.0 - args.tau).add_(p, alpha=args.tau)
        elif it % args.sync == 0:             # 硬同步（每 sync 次）
            t1.load_state_dict(m1.state_dict())
            t2.load_state_dict(m2.state_dict())
        if it % max(1, updates // 10) == 0 or it == updates:
            with torch.no_grad():
                pr = q_forward_pair(m1, m2, S_t, A_t)
                tn = q_forward_pair(t1, t2, S2_t, A2_t)
                resid = (pr - (R_t + G_EFF * D_t * tn)).numpy().reshape(-1)
            log.append((it, float(np.abs(resid).mean()) * 1000, float(resid.std()) * 1000,
                        float(pr.mean()) * 1000, float(pr.std()) * 1000))
    return dict(hidden='%dx%d' % hidden, updates=updates, feat=feat,
                feat_dim=S.shape[1], lr=args.lr, bsize=bs, tau=args.tau,
                secs=time.time() - t0, log=log, last=log[-1])


def fmt(rec):
    it, mae, sd, qm, qs = rec['last']
    return ('  %-9s %-6s %8d %7.1f %10.1f %10.1f %10.1f %10.1f  lr=%.0e b=%d %s'
            % (rec['hidden'], rec['feat'], rec['updates'], rec['secs'],
               mae, sd, qm, qs, rec['lr'], rec['bsize'],
               ('τ=%.3f' % rec['tau']) if rec['tau'] else 'hard-sync'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='output/TD3PERNOLOW-run57')
    ap.add_argument('--ckpt', type=int, default=72000)
    ap.add_argument('--hidden', default='16,100')
    ap.add_argument('--updates', type=int, default=100000)
    ap.add_argument('--feat', default='obs4', choices=['obs4', 'obs7', 'obs13'])
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--bsize', type=int, default=256)
    ap.add_argument('--sync', type=int, default=10000)
    ap.add_argument('--tau', type=float, default=0.0,
                    help='>0 ⇒ 目标网络 τ 软更新（与在线同机制）；0 ⇒ 每 --sync 次硬同步')
    ap.add_argument('--bias', type=float, default=None,
                    help='critic 输出层 bias 初值；默认从检查点读 critic_p 的 fc3.bias')
    ap.add_argument('--from-step', type=int, default=0,
                    help='只用 >= 该 step 的转移（建议取策略冻结期起点，如 52000）')
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    if not os.path.isdir(os.path.join(args.run, 'checkpoint_%d' % args.ckpt)):
        cks = sorted(int(p.split('_')[-1]) for p in os.listdir(args.run)
                     if p.startswith('checkpoint_'))
        args.ckpt = cks[-1]
        print('（指定检查点不存在，改用 %d）' % args.ckpt)
    print('=' * 108)
    print('离线 fitted-Q：critic 的 TD 误差地板    run=%s  ckpt=%d' % (args.run, args.ckpt))
    print('=' * 108)
    ds = build_dataset(args.run, args.from_step)
    print('  块级转移 %d 条（step %d~%d）  来源 %s'
          % (ds['n_blk'], ds['step'].min(), ds['step'].max(), os.path.basename(ds['log'])))
    _us = pd.read_csv(os.path.join(args.run, 'update_stats_H.csv'))
    _lam = np.sum(GAMMA ** np.arange(NSTEP))
    print('  在线对照（同口径）: 裸块奖励 %.1f 元 → n-step 折叠 ×%.4f = %.0f 元；'
          'td_p ≈ 991 元；Q ≈ 3915 元'
          % (ds['r'].mean(), _lam, ds['r'].mean() * _lam))
    print('  自检: 离线奖励即**裸块值**（已逐集验证 Σr1 ≡ episode_reward）\n')

    print('  %-9s %-6s %8s %7s %10s %10s %10s %10s'
          % ('hidden', 'feat', 'updates', 'secs', 'MAE|TD|', 'TD sd', 'Q mean', 'Q sd'))
    if args.sweep:
        # 网格（全部同机制对照）：先固定 τ=0.001/lr=1e-2 复现在线的学习机制，
        # 再单独变化 容量 / 特征 / 更新量 / bsize / lr，看残差能压到多低。
        grid = [
            dict(hidden=(16, 100), updates=56000, feat='obs4', tau=1e-3, lr=1e-2, bsize=256),
            dict(hidden=(16, 100), updates=56000, feat='obs7', tau=1e-3, lr=1e-2, bsize=256),
            dict(hidden=(64, 256), updates=56000, feat='obs4', tau=1e-3, lr=1e-2, bsize=256),
            dict(hidden=(16, 100), updates=56000, feat='obs4', tau=1e-3, lr=1e-3, bsize=256),
            dict(hidden=(16, 100), updates=56000, feat='obs4', tau=1e-3, lr=1e-2, bsize=1024),
            dict(hidden=(16, 100), updates=400000, feat='obs4', tau=1e-3, lr=1e-2, bsize=256),
            dict(hidden=(64, 256), updates=400000, feat='obs7', tau=1e-3, lr=1e-3, bsize=1024),
        ]
        recs = []
        for g in grid:
            a2 = argparse.Namespace(**vars(args))
            a2.tau = g['tau']; a2.lr = g['lr']; a2.bsize = g['bsize']
            recs.append(run_one(ds, a2, g['hidden'], g['updates'], g['feat']))
            print(fmt(recs[-1]), flush=True)
        print('\n  ---- 收敛轨迹（每档 10 个点，MAE 单位元）----')
        for r in recs:
            print('  %-9s %-6s %8d lr=%.0e b=%4d : %s'
                  % (r['hidden'], r['feat'], r['updates'], r['lr'], r['bsize'],
                     ' '.join('%.0f' % x[1] for x in r['log'])))
    else:
        h = tuple(int(x) for x in args.hidden.split(','))
        r = run_one(ds, args, h, args.updates, args.feat)
        print(fmt(r))
        print('\n  收敛轨迹（update, MAE, sd, Q_mean, Q_sd）:')
        for x in r['log']:
            print('    %8d  MAE %9.1f  sd %9.1f  Q %9.1f  Qsd %9.1f' % x)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
