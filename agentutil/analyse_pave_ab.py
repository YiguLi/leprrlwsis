# -*- coding: utf-8 -*-
"""aifflorpave A/B 一键分析：critic 几何指标 + 配对成绩。

判据（对应用户要检验的预言：几何正则 ⇒ 翻转率显著下降且成绩改善）
  ① 几何（离线，读各臂 checkpoint）
     * 相邻决策 ∇_aQ 的**符号翻转率**与 mean cos（文献 Base 9.0–35.5%）
     * **全 6×6 ∇²_aaQ 负定性率**（文献 Base 14–47%）
     * **跨维（5 台机组）∇_aQ 同号率**（物理上应高正相关）
     * 末端 TD 误差（critic 单位）
  ② 成绩（与基线逐日配对差 + 越界罚 + 越界率）
  ③ 输出 `output/_pave_ab_report.md`

用法::

    # 默认：把 ABPAVE2-* 全部臂与 ABPAVE2-BASE 比较，取各臂共同的最后一个 checkpoint
    python agentutil/analyse_pave_ab.py
    # 指定臂与 checkpoint
    python agentutil/analyse_pave_ab.py --arms ABPAVE2-BASE ABPAVE2-PAVE --ckpt 20000
"""
import argparse
import glob
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import torch                             # noqa: E402

from model import Critic                 # noqa: E402
from aifflorpave.aifflorpave_pave import CriticPAVE   # noqa: E402

torch.set_grad_enabled(False)
BAND = np.array([0.5] * 5 + [0.5])
LOW = np.array([0.0] * 5 + [-0.5])

#: checkpoint **不记录 critic 的激活类型** ⇒ 必须显式给出，否则会在错误的函数上量几何！
#: （relu 与 silu 的参数张量同构、可互载，但函数不同。）发现新臂请在此登记或用 --acts 覆盖。
DEFAULT_ACTS = {
    'ABPAVE2-BASE': 'relu', 'ABPAVE2-SILU': 'silu', 'ABPAVE2-PAVE': 'silu',
    'ABPAVE2-VFC': 'relu', 'ABPAVE2-NOCURV': 'silu', 'ABPAVE2-CURV4X': 'silu',
    'ABPAVE2-EVERY4': 'silu', 'ABPAVE2-MPR': 'relu', 'ABPAVE2-RELVFC': 'relu',
    'AB-H16N10': 'relu', 'AB-H64N10': 'relu', 'AB-H256N10': 'relu',
    'AB-H16N1': 'relu', 'AB-H64N1': 'relu',
}


def obs_h(o, mode=0, minute=0):
    a = np.fromstring(str(o).strip('[]'), sep=' ')
    b = np.array([a[0] / 1000., a[1] / 10., a[2] / 10., a[3] / 10.])
    if mode <= 0:
        return b
    ang = 2 * np.pi * (int(minute) % 1440) / 1440.
    return np.concatenate([b, [np.sin(ang), np.cos(ang)]])


def find_run(tag):
    ds = sorted(glob.glob(os.path.join('output', '%s-run*' % tag)))
    return ds[0] if ds else None


def load_ckpt(run, ckpt, act='relu'):
    """返回 (critic 列表, nb_states) —— 取 checkpoint_<ckpt>；`act` 必须与训练一致。"""
    sub = os.path.join(run, 'checkpoint_%d' % ckpt)
    if not os.path.isdir(sub):
        return None, None
    fps = [os.path.join(sub, 'critic_p_6.pkl'), os.path.join(sub, 'critic_p2_6.pkl')]
    if not all(os.path.isfile(f) for f in fps):
        return None, None
    sd = torch.load(fps[0], map_location='cpu')
    ns = sd['fc1.weight'].shape[1]
    nets = []
    for f in fps:
        s = torch.load(f, map_location='cpu')
        net = CriticPAVE(ns, 6, hidden1=s['fc1.weight'].shape[0],
                         hidden2=s['fc4.weight'].shape[0], act=act)
        try:
            net.load_state_dict(s)
        except Exception:
            return None, None
        net.eval()
        nets.append(net)
    return nets, ns


def csv(run, name):
    f = os.path.join(run, name)
    return pd.read_csv(f) if os.path.isfile(f) else None


def geometry(run, ckpt, act='relu', n_state=250, eg=0.01, eh=0.05):
    nets, ns = load_ckpt(run, ckpt, act=act)
    if nets is None:
        return None
    for cand in ('df_aifflorpave.csv', 'df_aifflor.csv'):
        df = csv(run, cand)
        if df is not None:
            break
    if df is None:
        return None
    sel = (df.step % 5 == 0) & (df.step > ckpt - 2000) & (df.step <= ckpt)
    idx = np.where(sel.values)[0][:n_state]
    if len(idx) < 40:
        return None
    X = torch.tensor(np.array([obs_h(o) for o in df.observation.values[idx]]), dtype=torch.float32)
    A = np.array([np.fromstring(str(x).strip('[]'), sep=' ') for x in df.original_action_H.values[idx]])

    def Q(M):
        m = M.shape[1]
        St = X.unsqueeze(1).expand(-1, m, -1).reshape(-1, ns)
        At = torch.tensor(M.reshape(-1, 6), dtype=torch.float32)
        return torch.min(nets[0]([St, At]), nets[1]([St, At])).reshape(len(idx), m).numpy()

    def grad(A0, e=eg):
        ev = [A0] + [np.column_stack([A0[:, c] + (e if c == j else 0) for c in range(6)]) for j in range(6)] \
                  + [np.column_stack([A0[:, c] - (e if c == j else 0) for c in range(6)]) for j in range(6)]
        q = Q(np.stack(ev, 1))
        return np.stack([(q[:, 1 + j] - q[:, 1 + 6 + j]) / (2 * e) for j in range(6)], axis=1)

    g = grad(A)
    cos = np.array([float(g[t - 1] @ g[t] / max(np.linalg.norm(g[t - 1]) * np.linalg.norm(g[t]), 1e-12))
                    for t in range(1, len(g))])
    gp = np.stack([grad(np.column_stack([A[:, c] + (eh if c == i else 0) for c in range(6)])) for i in range(6)], 0)
    gm = np.stack([grad(np.column_stack([A[:, c] - (eh if c == i else 0) for c in range(6)])) for i in range(6)], 0)
    H = (gp - gm) / (2 * eh)
    negdef = np.mean([np.linalg.eigvalsh(0.5 * (H[:, s, :] + H[:, s, :].T))[-1] < 0 for s in range(len(idx))])
    sw = np.sign(g[:, :5]); sw[sw == 0] = 1
    agree = np.mean([(sw[:, i] == sw[:, j]).mean() for i in range(5) for j in range(i + 1, 5)])
    return dict(n=len(idx), flip=float(np.mean(cos < 0)), cos=float(cos.mean()),
                negdef=float(negdef), sign_agree=float(agree),
                gscale=float((g ** 2).sum(-1).mean()))


def td_of(run, ckpt):
    u = csv(run, 'update_stats_H.csv')
    if u is None:
        return float('nan')
    u = u[(u.step > ckpt - 2000) & (u.step <= ckpt)]
    return float(u.td_p_mean.mean()) / 1000. if len(u) else float('nan')


def perf(run):
    for cand in ('df_aifflorpave.csv', 'df_aifflor.csv'):
        df = csv(run, cand)
        if df is not None:
            break
    if df is None:
        return None
    df = df[df.step > 1025].copy()
    df['day'] = df.step // 1440
    v = np.array([np.fromstring(str(x).strip('[]'), sep=' ') for x in df.original_action_H])
    return dict(day=df.groupby('day').r1.sum(), r1=df.r1.mean() * 5,
                pen=df.penalty_NN_H.mean() * 5, rev=df.revenue.mean() * 5,
                oob=100 * float(np.mean((v > BAND + 1e-9) | (v < LOW - 1e-9))),
                step=int(df.step.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='*', default=None)
    ap.add_argument('--base', default='ABPAVE2-BASE')
    ap.add_argument('--ckpt', type=int, default=0, help='0 = 取各臂共同的最后一个 checkpoint')
    ap.add_argument('--acts', nargs='*', default=[],
                    help='激活覆盖，形如 ABPAVE2-SILU=silu（默认用 DEFAULT_ACTS）')
    a = ap.parse_args()
    acts = dict(DEFAULT_ACTS)
    for kv in a.acts:
        if '=' in kv:
            k, v = kv.split('=', 1)
            acts[k] = v

    tags = a.arms or sorted({os.path.basename(p).split('-run')[0]
                             for p in glob.glob(os.path.join('output', 'ABPAVE2-*-run*'))})
    runs = {t: find_run(t) for t in tags}
    ckpts = {}
    for t, r in runs.items():
        if not r:
            continue
        ckpts[t] = sorted(int(re.search(r'checkpoint_(\d+)', p).group(1))
                          for p in glob.glob(os.path.join(r, 'checkpoint_*')))
    common = min((c[-1] for c in ckpts.values() if c), default=0)
    ck = a.ckpt or common
    lines = ['# aifflorpave A/B 报告', '',
             '臂: %s ｜ 基线: %s ｜ 比较 checkpoint: %d' % (', '.join(tags), a.base, ck), '',
             '## ① critic 几何（预测：正则 ⇒ 翻转率↓、负定性率↑）', '',
             '| 臂 | act | n | 翻转率 | mean cos | 6×6 负定性率 | 跨维同号率 | ‖∇_aQ‖² | TD |',
             '|---|---|---|---|---|---|---|---|---|']
    geo = {}
    for t in tags:
        r = runs.get(t)
        if not r:
            continue
        act = acts.get(t, 'relu')
        g = geometry(r, ck, act=act)
        if g:
            g['td'] = td_of(r, ck)
            geo[t] = g
            lines.append('| %s | %s | %d | **%.1f%%** | %.3f | **%.1f%%** | %.1f%% | %.3g | %.2f |'
                         % (t, act, g['n'], 100 * g['flip'], g['cos'], 100 * g['negdef'],
                            100 * g['sign_agree'], g['gscale'], g['td']))
    lines += ['', '## ② 成绩（与 %s 逐日配对）' % a.base, '',
              '| 臂 | 覆盖步数 | 日 r1 均值 | vs 基线 | 配对 SE | t | r1/块 | 罚/块 | 收益/块 | 越界% |',
              '|---|---|---|---|---|---|---|---|---|---|']
    P = {t: perf(runs[t]) for t in tags if runs.get(t)}
    base = P.get(a.base)
    for t in tags:
        p = P.get(t)
        if not p:
            continue
        if base is not None:
            d = (p['day'] - base['day']).dropna()
            m, se = (d.mean(), d.std(ddof=1) / np.sqrt(len(d))) if len(d) else (float('nan'), float('nan'))
            tv = '%.1f' % (m / se) if se and se > 0 else '-'
        else:
            m = se = float('nan'); tv = '-'
        lines.append('| %s | %d | %.1f | %.2f | %.2f | %s | %.1f | %.1f | %.1f | %.2f |'
                     % (t, p['step'], p['day'].mean(), m, se, tv, p['r1'], p['pen'], p['rev'], p['oob']))
    lines += ['', '> 判读：几何指标是"机制是否被修好"的判据；成绩是"是否值得"。',
              '> 若翻转率降了但成绩掉且负定性率没升 ⇒ Curv 权重不足（调 `--pave_w_curv`）。',
              '> 若翻转率不动 ⇒ 该正则无效（relu 路线见 RELVFC/MPR 臂）。']
    out = os.path.join('output', '_pave_ab_report.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('\n(已写入 %s)' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
