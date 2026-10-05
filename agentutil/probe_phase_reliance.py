# -*- coding: utf-8 -*-
"""量化"策略有多依赖**天内时间相位**（时钟）而不是**电价**"——把这个问题变成可跟踪的指标。

为什么要这个探针
----------------
本项目的日程是**确定性**的（`env.CalPrice` / `day_price_schedule` 六段），所以那 2 维
`sin/cos` 相位不只是"时间"，它是**整条未来价目的充分统计量** ⇒ 观测里的 `pr`（`obs[0]`）
在信息上**冗余** ⇒ 梯度下降会把信用全给时钟。实测（run62 最终权重）：把电价从 300 扫到 980，
actor 的输出几乎不动（方差 1e-5~3e-4），而只转动相位就能解释 **99–100%** 的行为变化。

⇒ 这属于**信息设计**问题，不是特征编码问题：把时钟编码得更好（mode 2/3、段号 one-hot）
只会强化它。要治，必须**破掉冗余**（去掉时钟 / 把前瞻信息改用价格表达 / 随机化日程）。
本探针提供两个判据，改前改后都要看：

指标
----
1. **反事实分解（主判据，与模型无关）**：把面板状态上的 actor 输出拆成
   * `Var_clock` = 把 `pr` **钉死**（分别钉 300/580/980 后取平均）、只让相位扫过全天
   * `Var_price` = 把相位**钉死**在若干参考时刻、只让 `pr` 扫过全天
   * `clock_share = Var_clock / (Var_clock + Var_price)` ⇒ **越高越糟**（1.0 = 纯时钟查找表）
2. **日志方差归因（行为侧）**：训练日志里 `applied_action_H` 的方差分别由
   minute-of-day（288 档）/ `pr`（3 档）/ `u`（30 档）解释的比例。
   注意：即使**去掉**时钟输入（mode 0），行为仍可能被 `pr` 驱动出日周期——那是**正确的**
   （价格本身是周期函数），所以判据看的是"**由价格解释**"是否顶上来。

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\probe_phase_reliance.py TD3PER-run54 AIFFLOR-run62
    ... --step 88000        # 只看某个 checkpoint（默认用 run 根目录的最终权重）
    ... --panel 200         # 反事实面板用多少条日志状态
    ... --json output/phase_reliance.json
"""
import argparse
import glob
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
import torch                                                  # noqa: E402

from runpaths import resolve_run, find_log_csv                # noqa: E402
from model import Actor                                       # noqa: E402
from td3per.td3per_myfunc import day_price_schedule           # noqa: E402

DIMS = ['a0', 'a1', 'a2', 'a3', 'a4', 'soc_signal']


def load_actor(d, step=None):
    """载入 H 层 actor；返回 (actor, 输入维, 权重来源描述)。"""
    base = d if step is None else os.path.join(d, 'checkpoint_%d' % step)
    p = os.path.join(base, 'actor_6.pkl')
    if not os.path.exists(p):
        raise SystemExit('找不到 %s' % p)
    sd = torch.load(p, map_location='cpu', weights_only=False)
    nin = int(sd['fc1.weight'].shape[1])
    act = Actor(nin, int(sd['fc4.weight'].shape[0]),
                hidden1=int(sd['fc1.weight'].shape[0]), hidden2=int(sd['fc2.weight'].shape[0]))
    act.load_state_dict(sd)
    act.eval()
    return act, nin, os.path.relpath(p, ROOT)


def panel_states(d, n, phase_mode, seed=0):
    """从日志抽面板状态：(u/10, E/10, P_G_prev/1000)。相位在反事实里另行生成。"""
    df = pd.read_csv(find_log_csv(d), usecols=['u', 'SOC', 'observation'])
    df = df.sample(min(n, len(df)), random_state=seed)
    obs = np.vstack([[float(x) for x in str(s).strip().strip('[]').split()][:4]
                     for s in df['observation'].values])
    return df['u'].values / 10.0, df['SOC'].values / 10.0, obs[:, 3] / 10.0


def counterfactual(act, nin, d, panel=200, seed=0):
    """反事实分解：Var_clock（pr 钉死）vs Var_price（相位钉死）。需要 nin>=6（含相位）。"""
    if nin < 6:
        return None
    PU, PE, PPG = panel_states(d, panel, 1, seed)
    n = len(PU)
    pr_day = day_price_schedule()
    mins = np.arange(0, 1440, 5)

    def sweep(pr_vec, ph_vec):
        X = np.empty((len(mins) * n, nin), dtype=np.float32)
        for i, m in enumerate(mins):
            ang = 2.0 * np.pi * ph_vec[i] / 1440.0
            blk = np.empty((n, nin), dtype=np.float32)
            blk[:, 0] = pr_vec[i] / 1000.0
            blk[:, 1], blk[:, 2], blk[:, 3] = PU, PE, PPG
            if nin >= 6:
                blk[:, 4], blk[:, 5] = np.sin(ang), np.cos(ang)
            X[i * n:(i + 1) * n] = blk
        with torch.no_grad():
            A = act(torch.tensor(X)).numpy()
        return A.reshape(len(mins), n, -1).mean(1)          # 面板均值 → (288, nb_actions)

    A_true = sweep(pr_day[mins], mins)
    A_clk = np.mean([sweep(np.full(len(mins), p), mins) for p in (300.0, 580.0, 980.0)], axis=0)
    A_pr = np.mean([sweep(pr_day[mins], np.full(len(mins), m0)) for m0 in (0, 360, 720, 1080)], axis=0)
    return dict(true=A_true.var(0), clock=A_clk.var(0), price=A_pr.var(0))


def log_attribution(d, enc=None):
    """日志侧方差归因：minute-of-day(288) / pr(3) / u(30) 各自解释的动作方差。"""
    df = pd.read_csv(find_log_csv(d), usecols=['phase_minute', 'pr', 'u', 'applied_action_H'])
    A = np.vstack([[float(x) for x in re.findall(r'-?\d+\.?\d*(?:e[-+]?\d+)?', str(s))][:5]
                   for s in df['applied_action_H'].values])
    pm = df['phase_minute'].values % 1440
    dec = pm % 5 == 0
    A, pm = A[dec], pm[dec]
    pr = df['pr'].values[dec]
    u = np.round(df['u'].values[dec] * 2) / 2

    def ve(x, key):
        g = pd.DataFrame({'x': x, 'k': key}).groupby('k')['x']
        return float(1 - (g.var(ddof=1).fillna(0) * g.size()).sum() / max(((x - x.mean()) ** 2).sum(), 1e-12))
    return {'clock': [ve(A[:, j], pm) for j in range(5)],
            'price': [ve(A[:, j], pr) for j in range(5)],
            'wind': [ve(A[:, j], u) for j in range(5)]}


def main():
    ap = argparse.ArgumentParser(description='策略对"天内时间相位"的依赖度（时钟 vs 电价）')
    ap.add_argument('runs', nargs='+', help='run 名（任意前缀，如 AIFFLOR-run62）或目录')
    ap.add_argument('--step', type=int, default=None, help='用该 checkpoint（默认 run 根目录权重）')
    ap.add_argument('--panel', type=int, default=200, help='反事实面板状态数（默认 200）')
    ap.add_argument('--json', default=None, help='把结果另存 JSON')
    args = ap.parse_args()

    out = {}
    for name in args.runs:
        d = resolve_run(name)
        tag = os.path.basename(d.rstrip('/\\'))
        act, nin, src = load_actor(d, args.step)
        print('\n' + '=' * 104)
        print('=== %s | actor 输入维 %d (%s) | 权重 %s ==='
              % (tag, nin, {4: '无相位(mode0)', 6: 'sin/cos(mode1)', 9: 'mode2'}.get(nin, '?'), src))
        print('=' * 104)
        rec = {'input_dim': nin, 'weights': src}
        cf = counterfactual(act, nin, d, args.panel)
        if cf is None:
            print('  反事实分解: 跳过（无相位输入）')
        else:
            print('  %-6s %12s %14s %14s %10s' % ('dim', 'var(真实)', 'var(纯时钟)', 'var(纯电价)', '时钟占比'))
            shares = []
            for j in range(min(cf['clock'].size, 6)):
                vc, vp = cf['clock'][j], cf['price'][j]
                sh = vc / max(vc + vp, 1e-18)
                shares.append(sh)
                print('  %-6s %12.3e %14.3e %14.3e %9.0f%%' % (DIMS[j], cf['true'][j], vc, vp, 100 * sh))
            rec['clock_share'] = [float(x) for x in shares]
            print('  ⇒ 平均时钟占比 %.0f%%（5 个风机维 %.0f%%）'
                  % (100 * np.mean(shares), 100 * np.mean(shares[:5])))
        la = log_attribution(d)
        print('  日志方差归因（决策点，%）：')
        print('    %-6s %10s %10s %10s' % ('dim', '时钟', '电价', '风速'))
        for j in range(5):
            print('    %-6s %9.1f%% %9.1f%% %9.1f%%'
                  % (DIMS[j], 100 * la['clock'][j], 100 * la['price'][j], 100 * la['wind'][j]))
        rec['log_attribution'] = la
        out[tag] = rec
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print('\n已写入 %s' % args.json)
    print('\n判据: `时钟占比` → 0 表示行为由电价驱动（目标）；≥90% 表示它是一张时钟查找表。')
    print('      mode0（无相位输入）时反事实分解不适用，请看"日志方差归因"里电价 vs 时钟的此消彼长。')


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    main()
