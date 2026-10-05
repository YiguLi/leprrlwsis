# -*- coding: utf-8 -*-
"""`--phase_feat` 配对 A/B 的对照报告（默认用于 aifflor 的 mode 0 vs mode 1）。

回答的问题：**去掉时钟输入（mode 0）是否消掉了"按时钟拟合"的行为，且代价可接受？**

判据（跑之前就定好，避免事后挑指标）
------------------------------------
① **配对性**：两臂必须只差 `--phase_feat`。外生量（`u` 风速、`pr` 电价）应**逐行全等**，
   每个 episode 重置点的 `SOC` 也应全等（同 seed ⇒ 同 RNG 流）；而 episode 内部的 `SOC`
   **应当不同**（动作不同 ⇒ 状态轨迹分化，这是实验要的）。
   若 ① 不成立，后面所有对比都不作数。
② **test9**：两臂 `df_test9.csv` 的 OVERALL 逐 step 对比（同 step 配对）。mode 0 **结构上
   不依赖任何相位约定**；mode 1 的相位是评测脚手架假造的（`rep_minute_for_price_level`），
   已实测同一权重换约定会摆动 6–9%。
③ **BESS 通道**：`degradation` / `bess_profit` 中位（元/episode）与净额。生产基线（run54）
   是退化 10.8k、现金 +0.98k ⇒ 净 **−9.9k 元/天**；去掉时钟不应让它恶化。
④ **归因**：日志里 `applied_action_H` 被 minute-of-day（时钟）/ `pr`（电价）/ `u`（风速）
   解释的方差比例（复用 `probe_phase_reliance.log_attribution`）。
   ⚠ 训练分布内 `pr` 是时钟的确定性函数 ⇒ 两者**观测等价**，单看④无法区分"靠价"与"靠钟"；
   真正能区分的是 ②（日程被改写的分布外评测）。④ 只用于看"风速解释度是否顶上来"。

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\ab_phase_feat_report.py \\
        --arm output/AIFFLORP0-run63 --arm output/AIFFLORP1-run64 --json output/ab_phase_feat.json
"""
import argparse
import json
import os
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
from probe_phase_reliance import log_attribution              # noqa: E402

EP = 1440          # 一个 episode 的分钟数
COLS = ['u', 'pr', 'SOC', 'degradation', 'bess_profit', 'r1', 'revenue', 'penalty_NN_H']


def arm_info(path):
    d = resolve_run(path)
    tag = os.path.basename(d.rstrip('/\\'))
    p = os.path.join(d, 'actor_6.pkl')
    nin = int(torch.load(p, map_location='cpu', weights_only=False)['fc1.weight'].shape[1])
    mode = {4: 0, 6: 1, 9: 2, 8: 4}.get(nin, '?')
    return d, tag, nin, mode


def load_arm(d):
    df = pd.read_csv(find_log_csv(d), usecols=COLS)
    n = len(df) // EP
    return df.iloc[:n * EP].reset_index(drop=True), n


def pairing(a, b, tag_a, tag_b):
    """① 配对性：外生量逐行全等；episode 重置点 SOC 全等；episode 内部 SOC 应分化。"""
    m = min(len(a), len(b))
    out = {}
    print('\n① 配对性（取两臂公共前缀 %d 行 = %.2f episode）' % (m, m / EP))
    for c in ('u', 'pr'):
        xa, xb = a[c].values[:m].astype(float), b[c].values[:m].astype(float)
        eq = float(np.mean(xa == xb)); mx = float(np.abs(xa - xb).max())
        ok = (mx == 0.0)
        print('   %-4s 逐行全等: %-5s  相等比例 %.6f  max|Δ| %.3g' % (c, 'OK' if ok else '**FAIL**', eq, mx))
        out['pair_' + c] = dict(equal_frac=eq, max_abs=mx, ok=bool(ok))
    k = min(len(a), len(b)) // EP
    ra = a['SOC'].values[np.arange(k) * EP].astype(float)
    rb = b['SOC'].values[np.arange(k) * EP].astype(float)
    reset_eq = float(np.mean(ra == rb))
    print('   episode 重置点 SOC（%d 个）: 全等比例 %.3f  max|Δ| %.3g  ⇒ %s'
          % (k, reset_eq, float(np.abs(ra - rb).max()),
             'OK（同 seed 的 RNG 流锁步）' if np.allclose(ra, rb, atol=1e-9) else '**FAIL**'))
    mid = a['SOC'].values.astype(float)[:m]
    mid_b = b['SOC'].values.astype(float)[:m]
    print('   episode 内部 SOC: 相等比例 %.3f（**应当**明显 <1：动作不同 ⇒ 轨迹分化）'
          % float(np.mean(mid == mid_b)))
    out['pair_reset_soc'] = dict(equal_frac=reset_eq, max_abs=float(np.abs(ra - rb).max()))
    out['pair_mid_soc_equal_frac'] = float(np.mean(mid == mid_b))
    return out


def test9(d):
    p = os.path.join(d, 'df_test9.csv')
    if not os.path.exists(p):
        return None
    t = pd.read_csv(p)
    ov = t[t['wind_level'] == 'OVERALL'] if 'wind_level' in t.columns else t
    return ov.groupby('step').agg(profit=('profit', 'mean'),
                                  profit_econ=('profit_econ', 'mean'),
                                  c1=('c1', 'mean'), pf_viol=('pf_viol', 'sum')).reset_index()


def bess_medians(d, n_ep):
    df = pd.read_csv(find_log_csv(d), usecols=['degradation', 'bess_profit', 'revenue'])
    df = df.iloc[:(len(df) // EP) * EP]
    g = df.groupby(np.arange(len(df)) // EP).sum()
    return dict(deg=float(g.degradation.median()), cash=float(g.bess_profit.median()),
                net=float((g.bess_profit - g.degradation).median()),
                rev=float(g.revenue.median()),
                deg_pct=float(100 * (g.degradation / g.revenue).median()))


def main():
    ap = argparse.ArgumentParser(description='--phase_feat 配对 A/B 对照报告')
    ap.add_argument('--arm', action='append', required=True,
                    help='run 名/目录，可给两次（建议按 mode 0 / mode 1 顺序）')
    ap.add_argument('--json', default=None, help='把结果写 JSON')
    args = ap.parse_args()

    infos = [arm_info(p) for p in args.arm]
    print('=' * 104)
    for d, tag, nin, mode in infos:
        print('臂 %-20s 输入维 %-2d → mode %-2s  |  %s' % (tag, nin, mode, d))
    print('=' * 104)
    rep = {tag: dict(input_dim=nin, mode=mode, dir=d) for d, tag, nin, mode in infos}

    # ---------------- ① 配对性 ----------------
    if len(infos) == 2:
        a, na = load_arm(infos[0][0])
        b, nb = load_arm(infos[1][0])
        rep['pairing'] = pairing(a, b, infos[0][1], infos[1][1])

    # ---------------- ② test9 ----------------
    ts = [(tag, test9(d)) for d, tag, _, _ in infos]
    if len(ts) == 2 and all(t is not None for _, t in ts):        # 两臂：同 step 配对
        print('\n② test9 OVERALL 逐 step（同 step 配对）')
        mrg = ts[0][1].rename(columns={'profit': 'p0', 'profit_econ': 'e0'})[
            ['step', 'p0', 'e0']].merge(
            ts[1][1].rename(columns={'profit': 'p1', 'profit_econ': 'e1'})[['step', 'p1', 'e1']],
            on='step', how='inner')
        print('   %-8s %11s %11s %9s | %11s %11s %9s'
              % ('step', ts[0][0][:10] + ' p', '', '', ts[1][0][:10] + ' p', '', ''))
        for _, r in mrg.tail(8).iterrows():
            print('   %-8d %11.1f %11.1f %9.1f | %11.1f %11.1f %9.1f'
                  % (r.step, r.p0, r.e0, r.p0 - r.p1, r.p1, r.e1, r.e1 - r.e0))
        tail = mrg.tail(min(5, len(mrg)))
        print('   末 %d 个公共评测点：profit 差（%s − %s）中位 %+.1f | profit_econ 差中位 %+.1f'
              % (len(tail), ts[0][0], ts[1][0], float((tail.p0 - tail.p1).median()),
                 float((tail.e0 - tail.e1).median())))
        rep['test9'] = dict(n_common=int(len(mrg)),
                            median_profit_diff=float((tail.p0 - tail.p1).median()),
                            median_econ_diff=float((tail.e0 - tail.e1).median()),
                            last=dict(step=int(mrg.step.iloc[-1]),
                                      p=[float(mrg.p0.iloc[-1]), float(mrg.p1.iloc[-1])],
                                      e=[float(mrg.e0.iloc[-1]), float(mrg.e1.iloc[-1])]))
        print('   ⚠ mode 1 的相位是评测脚手架假造的：已实测换约定同权重摆动 6–9%'
              '（profit 2036→1908 / profit_econ 1716→1566）。mode 0 无相位输入 ⇒ 该敏感性为 0。')
    else:                                                          # 单臂（或某臂缺 test9）：各报各的趋势
        print('\n② test9（单臂）')
        for tag, t in ts:
            if t is None:
                print('   %-20s 无 df_test9.csv' % tag)
                continue
            print('   %-20s n=%3d 轮 | profit_econ 首/中/末 = %8.1f / %8.1f / %8.1f | 最优 %8.1f'
                  % (tag, len(t), t.profit_econ.iloc[0], t.profit_econ.median(),
                     t.profit_econ.iloc[-1], t.profit_econ.max()))
            rep.setdefault('test9_single', {})[tag] = dict(
                n=int(len(t)), first=float(t.profit_econ.iloc[0]),
                median=float(t.profit_econ.median()), last=float(t.profit_econ.iloc[-1]),
                best=float(t.profit_econ.max()))

    # ---------------- ③ BESS 通道 ----------------
    print('\n③ BESS 通道（元/episode 中位；生产基线 run54 = 退化 10845 / 现金 +977 / 净 −9868）')
    print('   %-20s %12s %12s %12s %10s' % ('臂', 'degradation', 'bess_profit', '净额', '退化/收入'))
    rep['bess'] = {}
    for d, tag, _, _ in infos:
        m = bess_medians(d, None)
        rep['bess'][tag] = m
        print('   %-20s %12.0f %+12.0f %+12.0f %9.1f%%'
              % (tag, m['deg'], m['cash'], m['net'], m['deg_pct']))

    # ---------------- ④ 归因 ----------------
    print('\n④ 日志方差归因（决策点；clock=minute-of-day, price=pr, wind=u）')
    rep['attrib'] = {}
    for d, tag, _, _ in infos:
        la = log_attribution(d)
        rep['attrib'][tag] = la
        print('   %-20s 时钟 %s | 电价 %s | 风速 %s'
              % (tag,
                 ' '.join('%5.1f%%' % (100 * v) for v in la['clock']),
                 ' '.join('%5.1f%%' % (100 * v) for v in la['price']),
                 ' '.join('%5.1f%%' % (100 * v) for v in la['wind'])))
    print('   （训练分布内 pr 是时钟的确定性函数 ⇒ 两者观测等价；④ 主要看风速解释度是否上升）')

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, 'w', encoding='utf-8') as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=1)
        print('\n已写入 %s' % args.json)


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    main()
