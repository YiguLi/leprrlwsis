# -*- coding: utf-8 -*-
"""探针 2: 找出 run8 里 trackerr 最大的步, 判断"实际施加 ≠ 参考点"的成因。

只读。用法: python agentutil/_probe_trackerr_worst.py [csv 相对路径] [topN]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd

import hsrl_soctarget as hst

ETA = 0.98
DT = 1.0 / 60.0


def main(rel, topn=8):
    path = os.path.join(ROOT, rel)
    d = pd.read_csv(path)
    step = d["step"].to_numpy()
    tgt = d["soc_target"].to_numpy()
    pre = d["soc"].to_numpy()
    post = d["SOC"].to_numpy()
    wlo = d["pbrulemin"].to_numpy()
    whi = d["pbrulemax"].to_numpy()
    cmd = d["pbrule"].to_numpy()
    plo = d["pbphysicsmin"].to_numpy()
    phi = d["pbphysicsmax"].to_numpy()

    rows = []
    for i in range(len(d)):
        n = 5 - (step[i] % 5)
        tr, info = hst.soc_track_range(tgt[i], pre[i], post[i], wlo[i], whi[i], n)
        rows.append((tr, i, info, n))
    rows.sort(key=lambda t: -t[0])

    print("trackerr 统计: max=%.3f  p99=%.3f  p50=%.4f  mean=%.4f  (行数 %d)"
          % (rows[0][0], np.percentile([r[0] for r in rows], 99),
             np.percentile([r[0] for r in rows], 50),
             np.mean([r[0] for r in rows]), len(rows)))
    print("trackerr > 1e-9 的行数: %d (%.2f%%)"
          % (sum(1 for r in rows if r[0] > 1e-9), 100.0 * sum(1 for r in rows if r[0] > 1e-9) / len(rows)))
    print()

    for tr, i, info, n in rows[:topn]:
        side = ("band 在窗口上方 -> ref=whi" if info["band_lo"] > whi[i] else
                "band 在窗口下方 -> ref=wlo" if info["band_hi"] < wlo[i] else "F 非空")
        print("--- row %d step %d (n_rem=%d) trackerr=%.3f MW  [%s]" % (i, step[i], n, tr, side))
        print("    delta=%+.4f  g=%.6f  band=[%.4f, %.4f]  W=[%.4f, %.4f]  F=[%.4f, %.4f] empty=%s"
              % (info["delta"], info["gain"], info["band_lo"], info["band_hi"],
                 wlo[i], whi[i], info["F_lo"], info["F_hi"], info["gap_empty"]))
        print("    ref=[%.4f, %.4f]  cmd(pbrule)=%+.4f  pb_app(由 SOC 反解)=%+.4f  |cmd-app|=%.4f"
              % (info["ref_lo"], info["ref_hi"], cmd[i], info["pb_applied"],
                 abs(cmd[i] - info["pb_applied"])))
        print("    soc=%.6f -> SOC=%.6f  dSOC=%+.6f  (dSOC/eta/dt=%+.4f)  target=%.4f  phys=[%.3f, %.3f]"
              % (pre[i], post[i], post[i] - pre[i],
                 (post[i] - pre[i]) / (ETA * DT) if post[i] >= pre[i] else (post[i] - pre[i]) * ETA / DT,
                 tgt[i], plo[i], phi[i]))
        print("    SOC 触边界(max): %.1f/6.0   (SOC_next 列=%.6f)" % (max(pre[i], post[i]), d["soc_next"].to_numpy()[i]))
        print("    csv 旧列: soc_slack=%.6f soc_track_err=%.6f soc_deviation=%.6f r2=%.4f c2=%.6f"
              % (d["soc_slack"].to_numpy()[i], d["soc_track_err"].to_numpy()[i],
                 d["soc_deviation"].to_numpy()[i], d["r2"].to_numpy()[i], d["c2"].to_numpy()[i]))
        print()
    return 0


if __name__ == "__main__":
    rel = sys.argv[1] if len(sys.argv) > 1 else r"output\TD3PRRBB-run8\df_td3prrbb.csv"
    topn = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    sys.exit(main(rel, topn))
