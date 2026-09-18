# -*- coding: utf-8 -*-
"""探针: 验证 F=∅ 时"参考集合"是一个零宽区间 (纯粹的单个竭力值)。只读, 不改任何状态。"""
import io
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

import hsrl_soctarget as hst

ETA = 0.98
DT = 1.0 / 60.0


def soc_after(soc_prev, pb, dt=DT, eta=ETA):
    return soc_prev + eta * dt * max(pb, 0.0) - max(-pb, 0.0) * dt / eta


def main(csv_rel):
    path = os.path.join(ROOT, csv_rel)
    if not os.path.exists(path):
        print("缺少数据: %s" % path)
        return 1
    d = pd.read_csv(path)
    step = d["step"].to_numpy()
    tgt = d["soc_target"].to_numpy()
    pre = d["soc"].to_numpy()
    post = d["SOC"].to_numpy()
    wlo = d["pbrulemin"].to_numpy()
    whi = d["pbrulemax"].to_numpy()

    shown = 0
    n_empty = 0
    worst_rule = 0.0
    for i in range(len(d)):
        n = 5 - (step[i] % 5)
        tr, info = hst.soc_track_range(tgt[i], pre[i], post[i], wlo[i], whi[i], n)
        if not info["gap_empty"]:
            continue
        n_empty += 1
        # 规则动作 = clip(need, W) 一定落在参考点上 ⇒ trackerr 必须恒 0
        worst_rule = max(worst_rule, abs(tr))
        if shown < 3:
            shown += 1
            print("--- step %d (n_rem=%d) F=empty ---" % (step[i], n))
            print("  delta=%+.2f MWh  g=%.5f  B=[%.1f, %.1f] MW  W=[%.2f, %.2f] MW"
                  % (info["delta"], info["gain"], info["band_lo"], info["band_hi"], wlo[i], whi[i]))
            print("  参考最佳努力值 ref=[%.3f, %.3f] MW  (零宽: %.3f == %.3f -> %s)"
                  % (info["ref_lo"], info["ref_hi"], info["ref_lo"], info["ref_hi"],
                     info["ref_lo"] == info["ref_hi"]))
            print("  实际施加 pbrule=%+.3f MW  ->  trackerr=%.3f MW" % (info["pb_applied"], info["trackerr"]))
            base = info["ref_lo"]
            for test in (base, base - 0.5, base - 2.0, base + 0.5):
                tr2, inf2 = hst.soc_track_range(tgt[i], pre[i], soc_after(pre[i], test),
                                                wlo[i], whi[i], n)
                tag = "  <== 竭力 (不罚)" if abs(tr2) < 1e-12 else ""
                print("    施加 pb=%+7.3f MW -> trackerr=%7.3f MW%s" % (test, tr2, tag))
            print()

    print("F=empty 的步数: %d / %d (%.1f%%)" % (n_empty, len(d), 100.0 * n_empty / len(d)))
    print("这些步里 规则动作 的最大 trackerr = %.3e MW  (应恒为 0)" % worst_rule)

    print()
    print("实现处 hsrl_soctarget.py::soc_track_range():")
    src = io.open(os.path.join(ROOT, "hsrl_soctarget.py"), encoding="utf-8").read().splitlines()
    for i, line in enumerate(src):
        s = line.strip()
        if s.startswith("if gap_empty") or s.startswith("ref_lo = ref_hi") \
                or s.startswith("ref_lo, ref_hi") or s.startswith("trackerr = max"):
            print("  %4d | %s" % (i + 1, line.rstrip()))
    return 0


if __name__ == "__main__":
    rel = sys.argv[1] if len(sys.argv) > 1 else r"output\TD3PRRBB-run8\df_td3prrbb.csv"
    sys.exit(main(rel))
