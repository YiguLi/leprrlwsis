# -*- coding: utf-8 -*-
r"""一次性迁移（第 2 批）：BESS 缩放后**观察尺度**的重新标定。

为什么必须改
------------
第 1 批把容量 6 MWh -> 0.5 MWh、功率 10 MW -> 0.5 MW。而观察里的归一化因子
（`/10`）是按**旧量纲**标定的：

  * SOC 维：旧 SOC∈[0.5,5.5] ⇒ /10 后 [0.05,0.55]；新 SOC∈[0.042,0.458] ⇒ /10 后
    **[0.004,0.046]** ⇒ 有效分辨率掉 12 倍，且与目标维几乎无法区分；
  * 功率维：旧窗口 ±10 MW ⇒ /10 后 ±1；新窗口 ±0.5 MW ⇒ /10 后 **±0.05**。

新标定（取"该维满量程 ≈ 1"的常规做法）
--------------------------------------
  OBS_SOC_SCALE    10  -> **0.5**      = SOC 上限（容量），SOC/0.5 ∈ [0.083, 0.917]
  OBS_POWER_SCALE  10  -> **5.0**      机组功率量程 0~12 MW 的常用刻度
  8 个 myfunc 的 `pbrulemin/10, pbrulemax/10` -> `/5`（窗口 ±0.5 ⇒ ±0.1）
  `Normalize`/`Normalize_H` 里的 `obs[2]/10`（SOC）与 `obs[3]/10`（P_G）：
      SOC -> `/0.5`；P_G -> `/5`

⚠ 这些替换都要求**逐位精确匹配**；命中数不符则该组跳过并打印明细。

用法:    python agentutil/migrate_bess_obs_scale.py [--apply]
"""
import argparse
import glob
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)

SKIP_DIRS = ('agentutil', '__pycache__', 'output', '未命名', 'libraries', '.git')

RULES = [
    (r"^OBS_SOC_SCALE = 10\.0", "OBS_SOC_SCALE = 0.5", 1,
     'L 层 SOC 维刻度 -> 容量 0.5'),
    (r"^OBS_POWER_SCALE = 10\.0", "OBS_POWER_SCALE = 5.0", 1,
     'L 层功率维刻度 -> 5 MW'),
    # 8 个 myfunc 的 L 观测配料：pbrulemin/10, pbrulemax/10
    (r"rbb_info\['pbrulemin'\] / 10\.0", "rbb_info['pbrulemin'] / 5.0", None,
     "L 观测 pbrulemin/10 -> /5"),
    (r"rbb_info\['pbrulemax'\] / 10\.0", "rbb_info['pbrulemax'] / 5.0", None,
     "L 观测 pbrulemax/10 -> /5"),
    # 根 myfunc.build_L_obs 的 SOC 维（用"下一行是 u 注释"做唯一锚点，避免误伤其它 /10.0）
    (r"float\(observation\[2\]\) / 10\.0,\n(\s*float\(observation\[1\]\) / 10\.0,  # u)",
     "float(observation[2]) / 0.5,\n\\1", None,
     'myfunc.build_L_obs 的 SOC 维 -> /0.5'),
    # Normalize / Normalize_H：SOC 维与 P_G 维
    #   return np.array([obs[0] / 1000, obs[1] / 10, obs[2] / 10, obs[3] / 10])
    (r"obs\[2\] / 10, obs\[3\] / 10\]\)", "obs[2] / 0.5, obs[3] / 5.0])", None,
     'Normalize 的 SOC/P_G 维'),
    (r"obs\[2\] \* 10, obs\[3\] \* 10\]\)", "obs[2] * 0.5, obs[3] * 5.0])", None,
     'Original 的反归一化'),
]


def files():
    out = []
    for f in glob.glob('**/*.py', recursive=True):
        parts = f.replace('\\', '/').split('/')
        if any(p in SKIP_DIRS for p in parts):
            continue
        out.append(f)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    total = 0
    for pat, rep, want, desc in RULES:
        rx = re.compile(pat, re.M)
        hits = []
        for f in files():
            t = open(f, encoding='utf-8').read()
            n = sum(1 for _ in rx.finditer(t))
            if n:
                hits.append((f, n))
        n_all = sum(n for _, n in hits)
        ok = (want is None) or (n_all == want)
        print('%s %-42s 命中 %3d (期望 %s)  %s'
              % ('OK ' if ok else '❌ ', desc, n_all,
                 want if want is not None else '任意',
                 ', '.join('%s:%d' % (f.replace(chr(92), '/'), n) for f, n in hits[:8])))
        if not ok:
            print('   ⇒ 跳过该组')
            continue
        total += n_all
        if args.apply:
            for f, _ in hits:
                t = open(f, encoding='utf-8').read()
                open(f, 'w', encoding='utf-8', newline='\n').write(rx.sub(rep, t))
    print('\n合计 %d 处%s' % (total, '（已写盘）' if args.apply else '（dry-run）'))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
