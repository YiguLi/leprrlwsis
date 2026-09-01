"""
DBSCAN 聚类分段 — 训练轨迹前 N step 风速分类
============================================
对训练轨迹 CSV（默认 TD3PRLE-run144/df_td3prle.csv）前 N 步（默认 80）的
风速 u 做一维 DBSCAN 密度聚类（纯 numpy 实现，无 sklearn）：

  - 核心点：eps 邻域内点数 >= min_samples
  - 簇    ：核心点按密度连通性成簇（1D 下对排序后的核心点合并重叠邻域）
  - 边界点：非核心但落在某核心点 eps 邻域内 → 归属该簇
  - 噪声点：两者皆非 → NOISE
  - 簇数由数据密度自动决定（可能 != 3）；恰好 3 簇时按簇中心升序
    映射 LOW / MEDIUM / HIGH，否则用 CLUSTER_0/1/... 命名

用法:
  python data_piecewise/dbscan.py
  python data_piecewise/dbscan.py --eps 0.8 --min-samples 3
  python data_piecewise/dbscan.py --csv output/TD3PRLE-run144/df_td3prle.csv --steps 80 --out data_piecewise/dbscan_result.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# 强制 UTF-8 输出（避免 Windows 终端中文乱码）
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HSRL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(HSRL_ROOT, "output", "TD3PRLE-run144", "df_td3prle.csv")

# PRRBB 现有固定阈值（用于对比）
FIXED_LOW_MAX = 7.0
FIXED_HIGH_MIN = 11.0


def load_wind_first_steps(csv_path, steps):
    """读取训练轨迹 CSV 前 steps 步的风速 u（按 step 排序）。"""
    df = pd.read_csv(csv_path, usecols=["step", "u"])
    df = df.sort_values("step").head(steps)
    return df["step"].values, df["u"].values.astype(float)


def dbscan_1d(x, eps, min_samples):
    """一维 DBSCAN。

    返回 labels: int 数组，-1 = 噪声，>=0 = 簇编号（编号大小无顺序含义，
    由调用方按簇中心升序重排）。
    """
    n = x.size
    # ---- 核心点判定（O(n^2)，n=80 足够快）----
    core = np.zeros(n, dtype=bool)
    for i in range(n):
        core[i] = np.sum(np.abs(x - x[i]) <= eps) >= min_samples

    labels = np.full(n, -1, dtype=int)
    if not np.any(core):
        return labels  # 全部是噪声

    # ---- 核心点成簇：排序后，相邻核心点间距 <= eps 则同簇 ----
    core_idx = np.where(core)[0]
    core_idx = core_idx[np.argsort(x[core_idx])]
    cid = 0
    labels[core_idx[0]] = cid
    prev = core_idx[0]
    for cur in core_idx[1:]:
        if x[cur] - x[prev] <= eps:
            labels[cur] = cid
        else:
            cid += 1
            labels[cur] = cid
        prev = cur

    # ---- 边界点：非核心、但在某核心点 eps 邻域内 → 归属最近核心点的簇 ----
    for i in range(n):
        if labels[i] != -1:
            continue
        within = np.abs(x - x[i]) <= eps
        neigh = np.where(within & core)[0]
        if neigh.size > 0:
            best = neigh[np.argmin(np.abs(x[neigh] - x[i]))]
            labels[i] = labels[best]
    return labels


def fixed_level(u):
    """PRRBB 现有固定阈值分类（用于对比）。"""
    if u < FIXED_LOW_MAX:
        return "LOW"
    if u >= FIXED_HIGH_MIN:
        return "HIGH"
    return "MEDIUM"


def main():
    ap = argparse.ArgumentParser(description="DBSCAN 一维风速分段（训练轨迹前 N 步）")
    ap.add_argument("--csv", default=DEFAULT_CSV, help="训练轨迹 CSV（含 step, u 列）")
    ap.add_argument("--steps", type=int, default=80, help="取前多少步（默认 80）")
    ap.add_argument("--eps", type=float, default=0.8, help="邻域半径 eps（m/s）")
    ap.add_argument("--min-samples", dest="min_samples", type=int, default=3,
                    help="核心点最少邻域点数（默认 3）")
    ap.add_argument("--out", default=None, help="可选：输出每步分类 CSV")
    args = ap.parse_args()

    steps, u = load_wind_first_steps(args.csv, args.steps)
    print(f"数据: {args.csv}  前 {len(steps)} 步  风速范围 [{u.min():.3f}, {u.max():.3f}] m/s\n")

    labels = dbscan_1d(u, eps=args.eps, min_samples=args.min_samples)

    n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
    noise_mask = labels == -1
    print(f"== DBSCAN 结果 (eps={args.eps}, min_samples={args.min_samples}) ==")
    print(f"发现簇数: {n_clusters}   噪声点数: {int(noise_mask.sum())}/{len(u)}")

    if n_clusters == 0:
        print("没有任何簇，全部为噪声。请调大 eps 或调小 min_samples。")
        return

    # 簇按代表值（成员中位数）升序排序
    repr_val = [float(np.median(u[labels == c])) for c in range(n_clusters)]
    cluster_order = np.argsort(repr_val)
    # 恰好 3 簇时映射 LOW/MEDIUM/HIGH，否则 CLUSTER_x
    levels = ["LOW", "MEDIUM", "HIGH"] if n_clusters == 3 else \
        [f"CLUSTER_{pos}" for pos in range(n_clusters)]
    label_to_level = {int(cid): levels[pos] for pos, cid in enumerate(cluster_order)}

    print("\n簇区间（按风速升序）:")
    for cid in cluster_order:
        m = labels == cid
        rng = (u[m].min(), u[m].max())
        span = "   " if n_clusters == 3 else ""
        print(f"  {levels[int(np.where(cluster_order == cid)[0][0])]:8s}: "
              f"{rng[0]:.4f} ~ {rng[1]:.4f} m/s  (成员 {int(m.sum())}, 中位 {repr_val[cid]:.4f})")
    if noise_mask.sum():
        print(f"  NOISE    : {len(np.unique(u[noise_mask]))} 个不同风速值标记为噪声: "
              f"{', '.join(f'{v:.3f}' for v in np.sort(np.unique(u[noise_mask])))}")

    # ---- 每步分类 ----
    print("\n== 前 {} 步逐 step 分类 ==".format(len(steps)))
    print("  step     u(m/s)   DBSCAN档  | 固定7/11档  是否一致")
    n_diff = 0
    for st, ui, lab in zip(steps, u, labels):
        lv = label_to_level[int(lab)] if lab != -1 else "NOISE"
        fv = fixed_level(ui)
        diff = "  *" if (lv != "NOISE" and lv != fv) else ""
        if lv != "NOISE" and lv != fv:
            n_diff += 1
        print(f"  {st:>4d}  {ui:8.4f}   {lv:8s}  |   {fv:8s}  {diff}")
    print(f"\n与固定 7.0/11.0 规则不一致（非噪声）步数: {n_diff}/{len(steps)}")

    print("\n提示: DBSCAN 不预设簇数。若簇数 != 3，说明这 80 步风速的密度结构"
          "并非天然三档；噪声点则是密度孤立的风速，需额外决定归属策略"
          "（归入最近簇 / 单独 EXTREME 档）。")


if __name__ == "__main__":
    main()
