"""
DBSCAN 聚类分段 — 训练轨迹前 N 天风速分类（结果落盘）
====================================================
默认按训练轨迹**前 80 天**的风速分类（80 × 1440 = 115200 步）。

数据来源（与训练环境同一管线）：
  - 训练每天 = FFC_speed_noise_1.csv 的一列（1440 个风速值）；
  - 前 N 天的日索引 = fixed_day_sequence.csv 前 N 个值；
  - 拼接得到前 N 天训练轨迹风速序列。
  也可用 --csv 指定任意逐 step 轨迹 CSV（需含 step, u 列）。

算法：一维 DBSCAN（纯 numpy，searchsorted 向量化邻域计数，支持 11.5 万点）：
  - 核心点：eps 邻域内点数 >= min_samples
  - 簇    ：核心点按密度连通性成簇（1D 下按排序合并重叠邻域）
  - 边界点：非核心但落在某核心点 eps 邻域内 → 归属最近核心点的簇
  - 噪声点：两者皆非 → NOISE
  - 簇数由密度自动决定（可能 != 3）；恰好 3 簇时映射 LOW/MEDIUM/HIGH，
    否则 CLUSTER_0/1/...

结果储存（--outdir，默认 output/data_piecewise/）：
  - dbscan_<days>day.csv          每步分类：step, u, cluster_id, label(-1=噪声)
  - dbscan_<days>day_summary.json 配置 + 簇区间/成员数/噪声统计/与固定7-11对比

用法:
  python data_piecewise/dbscan.py                            # 前 80 天
  python data_piecewise/dbscan.py --eps 0.8 --min-samples 3
  python data_piecewise/dbscan.py --csv output/TD3PRLE-run144/df_td3prle.csv --days 41
"""

import argparse
import json
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
FFC_CSV = os.path.join(HSRL_ROOT, "FFC_speed_noise_1.csv")
DAY_SEQ_CSV = os.path.join(HSRL_ROOT, "fixed_day_sequence.csv")
STEPS_PER_DAY = 1440

# PRRBB 现有固定阈值（用于对比）
FIXED_LOW_MAX = 7.0
FIXED_HIGH_MIN = 11.0


def load_wind(days, csv=None, steps=None):
    """构建训练轨迹前 days 天的风速序列（默认 80 天 = 115200 步）。"""
    if csv is not None:
        df = pd.read_csv(csv, usecols=["step", "u"]).sort_values("step")
        n = len(df)
        want = steps if steps is not None else days * STEPS_PER_DAY
        take = min(want, n)
        if take < want and steps is None:
            print(f"  警告: 轨迹 CSV 仅 {n} 步（≈{n // STEPS_PER_DAY} 天），"
                  f"不足 {days} 天，按实际取前 {take} 步")
        df = df.head(take)
        return df["step"].values, df["u"].values.astype(float)

    wind = pd.read_csv(FFC_CSV)  # 1440 行 × 天数列
    seq = pd.read_csv(DAY_SEQ_CSV, header=None).values.flatten().astype(int)
    if days > len(seq):
        print(f"  警告: 日序列仅 {len(seq)} 天，按 {len(seq)} 天处理")
        days = len(seq)
    vals = [wind.iloc[:, d].values for d in seq[:days]]
    u = np.concatenate(vals)
    if steps is not None:
        u = u[:steps]
    return np.arange(u.size), u


def dbscan_1d(x, eps, min_samples):
    """一维 DBSCAN（向量化邻域计数 + 核心点成簇 + 边界点归属）。

    返回 labels: int 数组，-1 = 噪声，>=0 = 簇编号（编号无顺序含义）。
    """
    n = x.size
    xs = np.sort(x)
    # 每个点的 eps 邻域点数（searchsorted 向量化，O(n log n)）
    lo = np.searchsorted(xs, x - eps, side="left")
    hi = np.searchsorted(xs, x + eps, side="right")
    counts = hi - lo
    core = counts >= min_samples

    labels = np.full(n, -1, dtype=int)
    if not np.any(core):
        return labels  # 全部噪声

    # 核心点成簇：按值排序后，相邻核心点间距 <= eps 则同簇
    core_idx = np.where(core)[0]
    core_order = core_idx[np.argsort(x[core_idx])]
    cid = 0
    labels[core_order[0]] = cid
    prev = core_order[0]
    for cur in core_order[1:]:
        if x[cur] - x[prev] <= eps:
            labels[cur] = cid
        else:
            cid += 1
            labels[cur] = cid
        prev = cur

    # 边界点：非核心、但在某核心点 eps 邻域内 → 归属最近核心点的簇
    core_vals = x[core_order]
    core_lbl = labels[core_order]
    for i in np.where(labels == -1)[0]:
        v = x[i]
        lo2 = np.searchsorted(core_vals, v - eps, side="left")
        hi2 = np.searchsorted(core_vals, v + eps, side="right")
        if lo2 < hi2:
            cand = core_vals[lo2:hi2]
            k = lo2 + int(np.argmin(np.abs(cand - v)))
            labels[i] = core_lbl[k]
    return labels


def fixed_level(u):
    """PRRBB 现有固定阈值分类（用于对比，支持向量输入）。"""
    u = np.asarray(u)
    out = np.full(u.shape, "MEDIUM", dtype=object)
    out[u < FIXED_LOW_MAX] = "LOW"
    out[u >= FIXED_HIGH_MIN] = "HIGH"
    return out


def main():
    ap = argparse.ArgumentParser(description="DBSCAN 一维风速分段（训练轨迹前 N 天）")
    ap.add_argument("--csv", default=None,
                    help="可选：逐 step 轨迹 CSV（含 step, u 列）；缺省按训练管线重建前 N 天")
    ap.add_argument("--days", type=int, default=80, help="训练轨迹前多少天（默认 80）")
    ap.add_argument("--steps", type=int, default=None, help="可选：精确步数（默认 days×1440）")
    ap.add_argument("--eps", type=float, default=0.8, help="邻域半径 eps（m/s，默认 0.8）")
    ap.add_argument("--min-samples", dest="min_samples", type=int, default=3,
                    help="核心点最少邻域点数（默认 3）")
    ap.add_argument("--outdir", default=os.path.join(HSRL_ROOT, "output", "data_piecewise"),
                    help="结果输出目录（默认 output/data_piecewise/）")
    args = ap.parse_args()

    steps, u = load_wind(args.days, csv=args.csv, steps=args.steps)
    n = u.size
    print(f"风速序列: 前 {n} 步"
          + (f"（{args.days} 天 × {STEPS_PER_DAY}）" if args.steps is None else "")
          + f"  来源: {args.csv or '训练管线重建(FFC + fixed_day_sequence)'}")
    print(f"风速范围 [{u.min():.3f}, {u.max():.3f}] m/s\n")

    labels = dbscan_1d(u, eps=args.eps, min_samples=args.min_samples)

    n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
    noise_mask = labels == -1
    print(f"== DBSCAN 结果 (eps={args.eps}, min_samples={args.min_samples}) ==")
    print(f"发现簇数: {n_clusters}   噪声点数: {int(noise_mask.sum())}/{n} "
          f"({100.0 * noise_mask.mean():.2f}%)\n")

    if n_clusters == 0:
        print("没有任何簇，全部为噪声。请调大 --eps 或调小 --min-samples。")
        summary = {"method": "dbscan", "days": args.days, "steps": n,
                   "eps": args.eps, "min_samples": args.min_samples,
                   "data_source": args.csv or "train_pipeline(FFC+day_seq)",
                   "n_clusters": 0, "noise_count": int(n)}
        os.makedirs(args.outdir, exist_ok=True)
        tag = f"{args.days}day" if args.steps is None else f"{args.steps}step"
        json_path = os.path.join(args.outdir, f"dbscan_{tag}_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"摘要已储存: {json_path}")
        return

    # 簇按代表值（成员中位数）升序排序
    repr_val = [float(np.median(u[labels == c])) for c in range(n_clusters)]
    cluster_order = np.argsort(repr_val)
    levels = ["LOW", "MEDIUM", "HIGH"] if n_clusters == 3 else \
        [f"CLUSTER_{pos}" for pos in range(n_clusters)]
    label_to_level = {int(cid): levels[pos] for pos, cid in enumerate(cluster_order)}

    clusters_info = []
    print("簇区间（按风速升序）:")
    for cid in cluster_order:
        m = labels == cid
        rng = (u[m].min(), u[m].max())
        lv = levels[int(np.where(cluster_order == cid)[0][0])]
        clusters_info.append({
            "label": lv, "cluster_id": int(cid),
            "min": round(float(rng[0]), 6), "max": round(float(rng[1]), 6),
            "members": int(m.sum()), "median": round(repr_val[cid], 6),
        })
        print(f"  {lv:8s}: {rng[0]:.4f} ~ {rng[1]:.4f} m/s  "
              f"(成员 {int(m.sum())}, 中位 {repr_val[cid]:.4f})")
    if noise_mask.sum():
        nv = np.unique(u[noise_mask])
        print(f"  NOISE    : {nv.size} 个不同风速值, "
              f"范围 [{nv.min():.3f}, {nv.max():.3f}] m/s")

    level_arr = np.array([label_to_level[int(l)] if l != -1 else "NOISE"
                          for l in labels], dtype=object)
    non_noise = ~noise_mask
    fixed = fixed_level(u)
    mismatch = non_noise & (level_arr != fixed)
    print(f"\n与固定 7.0/11.0 不一致（仅非噪声步）: {int(mismatch.sum())}/{int(non_noise.sum())} "
          f"({100.0 * mismatch.sum() / max(int(non_noise.sum()), 1):.2f}%)")

    # 保存结果
    os.makedirs(args.outdir, exist_ok=True)
    tag = f"{args.days}day" if args.steps is None else f"{args.steps}step"
    csv_path = os.path.join(args.outdir, f"dbscan_{tag}.csv")
    json_path = os.path.join(args.outdir, f"dbscan_{tag}_summary.json")
    pd.DataFrame({
        "step": steps, "u": u, "cluster_id": labels, "label": level_arr,
    }).to_csv(csv_path, index=False)

    summary = {
        "method": "dbscan", "days": args.days, "steps": n,
        "eps": args.eps, "min_samples": args.min_samples,
        "data_source": args.csv or "train_pipeline(FFC+day_seq)",
        "n_clusters": n_clusters,
        "clusters": clusters_info,
        "noise_count": int(noise_mask.sum()),
        "noise_rate": round(float(noise_mask.mean()), 6),
        "mismatch_vs_fixed_7_11": int(mismatch.sum()),
        "mismatch_rate": round(float(mismatch.sum() / max(int(non_noise.sum()), 1)), 6),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n== 前 8 步预览 ==")
    for st, ui, lv in list(zip(steps, u, level_arr))[:8]:
        print(f"  step {st:>6d}  u={ui:8.4f}  {lv}")
    print("  ...")
    for st, ui, lv in list(zip(steps, u, level_arr))[-4:]:
        print(f"  step {st:>6d}  u={ui:8.4f}  {lv}")

    print(f"\n结果已储存:")
    print(f"  {csv_path}")
    print(f"  {json_path}")

    if n_clusters != 3:
        print("\n提示: DBSCAN 不预设簇数，本次发现簇数 != 3，说明这 80 天风速的密度结构"
              "并非天然三档；噪声点为密度孤立风速，需额外决定归属策略"
              "（归入最近簇 / 单独 EXTREME 档）。可调 --eps / --min-samples 观察结构变化。")


if __name__ == "__main__":
    main()
