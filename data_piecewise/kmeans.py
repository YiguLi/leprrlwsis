"""
KMeans 聚类分段 — 训练轨迹前 N 天风速分类（结果落盘）
====================================================
默认按训练轨迹**前 80 天**的风速分类（80 × 1440 = 115200 步）。

数据来源（与训练环境同一管线）：
  - 训练每天 = FFC_speed_noise_1.csv 的一列（1440 个风速值）；
  - 前 N 天的日索引 = fixed_day_sequence.csv 前 N 个值；
  - 拼接得到前 N 天训练轨迹风速序列。
  也可用 --csv 指定任意逐 step 轨迹 CSV（需含 step, u 列）。

算法：一维 KMeans(k=3)，纯 numpy（k-means++ 初始化 + Lloyd 迭代），
簇中心升序映射 LOW < MEDIUM < HIGH，档位边界 = 相邻簇中心中点（A4）。
质量指标：WCSS、轮廓系数（cumsum/searchsorted 向量化，支持 11.5 万点）。

结果储存（--outdir，默认 output/data_piecewise/）：
  - kmeans_<days>day.csv          每步分类：step, u, label, center
  - kmeans_<days>day_summary.json 配置 + 簇中心/边界/各档统计/质量指标/与固定7-11对比

用法:
  python data_piecewise/kmeans.py                      # 前 80 天
  python data_piecewise/kmeans.py --days 80 --k 3
  python data_piecewise/kmeans.py --csv output/TD3PRLE-run144/df_td3prle.csv --days 41
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
    """构建训练轨迹前 days 天的风速序列（默认 80 天 = 115200 步）。

    csv 为 None 时按训练数据管线重建：FFC_speed_noise_1.csv 的
    fixed_day_sequence.csv[0:days] 列拼接。
    返回 (step 数组, u 数组)。
    """
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


def kmeans_1d(x, k, seed=42, max_iter=300, tol=1e-9):
    """一维 k-means（k-means++ 初始化 + Lloyd 迭代）。返回 (centers, labels, n_iter)。"""
    rng = np.random.default_rng(seed)
    n = x.size
    centers = np.empty(k)

    # k-means++ 初始化
    centers[0] = x[rng.integers(n)]
    for j in range(1, k):
        dist = np.min(np.abs(x[:, None] - centers[:j][None, :]), axis=1)
        probs = dist ** 2
        s = probs.sum()
        probs = probs / s if s > 0 else np.full(n, 1.0 / n)
        centers[j] = x[rng.choice(n, p=probs)]

    # Lloyd 迭代
    labels = np.zeros(n, dtype=int)
    for it in range(max_iter):
        labels_new = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
        new_centers = np.array([
            x[labels_new == j].mean() if np.any(labels_new == j) else centers[j]
            for j in range(k)
        ])
        shift = float(np.abs(new_centers - centers).max())
        centers = new_centers
        labels = labels_new
        if shift < tol:
            return centers, labels, it + 1
    return centers, labels, max_iter


def wcss_1d(x, labels, centers):
    """簇内平方和（inertia）。"""
    return float(np.sum((x - centers[labels]) ** 2))


def _mean_abs_dist_to_sorted(y, x_pts):
    """x_pts 中每个点对排序数组 y 的平均绝对距离 (|x - y|)。"""
    m = y.size
    if m == 0:
        return None
    if m == 1:
        return np.abs(x_pts - y[0])
    cs = np.concatenate([[0.0], np.cumsum(y)])
    j = np.searchsorted(y, x_pts, side="right")
    left = x_pts * j - cs[j]
    right = (cs[m] - cs[j]) - x_pts * (m - j)
    return (left + right) / m


def silhouette_1d(x, labels):
    """一维轮廓系数（向量化，O(k·n log n)），[-1,1] 越大越紧致分离。"""
    n = x.size
    k = int(labels.max()) + 1
    if n < 2 or k < 2:
        return 0.0
    sorted_by_cluster = [np.sort(x[labels == c]) for c in range(k)]
    sizes = np.array([len(y) for y in sorted_by_cluster])
    a = np.zeros(n)
    b = np.full(n, np.inf)
    for c in range(k):
        y = sorted_by_cluster[c]
        m = y.size
        if m == 0:
            continue
        own = labels == c
        a[own] = _mean_abs_dist_to_sorted(y, x[own]) * m / max(m - 1, 1)
        d = _mean_abs_dist_to_sorted(y, x[~own])
        b[~own] = np.minimum(b[~own], d)
    s = np.where(np.isfinite(b), (b - a) / np.maximum(a, b), 0.0)
    s[sizes[labels] == 1] = 0.0  # 单点簇不计入
    return float(s.mean())


def fixed_level(u):
    """PRRBB 现有固定阈值分类（用于对比，支持向量输入）。"""
    u = np.asarray(u)
    out = np.full(u.shape, "MEDIUM", dtype=object)
    out[u < FIXED_LOW_MAX] = "LOW"
    out[u >= FIXED_HIGH_MIN] = "HIGH"
    return out


def main():
    ap = argparse.ArgumentParser(description="KMeans 一维风速分段（训练轨迹前 N 天）")
    ap.add_argument("--csv", default=None,
                    help="可选：逐 step 轨迹 CSV（含 step, u 列）；缺省按训练管线重建前 N 天")
    ap.add_argument("--days", type=int, default=80, help="训练轨迹前多少天（默认 80）")
    ap.add_argument("--steps", type=int, default=None, help="可选：精确步数（默认 days×1440）")
    ap.add_argument("--k", type=int, default=3, help="聚类簇数（默认 3）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    ap.add_argument("--outdir", default=os.path.join(HSRL_ROOT, "output", "data_piecewise"),
                    help="结果输出目录（默认 output/data_piecewise/）")
    args = ap.parse_args()

    steps, u = load_wind(args.days, csv=args.csv, steps=args.steps)
    n = u.size
    print(f"风速序列: 前 {n} 步"
          + (f"（{args.days} 天 × {STEPS_PER_DAY}）" if args.steps is None else "")
          + f"  来源: {args.csv or '训练管线重建(FFC + fixed_day_sequence)'}")
    print(f"风速范围 [{u.min():.3f}, {u.max():.3f}] m/s\n")

    centers, labels, n_iter = kmeans_1d(u, args.k, seed=args.seed)
    wcss = wcss_1d(u, labels, centers)
    sil = silhouette_1d(u, labels)

    order = np.argsort(centers)
    centers_sorted = centers[order]
    levels = ["LOW", "MEDIUM", "HIGH"][: args.k]
    label_to_level = {int(cid): levels[pos] for pos, cid in enumerate(order)}
    level_arr = np.array([label_to_level[l] for l in labels])
    boundaries = [(a + b) / 2.0 for a, b in zip(centers_sorted[:-1], centers_sorted[1:])]

    print("== KMeans 结果 ==")
    print(f"迭代: {n_iter}   WCSS: {wcss:.4f}   轮廓系数: {sil:.4f}\n")
    counts = {}
    for pos, c in enumerate(centers_sorted):
        cnt = int((labels == order[pos]).sum())
        counts[levels[pos]] = cnt
        m = labels == order[pos]
        print(f"  {levels[pos]:6s}: center={c:.4f} m/s  步数={cnt}  "
              f"范围[{u[m].min():.3f}, {u[m].max():.3f}]")
    if boundaries:
        print(f"\n  档位边界(中点): LOW_MAX ≈ {boundaries[0]:.4f}, "
              f"HIGH_MIN ≈ {boundaries[-1]:.4f}  (A4 落地阈值)")

    # 与固定 7/11 规则对比
    fixed = fixed_level(u)
    mismatch = level_arr != fixed
    print(f"\n与固定 7.0/11.0 规则不一致: {int(mismatch.sum())}/{n} "
          f"({100.0 * mismatch.mean():.2f}%)")

    # 保存结果
    os.makedirs(args.outdir, exist_ok=True)
    tag = f"{args.days}day" if args.steps is None else f"{args.steps}step"
    csv_path = os.path.join(args.outdir, f"kmeans_{tag}.csv")
    json_path = os.path.join(args.outdir, f"kmeans_{tag}_summary.json")
    pd.DataFrame({
        "step": steps, "u": u, "label": level_arr,
        "center": centers[labels],
    }).to_csv(csv_path, index=False)

    summary = {
        "method": "kmeans", "days": args.days, "steps": n, "k": args.k,
        "seed": args.seed, "data_source": args.csv or "train_pipeline(FFC+day_seq)",
        "centers": {levels[pos]: round(float(c), 6)
                    for pos, c in enumerate(centers_sorted)},
        "boundaries": [round(float(b), 6) for b in boundaries],
        "counts": counts, "wcss": round(wcss, 6),
        "silhouette": round(sil, 6),
        "mismatch_vs_fixed_7_11": int(mismatch.sum()),
        "mismatch_rate": round(float(mismatch.mean()), 6),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n== 前 8 步预览 ==")
    for st, ui, lv in list(zip(steps, u, level_arr))[:8]:
        print(f"  step {st:>6d}  u={ui:8.4f}  {lv}")
    print("  ...")
    for st, ui, lv in list(zip(steps, u, level_arr))[-4:]:
        print(f"  step {st:>6d}  u={ui:8.4f}  {lv}")

    print(f"\n结果已储存:")
    print(f"  {csv_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
