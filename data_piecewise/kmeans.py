"""
KMeans 聚类分段 — 训练轨迹前 N step 风速分类
============================================
对训练轨迹 CSV（默认 TD3PRLE-run144/df_td3prle.csv）前 N 步（默认 80）的
风速 u 做一维 KMeans(k=3) 聚类，把每一步风速划分为 LOW / MEDIUM / HIGH 三档。

纯 numpy 实现（项目环境未装 sklearn）：
  - k-means++ 初始化 + Lloyd 迭代（固定 seed，结果可复现）
  - 簇中心按数值升序映射为 LOW < MEDIUM < HIGH
  - 档位边界 = 相邻簇中心的中点（即方案 A4：KMeans 边界阈值化）
  - 附带 WCSS / 轮廓系数质量指标，并与固定 7.0/11.0 规则对比

用法:
  python data_piecewise/kmeans.py
  python data_piecewise/kmeans.py --csv output/TD3PRLE-run144/df_td3prle.csv --steps 80
  python data_piecewise/kmeans.py --k 3 --seed 42 --out data_piecewise/kmeans_result.csv
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


def kmeans_1d(x, k, seed=42, max_iter=300, tol=1e-9):
    """一维 k-means（k-means++ 初始化 + Lloyd 迭代）。

    返回 (centers, labels, n_iter)。
    """
    rng = np.random.default_rng(seed)
    n = x.size
    centers = np.empty(k)

    # ---- k-means++ 初始化 ----
    centers[0] = x[rng.integers(n)]
    for j in range(1, k):
        dist = np.min(np.abs(x[:, None] - centers[:j][None, :]), axis=1)
        probs = dist ** 2
        s = probs.sum()
        probs = probs / s if s > 0 else np.full(n, 1.0 / n)
        centers[j] = x[rng.choice(n, p=probs)]

    # ---- Lloyd 迭代 ----
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


def silhouette_1d(x, labels):
    """一维轮廓系数（numpy 实现，[-1, 1]，越大越紧致分离）。"""
    n = x.size
    if n < 2:
        return 0.0
    k = labels.max() + 1
    s = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        if same.sum() <= 1:
            s[i] = 0.0
            continue
        a = np.abs(x[same] - x[i]).mean()
        b = np.inf
        for j in range(k):
            if j == labels[i]:
                continue
            other = labels == j
            if other.sum() == 0:
                continue
            b = min(b, np.abs(x[other] - x[i]).mean())
        s[i] = (b - a) / max(a, b) if np.isfinite(b) else 0.0
    return float(s.mean())


def fixed_level(u):
    """PRRBB 现有固定阈值分类（用于对比）。"""
    if u < FIXED_LOW_MAX:
        return "LOW"
    if u >= FIXED_HIGH_MIN:
        return "HIGH"
    return "MEDIUM"


def main():
    ap = argparse.ArgumentParser(description="KMeans 一维风速分段（训练轨迹前 N 步）")
    ap.add_argument("--csv", default=DEFAULT_CSV, help="训练轨迹 CSV（含 step, u 列）")
    ap.add_argument("--steps", type=int, default=80, help="取前多少步（默认 80）")
    ap.add_argument("--k", type=int, default=3, help="聚类簇数（默认 3 = LOW/MEDIUM/HIGH）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    ap.add_argument("--out", default=None, help="可选：输出每步分类 CSV")
    args = ap.parse_args()

    steps, u = load_wind_first_steps(args.csv, args.steps)
    print(f"数据: {args.csv}  前 {len(steps)} 步  风速范围 [{u.min():.3f}, {u.max():.3f}] m/s\n")

    centers, labels, n_iter = kmeans_1d(u, args.k, seed=args.seed)
    wcss = wcss_1d(u, labels, centers)
    sil = silhouette_1d(u, labels)

    # 簇中心升序 → LOW < MEDIUM < HIGH
    order = np.argsort(centers)
    centers_sorted = centers[order]
    levels = ["LOW", "MEDIUM", "HIGH"][: args.k]
    label_to_level = {cid: levels[pos] for pos, cid in enumerate(order)}

    # 档位边界 = 相邻簇中心中点（A4: KMeans 边界阈值化）
    boundaries = []
    for a, b in zip(centers_sorted[:-1], centers_sorted[1:]):
        boundaries.append((a + b) / 2.0)

    print("== KMeans 结果 ==")
    print(f"迭代次数: {n_iter}   WCSS(inertia): {wcss:.4f}   轮廓系数: {sil:.4f}")
    print("\n簇中心（升序）与档位:")
    for pos, c in enumerate(centers_sorted):
        print(f"  {levels[pos]:6s}: center = {c:.4f} m/s  (样本数 {int((labels == order[pos]).sum())})")
    if boundaries:
        bstr = "  边界(中点): " + ", ".join(f"{b:.4f}" for b in boundaries)
        print(bstr)
    print(f"\n等效固定阈值（A4 落地值）: LOW_MAX ≈ {boundaries[0]:.4f}, HIGH_MIN ≈ {boundaries[-1]:.4f}"
          if boundaries else "\n（仅 1 簇，无边界）")

    # ---- 每步分类 ----
    print("\n== 前 {} 步逐 step 分类 ==".format(len(steps)))
    print("  step     u(m/s)   KMeans档   | 固定7/11档  是否一致")
    n_diff = 0
    for st, ui, lab in zip(steps, u, labels):
        lv = label_to_level[lab]
        fv = fixed_level(ui)
        diff = "  *" if lv != fv else ""
        if lv != fv:
            n_diff += 1
        print(f"  {st:>4d}  {ui:8.4f}   {lv:8s}  |   {fv:8s}  {diff}")
    print(f"\n与固定 7.0/11.0 规则不一致的步数: {n_diff}/{len(steps)}")

    # ---- 汇总 ----
    print("\n== 各档汇总 ==")
    for lv in levels:
        m = np.array([lv == label_to_level[l] for l in labels])
        if m.sum() == 0:
            continue
        print(f"  {lv:6s}: {int(m.sum()):3d} 步  风速范围 [{u[m].min():.3f}, {u[m].max():.3f}]")

    if args.out:
        out_df = pd.DataFrame({
            "step": steps, "u": u,
            "kmeans_label": [label_to_level[l] for l in labels],
            "center": centers[labels],
        })
        out_df.to_csv(args.out, index=False)
        print(f"\n已写出: {args.out}")


if __name__ == "__main__":
    main()
