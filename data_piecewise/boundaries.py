"""
风速分段边界统一计算 — percentile / kmeans / expert 三种方法
============================================================
供 prrbb / td3prrbb / prle / td3prle 的 main 在启动时按 --seg 选择：

  - percentile: 训练轨迹前 N 天风速的 33/66 百分位数（等频分段，每档约 1/3 曝光）
  - kmeans    : 训练轨迹前 N 天风速一维 KMeans(k=3) 簇中心中点（等方差分段）
  - expert    : 专家手工锚点（物理+统计+曝光率综合裁决，默认 8.5 / 12.0）

数据来源与 data_piecewise/kmeans.py 一致：由 FFC_speed_noise_1.csv +
fixed_day_sequence.csv 重建训练轨迹前 N 天风速（默认 80 天 = 115200 步）。
"""

import numpy as np

from data_piecewise.kmeans import load_wind, kmeans_1d

# 专家起点（物理锚点: 切入 3.5 / 额定≈12；统计共识带 8.7~9.2 / 11.8~12.2；
# 曝光均衡 26/42/31；轮廓 0.542 与 KMeans 最优几乎无差）
EXPERT_LOW_MAX = 8.5
EXPERT_HIGH_MIN = 12.0

METHODS = ('percentile', 'kmeans', 'expert')
PERCENTILES = (33, 66)
KMEANS_K = 3
KMEANS_SEED = 42


def compute_boundaries(method, days=80, csv=None, steps=None):
    """按方法计算 (low_max, high_min) 与信息字典。

    Parameters
    ----------
    method : str
        'percentile' | 'kmeans' | 'expert'
    days : int
        训练轨迹前多少天（默认 80）。
    csv : str or None
        可选：逐 step 轨迹 CSV（含 step, u 列），覆盖训练管线重建。
    steps : int or None
        可选：精确步数。

    Returns
    -------
    (low_max, high_min, info) : (float, float, dict)
    """
    method = str(method).lower()
    if method not in METHODS:
        raise ValueError(f"未知分段方法: {method!r}，可选 {METHODS}")

    if method == 'expert':
        return EXPERT_LOW_MAX, EXPERT_HIGH_MIN, {
            'method': 'expert', 'days': days,
            'low_max': EXPERT_LOW_MAX, 'high_min': EXPERT_HIGH_MIN,
            'note': f'专家锚点 {EXPERT_LOW_MAX}/{EXPERT_HIGH_MIN} m/s'
                    ' (切入3.5/额定≈12物理锚点 + 统计共识带 + 曝光均衡)',
        }

    _, u = load_wind(days, csv=csv, steps=steps)

    if method == 'percentile':
        p33 = float(np.percentile(u, PERCENTILES[0]))
        p66 = float(np.percentile(u, PERCENTILES[1]))
        return p33, p66, {
            'method': 'percentile', 'days': days, 'n': int(u.size),
            'percentiles': list(PERCENTILES),
            'low_max': p33, 'high_min': p66,
            'note': f'等频分段: 风速 {PERCENTILES[0]}/{PERCENTILES[1]} 分位 '
                    f'({p33:.4f}/{p66:.4f})，每档约 1/3 曝光',
        }

    # kmeans: 簇中心升序 → 相邻中心中点
    centers, labels, n_iter = kmeans_1d(u, KMEANS_K, seed=KMEANS_SEED)
    order = np.argsort(centers)
    cs = centers[order]
    low_max = float((cs[0] + cs[1]) / 2.0)
    high_min = float((cs[1] + cs[2]) / 2.0)
    counts = [int((labels == o).sum()) for o in order]
    return low_max, high_min, {
        'method': 'kmeans', 'days': days, 'n': int(u.size),
        'k': KMEANS_K, 'seed': KMEANS_SEED,
        'centers': [float(c) for c in cs],
        'counts': counts, 'iterations': n_iter,
        'low_max': low_max, 'high_min': high_min,
        'note': f'等方差分段: 簇中心中点 ({low_max:.4f}/{high_min:.4f})，'
                f'各档样本 {counts}',
    }


def apply_segmentation(method, set_thresholds_fn, days=80):
    """计算边界并应用到某 myfunc 模块的 set_wind_thresholds。

    Parameters
    ----------
    method : str
        'percentile' | 'kmeans' | 'expert'
    set_thresholds_fn : callable
        形如 set_wind_thresholds(low_max, high_min) -> (actual_low, actual_high)
    days : int
        训练轨迹前多少天（默认 80）。

    Returns
    -------
    (actual_low_max, actual_high_min, info)
    """
    low_max, high_min, info = compute_boundaries(method, days=days)
    actual_low, actual_high = set_thresholds_fn(low_max, high_min)
    print(f"[seg] 分段方法 = {method}: "
          f"LOW_MAX = {actual_low:.4f}, HIGH_MIN = {actual_high:.4f} m/s")
    print(f"[seg] {info['note']}")
    return actual_low, actual_high, info
