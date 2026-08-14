"""
PRRBB P_F 越限分析 —— 增加风速维度。

检查 P_F > 3.0 的越限在什么风速条件下发生，
按风速档位（低风中风高风）统计越限频率。
"""

import pandas as pd
import numpy as np

# ============================================================
# 配置
# ============================================================
CSV_PATH = 'output/PRRBB-run62/df_prrbb.csv'
STEP_START = 20000
STEP_END   = 25000

# 风速分档（与 prrbb_myfunc 一致）
WIND_LOW_MAX  = 7.0   # < 7.0 = 低风
WIND_HIGH_MIN = 11.0  # >= 11.0 = 高风, 中间 = 中风


def get_wind_level(u):
    if u < WIND_LOW_MAX:
        return 'LOW'
    elif u >= WIND_HIGH_MIN:
        return 'HIGH'
    else:
        return 'MEDIUM'


def extract_u(row):
    """
    从 observation 列提取风速 u（index=1）。
    observation 格式: [pr, u, E, P_G_last]
    - 若已是 list/ndarray: row['observation'][1]
    - 若是字符串: 解析后取 index 1
    - 若有单独 'u' 列则直接取
    """
    # 优先取单独列
    if 'u' in row.index and pd.notna(row['u']):
        return float(row['u'])

    obs = row['observation']
    if isinstance(obs, (list, np.ndarray)):
        return float(obs[1])
    if isinstance(obs, str):
        # numpy数组的字符串形式: "[0.58 7.2 3.0 5.1]" 或 "[0.58, 7.2, 3.0, 5.1]"
        s = obs.strip('[]')
        parts = s.replace(',', ' ').split()
        return float(parts[1])
    return np.nan


# ============================================================
# 读取数据
# ============================================================
df = pd.read_csv(CSV_PATH)
print(f"总行数: {len(df)}")
print(f"列名: {list(df.columns)}")

# 提取风速 u
df['u'] = df.apply(extract_u, axis=1)
df['wind_level'] = df['u'].apply(get_wind_level)

# 筛选区间
m = (df['step'] >= STEP_START) & (df['step'] <= STEP_END)
df_win = df[m].copy()
print(f"step {STEP_START}~{STEP_END} 区间行数: {len(df_win)}")

# ============================================================
# 1. P_F > 3.0 越限详情（含风速）
# ============================================================
THRESH = 3.0 + 1e-4
over = df_win[df_win['P_F'] > THRESH].copy()

print(f"\n{'='*70}")
print(f"P_F > 3.0 越限总数: {len(over)}")
print(f"{'='*70}")

if len(over) > 0:
    # 显示越限时的关键字段
    display_cols = ['step', 'P_F', 'u', 'wind_level', 'is_infeasible', 'inevitable_violation']
    available = [c for c in display_cols if c in over.columns]
    print(f"\n越限详情（前30条）:")
    print(over[available].head(30).to_string(index=False))

    # ============================================================
    # 2. 按风速档位统计越限
    # ============================================================
    print(f"\n--- 按风速档位统计越限 ---")
    total_by_level = df_win['wind_level'].value_counts()
    over_by_level  = over['wind_level'].value_counts()
    for level in ['LOW', 'MEDIUM', 'HIGH']:
        total_n = total_by_level.get(level, 0)
        over_n  = over_by_level.get(level, 0)
        rate    = over_n / total_n * 100 if total_n > 0 else 0
        print(f"  {level:6s}: 越限 {over_n:4d} / 总步数 {total_n:5d} = {rate:5.1f}%")

    # ============================================================
    # 3. 按风速分桶统计越限
    # ============================================================
    print(f"\n--- 按风速区间统计越限（1 m/s 桶） ---")
    bins = np.arange(0, np.ceil(df_win['u'].max()) + 1, 1.0)
    df_win['u_bin'] = pd.cut(df_win['u'], bins=bins, right=False)
    over['u_bin']   = pd.cut(over['u'], bins=bins, right=False)

    total_by_bin = df_win.groupby('u_bin', observed=False).size()
    over_by_bin  = over.groupby('u_bin', observed=False).size()
    rate_by_bin  = (over_by_bin / total_by_bin * 100).fillna(0)

    print(f"  {'风速区间':<12s} {'总步数':>6s} {'越限数':>6s} {'越限率':>7s}")
    print(f"  {'-'*35}")
    for iv in total_by_bin.index:
        if total_by_bin[iv] > 0:
            print(f"  {str(iv):<12s} {total_by_bin[iv]:>6d} "
                  f"{over_by_bin.get(iv, 0):>6d} {rate_by_bin[iv]:>6.1f}%")

    # ============================================================
    # 4. 越限时风速分布统计
    # ============================================================
    print(f"\n--- 越限时的风速分布 ---")
    print(f"  均值:   {over['u'].mean():.2f} m/s")
    print(f"  中位数: {over['u'].median():.2f} m/s")
    print(f"  标准差: {over['u'].std():.2f} m/s")
    print(f"  最小值: {over['u'].min():.2f} m/s")
    print(f"  最大值: {over['u'].max():.2f} m/s")

    # ============================================================
    # 5. 越限率最高的5个风速
    # ============================================================
    print(f"\n--- 越限率最高的5个整数风速 ---")
    top5 = rate_by_bin.sort_values(ascending=False).head(5)
    for iv, rate in top5.items():
        if rate > 0:
            print(f"  {str(iv):<12s} 越限率: {rate:.1f}%  "
                  f"(越限 {over_by_bin.get(iv, 0)} / 总 {total_by_bin[iv]})")

else:
    print("\n该区间内无 P_F > 3.0 越限。")

# ============================================================
# 6. 按 1440 步窗口 + 风速档位 聚合
# ============================================================
print(f"\n--- 每1440步窗口的越限次数及主导风速 ---")
df_win['window'] = df_win['step'] // 1440

# 每个窗口: 越限次数, 平均风速, 主导风速档位
window_stats = df_win.groupby('window').agg(
    violation_count = ('P_F', lambda x: (x > THRESH).sum()),
    total_steps      = ('P_F', 'count'),
    mean_u           = ('u', 'mean'),
    dominant_level   = ('wind_level', lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/A'),
).query('violation_count > 0')

if len(window_stats) > 0:
    print(window_stats.to_string())
else:
    print("  所有窗口均无越限。")

print(f"\n完成。")
