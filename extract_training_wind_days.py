"""
提取训练风速序列中指定 episode 对应的风速日，写入 TensorBoard。

训练使用固定随机序列 fixed_day_sequence.csv 决定每天训练用哪一列风速数据。
env.py 中的映射逻辑：
    self.j = FIXED_DAY_SEQUENCE[idx]          # 固定序列中的值
    self.wind_day = self.wind_noise.iloc[:, self.j]  # 按 iloc 位置取列

"第8天" = 第8个 episode 使用的风速列 = FIXED_DAY_SEQUENCE[7]

Usage:
    python extract_training_wind_days.py                # 默认第 8, 15, 25, 27 天
    python extract_training_wind_days.py --days 8 15    # 自定义天数
"""

import os
import argparse
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# ---- Paths ----
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXED_DAY_SEQ_PATH = os.path.join(_SCRIPT_DIR, 'fixed_day_sequence.csv')
WIND_CSV_PATH = os.path.join(_SCRIPT_DIR, 'FFC_speed_noise_1.csv')


def main():
    parser = argparse.ArgumentParser(
        description='Extract wind-speed days from training sequence → TensorBoard')
    parser.add_argument('--days', type=int, nargs='+', default=[8, 15, 25, 27],
                        help='Training episode indices (1-indexed). Default: 8 15 25 27')
    parser.add_argument('--output', type=str, default='runs/wind_training_days',
                        help='TensorBoard log directory')
    args = parser.parse_args()

    # ---- 1. Load fixed day sequence (same as env.py: FIXED_DAY_SEQUENCE) ----
    fixed_seq = pd.read_csv(FIXED_DAY_SEQ_PATH, header=None).values.flatten().tolist()
    print(f'fixed_day_sequence.csv: {len(fixed_seq)} entries (episode → iloc column position)')

    # ---- 2. Load wind CSV (same as env.py: pd.read_csv, NO index_col=0) ----
    wind_df = pd.read_csv(WIND_CSV_PATH)
    n_cols = wind_df.shape[1]
    print(f'Wind CSV: {n_cols} columns total '
          f'(iloc[:, 0] = Unnamed:0, iloc[:, 1] = col "0", ...)')

    # ---- 3. Resolve each requested training day ----
    writer = SummaryWriter(args.output)

    print(f'\n{"="*75}')
    print(f'  {"训练天数":>8s}  {"episode":>8s}  {"SEQ值":>6s}  {"iloc[:,N]":>10s}  {"CSV列名":>8s}')
    print(f'{"="*75}')

    for day_label in args.days:
        episode_idx = day_label - 1  # 1-indexed → 0-indexed
        if episode_idx < 0 or episode_idx >= len(fixed_seq):
            print(f'  【第{day_label}天】 OUT OF RANGE (valid: 1–{len(fixed_seq)}), skipped')
            continue

        iloc_pos = fixed_seq[episode_idx]  # same as env.py: self.j

        # 与 env.py 完全相同的方式取列：iloc[:, self.j]
        wind_series = wind_df.iloc[:, iloc_pos].values.astype(float)  # 1440 data points

        # 对应的 CSV 列名（仅用于显示参考）
        col_name = wind_df.columns[iloc_pos]

        print(f'  {"第"+str(day_label)+"天":>8s}  '
              f'{episode_idx:>8d}  {iloc_pos:>6d}  '
              f'{"iloc[:, "+str(iloc_pos)+"]":>10s}  '
              f'"{col_name}"')

        print(f'           → wind stats: '
              f'mean={wind_series.mean():.2f}, std={wind_series.std():.2f}, '
              f'min={wind_series.min():.2f}, max={wind_series.max():.2f}')

        tag = f'wind_speed/training_day_{day_label:02d}_iloc_{iloc_pos}'
        for minute, speed in enumerate(wind_series):
            writer.add_scalar(tag, speed, minute)

    writer.close()
    print(f'{"="*75}')
    print(f'\nDone. Run:  tensorboard --logdir={args.output}')


if __name__ == '__main__':
    main()
