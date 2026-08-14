# -*- coding: utf-8 -*-
"""
生成随机天数序列，用于训练中决定每个 episode 使用 FFC_speed_noise_1.csv
的哪一列风速数据（即 .iloc[:, j] 中的 j）。

CSV 结构：共 195 列，第 0 列为无名索引列，第 1–194 列为实际风速数据。
默认生成 1–194 的无重复随机排列，长度 = 194（与可用风速列数一致）。

若指定 --length 超过范围大小，则降级为 random.choices（允许重复）。

Usage:
    python generate_day_sequence.py                  # 默认：1–194 排列
    python generate_day_sequence.py --length 200     # 200 个值，有重复
    python generate_day_sequence.py --seed 123       # 自定义随机种子
    python generate_day_sequence.py --output my_seq.csv
"""

import argparse
import random
import os

# CSV 有 195 列：iloc[:, 0] 为无名索引，iloc[:, 1]~iloc[:, 194] 为风速数据
DEFAULT_RANGE_START = 1
DEFAULT_RANGE_END = 194       # inclusive
DEFAULT_LENGTH = 194          # 无重复排列，match 可用列数
DEFAULT_SEED = 64
DEFAULT_OUTPUT = 'fixed_day_sequence.csv'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        description='Generate random day-index sequence for wind-column selection.')
    parser.add_argument('--length', type=int, default=DEFAULT_LENGTH,
                        help=f'Sequence length (default: {DEFAULT_LENGTH})')
    parser.add_argument('--range-start', type=int, default=DEFAULT_RANGE_START,
                        help=f'Start of iloc range (default: {DEFAULT_RANGE_START})')
    parser.add_argument('--range-end', type=int, default=DEFAULT_RANGE_END,
                        help=f'End of iloc range, inclusive (default: {DEFAULT_RANGE_END})')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'Random seed for reproducibility (default: {DEFAULT_SEED})')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help=f'Output CSV filename (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--header', action='store_true',
                        help='Include CSV header line (off by default; env.py expects no header).')
    parser.add_argument('--allow-repeat', action='store_true',
                        help='Allow repeated values when --length > range size '
                             '(default: random.choices with replacement).')
    args = parser.parse_args()

    # ---- Validate ----
    if args.range_start > args.range_end:
        raise ValueError(f'range_start ({args.range_start}) > range_end ({args.range_end})')
    if args.length < 1:
        raise ValueError(f'length must be >= 1, got {args.length}')

    population = list(range(args.range_start, args.range_end + 1))
    range_size = len(population)

    # ---- Generate ----
    random.seed(args.seed)

    if args.length <= range_size:
        # 无重复排列
        sequence = random.sample(population, args.length)
        has_dupes = False
    elif args.allow_repeat:
        # 有放回抽样
        print(f'WARNING: length ({args.length}) > range size ({range_size}), '
              f'using random.choices WITH replacement.')
        sequence = random.choices(population, k=args.length)
        has_dupes = len(sequence) != len(set(sequence))
    else:
        raise ValueError(
            f'length ({args.length}) exceeds range size ({range_size}). '
            f'No-repeat permutation impossible. Use --allow-repeat for sampling with replacement.'
        )

    # ---- Write ----
    output_path = os.path.join(SCRIPT_DIR, args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        if args.header:
            f.write('day_index\n')
        for val in sequence:
            f.write(f'{val}\n')

    print(f'Generated sequence: length={len(sequence)}, '
          f'range=[{args.range_start}, {args.range_end}], seed={args.seed}')
    print(f'First 10: {sequence[:10]}')
    print(f'Last  10: {sequence[-10:]}')
    print(f'Has duplicates: {has_dupes}')
    print(f'Min={min(sequence)}, Max={max(sequence)}')
    print(f'Saved to: {output_path}')


if __name__ == '__main__':
    main()
