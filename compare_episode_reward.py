"""
Compare episode rewards between two projects using MAPE.

Supports two CSV formats:
  A) Training CSVs (df_episode_prrbb.csv, df_episode_prle.csv, etc.)
     → Columns: episode, episode_reward, ...
  B) Test-compare CSV (test_compare.csv)
     → Columns: episode, project, episode_reward, episode_profit, ...

Usage:
    python compare_episode_reward.py
    (all parameters entered interactively)
"""

import pandas as pd
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter


def compute_mape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.
    actual    — baseline (denominator)
    predicted — target / comparison
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    mask = np.abs(actual) > epsilon
    if not mask.any():
        return float('nan')
    pct_errors = np.abs((actual[mask] - predicted[mask]) / actual[mask])
    return float(np.mean(pct_errors) * 100.0)


def compute_smape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 1e-8) -> float:
    """Symmetric MAPE (bounded 0–200%), robust to near-zero values."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    denom = (np.abs(actual) + np.abs(predicted)) / 2.0 + epsilon
    return float(np.mean(np.abs(actual - predicted) / denom) * 100.0)


def main():
    print("=" * 60)
    print("  Episode Reward 对比工具 (MAPE / SMAPE / MAE / RMSE)")
    print("=" * 60)

    # ---- 1. CSV format ----
    print()
    print("CSV 格式:")
    print("  [1] 训练 CSV (每行一个 episode, 含 episode_reward 列)")
    print("  [2] 测试 CSV (test_compare.csv, 含 project 列需筛选)")
    fmt_choice = input("  选择 (1/2, 默认 1): ").strip()
    fmt = "1" if not fmt_choice else fmt_choice

    # ---- 2. CSV paths ----
    print("-" * 55)
    if fmt == "2":
        # Single CSV with project column
        csv_path = input("  CSV 路径: ").strip().strip('"')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'CSV not found: {csv_path}')
        df_all = pd.read_csv(csv_path, index_col=0)

        print(f"  可用项目: {sorted(df_all['project'].unique())}")
        baseline_proj = input("  Baseline 项目名 (如 PER): ").strip()
        target_proj = input("  Target   项目名 (如 PRLE): ").strip()

        df_base = df_all[df_all['project'] == baseline_proj].copy()
        df_targ = df_all[df_all['project'] == target_proj].copy()

        baseline_label = baseline_proj
        target_label = target_proj
    else:
        # Two separate training CSVs
        baseline_csv = input("  Baseline CSV 路径: ").strip().strip('"')
        target_csv = input("  Target   CSV 路径: ").strip().strip('"')

        if not os.path.exists(baseline_csv):
            raise FileNotFoundError(f'Baseline CSV not found: {baseline_csv}')
        if not os.path.exists(target_csv):
            raise FileNotFoundError(f'Target CSV not found: {target_csv}')

        df_base = pd.read_csv(baseline_csv, index_col=0)
        df_targ = pd.read_csv(target_csv, index_col=0)

        baseline_label = input("  Baseline 标签 (如 PRRBB): ").strip() or "Baseline"
        target_label = input("  Target   标签 (如 PRLE): ").strip() or "Target"

    # ---- 3. Metric column ----
    print("-" * 55)
    common_cols = set(df_base.columns) & set(df_targ.columns)
    numeric_cols = [c for c in common_cols
                    if df_base[c].dtype in ('float64', 'int64', 'float32', 'int32')]
    print(f"  可对比的数值列: {numeric_cols}")
    metric_col = input("  对比列名 (默认 episode_profit): ").strip()
    if not metric_col:
        metric_col = 'episode_profit'
    if metric_col not in df_base.columns or metric_col not in df_targ.columns:
        raise KeyError(f'Column "{metric_col}" not found in both CSVs')

    # ---- 4. Output ----
    print("-" * 55)
    tb_choice = input("  是否写入 TensorBoard? (y/n, 默认 y): ").strip().lower()
    write_tb = tb_choice != 'n'
    if write_tb:
        tb_dir = input("  TensorBoard 目录 (默认 runs/compare_episode_reward): ").strip()
        if not tb_dir:
            tb_dir = 'runs/compare_episode_reward'

    # ---- Load & align ----
    print(f'\nBaseline ({baseline_label}): {len(df_base)} episodes')
    print(f'  {metric_col}: mean={df_base[metric_col].mean():.2f}, '
          f'min={df_base[metric_col].min():.2f}, max={df_base[metric_col].max():.2f}')
    print(f'Target   ({target_label}):   {len(df_targ)} episodes')
    print(f'  {metric_col}: mean={df_targ[metric_col].mean():.2f}, '
          f'min={df_targ[metric_col].min():.2f}, max={df_targ[metric_col].max():.2f}')

    # Align by episode number
    merged = df_base[['episode', metric_col]].merge(
        df_targ[['episode', metric_col]],
        on='episode',
        suffixes=('_base', '_targ'),
        how='inner'
    )

    n = len(merged)
    n_base_only = len(df_base) - n
    n_targ_only = len(df_targ) - n

    print(f'\n对齐结果:')
    print(f'  {baseline_label}: {len(df_base)} episodes → 对齐 {n} 个, 丢弃 {n_base_only} 个')
    print(f'  {target_label}:   {len(df_targ)} episodes → 对齐 {n} 个, 丢弃 {n_targ_only} 个')
    print(f'  共同 episode 数: {n}')

    if n == 0:
        print('ERROR: No overlapping episodes found.')
        return

    actual = merged[f'{metric_col}_base'].values
    predicted = merged[f'{metric_col}_targ'].values

    # ---- Metrics ----
    mape = compute_mape(actual, predicted)
    smape = compute_smape(actual, predicted)
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mean_diff = float(np.mean(predicted - actual))
    rel_improvement = float(np.mean((predicted - actual) / np.abs(actual)) * 100.0)

    print(f'\n{"="*60}')
    print(f'  Metric                             Value')
    print(f'{"="*60}')
    print(f'  MAPE  (Mean Abs % Error)           {mape:.4f} %')
    print(f'  SMAPE (Symmetric MAPE)             {smape:.4f} %')
    print(f'  MAE   (Mean Absolute Error)        {mae:.4f}')
    print(f'  RMSE  (Root Mean Squared Err)      {rmse:.4f}')
    print(f'  Mean diff ({target_label} − {baseline_label})   {mean_diff:+.4f}')
    print(f'  Relative improvement               {rel_improvement:+.4f} %')
    print(f'{"="*60}')

    # ---- Per-episode improvement ----
    show_n = min(10, n)
    print(f'\nPer-episode ({metric_col}), first {show_n}:')
    header = f'  {"Ep":>5s}  {baseline_label:>10s}  {target_label:>10s}  {"Diff":>10s}  {"Impr%":>8s}'
    print(header)
    for i in range(show_n):
        impr = (predicted[i] - actual[i]) / (abs(actual[i]) + 1e-8) * 100
        print(f'  {merged["episode"].iloc[i]:>5d}  '
              f'{actual[i]:>10.2f}  {predicted[i]:>10.2f}  '
              f'{predicted[i]-actual[i]:>+10.2f}  {impr:>+7.2f}%')

    # ---- Save comparison CSV ----
    merged['diff'] = predicted - actual
    merged['improvement_pct'] = (predicted - actual) / (np.abs(actual) + 1e-8) * 100
    out_csv = f'compare_{baseline_label}_{target_label}_{metric_col}.csv'
    merged.to_csv(out_csv, index=False)
    print(f'\n对比明细已保存至: {out_csv}')

    # ---- TensorBoard ----
    if write_tb:
        writer = SummaryWriter(tb_dir)
        for i in range(n):
            ep = int(merged['episode'].iloc[i])
            writer.add_scalar(f'compare/{metric_col}_baseline', actual[i], ep)
            writer.add_scalar(f'compare/{metric_col}_target', predicted[i], ep)
            writer.add_scalar(f'compare/{metric_col}_diff', predicted[i] - actual[i], ep)
            writer.add_scalar(f'compare/{metric_col}_impr_pct',
                              (predicted[i] - actual[i]) / (abs(actual[i]) + 1e-8) * 100, ep)

        cumsum_base = np.cumsum(actual)
        cumsum_targ = np.cumsum(predicted)
        for i in range(n):
            writer.add_scalar(f'compare/cumavg_{baseline_label}', cumsum_base[i] / (i + 1), i)
            writer.add_scalar(f'compare/cumavg_{target_label}', cumsum_targ[i] / (i + 1), i)

        writer.add_scalar('summary/MAPE_pct', mape, 0)
        writer.add_scalar('summary/SMAPE_pct', smape, 0)
        writer.add_scalar('summary/MAE', mae, 0)
        writer.add_scalar('summary/RMSE', rmse, 0)
        writer.add_scalar('summary/rel_improvement_pct', rel_improvement, 0)
        writer.add_scalar('summary/aligned_episodes', n, 0)
        writer.close()

        print(f'TensorBoard logs → {tb_dir}')
        print(f'  Run:  tensorboard --logdir={tb_dir}')


if __name__ == '__main__':
    main()
