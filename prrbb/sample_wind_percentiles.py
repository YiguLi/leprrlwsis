# -*- coding: utf-8 -*-
"""
Randomly sample 5 days of wind speed data, calculate 33rd and 66th percentiles.
Pure stdlib — no pandas/numpy needed.
"""

import csv
import random
import math
import os
from pathlib import Path
import io

# ---- Config ----
DATA_ROOT = Path(r'c:\Users\yigul\Desktop\llm enhanced wsis\code\HSRL\FFC_Mast_Wind_Data')
WIND_COL = 'v80a (m/s)_mean'   # 80m mean wind speed
N_DAYS = 5                      # number of days to sample
SEED = 42                       # random seed (None = different each run)

random.seed(SEED)


def percentile_sorted(sorted_values, p):
    """Linear-interpolation percentile (matches numpy default)."""
    n = len(sorted_values)
    if n == 0:
        return float('nan')
    idx = (p / 100.0) * (n - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def read_wind_column(filepath):
    """
    Parse a _stat.csv file: skip '#' comment lines, read the CSV table,
    return list of float values from the WIND_COL column.
    """
    values = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
        # Skip comment lines, save first non-comment line as header
        for line in fh:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                header_line = stripped
                break

        # Read remaining rows
        body = fh.read()

    # Parse header
    header_cols = next(csv.reader(io.StringIO(header_line)))
    # Find target column
    try:
        wind_idx = header_cols.index(WIND_COL)
    except ValueError:
        # Debug: print available columns
        wind_cols = [c for c in header_cols if 'v80' in c.lower() or 'wind' in c.lower()]
        print(f"    DEBUG: WIND_COL not found. Columns with 'v80': {wind_cols}")
        print(f"    DEBUG: All columns ({len(header_cols)}): {header_cols[:5]}...")
        return values

    # Parse data rows
    reader = csv.reader(io.StringIO(body))
    for row in reader:
        if len(row) > wind_idx:
            try:
                values.append(float(row[wind_idx]))
            except ValueError:
                pass
    return values


# ---- Collect all CSV files ----
all_files = sorted(DATA_ROOT.rglob('*_stat.csv'))
print(f"Total files found: {len(all_files)}")

# ---- Randomly pick 5 days ----
chosen = random.sample(all_files, N_DAYS)

# ---- Read and collect wind speeds ----
all_wind_speeds = []
for f in chosen:
    vals = read_wind_column(str(f))
    if vals:
        day_str = f"{f.parent.name}/{f.stem}"
        print(f"  OK  {day_str}  -> {len(vals)} records, mean={sum(vals)/len(vals):.2f}")
        all_wind_speeds.extend(vals)
    else:
        print(f"  SKIP {f.stem}  (no data)")

# ---- Sort for percentile calculation ----
all_wind_speeds.sort()
wind = all_wind_speeds

print(f"\nTotal: {len(wind)} wind speed records across {N_DAYS} days")
print("(Theoretical max = 5 days * 144 ten-min intervals = 720)")

# ---- Statistics ----
if wind:
    mean_val = sum(wind) / len(wind)
    min_val = wind[0]
    max_val = wind[-1]
    var = sum((x - mean_val) ** 2 for x in wind) / len(wind)
    std_val = math.sqrt(var)
    p33 = percentile_sorted(wind, 33)
    p66 = percentile_sorted(wind, 66)

    print(f"\n===== RESULTS =====")
    print(f"5-day sequence length: {len(wind)}")
    print(f"Mean:       {mean_val:.4f} m/s")
    print(f"Std:        {std_val:.4f} m/s")
    print(f"Min:        {min_val:.4f} m/s")
    print(f"Max:        {max_val:.4f} m/s")
    print(f"33rd percentile:  {p33:.4f} m/s")
    print(f"66th percentile:  {p66:.4f} m/s")
    print(f"\nRandom seed = {SEED}  (change SEED for a different sample)")
else:
    print("\nERROR: No wind data collected!")
