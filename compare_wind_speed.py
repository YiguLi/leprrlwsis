"""
对比 FFC_speed_1.csv (无噪声) vs FFC_speed_noise_1.csv (有噪声)
选取第一天数据，写入 TensorBoard
"""

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# 读取两份数据
speed_1 = pd.read_csv("FFC_speed_1.csv", index_col=0)
speed_noise = pd.read_csv("FFC_speed_noise_1.csv", index_col=0)

# 选取第一天 (column '0')
day_smooth = speed_1["0"].values   # 1440 分钟
day_noise = speed_noise["0"].values

print(f"数据点数: {len(day_smooth)}")
print(f"smooth:  mean={day_smooth.mean():.2f}  std={day_smooth.std():.2f}  min={day_smooth.min():.2f}  max={day_smooth.max():.2f}")
print(f"noise:   mean={day_noise.mean():.2f}   std={day_noise.std():.2f}   min={day_noise.min():.2f}   max={day_noise.max():.2f}")

writer = SummaryWriter("runs/wind_speed_compare")

for minute in range(len(day_smooth)):
    writer.add_scalar("wind_speed/smooth", day_smooth[minute], minute)
    writer.add_scalar("wind_speed/noise",   day_noise[minute],  minute)
    # 差异
    writer.add_scalar("wind_speed/diff_noise_minus_smooth",
                      day_noise[minute] - day_smooth[minute], minute)

writer.close()
print("\n写入完成。运行: tensorboard --logdir=runs/wind_speed_compare")
