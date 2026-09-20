import os

import pandas as pd
import torch
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

# BESS 放电退化成本 (RMB/MWh 放电量) —— 全项目统一口径，任何退化成本计算
# 必须引用本常量，禁止写裸数字。
# 经济含义: 放电盈亏平衡电价 ≈ 490 RMB/MWh（低谷 300 充电、平/峰 580/980 放电
# 才有利可图），与 r1/训练奖励口径一致。
# 历史修正: 训练奖励 r1 一直用 490，但报告口径 episode_profit / test9 profit
# 曾误用 4.9（×100 低估放电退化），两套账不一致；现统一为 490。
BESS_DEG_RMB_PER_MWH = 490.0

def round_list(array,n):
    ans = array
    for i in range(len(array)):
        ans[i] = round(array[i],n)
    return ans

def prRed(prt):
    print("\033[91m {}\033[00m".format(prt))


def prGreen(prt):
    print("\033[92m {}\033[00m".format(prt))


def prYellow(prt):
    print("\033[93m {}\033[00m".format(prt))


def prLightPurple(prt):
    print("\033[94m {}\033[00m".format(prt))


def prPurple(prt):
    print("\033[95m {}\033[00m".format(prt))


def prCyan(prt):
    print("\033[96m {}\033[00m".format(prt))


def prLightGray(prt):
    print("\033[97m {}\033[00m".format(prt))


def prBlack(prt):
    print("\033[98m {}\033[00m".format(prt))


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(torch.from_numpy(ndarray),
                    volatile=volatile,
                    requires_grad=requires_grad).type(dtype)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) +
                                param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def share_update(target, source, tau=0.5):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        param.data.copy_(target_param.data)


def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir


def flush_csv_append(path, df, written):
    """把 `df` 中第 `written` 行之后的**新增行**以【追加】方式写入 CSV（首次写表头）。

    主循环里"只增不减"的日志表用::

        df_written = 0
        ...
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        if step % 50 == 0:
            df_written = flush_csv_append(path, df, df_written)

    为什么必须增量：全量 `df.to_csv(path)` 的单次写盘量 = O(累积行数) ⇒ 迭代数越大越卡
    （实测某次 67k 步的 df 表 63 MB、每 50 步重写一次，这正是"50 整倍数步数卡顿随训练步数
    增大而加剧"的根因）。只写增量后每次写盘量 = O(新增行) = 常数（整轮总写盘量 O(N)，
    而非 O(N²/间隔)）。

    index 语义不变：累积用 `ignore_index=True` ⇒ 全局 RangeIndex，切片追加后连续。
    返回新的"已写行数"，调用方存回计数器。
    """
    if df is None or len(df) <= written:
        return written
    chunk = df.iloc[written:]
    if written == 0:
        if os.path.exists(path):
            os.remove(path)                      # 覆盖旧 run 的同名表 (首次写表头)
        chunk.to_csv(path, mode='w', index=True)
    else:
        chunk.to_csv(path, mode='a', header=False, index=True)
    return len(df)
