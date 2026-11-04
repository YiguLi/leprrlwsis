#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""统一的"简化输出"开关（aifflor 同款）—— 由各 `*_main.py` 在训练前调用。

背景
----
aifflor / testpg 早就有 `--log_file`（详细逐行输出进 `<output>/train.log`，控制台只留
`--progress` 进度行）。其它用 **MATLAB/WFSim 环境**的方法没有这套东西，控制台会被每步的
`--H-- / r1 / c1 / price ...` 刷屏（一次训练几十万行），拖慢终端、也看不清进度。

本模块把那段逻辑抽出来，三个接口::

    run_io.setup(output_dir, log_file=0, progress=1, every=500, total=0)
    run_io.tick(step)          # 训练循环里 step += 1 之后调用
    run_io.close()             # 可选；进程退出时也会 flush

语义（与 aifflor 对齐）：
  * `log_file=0`（默认）⇒ **什么都不做**，输出与改造前逐位相同；
  * `log_file=1` ⇒ `sys.stdout` 重定向到 `<output>/train.log`（行缓冲），
    控制台只留一条提示 + 每 `every` 步的进度行；
  * 进度行只在 `log_file=1` 时生效（否则详细行仍会刷屏，进度行的意义不大）；
    `progress=0` 时连进度行也不打；
  * 进度行写到 `sys.__stdout__`（真实控制台），不受重定向影响。

进度行格式（与 aifflor 完全一致）::

      [TD3PRRBB-run3]  1500/200000 (  0.8%)   14.6 步/秒  已用    2 min  剩余约  227 min
"""
import os
import sys
import time

_STATE = {
    'active': False,
    'progress': False,
    'every': 500,
    'total': 0,
    'tag': '',
    't0': None,
    'fh': None,
    'old_stdout': None,
    #: 真实控制台的 fd 副本（`os.dup(1)`）—— fd 级重定向后仍要能打进度行
    'console_fd': None,
}


def _console_write(text):
    """写到**真实控制台**：优先用保存的 fd（fd 级重定向后依然有效）。

    编码用控制台自己的编码（Windows 下通常是 cp936 / utf-8）——直接写 utf-8 字节
    会在 cp936 控制台上把中文变成乱码。
    """
    fd = _STATE.get('console_fd')
    if fd is None:
        print(text, file=sys.__stdout__, flush=True)
        return
    enc = getattr(sys.__stdout__, 'encoding', None) or 'utf-8'
    try:
        data = (text + '\n').encode(enc, errors='replace')
    except LookupError:
        data = (text + '\n').encode('utf-8', errors='replace')
    try:
        os.write(fd, data)
    except OSError:
        print(text, file=sys.__stdout__, flush=True)


def setup(output_dir, log_file=0, progress=1, every=500, total=0):
    """在 `args.output = get_output_folder(...)` 之后调用一次。

    `log_file=1` 时做**两层**重定向：
      1. Python 层：`sys.stdout = <output>/train.log`；
      2. **fd 层**：`os.dup2(log_fd, 1)` —— MATLAB/WFSim 的 `disp(...)`、
         以及任何直接写 fd 1 的 C 扩展（torch/numpy 的 C 级输出）也会进 train.log。
         这是必要的：`WFSim.m:110/133` 的 `Performing N forward simulations..`
         是 **MATLAB 侧**的 `disp`，只重绑 `sys.stdout` 抓不到。
    """
    _STATE['active'] = bool(int(log_file or 0))
    _STATE['progress'] = bool(int(progress or 0)) and _STATE['active']
    _STATE['every'] = max(1, int(every or 500))
    _STATE['total'] = int(total or 0)
    _STATE['tag'] = os.path.basename(str(output_dir).rstrip('/\\'))
    _STATE['t0'] = time.time()
    if not _STATE['active']:
        return
    try:
        os.makedirs(str(output_dir), exist_ok=True)
        path = os.path.join(str(output_dir), 'train.log')
        _STATE['fh'] = open(path, 'w', encoding='utf-8', buffering=1)
        _STATE['old_stdout'] = sys.stdout
        # ---- fd 层：先留一个控制台副本，再把 fd 1 指到日志文件 ----
        try:
            _STATE['console_fd'] = os.dup(1)
            try:
                sys.stdout.flush()
            except Exception:                                 # noqa: BLE001
                pass
            os.dup2(_STATE['fh'].fileno(), 1)
        except OSError:
            _STATE['console_fd'] = None                       # 退化：只重绑 sys.stdout
        sys.stdout = _STATE['fh']
        _console_write('[log] 详细输出 → {}（控制台只显示进度与汇总；'
                       'MATLAB/WFSim 的 disp 也一并写入）'.format(path))
    except Exception as e:                                   # noqa: BLE001
        # 重定向失败就保持原样（与 aifflor 的处理一致）
        _STATE['active'] = False
        _STATE['progress'] = False
        _console_write('[log] 重定向失败, 保持控制台输出: {}'.format(e))


def set_total(total):
    """设置本次训练的总步数（用于百分比/ETA）。

    必须在 `train(...)` 调用前设：main 里 `args.train_iter_H` 往往在
    `get_output_folder()` **之后**才被 `1000*max_episode_length` / `--max_steps` 定稿。
    """
    _STATE['total'] = int(total or 0)


def tick(step):
    """训练循环里每步调用；只在 `step % every == 0` 且进度开启时打印。"""
    if not _STATE['progress']:
        return
    step = int(step)
    if step % _STATE['every'] != 0:
        return
    el = time.time() - _STATE['t0']
    rate = step / max(1e-9, el)
    total = _STATE['total']
    if total > 0:
        left = max(0.0, (total - step) / max(1e-9, rate))
        _console_write(
            '  [{}] {:>7}/{} ({:5.1f}%)  {:5.1f} 步/秒  已用 {:4.0f} min  剩余约 {:4.0f} min'
            .format(_STATE['tag'], step, total,
                    100.0 * step / max(1, total), rate, el / 60.0, left / 60.0))
    else:
        _console_write('  [{}] {:>7} 步  {:5.1f} 步/秒  已用 {:4.0f} min'
                       .format(_STATE['tag'], step, rate, el / 60.0))


def close():
    """恢复 stdout / fd 1 并关闭日志文件（幂等；进程退出也会 flush）。"""
    fd = _STATE.get('console_fd')
    if fd is not None:
        try:
            os.dup2(fd, 1)          # 先把 fd 1 指回控制台
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass
        _STATE['console_fd'] = None
    fh = _STATE.get('fh')
    if fh is not None:
        try:
            sys.stdout = _STATE.get('old_stdout') or sys.__stdout__
        except Exception:                                     # noqa: BLE001
            pass
        try:
            fh.flush()
            fh.close()
        except Exception:                                     # noqa: BLE001
            pass
    _STATE['fh'] = None
    _STATE['active'] = False
    _STATE['progress'] = False
