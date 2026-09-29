#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_five_days_regression.py —— P4② 回归：旧方法评测结果与"改动前"**逐位一致**。

为什么需要它（而不是直接跑两次真 MATLAB）
------------------------------------------
P4② 的判据是"同一 seed 下旧 9 个方法之一的评测结果与改动前逐位一致"。真跑需要
MATLAB（`env.PowerOutput → matlab.engine.WFSim_port`），而本项目文档已记录：

> MATLAB 启动时要写自己的偏好/临时目录，在受限文件沙箱（workspace-write）内会以
> `Fatal Startup Error` 失败；`matlab.engine.start_matlab()` 则表现为
> `UnicodeDecodeError`（引擎把 GBK 的中文启动横幅按 UTF-8 解码，掩盖了真实错误）。

⇒ 本脚本把 `matlab.engine.start_matlab` 换成**确定性 stub 引擎**（纯 Python、无 MATLAB），
于是可以在同一台受限机器上跑两条评测并**逐位比较**：

* 臂 **post** = 当前工作树（含本次全部改动：`env.py` 惰性 matlab / `physics` 形参 /
  标称动作类属性；`test_five_days.py` 的 env 解析 + `env.nominal()` + 引擎按需启动）；
* 臂 **pre**  = 把 `env.py` 与 `test_five_days.py` **从基线 commit 取回**（`git show`），
  跑完后 `git restore` 复原（不改写历史、不动工作树的其他文件）。

两条臂用**同一个 stub 引擎**（本脚本自带）⇒ 唯一变量就是那两个文件。
每条臂都在**独立子进程**里跑（模块缓存必须干净，否则 pre 臂会复用 post 臂已 import 的 env）。

判据：`five_day_steps.csv`（逐步明细）/ `five_day_days.csv` / `five_day_summary.csv`
逐列逐行严格相等（浮点按 `np.array_equal` ⇒ "逐位"）。

用法
----
    python agentutil/test_five_days_regression.py            # 全流程（含 git 取回基线）
    python agentutil/test_five_days_regression.py --days 2   # 更快
    python agentutil/test_five_days_regression.py --phase run --tag post   # 只跑一条臂（内部用）
"""
import argparse
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

os.environ.setdefault('MPLCONFIGDIR', os.path.join(ROOT, 'output', '_mplcache'))

# 基线 = 本次会话改动**之前**的最后一个 commit（yawflor 计划文档那次）
DEFAULT_BASE = '476e269'
FILES = ('env.py', 'test_five_days.py')
PROJ = 'td3per'
MODEL_DIR = os.path.join('output', 'TD3PER-run10')
SEED = 7


# ===========================================================================
# 确定性 stub 物理引擎（取代 MATLAB 的 WFSim_port）
# ===========================================================================
class _StubWFSim:
    """`WFSim_port(u, phi, a, phi2) → {'out': W}` 的确定性替代。

    单机 `Cp = 4a(1−a)²`（与 `WFSim_port.m` 同口径）+ 简化的串行尾流亏损。
    它**不需要物理正确**（两条臂用的是同一个），只需要：确定、有限、对动作敏感
    （这样策略轨迹、储能窗口、越限计数都会被真正走到）。
    """

    RHO = 1.20
    AREA = float(np.pi * 50.0 ** 2)

    def __init__(self):
        self.calls = 0

    def WFSim_port(self, u, phi, a, phi2):
        u0 = float(np.asarray(u).reshape(-1)[0])
        acts = [float(x) for x in np.asarray(a).reshape(-1)][:5]
        self.calls += 1
        if not (3.5 <= u0 <= 25):
            return {'out': 0.0}
        total, deficit = 0.0, 0.0
        for aj in acts:
            aj = min(max(aj, 0.0), 0.5)
            cp = 4.0 * aj * (1.0 - aj) ** 2
            total += 0.95 * 0.5 * self.RHO * self.AREA * cp * (u0 * (1.0 - deficit)) ** 3
            deficit = min(0.45, max(deficit, 0.12 * cp + 0.30 * deficit))
        return {'out': total}

    def quit(self):
        return None


def run_arm(tag, days):
    """跑一条臂（在**当前进程**内；由 `--phase run` 在子进程里调用）。"""
    import matlab.engine
    matlab.engine.start_matlab = lambda *a, **kw: _StubWFSim()   # 必须在 import 项目模块之前

    out_dir = os.path.join('output', '_p4_{}'.format(tag))
    sys.argv = ['test_five_days.py', '--{}'.format(PROJ), MODEL_DIR,
                '--days', str(days), '--seed', str(SEED),
                '--output', out_dir, '--no-tensorboard', '--device', 'cpu']
    import test_five_days as T
    print('=' * 78)
    print('P4 回归臂 [{}]: {} / {} / days={} / seed={} / out={}'
          .format(tag, PROJ, MODEL_DIR, days, SEED, out_dir))
    print('=' * 78)
    try:
        T.main()
    except SystemExit as e:                       # main() 只可能用 SystemExit 退出
        if e.code:
            raise
    print('[{}] 完成，输出目录 = {}'.format(tag, out_dir))
    return out_dir


# ===========================================================================
# 编排：post → pre（git 取回基线）→ 比较
# ===========================================================================
def _git(args, check=True):
    p = subprocess.run(['git'] + list(args), capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    if check and p.returncode != 0:
        raise RuntimeError('git {} 失败: {}'.format(' '.join(args), p.stderr))
    return p.stdout


def _restore():
    _git(['restore', '--source=HEAD', '--'] + list(FILES))


def _worktree_dirty():
    return bool(_git(['status', '--short', '--'] + list(FILES)).strip())


def _run_subprocess(tag, days, log):
    cmd = [sys.executable, os.path.join('agentutil',
                                        'test_five_days_regression.py'),
           '--phase', 'run', '--tag', tag, '--days', str(days)]
    with open(log, 'w', encoding='utf-8') as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return p.returncode


def _compare(dir_a, dir_b):
    import pandas as pd
    ok = True
    for name in ('five_day_steps.csv', 'five_day_days.csv', 'five_day_summary.csv'):
        pa, pb = os.path.join(dir_a, name), os.path.join(dir_b, name)
        if not (os.path.isfile(pa) and os.path.isfile(pb)):
            print('  %-24s **缺文件** A=%s B=%s'
                  % (name, os.path.isfile(pa), os.path.isfile(pb)))
            ok = False
            continue
        da, db = pd.read_csv(pa), pd.read_csv(pb)
        if da.shape != db.shape or list(da.columns) != list(db.columns):
            print('  %-24s **形状/列不同** A=%s B=%s' % (name, da.shape, db.shape))
            ok = False
            continue
        bad = []
        for c in da.columns:
            va, vb = da[c].to_numpy(), db[c].to_numpy()
            if va.dtype.kind in 'fc' and vb.dtype.kind in 'fc':
                eq = np.array_equal(va, vb, equal_nan=True)
            else:
                eq = np.array_equal(va.astype(str), vb.astype(str))
            if not eq:
                bad.append(c)
        print('  %-24s %s  %d 行 × %d 列%s'
              % (name, 'OK  ' if not bad else '**FAIL**', da.shape[0], da.shape[1],
                 '' if not bad else '  差异列: %s' % bad[:6]))
        ok = ok and not bad
    return ok


def main():
    ap = argparse.ArgumentParser(description='P4② 回归：旧方法评测逐位一致')
    ap.add_argument('--phase', choices=['all', 'run'], default='all')
    ap.add_argument('--tag', default='post')
    ap.add_argument('--days', type=int, default=5)
    ap.add_argument('--base', default=DEFAULT_BASE, help='基线 commit（改动前）')
    args = ap.parse_args()

    if args.phase == 'run':
        run_arm(args.tag, args.days)
        return 0

    if _worktree_dirty():
        print('⚠ 工作树的 %s 有未提交改动 —— 本脚本会临时覆盖它们；'
              '请先提交/快照再跑。' % ', '.join(FILES))
        return 2

    print('基线 commit = %s（%s）'
          % (args.base, _git(['log', '-1', '--format=%s', args.base]).strip()))
    print('模型 = %s / seed = %d / days = %d' % (MODEL_DIR, SEED, args.days))

    # ---- 臂 post：当前工作树 ----
    log_post = os.path.join('output', '_p4_post.log')
    rc = _run_subprocess('post', args.days, log_post)
    print('臂 post 退出码 = %d（日志 %s）' % (rc, log_post))
    if rc != 0:
        print('❌ post 臂失败，终止')
        return 1

    # ---- 臂 pre：取回基线的两个文件 ----
    try:
        for f in FILES:
            with open(f, 'w', encoding='utf-8', newline='') as fh:
                fh.write(_git(['show', '%s:%s' % (args.base, f)]))
        print('已取回基线版本: %s（跑完自动 restore）' % ', '.join(FILES))
        log_pre = os.path.join('output', '_p4_pre.log')
        rc = _run_subprocess('pre', args.days, log_pre)
        print('臂 pre 退出码 = %d（日志 %s）' % (rc, log_pre))
    finally:
        _restore()
        print('已 restore 工作树: %s（dirty=%s）'
              % (', '.join(FILES), _worktree_dirty()))
    if rc != 0:
        print('❌ pre 臂失败，终止')
        return 1

    # ---- 比较 ----
    print()
    print('=' * 78)
    print('P4② 逐位比较：post(当前) vs pre(%s)' % args.base)
    print('=' * 78)
    ok = _compare(os.path.join('output', '_p4_post'), os.path.join('output', '_p4_pre'))
    print()
    print('✅ P4② 通过：旧方法（%s）评测结果与改动前**逐位一致**' % PROJ if ok
          else '❌ P4② 失败：存在逐位差异（见上）')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
