# -*- coding: utf-8 -*-
"""aifflorpave 的 B/C/D 变体**串行**队列（用户要求："把 BCD 各种测试全跑起来，别并行太多"）。

行为
----
1) 先等第一批 4 条（ABPAVE2-BASE / -SILU / -PAVE / -VFC）跑到 30000 步。判据 = 纯文件读
   `output/<tag>-run*/update_stats_H.csv` 的最后一列 step（不依赖进程/权限）；带 8h 安全上限。
2) 然后**逐条**跑队列（同一时刻只 1 个训练进程，OMP_NUM_THREADS=3）：
     NOCURV  : silu + mpr .46 + vfc .22            （论文关键消融：只压 M 不保 μ）
     CURV4X  : 同上 + curv .0224                    （Curv 权重 ×4，第一旋钮）
     EVERY4  : 全 PAVE + --pave_every 4             （算力匹配：分离"正则"与"多走一步"）
     MPR     : relu + mpr .46                       （单因子 MPR，不换激活）
     RELVFC  : relu + mpr .46 + vfc .22             （单因子 MPR+VFC，"不动激活"路线上限）
   每个 30000 步、seed 101、16/100、nstep 10、rate 0.01、新 warmup 口径（bsize×4=1024）。
3) 幂等：已存在 ≥29990 步的 run ⇒ 跳过（可安全重跑续做）。
4) 锁：`output/_pave_queue.lock` 存在 ⇒ 直接退出（防重复启动；`--force` 可覆盖）。
5) 日志：`output/_pave_queue.log`（队列自身）+ `output/_<tag>.log`（每条臂）。

用法::

    python agentutil/run_pave_queue.py [--force] [--target 29990]
"""
import argparse
import glob
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

OUT = os.path.join(ROOT, 'output')
LOCK = os.path.join(OUT, '_pave_queue.lock')
QLOG = os.path.join(OUT, '_pave_queue.log')

PY = r'F:\anaconda\envs\wsis\python.exe'
if not os.path.isfile(PY):
    PY = sys.executable

BATCH1 = ['ABPAVE2-BASE', 'ABPAVE2-SILU', 'ABPAVE2-PAVE', 'ABPAVE2-VFC']

QUEUE = [
    dict(tag='ABPAVE2-NOCURV',
         args='--critic_act silu --pave_w_mpr 0.46 --pave_w_vfc 0.22',
         why='论文关键消融：只压 M 不保 μ（无 Curv）是否反而更差'),
    dict(tag='ABPAVE2-CURV4X',
         args='--critic_act silu --pave_w_mpr 0.46 --pave_w_vfc 0.22 --pave_w_curv 0.0224',
         why='Curv 权重 ×4：若 C 臂的 6×6 负定性率没升，这是第一旋钮'),
    dict(tag='ABPAVE2-EVERY4',
         args=('--critic_act silu --pave_w_mpr 0.46 --pave_w_vfc 0.22 '
               '--pave_w_curv 0.0056 --pave_every 4'),
         why='算力匹配变体（每 4 次 critic 更新加 1 次几何步）⇒ 分离"正则"与"多走一步"'),
    dict(tag='ABPAVE2-MPR',
         args='--critic_act relu --pave_w_mpr 0.46',
         why='单因子 MPR（不换激活）'),
    dict(tag='ABPAVE2-RELVFC',
         args='--critic_act relu --pave_w_mpr 0.46 --pave_w_vfc 0.22',
         why='单因子 MPR+VFC（relu，无 Curv）——"不动激活"路线的上限'),
]


def say(msg):
    line = '%s  %s' % (time.strftime('%m-%d %H:%M:%S'), msg)
    print(line, flush=True)
    with open(QLOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def last_step(tag):
    """该臂当前推进到第几步（读 update_stats_H.csv 最后一行的 step 列；无则 -1）。"""
    pat = os.path.join(OUT, '%s-run*' % tag, 'update_stats_H.csv')
    files = sorted(glob.glob(pat))
    if not files:
        return -1
    step = -1
    with open(files[0], 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2 and parts[1].isdigit():
                step = int(parts[1])
    return step


def run_arm(tag, args):
    """启动一条臂，返回 (Popen, log 路径, t0)。"""
    log = os.path.join(OUT, '_%s.log' % tag)
    cmd = [PY, os.path.join(ROOT, 'aifflorpave', 'aifflorpave_main.py'),
           '--env', tag, '--max_steps', '30000', '--seed', '101',
           '--hidden1', '16', '--hidden2', '100', '--nstep', '10', '--rate', '0.01'] + args.split()
    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    env['OMP_NUM_THREADS'] = '3'
    fh = open(log, 'w', encoding='utf-8')
    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
    return p, log, time.time(), fh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='忽略锁文件')
    ap.add_argument('--target', type=int, default=29990, help='完成判据步数')
    ap.add_argument('--wait_hours', type=float, default=8.0, help='等第一批的上限')
    ap.add_argument('--jobs', type=int, default=2,
                    help='队列内的并发条数（默认 2；每进程 3 线程 ⇒ 峰值 6/16 线程）')
    a = ap.parse_args()

    if os.path.exists(LOCK) and not a.force:
        print('队列已在运行（%s 存在）⇒ 退出（要强跑加 --force）' % LOCK)
        return 0
    os.makedirs(OUT, exist_ok=True)
    with open(LOCK, 'w', encoding='utf-8') as f:
        f.write('pid=%d  start=%s\n' % (os.getpid(), time.strftime('%Y-%m-%d %H:%M:%S')))

    try:
        say('队列启动 (pid=%d, python=%s)；等第一批 4 条到 %d 步（上限 %.1fh）'
            % (os.getpid(), PY, a.target, a.wait_hours))
        deadline = time.time() + a.wait_hours * 3600
        while time.time() < deadline:
            pending = [t for t in BATCH1 if last_step(t) < a.target]
            if not pending:
                say('第一批全部到 %d 步 ✓' % a.target)
                break
            say('等待中: ' + ' '.join('%s=%d' % (t, last_step(t)) for t in pending))
            time.sleep(180)
        else:
            say('⚠ 等待超时 ⇒ 不再等，直接开始队列（可能与第一批重叠 ⇒ 会慢，但数值不受影响）')

        todo = [arm for arm in QUEUE if last_step(arm['tag']) < a.target]
        for arm in QUEUE:
            if arm not in todo:
                say('跳过 %s（已完成 step=%d）' % (arm['tag'], last_step(arm['tag'])))
        n = len(QUEUE)
        while todo:
            group, todo = todo[:a.jobs], todo[a.jobs:]
            procs = []
            for arm in group:
                p, log, t0, fh = run_arm(arm['tag'], arm['args'])
                procs.append((p, arm, log, t0, fh))
                say('[%d/%d] 启动 %s —— %s' % (QUEUE.index(arm) + 1, n, arm['tag'], arm['why']))
                say('        args: %s' % arm['args'])
            for p, arm, log, t0, fh in procs:
                rc = p.wait()
                fh.close()
                say('[%d/%d] 结束 %s  exit=%s  wall=%.1fmin  step=%d  log=%s'
                    % (QUEUE.index(arm) + 1, n, arm['tag'], rc,
                       (time.time() - t0) / 60.0, last_step(arm['tag']), os.path.basename(log)))
        say('队列全部结束')
    finally:
        try:
            os.remove(LOCK)
        except OSError:
            pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
