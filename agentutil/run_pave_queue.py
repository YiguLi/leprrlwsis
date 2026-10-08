# -*- coding: utf-8 -*-
"""aifflorpave 的 B/C/D 变体**串行**队列（用户要求："把 BCD 各种测试全跑起来，别并行太多"）。

行为
----
1) 先等第一批 4 条（ABPAVE2-BASE / -SILU / -PAVE / -VFC）跑到 30000 步。判据 = 纯文件读
   `output/<tag>-run*/update_stats_H.csv` 的最后一列 step（不依赖进程/权限）；带 8h 安全上限。
2) 然后**每批 `--jobs` 条**（默认 2，每进程 3 线程 ⇒ 峰值 6/16 线程）跑队列：
     顺序按"诊断价值"排（首批 4 条在 ckpt 26000 的结果已定，见 PENDING_DECISIONS.md §5）：
     CURV4X   : silu + mpr .46 + vfc .22 + curv .0224   （Curv ×4：Curv 贡献随训练衰减到 ~0，先试剂量）
     CURVONLY : silu + curv .05（mpr/vfc 全 0）          （只做 μ 侧：能否避开成绩损失）
     NOVFC    : silu + mpr .46 + curv .0056（无 vfc）    （VFC 是不是"梯度塌到 0"的元凶）
     NOCURV   : silu + mpr .46 + vfc .22（无 curv）      （论文关键消融 ⇒ 预期复现塌陷，作反证锚点）
     EVERY4   : 全 PAVE + --pave_every 4                （算力匹配：分离"正则"与"多走一步"）
     MPR      : relu + mpr .46                          （单因子 MPR，不换激活）
     已删除 RELVFC（relu + mpr + vfc）：首批 VFC 臂（relu + vfc）已实测"梯度塌到 3e-5 + TD 29.2 +
     r1/块 −8096"的塌陷，RELVFC 只会复现同一现象，价值低于其它臂 ⇒ 先不跑（需要时再加回）。
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
    dict(tag='ABPAVE2-CURV4X',
         args='--critic_act silu --pave_w_mpr 0.46 --pave_w_vfc 0.22 --pave_w_curv 0.0224',
         why='Curv 权重 ×4 —— 实测 Curv 贡献随训练衰减到 ~0（53→0.089，权重固定）'),
    dict(tag='ABPAVE2-CURVONLY',
         args='--critic_act silu --pave_w_curv 0.05',
         why='只做 μ 侧（无 MPR/VFC）⇒ 能否在避开梯度塌陷的同时提高负定性率'),
    dict(tag='ABPAVE2-NOVFC',
         args='--critic_act silu --pave_w_mpr 0.46 --pave_w_curv 0.0056',
         why='去掉 VFC（"梯度塌到 0"的头号嫌疑），保留 MPR+Curv'),
    dict(tag='ABPAVE2-NOCURV',
         args='--critic_act silu --pave_w_mpr 0.46 --pave_w_vfc 0.22',
         why='论文关键消融：只压 M 不保 μ ⇒ 预期复现塌陷（反证锚点）'),
    dict(tag='ABPAVE2-EVERY4',
         args=('--critic_act silu --pave_w_mpr 0.46 --pave_w_vfc 0.22 '
               '--pave_w_curv 0.0056 --pave_every 4'),
         why='算力匹配变体（每 4 次 critic 更新加 1 次几何步）⇒ 分离"正则"与"多走一步"'),
    dict(tag='ABPAVE2-MPR',
         args='--critic_act relu --pave_w_mpr 0.46',
         why='单因子 MPR（不换激活）'),
]


def say(msg):
    line = '%s  %s' % (time.strftime('%m-%d %H:%M:%S'), msg)
    print(line, flush=True)
    with open(QLOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def last_step(tag):
    """该臂当前推进到第几步（读 update_stats_H.csv 最后一行的 step 列；无则 -1）。

    ⚠ 同一 tag 可能有**多个** run 目录（某条被静默终止后补跑就会这样）⇒ 取**步数最大**者，
    否则幂等判断会被那条残废的 run 永远钉在"未完成"，导致无限重跑。
    """
    best = -1
    for f in glob.glob(os.path.join(OUT, '%s-run*' % tag, 'update_stats_H.csv')):
        step = -1
        try:
            with open(f, 'r', encoding='utf-8', errors='replace') as fh:
                for line in fh:
                    parts = line.strip().split(',')
                    if len(parts) >= 2 and parts[1].isdigit():
                        step = int(parts[1])
        except OSError:
            continue
        best = max(best, step)
    return best


def run_arm(tag, args):
    """启动一条臂，返回 (Popen, log 路径, t0)。"""
    log = os.path.join(OUT, '_%s.log' % tag)
    # `-u`（无缓冲）：万一进程被静默终止，日志里也能留下 traceback（本次 NOCURV 就吃过这个亏）
    cmd = [PY, '-u', os.path.join(ROOT, 'aifflorpave', 'aifflorpave_main.py'),
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
                st = last_step(arm['tag'])
                note = ''
                if st < a.target:
                    note = ('  ⚠ **未跑满**（step=%d < %d）—— 可能被静默终止；'
                            '重跑本脚本会自动只补这一条（幂等）' % (st, a.target))
                say('[%d/%d] 结束 %s  exit=%s  观测到时距启动=%.1fmin  step=%d  log=%s%s'
                    % (QUEUE.index(arm) + 1, n, arm['tag'], rc,
                       (time.time() - t0) / 60.0, st, os.path.basename(log), note))
        say('队列全部结束')
    finally:
        try:
            os.remove(LOCK)
        except OSError:
            pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
