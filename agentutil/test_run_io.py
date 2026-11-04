#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`run_io` —— aifflor 同款"简化输出"开关的契约测试 + 12 个 main 的接线检查。

**不跑仿真**：只测 IO 行为与源码接线。
"""
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import run_io  # noqa: E402

N_OK = 0
N_BAD = 0


def ok(cond, msg):
    global N_OK, N_BAD
    if cond:
        N_OK += 1
        print('  ok   %s' % msg)
    else:
        N_BAD += 1
        print('  FAIL %s' % msg)
    return bool(cond)


MAINS = [
    'td3per/td3per_main.py', 'td3prrbb/td3prrbb_main.py',
    'td3prle/td3prle_main.py', 'td3prrlrb/td3prrlrb_main.py',
    'td3perrb/td3perrb_main.py', 'td3leprrlrb/td3leprrlrb_main.py',
    'td3perseg/td3perseg_main.py', 'prle/prle_main.py',
    'prrbb/prrbb_main.py', 'singleaif/singleaif_main.py',
    'yawflor/yawflor_main.py', 'aifflorpave/aifflorpave_main.py',
]


def section_a():
    print('\n[A] IO 语义（Python 层 + fd 层）')
    out = os.path.join(_HERE, '_tmp_run_io')
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)

    # --- log_file=0：什么都不做 ---
    old = sys.stdout
    run_io.setup(out, log_file=0, progress=1, every=10, total=100)
    same_stdout = sys.stdout is old
    run_io.tick(10)
    run_io.close()
    ok(same_stdout, 'log_file=0 不重定向 stdout')

    # --- log_file=1：Python 层 + fd 层都要重定向 ---
    #   这里把**真实 fd 1** 先指到一个临时"控制台"文件，这样就能在不干扰测试输出的
    #   情况下验证：MATLAB 那种直接写 fd 1 的输出会进 train.log，而进度行走控制台副本。
    con_path = os.path.join(_HERE, '_tmp_run_io_console.txt')
    con = open(con_path, 'w+b')
    sys.stdout.flush()          # 先把已缓冲的输出落到真实 fd1，避免测试日志错位
    real_fd = os.dup(1)
    os.dup2(con.fileno(), 1)
    try:
        old = sys.stdout
        run_io.setup(out, log_file=1, progress=1, every=10, total=100)
        redirected = sys.stdout is not old and hasattr(sys.stdout, 'name')
        has_fd = run_io._STATE.get('console_fd') is not None
        print('这句应该进 train.log')
        os.write(1, 'MATLAB-LIKE: Performing 5 forward simulations..\n'.encode('utf-8'))
        run_io.set_total(100)
        run_io.tick(10)
        run_io.tick(11)          # 非整步 ⇒ 不打印
        run_io.tick(20)
        run_io.close()
        restored = sys.stdout is old
    finally:
        os.dup2(real_fd, 1)
        os.close(real_fd)
        con.close()

    body = open(os.path.join(out, 'train.log'), encoding='utf-8').read()
    console = open(con_path, encoding='utf-8', errors='replace').read()
    shutil.rmtree(out, ignore_errors=True)
    os.remove(con_path)

    ok(redirected, 'log_file=1 把 sys.stdout 重定向到文件')
    ok(has_fd, 'log_file=1 同时做 fd 级重定向（os.dup2 → fd 1）')
    ok('这句应该进 train.log' in body, 'Python print 进 train.log')
    ok('MATLAB-LIKE: Performing 5 forward simulations..' in body,
       'MATLAB/WFSim 那类**直接写 fd 1** 的输出也进 train.log')
    lines = [x for x in console.splitlines() if '步/秒' in x]
    ok(len(lines) == 2, 'tick 每 every 步打印一次（10/20 各一行，实得 %d）' % len(lines))
    ok(bool(lines) and '10/100' in lines[0] and '20/100' in lines[1],
       '进度行含 step/total')
    ok(bool(lines) and ('[%s]' % os.path.basename(out)) in lines[0], '进度行带 run 名')
    ok(restored, 'close() 恢复 stdout')


def section_a2():
    """`--progress 0` 时不打进度行；train.log 里仍有 Python 输出。"""
    print('\n[A2] progress=0 静默')
    out = os.path.join(_HERE, '_tmp_run_io2')
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    con_path = os.path.join(_HERE, '_tmp_run_io2_console.txt')
    con = open(con_path, 'w+b')
    sys.stdout.flush()          # 先把已缓冲的输出落到真实 fd1，避免测试日志错位
    real_fd = os.dup(1)
    os.dup2(con.fileno(), 1)
    try:
        run_io.setup(out, log_file=1, progress=0, every=10, total=100)
        print('这句应该进 train.log')
        run_io.tick(10)
        run_io.tick(20)
        run_io.close()
    finally:
        os.dup2(real_fd, 1)
        os.close(real_fd)
        con.close()
    body = open(os.path.join(out, 'train.log'), encoding='utf-8').read()
    console = open(con_path, encoding='utf-8', errors='replace').read()
    shutil.rmtree(out, ignore_errors=True)
    os.remove(con_path)
    ok('这句应该进 train.log' in body, '详细输出写进了 train.log')
    ok('步/秒' not in console, 'progress=0 时不打进度行')


def section_b():
    print('\n[B] 12 个 main 的接线（静态）')
    for rel in MAINS:
        src = open(os.path.join(ROOT, rel), encoding='utf-8').read()
        flags = all(x in src for x in ("'--log_file'", "'--progress'",
                                       "'--progress_every'"))
        hooks = all(x in src for x in ('run_io.setup(', 'run_io.tick(step)',
                                       'run_io.set_total('))
        ok(flags and hooks, '%s: 三个 flag + setup/tick/set_total' % rel)
        ok("default=0" in src.split("'--log_file'")[1].split(')')[0],
           '%s: --log_file 默认 0（不改历史输出）' % rel)


def main():
    print('=' * 88)
    print('run_io（aifflor 同款简化输出）契约')
    print('=' * 88)
    section_a()
    section_a2()
    section_b()
    print('\n' + '=' * 88)
    print('结果：%d 项通过 / %d 项失败' % (N_OK, N_BAD))
    print('=' * 88)
    return 0 if N_BAD == 0 else 1


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
