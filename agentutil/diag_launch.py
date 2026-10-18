#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四段诊断实验的顺序启动器：S1 → S2 → S3 → E2，每段 70k 步后停。

* 每段独立子进程，stdout/stderr 落 `output/diag/logs/<stage>.log`；
* 任一段异常退出 **不阻塞** 后续段（记录 returncode，继续下一段）；
* 每段结束后自动跑 `agentutil/diag_report.py` 记分（追加 output/diag/summary.csv）；
* 状态写 `output/diag/status.json`（可随时查看进度）。

用法（建议后台跑）:
    python agentutil/diag_launch.py            # 四段顺序执行
    python agentutil/diag_launch.py --stages S1 S2    # 只跑指定段
    python agentutil/diag_launch.py --smoke    # 每段只跑 400 步（自检用）
"""
import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

PY = sys.executable
STEP = 70000
COMMON = ['--hsrl', '--max_episode_length', '1440', '--seed', '0']

STAGES = {
    # 实验 1-S1：固定风速 u=10 + 固定电价 580；无波动成本（λ≡0）；BESS 不参与；界罚 400
    'S1': dict(env={'HSRL_WIND_CSV': 'FFC_speed_1.csv'},
               argv=COMMON + ['--output', 'output/DIAG-S1', '--diag_u', '10',
                              '--diag_pr', '580', '--diag_freeze_g', '1',
                              '--diag_lambda_fixed', '0', '--pen_h', '400'],
               task='maxpower'),
    # 实验 1-S2：真实风速（clean 池 day 序列）+ 固定电价 580
    'S2': dict(env={'HSRL_WIND_CSV': 'FFC_speed_1.csv'},
               argv=COMMON + ['--output', 'output/DIAG-S2', '--diag_pr', '580',
                              '--diag_freeze_g', '1', '--diag_lambda_fixed', '0',
                              '--pen_h', '400'],
               task='maxpower'),
    # 实验 1-S3：真实风速 + 变电价（生产日程表 300/580/980）
    'S3': dict(env={'HSRL_WIND_CSV': 'FFC_speed_1.csv'},
               argv=COMMON + ['--output', 'output/DIAG-S3', '--diag_freeze_g', '1',
                              '--diag_lambda_fixed', '0', '--pen_h', '400'],
               task='maxpower'),
    # 实验 2：actor 只输出标量 s，a=0.33·s；λ 固定 2.0（actor）= 20（物理）；BESS 不参与
    #   ⚠ 必须用**加噪池**：clean 池下 λ=20 的最优解是 s≡1（没有可平抑的波动，权衡退化），
    #     加噪池下 DP 最优 s* 是内点（p50 0.48、p5 0.14）⇒ 才是真正的"收益 vs 波动"权衡。
    'E2': dict(env={'HSRL_WIND_CSV': 'FFC_speed_noise_1.csv'},
               argv=COMMON + ['--output', 'output/DIAG-E2', '--diag_scalar', '1',
                              '--diag_freeze_g', '1', '--diag_lambda_fixed', '2.0'],
               task='scalar'),
}

ap = argparse.ArgumentParser()
ap.add_argument('--stages', nargs='+', default=['S1', 'S2', 'S3', 'E2'])
ap.add_argument('--smoke', action='store_true', help='每段只跑 400 步')
a = ap.parse_args()
steps = 400 if a.smoke else STEP

def _find_run(out_dir):
    """main 会在 `--output` 下自动建 `AIFFLOR-runN`；返回真正含 df_hsrl.csv 的目录。"""
    import glob
    cands = [d for d in glob.glob(os.path.join(out_dir, 'AIFFLOR-run*'))
             if os.path.exists(os.path.join(d, 'df_hsrl.csv'))]
    if not cands:
        return out_dir
    return max(cands, key=lambda d: os.path.getmtime(os.path.join(d, 'df_hsrl.csv')))


os.makedirs(os.path.join('output', 'diag', 'logs'), exist_ok=True)
status_path = os.path.join('output', 'diag', 'status.json')
status = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'steps': steps, 'stages': {}}


def save_status():
    with open(status_path, 'w', encoding='utf-8') as f:
        json.dump(status, f, ensure_ascii=False, indent=2)


save_status()
for name in a.stages:
    cfg = STAGES[name]
    log = os.path.join('output', 'diag', 'logs', '%s.log' % name)
    env = dict(os.environ)
    env.update(cfg['env'])
    env['PYTHONIOENCODING'] = 'utf-8'
    cmd = [PY, os.path.join('aifflor', 'aifflor_main.py')] + cfg['argv'] + ['--max_steps', str(steps)]
    run_dir = cfg['argv'][cfg['argv'].index('--output') + 1]
    if a.smoke:                                   # 自检不污染正式产物目录
        run_dir = run_dir + '-SMOKE'
        i = cmd.index('--output')
        cmd[i + 1] = run_dir
    status['stages'][name] = dict(state='running', t0=time.strftime('%Y-%m-%d %H:%M:%S'))
    save_status()
    print('[launch] %s -> %s' % (name, ' '.join(cmd)), flush=True)
    t0 = time.time()
    rc = -1
    try:
        with open(log, 'w', encoding='utf-8', errors='replace') as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    except Exception as ex:                       # 启动失败也不阻塞后续
        print('[launch] %s 启动异常: %r' % (name, ex), flush=True)
    dur = time.time() - t0
    status['stages'][name].update(state='done' if rc == 0 else 'failed', rc=int(rc),
                                  t1=time.strftime('%Y-%m-%d %H:%M:%S'), sec=round(dur, 1))
    save_status()
    print('[launch] %s rc=%d 用时 %.0fs' % (name, rc, dur), flush=True)
    # 打分（失败也试一下，日志可能已经写了 CSV）
    rep = [PY, os.path.join('agentutil', 'diag_report.py'), '--run', _find_run(run_dir),
           '--task', cfg['task'], '--tag', name]
    try:
        r = subprocess.call(rep, env=env)
        print('[launch] %s report rc=%d' % (name, r), flush=True)
    except Exception as ex:
        print('[launch] %s report 异常: %r' % (name, ex), flush=True)

status['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
save_status()
print('[launch] 全部段结束，见 output/diag/status.json / summary.csv', flush=True)
