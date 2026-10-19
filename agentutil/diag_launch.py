#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一条命令顺序跑完四段诊断实验（S1 → S2 → S3 → E2），每段 70k 步后停。

特性：
  * **可续跑**（默认）：`output/diag/summary.csv` 里已有该段记录 ⇒ 自动跳过（`--no-resume` 强制重跑）
  * **心跳**：每 20 s 打印 `已跑步数 / 70k / 步速 / 预计剩余`
  * 单段失败**不阻塞**后续段（记录 rc，继续）
  * 每段结束自动跑 `agentutil/diag_report.py` 打分，追加 `output/diag/summary.csv`
  * 全部结束打印汇总表
  * 状态文件 `output/diag/status.json`

用法:
    python agentutil/diag_launch.py                     # 四段顺序（可续跑）
    python agentutil/diag_launch.py --stages S1 S2      # 只跑指定段
    python agentutil/diag_launch.py --no-resume         # 忽略已有记录，全部重跑
    python agentutil/diag_launch.py --smoke             # 每段 400 步自检（产物进 *-SMOKE）
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
COMMON = ['--hsrl', '--max_episode_length', '1440', '--seed', '0']   # 结构由 --arch/--leaky/--outinit 决定

STAGES = {
    # 实验 1-S1：固定风速 u=10 + 固定电价 580；无波动成本（λ≡0）；BESS 不参与；界罚 400
    'S1': dict(env={'HSRL_WIND_CSV': 'FFC_speed_1.csv'},
               argv=COMMON + ['--output', 'output/DIAG-S1', '--diag_u', '10',
                              '--diag_pr', '580', '--diag_freeze_g', '1',
                              '--diag_lambda_fixed', '0', '--pen_h', '400'],
               task='maxpower'),
    # 实验 1-S2：真实风速（clean 池）+ 固定电价 580
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
    #   ⚠ 必须用加噪池：clean 池下 λ=20 的最优是 s≡1（权衡退化），加噪池下 DP 最优 s* 是内点。
    'E2': dict(env={'HSRL_WIND_CSV': 'FFC_speed_noise_1.csv'},
               argv=COMMON + ['--output', 'output/DIAG-E2', '--diag_scalar', '1',
                              '--diag_freeze_g', '1', '--diag_lambda_fixed', '2.0'],
               task='scalar'),
}

ap = argparse.ArgumentParser()
ap.add_argument('--stages', nargs='+', default=['S1', 'S2', 'S3', 'E2'])
ap.add_argument('--smoke', action='store_true', help='每段只跑 400 步')
ap.add_argument('--no-resume', action='store_true', help='忽略 summary.csv 里已有记录')
ap.add_argument('--step-cap', type=int, default=STEP)
ap.add_argument('--extra', nargs='+', default=[], help='额外透传给 aifflor_main 的参数（原样追加）')
ap.add_argument('--prod-arch', action='store_true', help='同 --arch v1（历史结构）')
ap.add_argument('--arch', default='v2', choices=['v1', 'v1af', 'v2'], help='critic 结构（生产默认 v2）')
ap.add_argument('--leaky', action='store_true', help='critic 用 LeakyReLU(0.01)')
ap.add_argument('--outinit', default='3e3', choices=['3e3', 'fanin'], help='输出层权重初始化')
ap.add_argument('--bias', action='store_true', help='启用 critic 输出层 bias 初始化（生产默认关）')
ap.add_argument('--no-bias', action='store_true', help='（兼容旧用法；现已是默认）')
a = ap.parse_args()
steps = 400 if a.smoke else a.step_cap
summary = os.path.join('output', 'diag', 'summary.csv')
os.makedirs(os.path.join('output', 'diag', 'logs'), exist_ok=True)
status_path = os.path.join('output', 'diag', 'status.json')


def done_tags():
    if a.no_resume or not os.path.exists(summary):
        return set()
    try:
        import csv
        with open(summary, encoding='utf-8') as fh:
            return {r.get('task') and r.get('run') for r in csv.DictReader(fh)}
    except Exception:
        return set()


def steps_from_log(log_path):
    """从训练日志末尾解析 `step:NNNN`（每步一行标记）⇒ 当前进度，O(1)。"""
    import re
    try:
        with open(log_path, 'rb') as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 16384))
            txt = fh.read().decode('utf-8', 'replace')
        m = re.findall(r'step:(\d+)', txt)
        return (int(m[-1]) + 1) if m else 0
    except Exception:
        return 0


def find_run(out_dir):
    import glob
    cands = [d for d in glob.glob(os.path.join(out_dir, 'AIFFLOR-run*'))
             if os.path.exists(os.path.join(d, 'df_hsrl.csv'))]
    return max(cands, key=lambda d: os.path.getmtime(os.path.join(d, 'df_hsrl.csv'))) if cands else out_dir


already = done_tags()
status = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'steps': steps, 'stages': {}}


def save_status():
    with open(status_path, 'w', encoding='utf-8') as f:
        json.dump(status, f, ensure_ascii=False, indent=2)


save_status()
t_all = time.time()
for name in a.stages:
    cfg = STAGES[name]
    arch = 'v1' if a.prod_arch else a.arch
    suf = {'v1': '', 'v1af': '-af', 'v2': '-v2'}[arch]
    suf += '-lk' if a.leaky else ''
    suf += '-fi' if a.outinit == 'fanin' else ''
    suf += '' if a.bias else '-nb'
    tag = name + suf + ('-smoke' if a.smoke else '')
    log = os.path.join('output', 'diag', 'logs', '%s.log' % tag)
    run_dir = cfg['argv'][cfg['argv'].index('--output') + 1]
    if a.smoke:
        run_dir += '-SMOKE'
    if tag in already:
        print('[skip] %s 已有记录（%s）' % (tag, summary), flush=True)
        status['stages'][name] = dict(state='skipped')
        save_status()
        continue
    env = dict(os.environ)
    env.update(cfg['env'])
    env['PYTHONIOENCODING'] = 'utf-8'
    out_arg = run_dir if a.smoke else cfg['argv'][cfg['argv'].index('--output') + 1]
    _argv = list(cfg['argv'])
    _argv += ['--critic_arch', arch]
    _argv += ['--critic_act', 'leaky' if a.leaky else 'relu']
    _argv += ['--critic_out_init', a.outinit]
    _argv += list(a.extra)
    cmd = [PY, os.path.join('aifflor', 'aifflor_main.py')] + _argv
    cmd[cmd.index('--output') + 1] = out_arg
    cmd += ['--critic_bias_init', 'estimate' if a.bias else 'none']
    cmd += ['--max_steps', str(steps)]
    # ⚠ main 会**追加**写入已有的 df_hsrl.csv（不自动换 run 号）⇒ 每段开跑前必须清空旧产物，
    #   否则续跑/重跑会把两段数据混在同一个 csv 里（心跳行数>100% 就是这个症状）。
    import shutil
    if os.path.exists(out_arg):
        shutil.rmtree(out_arg, ignore_errors=True)
    print('\n[launch] %s -> %s' % (name, ' '.join(cmd)), flush=True)
    status['stages'][name] = dict(state='running', t0=time.strftime('%Y-%m-%d %H:%M:%S'))
    save_status()
    t0 = time.time()
    rc = -1
    with open(log, 'w', encoding='utf-8', errors='replace') as fh:
        try:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
            last = -1
            while proc.poll() is None:
                time.sleep(20)
                n = steps_from_log(log)
                if n != last:
                    el = time.time() - t0
                    rate = n / max(el, 1e-9)
                    eta = (steps - n) / rate / 60.0 if rate > 1e-9 else float('nan')
                    print('  [%s] %6d/%d (%5.1f%%)  %5.1f 步/秒  已用 %4.0f min  剩余约 %4.0f min'
                          % (tag, n, steps, 100.0 * n / steps, rate, el / 60.0, eta), flush=True)
                    last = n
            rc = proc.returncode
        except Exception as ex:
            print('  [%s] 启动异常: %r' % (tag, ex), flush=True)
    dur = time.time() - t0
    status['stages'][name].update(state='done' if rc == 0 else 'failed', rc=int(rc),
                                  t1=time.strftime('%Y-%m-%d %H:%M:%S'), sec=round(dur, 1))
    save_status()
    print('  [%s] rc=%d 用时 %.0f min' % (tag, rc, dur / 60.0), flush=True)
    rep = [PY, os.path.join('agentutil', 'diag_report.py'), '--run', find_run(out_arg),
           '--task', cfg['task'], '--tag', tag]
    r = subprocess.call(rep, env=env)
    if r != 0:
        print('  [%s] 打分失败 rc=%d（训练产物仍在 %s）' % (tag, r, out_arg), flush=True)

status['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
status['total_min'] = round((time.time() - t_all) / 60.0, 1)
save_status()
print('\n' + '=' * 96)
print('全部结束（%.0f min）。汇总：' % status['total_min'])
if os.path.exists(summary):
    import pandas as pd
    df = pd.read_csv(summary)
    cols = [c for c in ['run', 'task', 'eta', 'pattern_err', 'viol_pct', 'a_std_step', 'r2_dua',
                        'regret_pct', 's_mean', 's_star_mean', 'J_policy', 'J_star', 'n'] if c in df.columns]
    print(df[cols].to_string(index=False, float_format=lambda v: '%.3f' % v))
else:
    print('（还没有 summary.csv —— 每段都失败了？看 output/diag/logs/*.log）')
print('=' * 96)
