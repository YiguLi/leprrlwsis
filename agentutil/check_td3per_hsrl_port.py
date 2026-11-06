# -*- coding: utf-8 -*-
"""静态自检：td3per 的 --hsrl 移植（不 import env / 不启 MATLAB / 不跑仿真）。

检查三件事：
  1. `td3per_main.py` 里所有 `args.<name>` / `getattr(args, '<name>')` 读取都能在
     argparse（含 hsrl_warmup / td3per_branch 挂上去的）里找到；
  2. `hsrl.<name>` 引用的属性在 `td3per_hsrl` 模块里真的存在；
  3. `train_hsrl()` / `run_branch_block_hsrl()` / `SP_TD3` 的新 kwarg 都在。
"""
import ast
import importlib
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)

MAIN = os.path.join(ROOT, 'td3per', 'td3per_main.py')
BRANCH = os.path.join(ROOT, 'td3per', 'td3per_branch.py')
WARM = os.path.join(ROOT, 'hsrl_warmup.py')

fail = 0


def defined_args(*paths):
    names = set()
    pat = re.compile(r"add_argument\(\s*'--([A-Za-z0-9_\-]+)'")
    for p in paths:
        for m in pat.finditer(open(p, encoding='utf-8').read()):
            names.add(m.group(1).replace('-', '_'))
    return names


def used_args(path):
    src = open(path, encoding='utf-8').read()
    used = set(re.findall(r'args\.([A-Za-z_][A-Za-z0-9_]*)', src))
    used |= set(re.findall(r"getattr\(args,\s*'([A-Za-z0-9_]+)'", src))
    return used


cli = defined_args(MAIN, BRANCH, WARM)
# 训练/评测脚本里常见的非 CLI 派生属性（main 自己赋的）
derived = set(re.findall(r'^\s*args\.([A-Za-z_][A-Za-z0-9_]*)\s*=',
                         open(MAIN, encoding='utf-8').read(), re.M))
missing = sorted(used_args(MAIN) - cli - derived)
print('CLI 定义数 %d | main 读取数 %d | 未定义 %d'
      % (len(cli), len(used_args(MAIN)), len(missing)))
if missing:
    fail += 1
    print('  ✗ 未定义:', missing)
else:
    print('  ✓ argparse / getattr 读取全部有定义')

# ---- hsrl 模块属性 ----
hsrl = importlib.import_module('td3per.td3per_hsrl')
src = open(MAIN, encoding='utf-8').read()
attrs = sorted(set(re.findall(r'\bhsrl\.([A-Za-z_][A-Za-z0-9_]*)', src)))
bad = [a for a in attrs if not hasattr(hsrl, a)]
print('\nhsrl.<name> 引用 %d 个' % len(attrs))
if bad:
    fail += 1
    print('  ✗ 模块里不存在:', bad)
else:
    print('  ✓ 全部存在:', ', '.join(attrs))

# ---- 必需函数 ----
need = ['train_hsrl', 'run_branch_block_hsrl', 'goal']
main_ns = ast.parse(src)
defs = {n.name for n in ast.walk(main_ns) if isinstance(n, ast.FunctionDef)}
print('\nmain 定义 train_hsrl:', 'train_hsrl' in defs)
if 'train_hsrl' not in defs:
    fail += 1
bsrc = open(BRANCH, encoding='utf-8').read()
print('branch 定义 run_branch_block_hsrl:', 'def run_branch_block_hsrl(' in bsrc)
print('branch 定义 goal:', 'def goal(' in bsrc)
if 'def run_branch_block_hsrl(' not in bsrc or 'def goal(' not in bsrc:
    fail += 1

# ---- SP_TD3 新 kwarg ----
from td3per.td3per_ddpg import SP_TD3
import inspect
params = set(inspect.signature(SP_TD3.__init__).parameters)
for k in ('wt_noise_mode', 'wt_noise_min', 'wt_noise_max', 'cost_clip'):
    ok = k in params
    print('SP_TD3 接受 %-14s %s' % (k, '✓' if ok else '✗'))
    if not ok:
        fail += 1

# ---- argparse：可构建 + --hsrl 默认值 ----
# 只 exec `parser.add_argument(...)` 那一段（不含 __main__ 的运行逻辑），
# 因此不 import env、不启 MATLAB、不跑仿真。
import argparse as _argparse
import hsrl_cost as _hc
import hsrl_soctarget as _hst
import hsrl_warmup as _hw

src_lines = open(MAIN, encoding='utf-8').read().splitlines(True)
i0 = next(i for i, l in enumerate(src_lines)
          if l.startswith('    parser = argparse.ArgumentParser('))
i1 = next(i for i, l in enumerate(src_lines)
          if l.startswith('    args = parser.parse_args()'))
snippet = 'if True:\n' + ''.join(src_lines[i0:i1])
ns = dict(argparse=_argparse, np=__import__('numpy'), hc=_hc, hst=_hst,
          hwarm=_hw, hsrl=hsrl, tbr=__import__('td3per.td3per_branch',
                                              fromlist=['x']),
          SOC_TOL=__import__('td3per.td3per_myfunc', fromlist=['x']).SOC_TOL,
          VIOL_DERATE_COEF=__import__('td3per.td3per_myfunc',
                                      fromlist=['x']).VIOL_DERATE_COEF)
exec(compile(snippet, '<argparse-block>', 'exec'), ns)
p = ns['parser']
a = p.parse_args(['--hsrl', '--env', 'X', '--branch_every', '5',
                  '--max_steps', '100'])
print('\nargparse 构建/解析: ✓')
want = {'hsrl': True, 'env': 'X', 'branch_every': 5,
        'pen_h': 100.0, 'noise_zero_step': 115200,
        'wt_noise_max': 0.05, 'cost_clip': 0, 'c1_bound_coef': 0.0,
        'hsrl_lam1_min': hsrl.HSRL_LAM1_PHYS_MIN,
        'hsrl_lam1_max': hsrl.HSRL_LAM1_PHYS_MAX,
        'hsrl_d_block_l': hsrl.HSRL_D_BLOCK_L,
        'hsrl_sig_noise_max': 0.5, 'hsrl_sig_noise_min': 0.025,
        'hsrl_test_wind': 'fixed'}
for k, v in want.items():
    got = getattr(a, k, '<missing>')
    ok = (got == v)
    print('  %-20s = %-10r %s' % (k, got, '✓' if ok else '✗ (want %r)' % (v,)))
    if not ok:
        fail += 1

# ---- 运行期装配（exec `if args.hsrl:` 那一段，不碰 env / MATLAB）----
#   该段只改写 args + DIAG，因此可以安全地单独执行。
src_all = ''.join(src_lines)
i2 = src_all.index('    if args.hsrl:\n        assert not args.nolow')
i3 = src_all.index('    # =================================================================\n'
                   '    # warmup7200', i2)
# 模拟真实 main 的第一步：解析 warmup（默认 7200 步）
a = _hw.resolve_warmup(a, verbose=False)
wiring = 'if True:\n' + src_all[i2:i3]
wired = dict(args=a, DIAG=dict(lam_fixed=-1.0, pen_h=100.0, center=0),
             sys=__import__('sys'), hsrl=hsrl,
             hc=_hc, np=ns['np'], prCyan=lambda *x, **k: None,
             prYellow=lambda *x, **k: None)
# 模拟“未显式指定 --warmup_wt_sched”
wired['sys'] = type('S', (), {'argv': ['x', '--hsrl']})()
exec(compile(wiring, '<hsrl-wiring>', 'exec'), wired)
wa = wired['args']
checks = [('wt_noise_mode=wide', wa.wt_noise_mode, 'wide'),
          ('epsilon=21600', wa.epsilon, 21600),
          ('cost_clip=1', wa.cost_clip, 1),
          ('warmup_wt_sched=unified', wa.warmup_wt_sched, 'unified'),
          ('DIAG.pen_h=100', wired['DIAG']['pen_h'], 100.0),
          ('DIAG.center=0', wired['DIAG']['center'], 0),
          ('DIAG.lam_fixed=-1', wired['DIAG']['lam_fixed'], -1.0)]
print('\n运行期装配（--hsrl 覆盖 aifflor 值）:')
for name, got, want in checks:
    ok = (got == want)
    print('  %-24s = %-10r %s' % (name, got, '✓' if ok else '✗ (want %r)' % (want,)))
    if not ok:
        fail += 1

# ---- 未定义全局名（粗粒度 pyflakes）----
#   抓"从 aifflor 搬过来时漏了某个 import"这类错误（例: `_time_mod`）。
import builtins as _bi
import util as _util_mod          # `from util import *` 的源（prCyan/round_list/...）
for path in (MAIN, BRANCH,
             os.path.join(ROOT, 'td3per', 'td3per_hsrl.py'),
             os.path.join(ROOT, 'td3per', 'td3per_ddpg.py'),
             os.path.join(ROOT, 'env.py')):
    tree = ast.parse(open(path, encoding='utf-8').read())
    bound = set(dir(_bi)) | set(dir(_util_mod)) | {'__file__', '__name__', '__doc__'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for al in node.names:
                bound.add((al.asname or al.name).split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            for al in node.names:
                bound.add(al.asname or al.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                bound.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    undefined = sorted({n.id for n in ast.walk(tree)
                        if isinstance(n, ast.Name)
                        and isinstance(n.ctx, ast.Load)
                        and n.id not in bound})
    print('未定义名 %-28s %s' % (os.path.basename(path),
                              '✓' if not undefined else '✗ ' + str(undefined)))
    if undefined:
        fail += 1

print('\nRESULT', 'PASS' if fail == 0 else 'FAIL(%d)' % fail)
sys.exit(0 if fail == 0 else 1)
