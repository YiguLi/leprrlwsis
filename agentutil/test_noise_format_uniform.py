# -*- coding: utf-8 -*-
"""核对 TD3 全族的**噪声格式统一性**（应全部为 td3per 格式）。

判据（逐个 ddpg 文件）：
  1. 形参含 `wt_noise=0.05, sig_noise_max=0.2, sig_noise_min=0.01` 与 `wt_noise_floor=0.01`
  2. 有 `self.explore_amp` / `self.explore_floor` 的赋值
  3. amp 公式为 `self.explore_amp * eps + self.explore_floor * (1.0 - eps)`
  4. **不再残留**旧的 `noise_halfwidth` / `noise_floor` 噪声路径
"""
import glob
import os
import re

FAMILY = ('td3per', 'td3perrb', 'td3prle', 'td3prrbb', 'td3prrlrb',
          'td3leprrlrb', 'singleaif', 'aifflor', 'yawflor')

print('%-40s %-6s %-6s %-6s %-6s %-6s' %
      ('ddpg 文件', '形参', 'amp_e', 'amp式', '旧噪声', '判定'))
bad = []
for pkg in FAMILY:
    f = '%s/%s_ddpg.py' % (pkg, pkg)
    if not os.path.isfile(f):
        print('%-40s (缺文件)' % f)
        continue
    s = open(f, encoding='utf-8').read()
    a = 'wt_noise=0.05, sig_noise_max=0.2, sig_noise_min=0.01' in s
    b = 'self.explore_amp' in s and 'self.explore_floor' in s
    c = 'self.explore_amp * eps + self.explore_floor * (1.0 - eps)' in s
    d = ('noise_halfwidth' in s) or ('self.noise_floor' in s)
    ok = a and b and c and not d
    print('%-40s %-6s %-6s %-6s %-6s %s'
          % (f, 'Y' if a else 'N', 'Y' if b else 'N', 'Y' if c else 'N',
             'Y' if d else '-', 'OK' if ok else '**不一致**'))
    if not ok:
        bad.append(f)

print()
# 逐维幅度表核对（对 H 层 6 维）
print('=' * 96)
print('逐维幅度表（H 层 nb_actions=6, nb_turbine=5）')
print('=' * 96)
for pkg in FAMILY:
    f = '%s/%s_ddpg.py' % (pkg, pkg)
    if not os.path.isfile(f):
        continue
    s = open(f, encoding='utf-8').read()
    m = re.search(r'amp_e = np\.array\(\[self\.wt_noise\] \* self\.nb_turbine \+ '
                  r'\[self\.sig_noise_max\],', s)
    print('  %-16s %s' % (pkg, '含 nt+1 分支（H 层逐维）' if m else '**无该分支**'))
    if not m:
        bad.append(f + ' (缺 nt+1 分支)')

print()
print('%s' % ('TD3 全族噪声格式统一 ✅' if not bad else '**不一致**: %s' % bad))
