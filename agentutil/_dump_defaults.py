# -*- coding: utf-8 -*-
"""抽取各 main 的关键超参默认值（用于核对 7f9dbbf 之后的真实基线）。"""
import glob

KEYS = ['--nstep', '--rmsize', '--bsize', '--utd_h', '--utd_l', '--epsilon',
        '--l_epsilon', '--wt_noise', '--sig_noise_max', '--sig_noise_min',
        '--wt_noise_floor']
SHORT = ['nstep', 'rmsize', 'bsize', 'utd_h', 'utd_l', 'eps', 'l_eps',
         'wt_n', 'sig_max', 'sig_min', 'wt_floor']


def defaults_of(path):
    s = open(path, encoding='utf-8').read()
    d = {}
    for k in KEYS:
        i = s.find("add_argument('%s'" % k)
        if i < 0:
            i = s.find('add_argument("%s"' % k)
        if i < 0:
            continue
        j = s.find('default=', i)
        if j < 0:
            continue
        k2 = s.find(',', j)
        d[k] = s[j + 8:k2].strip()
    return d


files = sorted(glob.glob('*/[a-z]*_main.py'))
print('%-32s %s' % ('file', ' '.join('%-8s' % x for x in SHORT)))
for f in files:
    d = defaults_of(f)
    print('%-32s %s' % (f, ' '.join('%-8s' % d.get(k, '-') for k in KEYS)))
