# -*- coding: utf-8 -*-
"""最强证据：把改前的 ddpg 从 git 取出来（临时副本），与改后跑**同一次测量**对比。

做法：`git show HEAD:<path>` 取出改前源码 → 写成 `agentutil/_before_<name>.py`，
      用 importlib 加载，然后与改后版本在同一 ε 下各测 6000 次动作，比较半宽。
      测完删除临时文件。
"""
import importlib.util
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

TARGET = 'td3leprrlrb/td3leprrlrb_ddpg.py'
CLASS = 'SP_TD3_LEPRRLRB'
TMP = os.path.join(_HERE, '_before_leprrlrb_ddpg.py')

CASES = [
    ('td3prle/td3prle_ddpg.py', 'td3prle.td3prle_ddpg', 'SP_TD3_PRLE',
     os.path.join(_HERE, '_before_prle_ddpg.py')),
    ('td3prrbb/td3prrbb_ddpg.py', 'td3prrbb.td3prrbb_ddpg', 'SP_TD3',
     os.path.join(_HERE, '_before_prrbb_ddpg.py')),
    ('td3leprrlrb/td3leprrlrb_ddpg.py', 'td3leprrlrb.td3leprrlrb_ddpg',
     'SP_TD3_LEPRRLRB', os.path.join(_HERE, '_before_leprrlrb_ddpg.py')),
]


class Args:
    pass


def mkargs():
    a = Args()
    a.cuda = False
    a.seed = 0
    a.hidden1 = 32
    a.hidden2 = 32
    a.init_w = 3e-3
    a.rate = 1e-3
    a.prate = 1e-3
    a.lr_lam = 1e-3
    a.discount = 0.98
    a.tau = 0.001
    a.bsize = 64
    a.rmsize = 200
    a.PER_alpha = 0.6
    a.PER_beta = 0.4
    a.PER_epsilon = 0.01
    a.beta_increment_per_sampling = 0.001
    a.epsilon = 50000
    a.lam = 0.1
    a.max_episode_length = 1440
    a.nb_turbine = 5
    a.cost_single_limit = 1000.0
    a.ou_mu = 0.0
    a.ou_sigma = 0.2
    a.ou_theta = 0.15
    a.policy_delay = 2
    a.policy_noise = 0.1
    a.noise_clip = 0.3
    a.cost_critic_mode = 'max'
    a.nstep = 1
    a.target_agg = 'min'
    return a


def measure(K, eps, n=6000, **kw):
    ag = K(nb_states=6, nb_actions=6, args=mkargs(), **kw)
    ag.is_training = True
    ag.epsilon = eps
    ag.depsilon = 0.0
    s = np.zeros(6, dtype=np.float32)
    acts = np.array([ag.select_action(s, decay_epsilon=False) for _ in range(n)])
    return (acts.max(axis=0) - acts.min(axis=0)) / 2.0


def load_before(target, tmp):
    src = subprocess.check_output(['git', 'show', 'HEAD:%s' % target])
    open(tmp, 'wb').write(src)
    spec = importlib.util.spec_from_file_location('_before_x', tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['_before_x'] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    allok = True
    for target, after_mod_name, cls, tmp in CASES:
        print('=' * 96)
        print('改前 vs 改后：%s' % target)
        print('=' * 96)
        before_mod = load_before(target, tmp)
        Before = getattr(before_mod, cls)
        after_mod = __import__(after_mod_name, fromlist=[cls])
        After = getattr(after_mod, cls)
        hb1 = ha1 = None
        for eps in (1.0, 0.5, 0.0):
            try:
                hb = measure(Before, eps)
                hb_s = np.array2string(hb, precision=5)
            except Exception as e:
                hb_s = 'n/a(%s)' % type(e).__name__
            ha = measure(After, eps, noise_floor=0.01)
            print('  ε=%-4.1f 改前 %-40s 改后 %s'
                  % (eps, hb_s, np.array2string(ha, precision=5)))
            if eps == 1.0:
                hb1, ha1 = hb, ha
            if eps == 0.0:
                ha0 = ha
        same = bool(np.allclose(hb1, ha1, atol=2e-3))
        pos = bool(np.all(ha0 > 0))
        print('  ε=1 改前≈改后（起始幅度不变）: %s' % ('OK' if same else '**不一致**'))
        print('  ε=0 改后 > 0（不再停止探索）: %s (最小 %.5f)'
              % ('OK' if pos else '**FAIL**', float(ha0.min())))
        allok = allok and same and pos
        try:
            os.remove(tmp)
        except OSError:
            pass
        print()
    print('%s' % ('三个方法全部通过 ✅' if allok else '**有失败** ❌'))
    return 0 if allok else 1


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
