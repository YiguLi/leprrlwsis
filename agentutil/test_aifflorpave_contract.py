# -*- coding: utf-8 -*-
"""`aifflorpave` arm 的不变量测试（4 组）。

① `CriticPAVE(act='relu')` 与 `model.Critic` **同 seed 下参数/输出逐位相同**
   ⇒ "只换激活"是单因子，基线结构没被动过。
② `SP_TD3_PAVE` 在 `critic_act=relu, pave_w_* =0` 时与 `SP_TD3` **逐位相同**
   ⇒ 关掉 PAVE 时本 arm ≡ aifflor 基线。
③ 几何步（`_pave_step`）的梯度**只落在 profit 双 critic**：actor 与 cost 双 critic
   的 grad 恒为 None；且三项损失有限；ReLU 下 Curv ≡ 0（分段线性 ⇒ ∇²_aaQ≡0）。
④ `agentutil/port_aifflorpave.py` 的 7 处替换**各命中 1 次**，且反向还原**逐位等于**
   `aifflor/aifflor_main.py`（"只差这几处"的机器可读证明）。

用法::

    python agentutil/test_aifflorpave_contract.py     # exit 0 = PASS
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402

from model import Critic                                        # noqa: E402
from aifflor.aifflor_ddpg import SP_TD3                         # noqa: E402
from aifflorpave.aifflorpave_pave import CriticPAVE, pave_terms  # noqa: E402
from aifflorpave.aifflorpave_ddpg import SP_TD3_PAVE            # noqa: E402

NS, NA = 4, 6
FAIL = []


def check(cond, msg):
    print(('  OK   ' if cond else '  FAIL ') + msg)
    if not cond:
        FAIL.append(msg)


class _Args(object):
    """SP_TD3 构造所需的最小 args stub（缺失字段一律 0/False）。"""

    def __init__(self, **kw):
        self.cuda = False
        self.seed = 101
        self.hidden1, self.hidden2, self.init_w = 16, 100, 3e-3
        self.rate, self.prate = 0.01, 1e-4
        self.bsize, self.rmsize = 64, 4000
        self.discount, self.tau = 0.95, 0.001
        self.nb_turbine = 5
        self.PER_alpha, self.PER_beta, self.PER_epsilon = 0.6, 0.4, 0.01
        self.beta_increment_per_sampling = 3e-5
        self.cost_single_limit = 1000.0
        self.ou_theta, self.ou_sigma, self.ou_mu = 0.15, 0.2, 0.0
        self.lam, self.lr_lam = 3.0, 0.1
        # ddpg.__init__ 还会读这两个（missing ⇒ __getattr__ 给 0.0 ⇒ 除零）
        self.epsilon = 4000
        self.max_episode_length = 1440
        self.output = None          # 关闭 pave_stats.csv 落盘（测试不留产物）
        for k, v in kw.items():
            setattr(self, k, v)

    def __getattr__(self, k):
        return 0.0


def sd_equal(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    if list(sa.keys()) != list(sb.keys()):
        return False
    return all(torch.equal(sa[k], sb[k]) for k in sa)


# ------------------------------------------------------------------ ①
print('=' * 78)
print('① CriticPAVE(relu) ≡ model.Critic（同 seed, 参数与输出逐位）')
torch.manual_seed(7)
c0 = Critic(NS, NA, hidden1=16, hidden2=100)
torch.manual_seed(7)
c1 = CriticPAVE(NS, NA, hidden1=16, hidden2=100, act='relu')
check(sd_equal(c0, c1), '参数逐位相同')
xs = [torch.randn(8, NS), torch.randn(8, NA)]
check(torch.equal(c0(xs), c1(xs)), '前向输出逐位相同')
c2 = CriticPAVE(NS, NA, hidden1=16, hidden2=100, act='silu')
check(not torch.equal(c1(xs), c2(xs)), 'silu 版输出与 relu 不同（激活确实生效）')

# ------------------------------------------------------------------ ②
print('=' * 78)
print('② SP_TD3_PAVE(critic_act=relu, pave_w_*=0) ≡ SP_TD3（逐位）')
a_base = _Args()
a_pave = _Args()
base = SP_TD3(NS, NA, a_base, nstep=1, target_agg='min')
pave = SP_TD3_PAVE(NS, NA, a_pave, critic_act='relu', pave_w_mpr=0.0, pave_w_vfc=0.0,
                   pave_w_curv=0.0, nstep=1, target_agg='min')
check(not pave.pave_enabled, 'pave_enabled=False')
check(sd_equal(base.actor, pave.actor), 'actor 参数逐位相同')
for nm in ('critic_p', 'critic_p2', 'critic_c', 'critic_c2'):
    check(sd_equal(getattr(base, nm), getattr(pave, nm)), '%s 参数逐位相同' % nm)
s = np.zeros(NS, dtype=np.float32)
check(np.array_equal(base.actor(torch.tensor([s])).detach().numpy(),
                     pave.actor(torch.tensor([s])).detach().numpy()), 'actor 输出逐位相同')
check(type(pave.critic_p).__name__ == 'Critic', 'pave off 时 critic 仍是 model.Critic')

# ------------------------------------------------------------------ ③
print('=' * 78)
print('③ 几何步：梯度只落 profit 双 critic；三项有限；ReLU 下 Curv≡0')
torch.manual_seed(3)
pv = SP_TD3_PAVE(NS, NA, _Args(), critic_act='silu', pave_w_mpr=1.0, pave_w_vfc=1.0,
                 pave_w_curv=1.0, nstep=1, target_agg='min', pave_log_every=0)
check(pv.pave_enabled and pv.pave_curv_on, 'silu 下 Curv 生效标志')
check(type(pv.critic_p).__name__ == 'CriticPAVE', 'profit critic 已换成 CriticPAVE')
check(type(pv.critic_c).__name__ == 'Critic', 'cost critic 未被动过（仍是 model.Critic）')
rng = np.random.RandomState(0)
for _ in range(200):
    pv.observe_jump(rng.randn(NS).astype(np.float32), rng.randn(NA).astype(np.float32),
                    float(rng.randn()), float(abs(rng.randn())),
                    rng.randn(NS).astype(np.float32), 0.0)
for p in pv.actor.parameters():
    p.grad = None
for m in (pv.critic_c, pv.critic_c2):
    for p in m.parameters():
        p.grad = None
pv._pave_step()
check(all(p.grad is None for p in pv.actor.parameters()), 'actor 梯度仍为 None')
check(all(p.grad is None for m in (pv.critic_c, pv.critic_c2) for p in m.parameters()),
      'cost 双 critic 梯度仍为 None')
check(any(p.grad is not None for p in pv.critic_p.parameters()), 'profit critic 收到梯度')
e = pv._pave_ema
check(all(np.isfinite(v) for v in e), '三项+loss 有限: ' + str(tuple(round(v, 6) for v in e)))
check(e[0] >= 0 and e[1] >= 0 and e[2] >= 0, '三项均非负')
# ReLU 下 Curv ≡ 0
s_b = torch.randn(16, NS)
a_b = torch.randn(16, NA)
s2_b = torch.randn(16, NS)
m_relu = CriticPAVE(NS, NA, hidden1=16, hidden2=100, act='relu')
_m, _v, cur_relu, _g = pave_terms([m_relu], s_b, a_b, s2_b, do_curv=True)
check(float(cur_relu) == 0.0, 'ReLU 下 Curv 恒为 0（实测 %.3g）' % float(cur_relu))
m_silu = CriticPAVE(NS, NA, hidden1=16, hidden2=100, act='silu')
_m2, _v2, cur_silu, _g2 = pave_terms([m_silu], s_b, a_b, s2_b, do_curv=True)
check(float(cur_silu) > 0, 'SiLU 下 Curv > 0（实测 %.3g）' % float(cur_silu))
check(float(_m) >= 0 and float(_v) >= 0, 'MPR/VFC 非负')

# ------------------------------------------------------------------ ④
print('=' * 78)
print('④ port 审计：7 处替换各命中 1 次 + 反向还原逐位等于 aifflor_main.py')
import port_aifflorpave as port                                    # noqa: E402
src = open(port.SRC, encoding='utf-8').read()
dst, hits = port.build(src)
check([n for _t, n in hits] == [1] * 7, '7 处替换各命中 1 次: %s' % [n for _t, n in hits])
check(port.revert(dst) == src, '反向还原逐位等于源文件')
check(os.path.isfile(port.DST), '生成文件存在: %s' % port.DST)
check(port.revert(open(port.DST, encoding='utf-8').read()) == src, '磁盘上的 main 也可逆')

print('=' * 78)
print('TEST PASSED' if not FAIL else '**TEST FAILED**: %s' % FAIL)
sys.exit(0 if not FAIL else 1)
