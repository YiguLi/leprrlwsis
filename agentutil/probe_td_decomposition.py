# -*- coding: utf-8 -*-
r"""TD 误差的成分分解：critic 的**预测**与 **TD 目标**在同一批状态下各有多大的离散度？

动机（用户直觉）
----------------
"本步盈利只与本步动作相关，未来盈利也只与未来各步相关 ⇒ 没理由 TD 这么大"。
这个直觉逼出一个可测的问题：`td_p ≈ 1000 元` 到底是
  (A) **目标本身离散大**（值函数真的在状态间变化很大，而网络没拟合上）——可修；
  (B) **预测太平**（网络输出几乎是常数，只有 bias 在动）——可修且症状明确；
  (C) **两者都大但不相干**（自举闭环噪声）——难修。
用末检查点的 `critic_p`（真权重）+ 日志里的真实状态，直接算：
    pred = Q(s, a)                      ← critic 前向
    tgt  = r + γ^n·min(Q_p1',Q_p2')(s',π'(s'))   ← 复现 update_policy 的 target
    TD   = tgt − pred
并报告 pred / tgt 的 sd、相关系数、以及 r 与 tgt 的关系。

⚠ 只读；需要 torch；不启 MATLAB。
用法: python agentutil/probe_td_decomposition.py [output/TD3PERNOLOW-run57]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from model import Critic  # noqa: E402
from td3per.td3per_myfunc import Normalize_H  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else 'output/TD3PERNOLOW-run57'
NSTEP, GAMMA, BLOCK = 5, 0.95, 5
G_EFF = GAMMA ** NSTEP
CKPTS = sorted(int(p.split('_')[-1]) for p in os.listdir(RUN)
               if p.startswith('checkpoint_'))
CK = CKPTS[-1]
CDIR = os.path.join(RUN, 'checkpoint_%d' % CK)
print('=' * 96)
print('TD 分解: %s   checkpoint=%d' % (RUN, CK))
print('=' * 96)


def load_critic(path, nb_states, nb_actions):
    """Critic 的层序（见 model.py）：fc1(state→h1) → fc4(h1→h2) → fc2(h2+nb_a→80)
    → fc5(80→50) → fc3(50→1)。
    ⇒ `nb_actions` 要从 **fc2** 的输入维反推（`fc4.weight.shape[0]` 是 hidden2，不是动作维）。"""
    sd = torch.load(path, map_location='cpu', weights_only=True)
    h1 = sd['fc1.weight'].shape[0]
    h2 = sd['fc4.weight'].shape[0]
    nb_a = sd['fc2.weight'].shape[1] - h2
    m = Critic(nb_states, nb_a, hidden1=h1, hidden2=h2)
    m.load_state_dict(sd)          # 形状不符会直接抛错 ⇒ 这里的 nb_a 必是自洽解
    m.eval()
    return m


def load_actor(path):
    from model import Actor
    sd = torch.load(path, map_location='cpu', weights_only=True)
    m = Actor(sd['fc1.weight'].shape[1], sd['fc4.weight'].shape[0],
              hidden1=sd['fc1.weight'].shape[0], hidden2=sd['fc2.weight'].shape[0])
    m.load_state_dict(sd)
    m.eval()
    return m


def has(d, f):
    return os.path.exists(os.path.join(d, f))


act_f = 'actor_5.pkl' if has(CDIR, 'actor_5.pkl') else 'actor_6.pkl'
actor = load_actor(os.path.join(CDIR, act_f))
actor_t = load_actor(os.path.join(CDIR, act_f))       # 末检查点：target≈online（τ 滑动已久）
cp1 = load_critic(os.path.join(CDIR, 'critic_p_5.pkl'), 4, 5)
cp2 = load_critic(os.path.join(CDIR, 'critic_p2_5.pkl'), 4, 5)
nb_states = cp1.fc1.weight.shape[1]
nb_a = cp1.fc2.weight.shape[1] - cp1.fc4.weight.shape[0]
print('  actor 输入维(nb_states_H) = %d, 动作维(nb_actions_H) = %d' % (nb_states, nb_a))

# ---- 取真实状态：从日志末段抽块级决策点 ----
cols = ['step', 'observation', 'observation2', 'phase_minute', 'c1']
LOG = os.path.join(RUN, 'df_td3pernolow.csv')
acc = []
for ch in pd.read_csv(LOG, usecols=cols, chunksize=200000,
                      encoding='utf-8-sig', low_memory=False):
    acc.append(ch)
df = pd.concat(acc, ignore_index=True)
df = df[df['step'] > df['step'].max() - 40000]          # 末段（缓冲区覆盖的范围）
blk = df.iloc[::BLOCK].reset_index(drop=True)
if len(blk) > 3000:
    blk = blk.iloc[np.linspace(0, len(blk) - 1, 3000).astype(int)].reset_index(drop=True)
print('  块级样本 %d 个' % len(blk))


def parse(s):
    return np.array([[float(x) for x in t.strip().strip('[]').split()] for t in s.astype(str)])


o1 = parse(blk['observation'])
o2 = parse(blk['observation2'])
pm = blk['phase_minute'].to_numpy(int)


def norm(o, minute):
    out = []
    for i in range(len(o)):
        base = np.array([o[i, 0] / 1000, o[i, 1] / 10, o[i, 2] / 10, o[i, 3] / 10])
        if nb_states == 4:
            out.append(base)
        else:
            ang = 2 * np.pi * (int(minute[i]) % 1440) / 1440
            out.append(np.concatenate([base, [np.sin(ang), np.cos(ang)]]))
    return np.asarray(out, dtype=np.float32)


S = norm(o1, pm)
S2 = norm(o2, pm + 1)
with torch.no_grad():
    a = actor(torch.as_tensor(S)).numpy()                    # 策略动作（评估态）
    a_next = actor_t(torch.as_tensor(S2)).numpy()
    q1 = cp1([torch.as_tensor(S), torch.as_tensor(a)]).numpy().reshape(-1)
    q2 = cp2([torch.as_tensor(S), torch.as_tensor(a)]).numpy().reshape(-1)
    n1 = cp1([torch.as_tensor(S2), torch.as_tensor(a_next)]).numpy().reshape(-1)
    n2 = cp2([torch.as_tensor(S2), torch.as_tensor(a_next)]).numpy().reshape(-1)
pred = np.minimum(q1, q2)
bootstrap = np.minimum(n1, n2)
# 近似 target：r 用 critic 空间（s_p=1000 归一化）的块级 r1 —— 日志无 r1 列时用 0 占位
r_raw = np.full(len(blk), np.nan)

print('\n---- critic 预测 vs 自举项（cp 空间，即 s_p=1000 归一化后的 Q）----')
for nm, v in (('pred  = min(Q1,Q2)(s,a)', pred),
              ('boot  = min(Q1,Q2)(s2,a2)', bootstrap)):
    print('  %-28s mean %10.4f  sd %9.4f  min %9.4f  max %9.4f'
          % (nm, v.mean(), v.std(), v.min(), v.max()))
print('  两者相关 r = %.4f' % np.corrcoef(pred, bootstrap)[0, 1])
print('  差值 sd = %.4f   ⇒ n-step target 的"自举项"离散度 = %.4f'
      % ((bootstrap - pred).std(), bootstrap.std()))
g_eff_scale = G_EFF
print('\n  折算：target = r/1000 + %.4f·boot ⇒ 自举项对 target 的贡献 sd = %.4f'
      % (g_eff_scale, g_eff_scale * bootstrap.std()))
print('  critic 空间里的 td_p（日志口径）末段均值 = %.4f（≈991 元 / 1000）'
      % (pd.read_csv(os.path.join(RUN, 'update_stats_H.csv'))['td_p_mean'].tail(200).mean() / 1000))

print('\n---- 残差分解：预测在同一风况带内还有多大离散？----')
u = o1[:, 1]                                    # 稳态风速（第 2 维）
pr = o1[:, 0]
resid = bootstrap - pred
bins = [(0, 7), (7, 9), (9, 11), (11, 13), (13, 16), (16, 30)]
print('  %-12s %6s %12s %12s %12s' % ('u 区间', 'n', 'pred sd', 'resid sd', 'resid/predsd'))
for lo, hi in bins:
    m = (u >= lo) & (u < hi)
    if m.sum() < 30:
        continue
    print('  %-12s %6d %12.4f %12.4f %12.2f'
          % ('[%d,%d)' % (lo, hi), int(m.sum()), pred[m].std(), resid[m].std(),
             resid[m].std() / max(pred[m].std(), 1e-9)))
print('  %-12s %6d %12.4f %12.4f %12.2f'
      % ('全体', len(pred), pred.std(), resid.std(), resid.std() / max(pred.std(), 1e-9)))
print('\n  ⇒ 若"带内 resid sd" ≈ "全体 resid sd"，说明残差不来自风况（风已被状态覆盖），')
print('     而来自**同一风况下状态仍无法确定未来**（观测只有 4 维：pr/u/SOC/P_G）。')

print('\n---- 同一风况、同一电价下 Q 的剩余离散（状态维已榨干的程度）----')
uk = np.round(u, 1)
pk = np.round(pr, 0)
g = pd.DataFrame({'uk': uk, 'pk': pk, 'q': pred, 'b': bootstrap}).groupby(['uk', 'pk'])
sz = g.size()
big = sz[sz >= 5].index
if len(big):
    sub = g.get_group(big[0])
    print('  例：u≈%.1f, pr=%.0f 的 %d 个样本里  pred sd = %.4f（全体 %.4f）'
          % (big[0][0], big[0][1], len(sub), sub['q'].std(), pred.std()))
    sds = [g.get_group(k)['q'].std() for k in big]
    print('  所有 (u,pr) 格（n≥5, 共 %d 格）内 pred sd 的中位 = %.4f  ⇒ 占全体 sd 的 %.0f%%'
          % (len(sds), float(np.median(sds)), 100 * np.median(sds) / pred.std()))
    print('  ⇒ 这个比例就是"状态里还剩下多少不能被 (u,pr) 解释"的值函数变化。')

print('\n---- 决定性对照：critic vs 同特征上的**朴素最优预测**（线性闭合解）----')
# 目标 t = boot（自举项）；用状态特征直接预测它。
#   * 若 critic 的 td 与"线性最优"可比 ⇒ critic 已接近该特征集的能力上限（残差是**不可约**的）
#   * 若 critic 明显更差 ⇒ 是网络/优化没学好（可修）
feat = np.column_stack([
    np.ones(len(S)),
    u,                                      # 风速
    u ** 2, u ** 3,
    pr / 1000.0,                            # 电价
    o1[:, 2] / 10.0,                        # SOC
    o1[:, 3] / 10.0,                        # P_G
    a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4],   # 动作
])
tgt = bootstrap
coef, *_ = np.linalg.lstsq(feat, tgt, rcond=None)
lin = feat @ coef
res_lin = tgt - lin
res_crit = tgt - pred
print('  线性最优（%d 个特征）: 残差 sd = %.4f   预测 sd = %.4f'
      % (feat.shape[1], res_lin.std(), lin.std()))
print('  critic 网络            : 残差 sd = %.4f   预测 sd = %.4f'
      % (res_crit.std(), pred.std()))
print('  朴素常数（只用均值）    : 残差 sd = %.4f' % tgt.std())
print()
print('  ⇒ critic 残差 / 线性最优残差 = %.2f' % (res_crit.std() / res_lin.std()))
print('     critic 残差 / 常数残差    = %.2f' % (res_crit.std() / tgt.std()))
print('     线性最优已解释的方差占比  = %.1f%%'
      % (100 * (1 - res_lin.var() / tgt.var())))
print('     critic 已解释的方差占比   = %.1f%%'
      % (100 * (1 - res_crit.var() / tgt.var())))


