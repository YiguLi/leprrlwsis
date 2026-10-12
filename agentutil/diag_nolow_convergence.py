# -*- coding: utf-8 -*-
"""NOLOW 分支（td3per --nolow）**收敛诊断**：回答"为什么策略难以收敛到优秀区间、怎么加快"。

只读（不需 MATLAB / 不需 torch 之外的东西）。四段输出，每段都给出**机器可读判据**：

[A] update_stats_H.csv  —— H 层学习动力学
    td_p/td_c（TD 误差）、|Q| 电平、**自洽比 Qbar/anchor**（anchor = EMA(块奖励)/(1−γ)）、
    奖励电平 vs 滚动锚、λ1（对偶价格）轨迹。
    判据：自洽比 → 1 表示值函数与真实回报一致（收敛）；持续 < 0.6 表示 critic 严重欠拟合。

[B] df_episode_*.csv —— episode 层面
    profit / cost / bess_profit 对 **wind_day 做同天配对**（不同 episode 风况不同，
    直接看 raw 曲线会被风况噪声淹没）⇒ 输出"同天去均值后"的残差趋势，这才是策略进步的真信号。
    判据：残差趋势斜率 t 值；|t|<2 且幅度 <1% ⇒ 停留在平台。

[C] df_test9.csv —— 固定 9 用例面板（跨 run 可比）
    逐检查点 profit_econ / profit / c1 / penalty_NN_H / turbine_profit。
    这是"优秀区间"的直接度量。

[D] actor_*.pkl / checkpoint_*/actor_6.pkl —— 策略漂移
    取日志每个 5 步块首状态（子采样），用各检查点 actor 算动作 ⇒
    rms(k) = sqrt(mean_s ||a_{i+k} − a_i||²) 的 **标度律**：
      α≈1 匀速漂移（未收敛） / α≈0.5 定点随机游走（已收敛） / α≈0 冻结
    另报相邻增量的方向余弦 cos(Δ_i·Δ_{i+1})（≈0 游走 / →1 有向漂移）。

用法
----
    python agentutil/diag_nolow_convergence.py --run output/TD3PERNOLOW-run57
    python agentutil/diag_nolow_convergence.py --run run57 run55 run56      # 前缀匹配
"""
import argparse
import glob
import os
import pickle
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 40)

GAMMA = 0.95          # H 层折扣（CLI --discount）
BLOCK = 5             # H 层每 5 步决策一次 ⇒ 一条 transition 覆盖一个块
# ⚠ H 层的 n 步回报按 **块** 生效（td3per_ddpg `_boot = γ^nstep`，nstep 默认 5），
#   所以无限视野锚点必须用 γ_eff = γ^nstep，而不是 γ：
#     1/(1−γ)   = 20   块  ← 错（那是"每步一次决策"的口径）
#     1/(1−γ^5) = 4.42 块  ← 对（H 层每块才折扣一次）
GAMMA_EFF = GAMMA ** 5
HORIZON = 1.0 / (1.0 - GAMMA_EFF)


# ---------------------------------------------------------------- helpers
def find_run(spec, base='output'):
    """spec 可以是完整路径、run 名、或唯一子串。"""
    if os.path.isdir(spec):
        return spec
    cand = os.path.join(base, spec)
    if os.path.isdir(cand):
        return cand
    hits = [os.path.join(base, d) for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d)) and spec in d]
    if not hits:
        raise SystemExit('找不到 run: %s' % spec)
    hits.sort(key=lambda p: -os.path.getmtime(p))
    return hits[0]


def log_csv(d):
    """逐步日志表：优先 df_td3per*.csv（排除 episode / test9）。"""
    for pat in ('df_td3pernolow.csv', 'df_td3per.csv', 'df_td3perseg*.csv'):
        p = os.path.join(d, pat)
        if os.path.exists(p):
            return p
    for p in sorted(glob.glob(os.path.join(d, 'df_*.csv'))):
        b = os.path.basename(p)
        if 'episode' in b or 'test9' in b:
            continue
        with open(p, 'r', encoding='utf-8-sig') as f:
            head = f.readline()
        if 'original_action_H' in head and 'turbine_profit' in head:
            return p
    return None


def ckpt_steps(d):
    out = []
    for p in os.listdir(d):
        if p.startswith('checkpoint_') and os.path.isdir(os.path.join(d, p)):
            try:
                out.append(int(p.split('_')[-1]))
            except ValueError:
                pass
    return sorted(out)


def load_actor(path):
    """checkpoint 存的是 `actor.state_dict()`（torch.save 的 OrderedDict），
    必须按 `model.Actor` 的形状重建后 `load_state_dict` —— 存的是**参数字典不是模块**。"""
    import torch
    from model import Actor
    sd = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(sd, dict) or 'fc1.weight' not in sd:
        raise RuntimeError('%s 不是 Actor state_dict（键: %s）'
                           % (path, list(sd)[:5] if isinstance(sd, dict) else type(sd)))
    nb_states = sd['fc1.weight'].shape[1]
    nb_actions = sd['fc4.weight'].shape[0]
    h1 = sd['fc1.weight'].shape[0]
    h2 = sd['fc2.weight'].shape[0]
    m = Actor(nb_states, nb_actions, hidden1=h1, hidden2=h2)
    m.load_state_dict(sd)
    m.eval()
    return m


def actor_acts(model, X):
    import torch
    with torch.no_grad():
        t = torch.as_tensor(np.asarray(X, dtype=np.float32))
        y = model(t)
        return np.asarray(y.detach().cpu().numpy(), dtype=float)


def hdr(t):
    print('\n' + '=' * 100)
    print(t)
    print('=' * 100)


def tstat(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return float('nan'), float('nan')
    idx = np.arange(n)
    b, a = np.polyfit(idx, x, 1)
    resid = x - (a + b * idx)
    se = np.sqrt((resid ** 2).sum() / (n - 2) / ((idx - idx.mean()) ** 2).sum())
    return b, (b / se if se > 0 else float('nan'))


# ---------------------------------------------------------------- [A]
def sec_A(d):
    p = os.path.join(d, 'update_stats_H.csv')
    if not os.path.exists(p):
        print('\n[A] 无 update_stats_H.csv，跳过')
        return None
    hdr('[A] H 层学习动力学  (%s)' % os.path.basename(p))
    cols = ['step', 'td_p_mean', 'td_c_mean', 'q_p_mean', 'q_p2_mean',
            'q_c_mean', 'target_p_mean', 'target_c_mean', 'reward_mean',
            'cost_mean', 'weight_mean']
    df = pd.read_csv(p, usecols=lambda c: c in cols)
    if 'step' not in df:
        df = pd.read_csv(p)
    n = len(df)
    print('  条数 %d，step 范围 %d ~ %d' % (n, df['step'].min(), df['step'].max()))
    print('  折扣口径: γ=%.2f/块 × nstep=5 ⇒ γ_eff=γ^5=%.4f ⇒ 无限视野 %.2f 块'
          % (GAMMA, GAMMA_EFF, HORIZON))

    # Qbar = 双 critic 平均；anchor = EMA(块奖励)×1/(1−γ^5)
    qbar = df[['q_p_mean', 'q_p2_mean']].mean(axis=1)
    df['Qbar'] = qbar
    r = df['reward_mean'].to_numpy(float)
    ema, al = [], 1.0 / 50.0
    cur = r[0]
    for v in r:
        cur = (1 - al) * cur + al * v
        ema.append(cur)
    df['块奖励EMA'] = np.array(ema)
    df['anchor'] = df['块奖励EMA'] * HORIZON
    df['自洽比'] = qbar / df['anchor']
    df['relTD_p'] = df['td_p_mean'] / np.maximum(np.abs(df['q_p_mean']), 1e-9)
    df['relTD_c'] = df['td_c_mean'] / np.maximum(np.abs(df['q_c_mean']), 1e-9)
    df['块奖励EMA'] = np.array(ema)

    show = ['step', 'Qbar', 'anchor', '自洽比', 'td_p_mean', 'q_p_mean',
            'td_c_mean', 'q_c_mean', '块奖励EMA', 'cost_mean', 'weight_mean']
    k = max(1, n // 12)
    print('\n  ---- 每 %d 个更新点采样一次 ----' % k)
    print(df[show].iloc[::k].to_string(index=False,
                                       float_format=lambda v: '%10.3f' % v))

    print('\n  ---- 前后半段对比 ----')
    h = n // 2
    for c in ('td_p_mean', 'td_c_mean', 'q_p_mean', 'q_c_mean', '自洽比',
              '块奖励EMA', 'cost_mean', 'weight_mean'):
        a, b = df[c].iloc[:h].mean(), df[c].iloc[h:].mean()
        print('    %-12s 前半 %12.4f → 后半 %12.4f  (%+.1f%%)'
              % (c, a, b, 100 * (b - a) / abs(a) if abs(a) > 1e-12 else float('nan')))
    b, t = tstat(df['自洽比'])
    print('    自洽比线性趋势: 斜率 %+.3e/更新点  t=%+.2f' % (b, t))
    b, t = tstat(df['td_p_mean'])
    print('    td_p 线性趋势 : 斜率 %+.3e  t=%+.2f  (应 <0 = critic 在学)'
          % (b, t))
    # 重量：最后 25% 段的自洽比
    tail = df['自洽比'].iloc[-max(1, n // 4):]
    print('    末 25%% 自洽比: 均值 %.3f  范围 %.3f ~ %.3f'
          % (tail.mean(), tail.min(), tail.max()))
    print('    相对 TD 误差 td_p/|Q_p|: 前 5%%=%.3f  中段=%.3f  末 5%%=%.3f （应下降）'
          % (df['relTD_p'].iloc[:max(1, n // 20)].mean(),
             df['relTD_p'].iloc[n // 2 - n // 40:n // 2 + n // 40].mean(),
             df['relTD_p'].iloc[-max(1, n // 20):].mean()))
    print('    ⚠ weight_mean 是 **PER 重要性权重**（β 0.4→1 退火），不是 λ1；'
          ' λ1 在逐步日志 df_*.csv 里')
    return df


def sec_A2(d, step_csv):
    """λ1 / d_block / ema_c1 的对偶诊断（逐步日志，分块统计）。"""
    if step_csv is None:
        return
    hdr('[A2] 对偶变量 λ1 / 预算 d_block_H / EMA 实际成本')
    want = ['step', 'lam1', 'lam2', 'd_block_H', 'ema_c1_H', 'c1', 'P_F',
            'penalty_NN_H', 'clip_amount']
    head = pd.read_csv(step_csv, nrows=0, encoding='utf-8-sig').columns
    cols = [c for c in want if c in head]
    if 'lam1' not in cols:
        print('  无 lam1 列，跳过')
        return
    acc = []
    for ch in pd.read_csv(step_csv, usecols=cols, chunksize=200000,
                          encoding='utf-8-sig', low_memory=False):
        for c in cols:
            if ch[c].dtype == object:
                ch[c] = pd.to_numeric(ch[c], errors='coerce')
        acc.append(ch)
    df = pd.concat(acc, ignore_index=True)
    # λ1 的观测区间反推 lam_scale（λ_raw = lam1 / kl）
    lo, hi = float(df['lam1'].min()), float(df['lam1'].max())
    print('  λ1(actor 空间) 范围 %.6g ~ %.6g  均值 %.6g  唯一值 %d 个'
          % (lo, hi, df['lam1'].mean(), df['lam1'].nunique()))
    try:
        import hsrl_cost as hc
        kl = hc.lam_scale()
        print('  lam_scale=s_c/s_p=%.6g ⇒ λ_raw=λ1/kl ∈ [%.6g, %.6g]；'
              '物理区间 [%.4f, %.4f] 初值 %.4f'
              % (kl, lo / kl, hi / kl, hc.LAM_MIN, hc.LAM_MAX, hc.LAM_INIT))
        floor, ceil = hc.LAM_MIN * kl, hc.LAM_MAX * kl
    except Exception as e:                     # noqa: BLE001
        print('  （未能 import hsrl_cost: %s）' % e)
        floor = ceil = None
    if floor is not None:
        at_f = float((df['lam1'] <= floor * (1 + 1e-6)).mean())
        at_c = float((df['lam1'] >= ceil * (1 - 1e-6)).mean())
        print('  λ1 贴下界占比 %.1f%%  贴上界占比 %.1f%%' % (100 * at_f, 100 * at_c))
    if 'd_block_H' in df:
        print('  d_block_H: %s 处不同取值（curriculum 是否触发）'
              % sorted(df['d_block_H'].dropna().unique())[:6])
    if 'ema_c1_H' in df and 'd_block_H' in df:
        m = df['d_block_H'] > 0
        resid = (df.loc[m, 'ema_c1_H'] / df.loc[m, 'd_block_H'] - 1)
        print('  对偶残差 EMA(c1)/d_block − 1: 末 10%% 均值 %+.3f （>0 违反预算，'
              '<0 预算宽松 ⇒ 约束不起作用）' % resid.iloc[-max(1, len(resid) // 10):].mean())
    if 'c1' in df:
        print('  步级 c1: 均值 %.4f  中位 %.4f  p95 %.4f  max %.4f；c1=0 占比 %.1f%%'
              % (df['c1'].mean(), df['c1'].median(), df['c1'].quantile(.95),
                 df['c1'].max(), 100 * (df['c1'] <= 1e-12).mean()))
    if 'penalty_NN_H' in df:
        print('  步级越界罚: 均值 %.4f  p95 %.3f  max %.3f  非零占比 %.2f%%'
              % (df['penalty_NN_H'].mean(), df['penalty_NN_H'].quantile(.95),
                 df['penalty_NN_H'].max(),
                 100 * (df['penalty_NN_H'] > 1e-9).mean()))
    return df


# ---------------------------------------------------------------- [B]
def sec_B(d):
    ep = [p for p in glob.glob(os.path.join(d, 'df_episode*.csv'))]
    if not ep:
        print('\n[B] 无 df_episode*.csv，跳过')
        return None
    hdr('[B] episode 层面（同风况日配对）  (%s)' % os.path.basename(ep[0]))
    df = pd.read_csv(ep[0])
    print('  %d 个 episode，wind_day %d 个不同天' % (len(df), df['wind_day'].nunique()))
    # 每个 wind_day 的期望电平 = 该天所有 episode 的均值；残差 = 观测 - 当天均值
    for c in ('episode_profit', 'episode_reward', 'bess_profit', 'episode_cost'):
        g = df.groupby('wind_day')[c]
        df['res_' + c] = df[c] - g.transform('mean')
    # 只用出现 >=2 次的天做残差（1 次的天残差恒 0）
    cnt = df.groupby('wind_day')['episode'].transform('size')
    ok = cnt >= 2
    print('  参与残差配对的天覆盖 %d/%d 个 episode' % (int(ok.sum()), len(df)))
    for c in ('episode_profit', 'episode_reward', 'bess_profit', 'episode_cost'):
        b, t = tstat(df.loc[ok, 'res_' + c])
        raw_b, raw_t = tstat(df[c])
        mu = df[c].abs().mean()
        print('    %-16s 残差斜率 %+12.2f/ep (t=%+5.2f, 相对 %.2f%%) | raw 斜率 %+12.2f (t=%+5.2f)'
              % (c, b, t, 100 * b / mu if mu else float('nan'), raw_b, raw_t))
    # 关键：profit 的方差由风况解释的比例
    for c in ('episode_profit', 'episode_reward'):
        ss_tot = ((df[c] - df[c].mean()) ** 2).sum()
        ss_res = (df['res_' + c] ** 2).sum()
        print('    %-16s 风况日解释方差 R² = %.3f （=1 - SSres/SStot）' % (c, 1 - ss_res / ss_tot))
    print('\n  ---- 逐 episode ----')
    print(df[['episode', 'wind_day', 'episode_profit', 'episode_cost',
              'bess_profit', 'episode_reward']].to_string(index=False,
                                                          float_format=lambda v: '%11.1f' % v))
    return df


# ---------------------------------------------------------------- [C]
def sec_C(d):
    p = os.path.join(d, 'df_test9.csv')
    if not os.path.exists(p):
        print('\n[C] 无 df_test9.csv，跳过')
        return None
    hdr('[C] 固定 9 用例面板（跨 run 可比的"优秀区间"度量）')
    df = pd.read_csv(p, encoding='utf-8-sig')
    if len(df) % 45 != 0:
        print('  ⚠ 行数 %d 不是 45 的整数倍（评测被中断）' % len(df))
    k = len(df) // 45
    rows = []
    for i in range(k):
        s = df.iloc[i * 45:(i + 1) * 45]
        if len(s) == 0:
            continue
        rows.append(dict(step=int(s['step'].iloc[0]),
                         profit=float(s['profit'].sum()),
                         profit_econ=float(s['profit_econ'].sum()),
                         r1=float(s['r1'].mean()),
                         c1=float(s['c1'].mean()),
                         penalty=float(s['penalty_NN_H'].sum()),
                         turbine=float(s['turbine_profit'].sum()),
                         derate=float(s['derate_loss'].sum()),
                         pf_viol=int(s['pf_viol'].sum()),
                         cvm=int(s['cvm'].sum()),
                         soc_end_min=float(s['soc_end'].min()),
                         soc_end_max=float(s['soc_end'].max())))
    R = pd.DataFrame(rows)
    print('  共 %d 轮（每轮 45 步 = 9 用例 × 5 步），共 %d 行' % (len(R), len(df)))
    print(R.to_string(index=False, float_format=lambda v: '%11.2f' % v))
    print('\n  ---- 趋势（线性斜率 / t）----')
    for c in ('profit', 'profit_econ', 'penalty', 'c1', 'turbine', 'pf_viol'):
        b, t = tstat(R[c])
        a2 = R[c].iloc[:max(1, len(R) // 2)].mean()
        b2 = R[c].iloc[len(R) // 2:].mean()
        print('    %-12s 斜率 %+11.3f/轮 (t=%+5.2f)  前半 %11.2f → 后半 %11.2f'
              % (c, b, t, a2, b2))
    # 惩罚分解：越界量
    x = R['penalty'] / 400.0
    print('\n  越界总量 Σ前5维越界（= penalty/400）: 均值 %.4f  末轮 %.4f  最大 %.4f'
          % (x.mean(), x.iloc[-1], x.max()))
    print('  末轮 profit_econ = %.2f 元 /45 步  ⇒ 日均 %.0f 元（×1440/45）'
          % (R['profit_econ'].iloc[-1], R['profit_econ'].iloc[-1] * 32))
    # 与最优可比：wf 上界（turbine 满发）只能事后讨论，此处只报电平
    return R


# ---------------------------------------------------------------- [D]
def sec_D(d, step_csv, max_ck=8, sub=1500):
    cks = ckpt_steps(d)
    if not cks:
        print('\n[D] 无 checkpoint，跳过')
        return None
    hdr('[D] 策略漂移：rms(k) 标度律 + 增量方向余弦')
    need = ['norm_observation_L', 'original_action_H']
    cols = None
    with open(step_csv, 'r', encoding='utf-8-sig') as f:
        cols = f.readline().strip().split(',')
    usecols = [c for c in ('step', 'phase_minute', 'phase_sin', 'phase_cos',
                           'observation') if c in cols]
    print('  读日志（仅 %d 列）…' % len(usecols))
    df = pd.read_csv(step_csv, usecols=usecols)
    blk = df.iloc[::BLOCK].reset_index(drop=True)
    if len(blk) > sub:
        idx = np.linspace(0, len(blk) - 1, sub).astype(int)
        blk = blk.iloc[idx].reset_index(drop=True)
    print('  块级样本 %d 个（子采样自 %d 块）' % (len(blk), len(df) // BLOCK))

    # 组装 H 观测：走生产同一函数 Normalize_H(obs, minute, phase_mode)
    #   日志 observation 列格式如 "[ 300.  10.5  3.   3.5]"
    #   维数由 checkpoint 的 fc1.weight.shape[1] 反推（--phase_feat 0/1/2 ⇒ 4/6/9）
    obs = blk['observation'].astype(str)
    parsed = np.array([[float(x) for x in s.strip().strip('[]').split()] for s in obs])
    minute = (blk['phase_minute'].to_numpy(int) if 'phase_minute' in blk
              else np.zeros(len(blk), dtype=int))
    from td3per.td3per_myfunc import Normalize_H
    import torch as _torch
    _first = None
    for _s in ckpt_steps(d):
        _c = os.path.join(d, 'checkpoint_%d' % _s)
        for _f in ('actor_4.pkl', 'actor_6.pkl', 'actor_5.pkl', 'actor_9.pkl'):
            if os.path.exists(os.path.join(_c, _f)):
                _first = os.path.join(_c, _f)
                break
        if _first:
            break
    _sd = _torch.load(_first, map_location='cpu', weights_only=True)
    want_dim = int(_sd['fc1.weight'].shape[1])
    phase_mode = {4: 0, 6: 1, 9: 2}.get(want_dim, -1)
    if phase_mode < 0:
        raise SystemExit('未知 nb_states_H=%d' % want_dim)
    X = np.array([Normalize_H(parsed[i, :4], int(minute[i]), phase_mode)
                  for i in range(len(parsed))], dtype=float)
    print('  H 观测维数 = %d（phase_mode=%d，由 %s 的 fc1 反推）'
          % (X.shape[1], phase_mode, os.path.basename(_first)))

    # 选检查点：末 8 个 + 均匀跨度
    sel = cks if len(cks) <= max_ck else [cks[i] for i in
                                          np.linspace(0, len(cks) - 1, max_ck).astype(int)]
    As = {}
    for s in sel:
        p = os.path.join(d, 'checkpoint_%d' % s)
        act_p = None
        for f in ('actor_4.pkl', 'actor_6.pkl', 'actor_5.pkl', 'actor_9.pkl'):
            if os.path.exists(os.path.join(p, f)):
                act_p = os.path.join(p, f)
                break
        if act_p is None:
            continue
        try:
            mdl = load_actor(act_p)
            y = actor_acts(mdl, X)
        except Exception as e:                 # noqa: BLE001
            import traceback
            print('    checkpoint_%d actor 失败: %s' % (s, e))
            traceback.print_exc(limit=3)
            continue
        if y is None:
            continue
        As[s] = y
    if len(As) < 3:
        print('  ⚠ 可用检查点 <3，跳过 [D]')
        return None
    steps = sorted(As)
    print('  可用检查点: %s' % steps)

    print('\n  ---- 相邻检查点之间的策略位移 ----')
    print('    %-16s %10s %10s %10s %10s' % ('区间', 'rms位移', '相对尺度', 'cos(Δ_i,Δ_i+1)', '维度'))
    P = {s: As[s].mean(axis=0) for s in steps}
    A = {s: As[s] for s in steps}
    for i in range(len(steps) - 1):
        a, b = steps[i], steps[i + 1]
        d1 = P[b] - P[a]
        if i + 2 < len(steps):
            d2 = P[steps[i + 2]] - P[b]
            cs = float(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-12))
        else:
            cs = float('nan')
        rms = float(np.sqrt(((A[b] - A[a]) ** 2).mean()))
        print('    %6d→%-6d %10.5f %10.3f %10.3f %10d'
              % (a, b, rms, rms / 0.5, cs, A[a].shape[1]))

    # rms(k) 标度律（在同一批检查点上）
    print('\n  ---- rms(k) = sqrt(mean_s ||a_{i+k} − a_i||²) 标度律 ----')
    for k in range(1, min(5, len(steps))):
        vals = []
        for i in range(len(steps) - k):
            vals.append(((A[steps[i + k]] - A[steps[i]]) ** 2).mean())
        rms = float(np.sqrt(np.mean(vals)))
        print('    k=%d  rms=%10.5f' % (k, rms))
    ks, rs = [], []
    for k in range(1, min(5, len(steps))):
        vals = [((A[steps[i + k]] - A[steps[i]]) ** 2).mean()
                for i in range(len(steps) - k)]
        ks.append(np.log(k))
        rs.append(np.log(float(np.sqrt(np.mean(vals))) + 1e-12))
    if len(ks) >= 2:
        al, _ = np.polyfit(ks, rs, 1)
        print('    ⇒ α = %.3f   (≈1 匀速漂移/未收敛；≈0.5 定点随机游走/已收敛；≈0 冻结)' % al)
    # 动作到界面的距离
    last = A[steps[-1]]
    print('\n  ---- 末检查点动作统计（评估态，无噪声） ----')
    names = ['a1', 'a2', 'a3', 'a4', 'a5']
    for j in range(last.shape[1]):
        v = last[:, j]
        nm = names[j] if j < len(names) else 'sig'
        print('    %-4s mean %+7.4f sd %6.4f  min %+7.4f max %+7.4f  '
              '越下界 %5.2f%% 越上界 %5.2f%%'
              % (nm, v.mean(), v.std(), v.min(), v.max(),
                 100 * (v < 0).mean(), 100 * (v > 0.5).mean()))
    return As


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', nargs='+', default=['output/TD3PERNOLOW-run57'])
    ap.add_argument('--base', default='output')
    ap.add_argument('--skip-D', action='store_true')
    args = ap.parse_args()

    for spec in args.run:
        d = find_run(spec, args.base)
        print('\n' + '#' * 100)
        print('# run: %s' % d)
        print('#' * 100)
        step_csv = log_csv(d)
        print('  逐步日志: %s' % (os.path.basename(step_csv) if step_csv else '—'))
        print('  checkpoint: %d 个, 最大 step=%s'
              % (len(ckpt_steps(d)), max(ckpt_steps(d)) if ckpt_steps(d) else '—'))
        sec_C(d)
        sec_A(d)
        sec_A2(d, step_csv)
        sec_B(d)
        if not args.skip_D and step_csv:
            sec_D(d, step_csv)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
