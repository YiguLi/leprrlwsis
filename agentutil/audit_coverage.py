#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""九方法覆盖审计 + H 层 η1 分析（不需要 MATLAB）。

回答两个问题：

Q1. **H 层 c1 的 EMA 对偶步长 η1 是否面临 η2 那样的困境？**
    η2 的困境来自"加性对偶 + 被比较的两个量同时缩小 k 倍"：
        λ += η·(ema − d)，若 ema,d 都 ×1/k 而 η 不变 ⇒ 增量 ×1/k，对偶慢 k 倍。
    H 层是**乘性**对偶 λ ← λ·exp(η·(ema/d − 1))，只看**比值**，因此：
        * 现状（ema,d 都保持原始口径）⇒ 比值不变 ⇒ η1 无需改；
        * 即使把 H 对偶也搬进归一化口径（ema/s, d/s）⇒ 比值仍不变 ⇒ η1 仍无需改。
    本脚本用同一段合成 c1 序列把四种组合实算一遍，把"乘性免疫 / 加性不免疫"量化出来。

Q2. **归一化(--value_scale) 与 bias 策略是否对九个方法都生效？**
    逐个方法核对：main 接线、H agent 类的换算点、L 层(6 个 RL 下层)显式不缩放、
    bias 估计器与 H agent 的 (γ, s) 是否同源、bias 应用是否发生在首次策略更新之前、
    critic 是否确有 fc3 输出层（bias 策略的前提）。

用法:  python agentutil/audit_coverage.py
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)                                         # 相对数据/输出路径以仓库根为基准
sys.path.insert(0, ROOT)                               # 项目模块 (hsrl_*, myfunc, util, td3*/...)
sys.path.insert(0, _HERE)                              # agentutil 内互相 import

import numpy as np                                     # noqa: E402

import hsrl_cost as hc                                 # noqa: E402
import test_five_days as T                             # noqa: E402  (注册表, import 不启动 MATLAB)

FAILS = []


def src(p):
    return open(p, encoding='utf-8').read()


def need(cond, tag, detail=''):
    if not cond:
        FAILS.append('{} {}'.format(tag, detail))
    return cond


# =============================================================================
print('=' * 92)
print('Q1. H 层 η1：对偶形式决定它是否面临 η2 的困境')
print('=' * 92)

# --- 代码事实 ---
dual_src = src('hsrl_cost.py')
need('agent.lam *= float(np.exp(float(agent.ema_lam_lr) * (ratio - 1.0)))' in dual_src,
     '[η1/代码]', 'hc.dual_update 的 uniform 分支应为乘性(比值)形式')
need("ratio = float(agent.ema_cost) / max(float(d_block), 1e-9)" in dual_src,
     '[η1/代码]', '乘性对偶应只用 EMA/d 的比值')
eta1 = hc.ETA_LAM
print('  H 层: 乘性 λ·exp(η1·(EMA/d − 1)), η1 = {} (= hc.ETA_LAM)'.format(eta1))
print('  L 层: 加性 λ += η2·(EMA − d),     η2 = 1.0 (÷100 口径同步 ×100)')

# 各 main 是否把**原始** EMA/d 交给对偶
for proj, spec in sorted(T.PROJECTS.items()):
    t = src(spec['ref'])
    need(re.search(r'update_ema_cost\(sum_c1\)', t) is not None,
         '[η1/{}/原始EMA]'.format(proj), 'agent_H.update_ema_cost 应吃原始 sum_c1')
    need('hc.dual_update(agent_H, d_block_H)' in t,
         '[η1/{}/对偶]'.format(proj), 'H 对偶应走 hc.dual_update')
    need(not re.search(r'update_ema_cost\(\s*(hc\.)?v_scale', t),
         '[η1/{}/未缩放]'.format(proj), 'EMA 入口不得被 /s 缩放')

# --- 数值实验: 四种组合 ---
# 造一段"先持续越限(30 MW·step, 比值 2) 再回到正常(5)"的 c1 序列 —— 让 λ 离开
# 边界来回走, 四种组合才有区分度(若全程都在边界上, 轨迹会被 clip 掩盖)。
c1 = np.concatenate([np.full(80, 30.0), np.full(320, 5.0)])
ALPHA, LAM0 = hc.ALPHA_EMA, hc.LAM_INIT
LMIN, LMAX = hc.LAM_MIN, hc.LAM_MAX


def run_dual(c1_block, d, eta, mode, scale=1.0):
    """scale: 把 EMA 与 d 同时乘 1/scale（模拟"对偶通道也搬进归一化口径"）。"""
    lam, ema = LAM0, None
    out = []
    for x in c1_block:
        x = x / scale
        ema = x if ema is None else (1 - ALPHA) * ema + ALPHA * x
        d_s = d / scale
        if mode == 'mul':
            lam *= float(np.exp(eta * (ema / max(d_s, 1e-9) - 1.0)))
        else:
            lam += eta * (ema - d_s)
        lam = min(max(lam, LMIN), LMAX)
        out.append(lam)
    return np.array(out)


A = run_dual(c1, hc.D_BLOCK, eta1, 'mul')                          # 现状: 原始口径 + 乘性
B = run_dual(c1, hc.D_BLOCK, eta1, 'mul', scale=1000.0)            # 对偶也归一化 + 乘性
E = run_dual(c1, hc.D_BLOCK, eta1, 'add')                          # 加性 + 原始口径(旧形式)
D = run_dual(c1, hc.D_BLOCK, eta1 * 1000.0, 'add', scale=1000.0)   # 加性 + 归一化 + η×1000
C = run_dual(c1, hc.D_BLOCK, eta1, 'add', scale=1000.0)            # 加性 + 归一化 + η 不补
print('\n  同一段合成 c1 ({} 块: 前 80 块 30 MW·step 越限, 之后 5), d_block_H={}, '
      'λ0={:.3f}, λ∈[{:.3f},{:.3f}]:'.format(len(c1), hc.D_BLOCK, LAM0, LMIN, LMAX))
for tag, arr in (('A 原始口径 + 乘性 (现状)', A), ('B 归一化口径 + 乘性', B),
                 ('E 原始口径 + 加性 (旧形式)', E), ('D 归一化 + 加性 (η×1000)', D),
                 ('C 归一化 + 加性 (η 不补)', C)):
    print('    {:<28s} λ_end={:.4f}  λ_max={:.4f}  λ_std={:.5f}'
          .format(tag, arr[-1], arr.max(), arr.std()))
print('    |A−B|max = {:.3e}  ⇒ 乘性对"EMA,d 同比例缩放"逐位免疫'
      .format(np.abs(A - B).max()))
print('    |E−D|max = {:.3e}  ⇒ 加性只要 η×k 就与旧口径逐位一致 (η2 的处理方式)'
      .format(np.abs(E - D).max()))
print('    |E−C|max = {:.3e}  ⇒ 加性不补 η 则慢到几乎不动 (η2 的困境)'
      .format(np.abs(E - C).max()))
need(np.abs(A - B).max() < 1e-12, '[η1/结论]', '乘性对偶对同比例缩放免疫')
need(np.abs(E - D).max() < 1e-12, '[η1/结论]', '加性对偶补 η×k 后与旧口径一致')
need(np.abs(E - C).max() > 1.0, '[η1/结论]', '加性对偶不补 η 会明显变慢(η2 的困境)')
print('  ⇒ 结论: H 层**不需要**改 η1 —— 乘性对偶只看 EMA/d 比值, 而现状连口径都没动;\n'
      '     即便将来把 H 对偶也搬进归一化口径, 比值不变 ⇒ 依旧不需要。\n'
      '     反之 L 层是加性对偶, 又真的把 c2 与 d_block_L 同除了 100 ⇒ 必须 η2×100。')

# =============================================================================
print('=' * 92)
print('Q2. 归一化 + bias 策略 对九方法的覆盖')
print('=' * 92)
print('{:<11s} {:>4s} {:>4s} {:>4s} {:>4s} {:>4s} {:>4s} {:>4s} {:>4s} {:>4s} {:>4s}'
      .format('方法', 'arg', 'set', 'wire', 'cls', 'TD', 'log', 'bias', 'bfst', 'L1.0', 'γ/s'))
rows = []
for proj, spec in sorted(T.PROJECTS.items()):
    main_t = src(spec['ref'])
    mod, cls = spec['agent']
    agent_t = src(mod.replace('.', '/') + '.py')

    c = {}
    c['arg'] = main_t.count("parser.add_argument('--value_scale'") == 1
    c['set'] = main_t.count('hc.set_value_scale(args.value_scale)') == 1
    c['wire'] = main_t.count('agent_H.value_scale = hc.value_scale()') == 1
    # 该项目的 H agent 类里换算点是否齐全
    c['cls'] = (agent_t.count('self.value_scale = 1.0') == 1
                and agent_t.count('/ _S') == 2
                and agent_t.count('* _S - self.cost_limit') == 1)
    c['TD'] = (re.search(r'TD_errors_p = abs\([^\n]*\) \* _S', agent_t) is not None
               and re.search(r'TD_errors_c = abs\([^\n]*\) \* _S', agent_t) is not None)
    c['log'] = agent_t.count('_d[_k] = float(_d[_k]) * _S') == 1
    # bias 策略: 开关 + 估计器 + 应用点"在首次策略更新之前"
    c['bias'] = (main_t.count("parser.add_argument('--critic_bias_init'") == 1
                 and main_t.count('hinit.BiasInitTracker.from_args(args)') == 1
                 and 'bias_init.on_block_end(' in main_t)
    i_apply = main_t.find('bias_init.on_block_end(')
    i_obs = main_t.find('agent_H.observe_jump(')
    i_upd = main_t.find('agent_H.update_policy()')
    c['bfst'] = (0 <= i_apply < i_obs) and (i_upd < 0 or i_apply < i_upd)
    # L 层: RL 下层必须显式 1.0; 规则下层不应有 agent_L
    if spec.get('lower') == 'rl':
        c['L1.0'] = main_t.count('agent_L.value_scale = 1.0') == 1
    else:
        c['L1.0'] = 'agent_L = ' not in main_t
    # 估计器的 γ 与 s 是否**显式绑定**到 H agent（结构上不可能分叉）
    has_bind = ('bias_init.discount = float(agent_H.discount)' in main_t and
                'bias_init.value_scale =' in main_t)
    disc_override = re.search(r'agent_H\.discount\s*=', main_t) is not None
    c['γ/s'] = has_bind and not disc_override
    rows.append((proj, c))
    print('{:<11s} {}'.format(proj, '  '.join(
        (' ✓  ' if c[k] else ' ✗  ') for k in
        ('arg', 'set', 'wire', 'cls', 'TD', 'log', 'bias', 'bfst', 'L1.0', 'γ/s'))))
    for k, ok in c.items():
        need(ok, '[{}/{}]'.format(proj, k))

print('\n  列含义: arg=--value_scale 存在; set=hc.set_value_scale 生效; wire=agent_H 接线;')
print('          cls=该项目的 H agent 类含 /s 换算; TD/日志=原始口径还原; bias=开关+估计器+应用;')
print('          bfst=bias 应用在首次策略更新之前; L1.0=RL 下层显式不缩放/规则下层无 agent_L;')
print('          γ/s=估计器的 γ 与 s 绑定到 H agent 本身(结构上不可能不一致)')

# --- critic 是否有 fc3 输出层 (bias 策略的前提) ---
print('\n  bias 策略前提: 各 H agent 的 critic 必须是 model.Critic (含 fc3 输出层)')
for proj, spec in sorted(T.PROJECTS.items()):
    mod, cls = spec['agent']
    t = src(mod.replace('.', '/') + '.py')
    m = re.search(r'from model import \(?([^\n)]*)\)?', t)
    need(m is not None and 'Critic' in m.group(1), '[fc3/{}]'.format(proj),
         'agent 应从 model 导入 Critic')
print('    {} 个 agent 模块均从 model 导入 Critic(含 fc3)'.format(len(T.PROJECTS)))

# =============================================================================
print('=' * 92)
print('Q3. SOC 目标支: soc_target 必须"先赋值后使用" (防 UnboundLocalError)')
print('=' * 92)
# 2026 年那次改造曾在 td3prrbb 踩过: 块末 gap 罚块被插在 soc_target 赋值之前 ⇒
# 一训练就 UnboundLocalError。这里静态钉住顺序 (只查 train() 体内)。
# 注: td3prle 没有块末 gap 罚 (它的追踪优化在下层 c2 的二次势里), 故不在检查范围。
GAP_MAINS = ['td3per', 'td3prrbb', 'prle', 'td3perrb', 'td3prrlrb', 'td3leprrlrb']
for proj in GAP_MAINS:
    t = src(T.PROJECTS[proj]['ref'])
    i = t.index('def train')
    j = t.find('\nif __name__')
    body = t[i:j if j > 0 else len(t)]
    d = body.find('soc_target = ')
    u = body.find('abs(soc_target')
    need(d >= 0, '[顺序/{}]'.format(proj), '未找到 soc_target 赋值')
    need(u >= 0, '[顺序/{}]'.format(proj), '未找到块末 gap 罚 (abs(soc_target ...))')
    need(d < u, '[顺序/{}]'.format(proj),
         'soc_target 在赋值之前被使用 (赋值@{}, 使用@{})'.format(d, u))
    need('sum_gap_pen = 0.0' in body, '[顺序/{}]'.format(proj), '缺 sum_gap_pen 累加器初始化')
    print('  {:<12s} 赋值@{} < 使用@{}  ✓  sum_gap_pen 已初始化'.format(proj, d, u))

# =============================================================================
print('=' * 92)
if FAILS:
    print('审计未通过 ({} 项):'.format(len(FAILS)))
    for f in FAILS:
        print('  ✗', f)
    sys.exit(1)
print('AUDIT OK —— η1 无需改动; 归一化 + bias 策略对 9 个方法均已生效; SOC 支取用顺序正确')
