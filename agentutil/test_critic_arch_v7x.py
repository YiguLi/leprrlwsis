#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""v7.1 / v7.2 / v7.3 的 critic-actor 结构契约（**纯静态**，不训练、不推进仿真）。

口径（2026-11-06，用户指定）::

    v7.1 : critic cat(s,a) → 500 → 400 → 500 → 1
           actor  s → 500 → 400 → 500 → a

    v7.2 : critic (s→128 ∥ a→128) → cat(256) → 400 → 300 → 400 → 1
           actor  s → 128 → 400 → 300 → 400 → a

    v7.3 : critic (s→128 ∥ a→128) → cat(256) → 500 → 400 → 500 → 1
           actor  s → 128 → 500 → 400 → 500 → a

其中 256 = 128 + 128，是**拼接宽度**，不是额外隐藏层（用户确认）。

段：
  A 参数量 / 层形状（H、wind5、L、feat_power；--hidden1/--hidden2 不生效）
  B 前向 / 反向 / 通路独立性（v7.1 的动作只在首层；v7.2/v7.3 的动作只过 a1）
  C infer_arch / feat_from_sd / build_critic_from_sd 往返（不与 v6/v7/v6old 混淆）
  D actor 规格（层形状/参数量/可选 fc0 编码器/输出层置零）
  E SP_TD3 端到端（8 个 critic、H/L actor、select_action、checkpoint 往返、
    评测侧从 actor_pre 恢复 v7.2/v7.3 的 actor）
  F CLI / 评测接线（静态 AST / 源码检查，不 import aifflor_main）
  G 评测侧 checkpoint 探测（test_five_days._detect_* 的纯读取往返）

用法::

    python agentutil/test_critic_arch_v7x.py
"""
import ast
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np                                       # noqa: E402
import torch                                             # noqa: E402

import model as M                                        # noqa: E402
import hsrl_agent_init as hinit                          # noqa: E402

N_OK = 0
N_BAD = 0

ARCHS = ('v7.1', 'v7.2', 'v7.3')


def ok(cond, msg):
    global N_OK, N_BAD
    if cond:
        N_OK += 1
        print('  ok   %s' % msg)
    else:
        N_BAD += 1
        print('  FAIL %s' % msg)
    return bool(cond)


def npar(m):
    return sum(p.numel() for p in m.parameters())


def critic_expected(arch, ns, na, feat=False):
    """按层表算参数量（含 bias）。"""
    a = na + (1 if feat else 0)
    if arch == 'v7.1':
        h1, h2, h3 = M.V71_DIMS
        return ((ns + a) * h1 + h1) + (h1 * h2 + h2) + (h2 * h3 + h3) + (h3 + 1)
    cfg = M.V72_DIMS if arch == 'v7.2' else M.V73_DIMS
    s, aa = cfg['s'], cfg['a']
    m1, m2, m3 = cfg['merge']
    return ((ns * s + s) + (a * aa + aa)
            + ((s + aa) * m1 + m1) + (m1 * m2 + m2) + (m2 * m3 + m3)
            + (m3 + 1))


def actor_expected(arch, ns, na):
    """按 actor 规格算参数量（含 bias）。"""
    h1, h2, pre = M.actor_spec(arch)
    n = 0
    last = ns
    if pre:
        n += ns * pre + pre
        last = pre
    return (n + last * h1 + h1 + h1 * h2 + h2 + h2 * h1 + h1 + h1 * na + na)


def batch(ns, na, n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, ns, generator=g)
    a = torch.rand(n, na, generator=g) * 0.5
    return x, a


def critic_shapes(arch, ns, na, feat=False):
    a = na + (1 if feat else 0)
    if arch == 'v7.1':
        h1, h2, h3 = M.V71_DIMS
        return {
            'h1': (h1, ns + a), 'h2': (h2, h1), 'h3': (h3, h2), 'out': (1, h3),
        }
    cfg = M.V72_DIMS if arch == 'v7.2' else M.V73_DIMS
    s, aa = cfg['s'], cfg['a']
    m1, m2, m3 = cfg['merge']
    return {
        's1': (s, ns), 'a1': (aa, a),
        'm1': (m1, s + aa), 'm2': (m2, m1), 'm3': (m3, m2), 'out': (1, m3),
    }


def actor_shapes(arch, ns, na):
    h1, h2, pre = M.actor_spec(arch)
    out = {}
    if pre:
        out['fc0'] = (pre, ns)
        in0 = pre
    else:
        in0 = ns
    out.update({
        'fc1': (h1, in0), 'fc2': (h2, h1), 'fc3': (h1, h2), 'fc4': (na, h1),
    })
    return out


# ---------------------------------------------------------------------------
def section_a():
    print('\n[A] 参数量 / 层形状（v7.1 首层拼接；v7.2/v7.3 双 128 编码器 + 256 拼接）')
    print('  %-8s %-8s %10s %10s' % ('arch', '(ns,na)', '参数量', '理论值'))
    for arch in ARCHS:
        for ns, na in ((4, 6), (8, 6), (6, 1), (6, 7)):
            net = M.Critic(ns, na, arch=arch)
            n = npar(net)
            exp = critic_expected(arch, ns, na)
            print('  %-8s (%d,%d)%3s %10d %10d' % (arch, ns, na, '', n, exp))
            ok(n == exp, '%s(%d,%d) 参数量 = %d' % (arch, ns, na, exp))
            got = {name: tuple(getattr(net, name).weight.shape)
                   for name in critic_shapes(arch, ns, na)}
            ok(got == critic_shapes(arch, ns, na),
               '%s(%d,%d) 层形状正确（%s）' % (arch, ns, na, got))
        # 宽度与 --hidden1/--hidden2 无关
        na_ = npar(M.Critic(4, 6, hidden1=16, hidden2=100, arch=arch))
        nb_ = npar(M.Critic(4, 6, hidden1=512, hidden2=512, arch=arch))
        ok(na_ == nb_ == critic_expected(arch, 4, 6),
           '%s 宽度与 --hidden1/--hidden2 无关（%d）' % (arch, na_))
        # feat_power：动作输入 +1 维
        nf = npar(M.Critic(4, 6, arch=arch, feat_power=True))
        ok(nf == critic_expected(arch, 4, 6, feat=True),
           '%s + feat_power 首层动作输入 = na+1（%d 参数）' % (arch, nf))
        # 结构键互斥（供 infer_arch 唯一识别）
        if arch == 'v7.1':
            net = M.Critic(4, 6, arch=arch)
            ok(hasattr(net, 'h1') and not hasattr(net, 's1') and not hasattr(net, 'fc1'),
               'v7.1 只有 h1/h2/h3/out（无 s1/fc1）')
        else:
            net = M.Critic(4, 6, arch=arch)
            ok(hasattr(net, 's1') and hasattr(net, 'a1') and not hasattr(net, 'h1'),
               '%s 是双流 s1/a1（无 h1）' % arch)
    # 别名 7.1/7.2/7.3
    for arch in ARCHS:
        alias = arch[1:]
        net = M.Critic(4, 6, arch=alias)
        ok(net.arch == arch, 'arch=%s 归一成 %s' % (alias, arch))


def section_b():
    print('\n[B] 前向 / 反向 / 通路独立性')
    ns, na = 4, 6
    for arch in ARCHS:
        torch.manual_seed(0)
        net = M.Critic(ns, na, arch=arch)
        x, a = batch(ns, na, 64, seed=1)
        q = net([x, a])
        ok(tuple(q.shape) == (64, 1), '%s 前向形状 = (N,1)' % arch)
        ok(bool(torch.isfinite(q).all()), '%s 前向输出有限' % arch)
        target = torch.randn_like(q)
        ((q - target) ** 2).mean().backward()
        bad = [n for n, p in net.named_parameters()
               if p.grad is None or not bool(torch.isfinite(p.grad).all())]
        ok(not bad, '%s 全部参数梯度存在且有限（坏层: %s）' % (arch, bad or '无'))
        zero = [n for n, p in net.named_parameters() if float(p.grad.abs().max()) == 0.0]
        ok(not zero, '%s 全部参数梯度非严格 0（恒零层: %s）' % (arch, zero or '无'))
        # ∂Q/∂a 非零
        a2 = a.clone().requires_grad_(True)
        net.zero_grad(set_to_none=True)
        g, = torch.autograd.grad(net([x, a2]).sum(), a2)
        ok(float(g.abs().mean()) > 0.0,
           '%s ∂Q/∂a 非零（mean|g|=%.3e）' % (arch, float(g.abs().mean())))
        # 动作通路：置零对应入口 ⇒ Q 与 a 严格无关
        probe = M.Critic(ns, na, arch=arch)
        with torch.no_grad():
            if arch == 'v7.1':
                probe.h1.weight[:, ns:].zero_()
            else:
                probe.a1.weight.zero_()
        qa = probe([x, a]); qb = probe([x, a + 0.3])
        ok(bool(torch.equal(qa.detach(), qb.detach())),
           '%s 动作入口置零后 Q 与 a 严格无关（|ΔQ|=%.3e）'
           % (arch, float((qa - qb).detach().abs().max())))
        # 状态通路
        probe2 = M.Critic(ns, na, arch=arch)
        with torch.no_grad():
            if arch == 'v7.1':
                probe2.h1.weight[:, :ns].zero_()
            else:
                probe2.s1.weight.zero_()
        qx = probe2([x, a]); qy = probe2([x + 0.5, a])
        ok(bool(torch.equal(qx, qy)),
           '%s 状态入口置零后 Q 与 state 严格无关' % arch)
        # 初始尺度：不塌陷、不爆炸
        net.eval()
        with torch.no_grad():
            q = net([x, a])
        qsd = float(q.std())
        ok(np.isfinite(qsd) and 1e-6 < qsd < 100.0,
           '%s 初始 Q 的 sd 有限且不爆炸（%.4f）' % (arch, qsd))


def section_c():
    print('\n[C] infer_arch / feat_from_sd / build_critic_from_sd 往返')
    cases = [
        ('v1', M.Critic(4, 6, hidden1=16, hidden2=100, arch='v1')),
        ('v1af', M.Critic(4, 6, hidden1=16, hidden2=100, arch='v1af')),
        ('v2', M.Critic(4, 6, hidden1=16, hidden2=100, arch='v2')),
        ('v3', M.Critic(4, 6, hidden1=16, hidden2=100, arch='v3')),
        ('v4', M.Critic(4, 6, hidden1=16, hidden2=100, arch='v4')),
        ('v5', M.Critic(4, 6, hidden1=16, hidden2=100, arch='v5')),
        ('v6', M.Critic(4, 6, arch='v6')),
        ('v6old', M.Critic(4, 6, arch='v6old')),
        ('v7', M.Critic(4, 6, arch='v7')),
        ('v7.1', M.Critic(4, 6, arch='v7.1')),
        ('v7.2', M.Critic(4, 6, arch='v7.2')),
        ('v7.3', M.Critic(4, 6, arch='v7.3')),
    ]
    for name, net in cases:
        ok(M.infer_arch(net.state_dict(), 4, 6) == name,
           'infer_arch(%s) = %s' % (name, name))
    for arch in ARCHS:
        # L 层（na=1）也要认得出
        ok(M.infer_arch(M.Critic(5, 1, arch=arch).state_dict(), 5, 1) == arch,
           'infer_arch(%s L 层 ns=5, na=1) = %s' % (arch, arch))
        # feat_power 往返
        nf = M.Critic(4, 6, arch=arch, feat_power=True)
        ok(M.feat_from_sd(nf.state_dict(), 4, 6) is True,
           'feat_from_sd(%s + feat) = True' % arch)
        ok(M.feat_from_sd(M.Critic(4, 6, arch=arch).state_dict(), 4, 6) is False,
           'feat_from_sd(%s 无 feat) = False' % arch)
        built = M.build_critic_from_sd(nf.state_dict(), 4, 6)
        ok(built.arch == arch, 'build_critic_from_sd 重建出 %s' % arch)
        x, a = batch(4, 6, 32, seed=3)
        nf.eval(); built.eval()
        with torch.no_grad():
            ok(bool(torch.equal(nf([x, a]), built([x, a]))),
               '重建后的 %s 前向逐位一致' % arch)


def section_d():
    print('\n[D] actor 规格（v7.1：500-400-500；v7.2/3：128 编码器 + 主干）')
    for arch in ARCHS:
        h1, h2, pre = M.actor_spec(arch)
        exp = (500, 400, None) if arch == 'v7.1' else (
            (400, 300, 128) if arch == 'v7.2' else (500, 400, 128))
        ok((h1, h2, pre) == exp, 'actor_spec(%s) = %s' % (arch, exp))
        act = M.Actor(4, 6, hidden1=h1, hidden2=h2, pre_hidden=pre)
        ok(npar(act) == actor_expected(arch, 4, 6),
           '%s actor 参数量 = %d' % (arch, actor_expected(arch, 4, 6)))
        got = {name: tuple(getattr(act, name).weight.shape)
               for name in actor_shapes(arch, 4, 6)}
        ok(got == actor_shapes(arch, 4, 6),
           '%s actor 层形状正确（%s）' % (arch, got))
        if pre:
            ok(act.fc0 is not None, '%s actor 有 fc0 state 编码器' % arch)
        else:
            ok(act.fc0 is None, '%s actor 没有 fc0（历史三层形状）' % arch)
    # actor 输出层置零（hsrl_agent_init 的 fc4 口径）
    act = M.Actor(4, 6, hidden1=500, hidden2=400)
    holder = types.SimpleNamespace(actor=act, actor_target=M.Actor(4, 6, hidden1=500,
                                                                  hidden2=400))
    done = hinit.zero_actor_output(holder, verbose=False)
    ok(done == ['actor', 'actor_target'], 'zero_actor_output 认得 fc4（%s）' % done)
    ok(float(act.fc4.weight.abs().max()) == 0.0
       and float(act.fc4.bias.abs().max()) == 0.0, 'fc4 权重与 bias 都被置零')
    # critic 输出头
    for arch in ARCHS:
        net = M.Critic(4, 6, arch=arch)
        ok(hinit._out_head(net) is net.out, '_out_head(%s) = out' % arch)


def _args(arch, **kw):
    base = dict(
        seed=0, cuda=False, hidden1=16, hidden2=100, init_w=3e-3,
        prate=1e-4, rate=3e-3, rmsize=256, bsize=16, tau=1e-3, discount=0.98,
        epsilon=4000, lam=1.0, lr_lam=0.0, cost_single_limit=1000.0,
        PER_alpha=0.6, PER_beta=0.4, PER_epsilon=0.01,
        beta_increment_per_sampling=0.0, nb_turbine=5,
        ou_theta=0.15, ou_mu=0.0, ou_sigma=0.2, max_episode_length=1440,
        critic_arch=arch, critic_act='relu', critic_out_init='3e3',
        critic_feat_power=0, critic_feat_power_k=1.357e-3, cost_clip=1,
        farm_level=0, scalar_wt=0, na_box=5,
        aux_model=0, critic_action_first=0, critic_bias_init='none',
        warmup_wt_sched='lhs',
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def section_e():
    print('\n[E] aifflor SP_TD3 端到端（随机数据，无 PER；不推进仿真）')
    from aifflor.aifflor_ddpg import SP_TD3

    names = ('critic_p', 'critic_target_p', 'critic_p2', 'critic_target_p2',
             'critic_c', 'critic_target_c', 'critic_c2', 'critic_target_c2')
    for arch in ARCHS:
        torch.manual_seed(0)
        args = _args(arch)
        ag = SP_TD3(nb_states=4, nb_actions=6, args=args, actor_arch=arch)
        ok(all(getattr(ag, n).arch == arch for n in names),
           '%s: 8 个 critic/target 全部是该结构' % arch)
        ok(npar(ag.critic_p) == critic_expected(arch, 4, 6),
           '%s: critic_p 参数 = %d' % (arch, critic_expected(arch, 4, 6)))
        shp = {name: tuple(getattr(ag.actor, name).weight.shape)
               for name in actor_shapes(arch, 4, 6)}
        ok(shp == actor_shapes(arch, 4, 6),
           '%s: H actor 层形状正确（%s）' % (arch, shp))
        ok(npar(ag.actor) == actor_expected(arch, 4, 6),
           '%s: H actor 参数 = %d' % (arch, actor_expected(arch, 4, 6)))
        # L 层不放大：actor 仍 --hidden1/--hidden2
        agL = SP_TD3(nb_states=6, nb_actions=1, args=_args(arch))
        ok(npar(agL.critic_p) == critic_expected(arch, 6, 1),
           '%s: L critic 参数 = %d' % (arch, critic_expected(arch, 6, 1)))
        ok(tuple(agL.actor.fc1.weight.shape) == (16, 6)
           and npar(agL.actor) == 3445
           and getattr(agL.actor, 'fc0', None) is None,
           '%s: L actor 仍 16/100（不被放大）' % arch)
        # select_action
        ag.is_training = True
        ag.epsilon = 0.0
        s = np.random.RandomState(0).rand(4).astype(np.float32)
        a = ag.select_action(s)
        ok(np.asarray(a).shape == (6,), '%s: select_action 返回 (6,)' % arch)

    # checkpoint 往返（每个 arch 都存/读；v7.2 额外走评测侧 actor_pre 路径）
    import shutil
    d = os.path.join(_HERE, '_tmp_v7x_ckpt')
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    try:
        for arch in ARCHS:
            ag = SP_TD3(nb_states=4, nb_actions=6, args=_args(arch), actor_arch=arch)
            ag.save_model(d)
            ag2 = SP_TD3(nb_states=4, nb_actions=6, args=_args(arch), actor_arch=arch)
            ag2.load_weights(d)
            same_actor = all(bool(torch.equal(getattr(ag.actor, n).weight,
                                              getattr(ag2.actor, n).weight))
                             for n in actor_shapes(arch, 4, 6))
            ok(same_actor, '%s: actor checkpoint 能存能读（含 fc0 时逐位一致）' % arch)
            ok(bool(torch.equal(ag.critic_p.out.weight, ag2.critic_p.out.weight)),
               '%s: critic_p checkpoint 能存能读' % arch)
        # 评测路径：只给 args.actor_pre，不传 actor_arch
        arch = 'v7.2'
        ag = SP_TD3(nb_states=4, nb_actions=6, args=_args(arch), actor_arch=arch)
        ag.save_model(d)
        ev = _args(arch, hidden1=400, hidden2=300, actor_pre=128)
        ag_ev = SP_TD3(nb_states=4, nb_actions=6, args=ev)   # actor_arch=None
        ag_ev.load_weights(d)
        ok(tuple(ag_ev.actor.fc0.weight.shape) == (128, 4),
           '评测侧 actor_pre=128 能恢复 v7.2 的 fc0 编码器')
        ok(bool(torch.equal(ag_ev.actor.fc4.weight, ag.actor.fc4.weight)),
           '评测侧恢复后 actor 输出层逐位一致')
        # L 层不能被同一个 args.actor_pre 误加编码层
        agL_ev = SP_TD3(nb_states=5, nb_actions=1, args=ev)
        ok(getattr(agL_ev.actor, 'fc0', None) is None,
           '评测侧 actor_pre 只作用于 H，L actor 仍无 fc0')
    finally:
        shutil.rmtree(d, ignore_errors=True)


def section_f():
    print('\n[F] CLI / 评测接线（静态 AST / 源码，不 import aifflor_main）')
    path = os.path.join(ROOT, 'aifflor', 'aifflor_main.py')
    with open(path, encoding='utf-8') as f:
        src = f.read()
    tree = ast.parse(src, filename=path)
    arg = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument' and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == '--critic_arch'):
            arg = node
            break
    ok(arg is not None, 'aifflor_main 注册了 --critic_arch')
    if arg is not None:
        kw = {k.arg: k.value for k in arg.keywords}
        choices = [e.value for e in kw['choices'].elts]
        default = kw['default'].value
        type_src = ast.unparse(kw['type'])
        ok('v7.1' in choices and 'v7.2' in choices and 'v7.3' in choices,
           '--critic_arch choices 含 v7.1/v7.2/v7.3（%s）' % choices)
        ok(default == 'v7', '--critic_arch 默认仍 = v7')
        ok(type_src == '_normalize_arch', '--critic_arch 用 _normalize_arch 接受 7.x 别名（%s）'
           % type_src)
    ok('_actor_spec(_arch_eff)' in src, 'main 用 actor_spec 选 actor 宽度')
    ok('actor_arch=(_arch_eff if _aspec is not None else None)' in src,
       'main 只给 H 层传 actor_arch（L 层不放大）')
    with open(os.path.join(ROOT, 'aifflor', 'aifflor_ddpg.py'), encoding='utf-8') as f:
        dsrc = f.read()
    ok('actor_spec' in dsrc and 'args.actor_pre' in dsrc,
       'aifflor_ddpg 用 actor_spec 并按 args.actor_pre 恢复评测/旧 checkpoint')
    with open(os.path.join(ROOT, 'test_five_days.py'), encoding='utf-8') as f:
        tsrc = f.read()
    ok('def _detect_actor_pre' in tsrc and '_detect_actor_pre(model_dir, 6)' in tsrc,
       'test_five_days 从 actor_*.pkl 的 fc0 探测 v7.2/v7.3 编码器')
    ok("('fc0.weight', 'fc1.weight', '0.weight')" in tsrc,
       'test_five_days 的 _detect_nb_states 先读 fc0（否则会把 128 当观测维）')
    ok('ns.actor_pre' in tsrc and 'ns.actor_pre = int(actor_pre)' in tsrc,
       'test_five_days 把 actor_pre 塞进 args')
    ok('_detect_hidden(model_dir, 1)' in tsrc and 'l_args = deepcopy(args)' in tsrc,
       'test_five_days 按 actor_1.pkl 单独构造 L（不用 H 的宽度/编码器）')
    # infer_arch 的文档/识别键
    ok(M.infer_arch(M.Critic(4, 6, arch='v7.1').state_dict(), 4, 6) == 'v7.1',
       'infer_arch 不把 v7.1 认成 v7')
    ok(M.infer_arch(M.Critic(4, 6, arch='v7.2').state_dict(), 4, 6) == 'v7.2',
       'infer_arch 不把 v7.2 认成 v6old/v4')
    ok(M.infer_arch(M.Critic(4, 6, arch='v7.3').state_dict(), 4, 6) == 'v7.3',
       'infer_arch 不把 v7.3 认成 v7.2')


def section_g():
    print('\n[G] 评测侧 checkpoint 探测（test_five_days 的纯读取函数）')
    import shutil
    import test_five_days as T

    d = os.path.join(_HERE, '_tmp_v7x_eval')
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    try:
        for arch in ARCHS:
            h1, h2, pre = M.actor_spec(arch)
            act = M.Actor(4, 6, hidden1=h1, hidden2=h2, pre_hidden=pre)
            torch.save(act.state_dict(), os.path.join(d, 'actor_6.pkl'))
            torch.save(M.Critic(4, 6, arch=arch).state_dict(),
                       os.path.join(d, 'critic_p_6.pkl'))
            ok(T._detect_nb_states(d, 6) == 4,
               '%s: _detect_nb_states 从 fc0/fc1 读出观测维 4' % arch)
            ok(T._detect_hidden(d, 6) == (h1, h2),
               '%s: _detect_hidden = %s' % (arch, (h1, h2)))
            ok(T._detect_actor_pre(d, 6) == pre,
               '%s: _detect_actor_pre = %s' % (arch, pre))
            ok(T._detect_critic_arch(d, 6, 4) == arch,
               '%s: _detect_critic_arch = %s' % (arch, arch))
        # L 层：16/100、无 fc0
        torch.save(M.Actor(6, 1, hidden1=16, hidden2=100).state_dict(),
                   os.path.join(d, 'actor_1.pkl'))
        ok(T._detect_hidden(d, 1) == (16, 100), 'L: _detect_hidden = (16,100)')
        ok(T._detect_actor_pre(d, 1) is None, 'L: 没有 fc0')
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    print('=' * 96)
    print('v7.1 / v7.2 / v7.3 critic-actor 结构契约')
    print('  v7.1: cat(s,a) 500-400-500      + actor s-500-400-500-a')
    print('  v7.2: (s-128,a-128)-cat256-400-300-400 + actor s-128-400-300-400-a')
    print('  v7.3: (s-128,a-128)-cat256-500-400-500 + actor s-128-500-400-500-a')
    print('=' * 96)
    print('torch %s | cuda %s | V71_DIMS=%s V72=%s V73=%s'
          % (torch.__version__, torch.cuda.is_available(),
             M.V71_DIMS, M.V72_DIMS, M.V73_DIMS))
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
    section_f()
    section_g()
    print('\n' + '=' * 96)
    print('结果：%d 项通过 / %d 项失败' % (N_OK, N_BAD))
    print('=' * 96)
    return 0 if N_BAD == 0 else 1


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
