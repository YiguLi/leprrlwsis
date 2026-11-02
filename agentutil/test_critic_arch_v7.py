#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""v7 critic 结构契约 + 与 v1/v1af/v6 的静态对照（**不训练、不推进仿真**）。

v7 口径（2026-11-02，用户指定）::

    input = cat(state, action)  →  400 → 300 → 400 → 1

与 v1af 的关系：同一"动作首层进网"动机，但宽度是显式规格 `model.V7_DIMS`，
且不像 v1af 那样把参数全压在首层（v1af 默认 hidden1=16 ⇒ 4·16=64→300→80→50→1；
hidden1=100 才是 400→300→80→50→1）。

段：
  A 接线/参数量：各 (ns,na) 下 v7 的层表/参数量；`--hidden1/--hidden2` 不生效
  B 前向/反向：输出形状 (N,1)、有限；全参数梯度存在且有限；**动作只在第一层进入**
    （把 h1 的动作列置零 ⇒ Q 与 a 严格无关）
  C 初始化尺度：各层激活 sd / 死单元比例 / 初始 Q 的 sd（v1/v1af/v6/v7 对照）
  D `infer_arch` / `feat_from_sd` 往返：v7 与 v1/v1af/v2/v4/v5/v6 互不误判
  E `hsrl_agent_init._out_head(v7) = out`；`init_critic_bias` 真能落地
  F 前向耗时（batch 256 × 200 次，CPU）

用法::

    python agentutil/test_critic_arch_v7.py
"""
import os
import sys
import time
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


def v7_expected(ns, na, feat=False):
    """input → 400 → 300 → 400 → 1 的参数量（含 bias）。"""
    a = na + (1 if feat else 0)
    return ((ns + a) * 400 + 400) + (400 * 300 + 300) + (300 * 400 + 400) + (400 + 1)


def realistic_batch(ns, na, n=2000, seed=0):
    """aifflor 量级的输入：state ~ U(0,1)（已归一化），action 前 5 维 ~ U(0,0.5)。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, ns, generator=g)
    a = torch.rand(n, na, generator=g) * 0.5
    return x, a


def hidden_stats(net, xs, layers=('h1', 'h2', 'h3')):
    """返回 (逐层激活 sd, 逐层死单元比例) + 输出 sd；仅对 relu 全连接栈有意义。

    * v7：z = cat(x, a) 直接进 h1；
    * v6：先走两条支路 s1/s2、a1/a2，再进 merge 栈 m1/m2/m3（layers 里传 m1/m2/m3）。
    """
    net.eval()
    with torch.no_grad():
        if layers and layers[0].startswith('m'):
            hs = net.relu(net.s2(net.relu(net.s1(xs[0]))))
            ha = net.relu(net.a2(net.relu(net.a1(xs[1]))))
            z = torch.cat([hs, ha], 1)
        else:
            z = torch.cat([xs[0], xs[1]], 1)
        stats = []
        for name in layers:
            z = net.relu(getattr(net, name)(z))
            stats.append((float(z.std()), float((z <= 0).all(0).float().mean())))
        q = net.out(z)
    return stats, float(q.std())


# ---------------------------------------------------------------------------
def section_a():
    print('\n[A] 接线 / 参数量（v7 宽度写死在 V7_DIMS，隐藏层 h1/h2/h3 + 输出 out）')
    print('  %-10s %10s %10s %14s' % ('(ns,na)', '参数量', '理论值', 'h1 输入维'))
    for ns, na in ((4, 6), (8, 6), (5, 1), (6, 7)):
        net = M.Critic(ns, na, arch='v7')
        n = npar(net)
        exp = v7_expected(ns, na)
        print('  (%d,%d)%5s %10d %10d %14d'
              % (ns, na, '', n, exp, net.h1.in_features))
        ok(n == exp, 'v7(%d,%d) 参数量 = %d（input→400→300→400→1）' % (ns, na, exp))
        ok(net.h1.in_features == ns + na, 'v7(%d,%d) 首层输入 = ns+na' % (ns, na))
        ok(tuple(net.h2.weight.shape) == (300, 400)
           and tuple(net.h3.weight.shape) == (400, 300)
           and tuple(net.out.weight.shape) == (1, 400),
           'v7(%d,%d) 层形状 = 400→300→400→1' % (ns, na))
        ok(not hasattr(net, 'fc1') and not hasattr(net, 's1'),
           'v7(%d,%d) 没有 fc1/s1（判据唯一，供 infer_arch 用）' % (ns, na))
    # hidden1/hidden2 不生效（与 v6 同口径）
    n_a = npar(M.Critic(4, 6, hidden1=16, hidden2=100, arch='v7'))
    n_b = npar(M.Critic(4, 6, hidden1=512, hidden2=512, arch='v7'))
    ok(n_a == n_b == v7_expected(4, 6),
       'v7 宽度与 --hidden1/--hidden2 无关（16/100 与 512/512 都是 %d）' % n_a)
    # feat_power：动作输入 +1 维
    nf = npar(M.Critic(4, 6, arch='v7', feat_power=True))
    ok(nf == v7_expected(4, 6, feat=True), 'v7 + feat_power 首层输入 = ns+na+1（%d 参数）' % nf)


def section_b():
    print('\n[B] 前向 / 反向 / 动作入口')
    ns, na = 4, 6
    torch.manual_seed(0)
    net = M.Critic(ns, na, arch='v7')
    x, a = realistic_batch(ns, na, 64, seed=1)
    q = net([x, a])
    ok(tuple(q.shape) == (64, 1), '前向形状 = (N,1)')
    ok(bool(torch.isfinite(q).all()), '前向输出有限')
    # 反向：每个参数都有有限非零梯度
    target = torch.randn_like(q)
    ((q - target) ** 2).mean().backward()
    bad = [n for n, p in net.named_parameters()
           if p.grad is None or not bool(torch.isfinite(p.grad).all())]
    ok(not bad, '全部参数梯度存在且有限（坏层: %s）' % (bad or '无'))
    zero = [n for n, p in net.named_parameters() if float(p.grad.abs().max()) == 0.0]
    ok(not zero, '全部参数梯度非严格 0（恒零层: %s）' % (zero or '无'))
    # ∂Q/∂a 非零（动作通路存在）
    a2 = a.clone().requires_grad_(True)
    net.zero_grad(set_to_none=True)
    g, = torch.autograd.grad(net([x, a2]).sum(), a2)
    ok(float(g.abs().mean()) > 0.0, '∂Q/∂a 非零（mean|g|=%.3e）' % float(g.abs().mean()))
    # **动作只在第一层进入**：置零 h1 的动作列 ⇒ Q 与 a 严格无关
    probe = M.Critic(ns, na, arch='v7')
    with torch.no_grad():
        probe.h1.weight[:, ns:].zero_()
    qa = probe([x, a]); qb = probe([x, a + 0.3])
    ok(bool(torch.equal(qa.detach(), qb.detach())),
       'h1 动作列置零后 Q 与 a 严格无关 ⇒ 动作只在首层进入（|ΔQ|=%.3e）'
       % float((qa - qb).detach().abs().max()))
    # 状态通路同理（置零 h1 的状态列 ⇒ Q 与 state 无关）
    probe2 = M.Critic(ns, na, arch='v7')
    with torch.no_grad():
        probe2.h1.weight[:, :ns].zero_()
    qx = probe2([x, a]); qy = probe2([x + 0.5, a])
    ok(bool(torch.equal(qx, qy)), 'h1 状态列置零后 Q 与 state 严格无关（接线对称）')


def section_c():
    print('\n[C] 初始化尺度（ns=4, na=6；state~U(0,1), a~U(0,0.5)；2000 样本）')
    print('  %-16s %8s %8s %-34s %9s'
          % ('结构', '参数', 'Q sd', '逐层 sd / 死单元%', 'Q 均值'))
    x, a = realistic_batch(4, 6, 2000, seed=0)
    rows = [
        ('v1 (16/100)', M.Critic(4, 6, hidden1=16, hidden2=100, arch='v1'), ()),
        ('v1af (16/100)', M.Critic(4, 6, hidden1=16, hidden2=100, arch='v1af'), ()),
        ('v1af (100/300)', M.Critic(4, 6, hidden1=100, hidden2=300, arch='v1af'), ()),
        ('v6', M.Critic(4, 6, arch='v6'), ('m1', 'm2', 'm3', 'm4')),
        ('v7', M.Critic(4, 6, arch='v7'), ('h1', 'h2', 'h3')),
    ]
    for name, net, layers in rows:
        net.eval()
        with torch.no_grad():
            q = net([x, a])
        if layers:
            st, qsd = hidden_stats(net, (x, a), layers)
            cell = ' '.join('%s %.4f/%.0f%%' % (n, sd, 100 * dead)
                            for n, (sd, dead) in zip(layers, st))
        else:
            st, qsd = [], float(q.std())
            cell = '-'
        print('  %-16s %8d %8.4f %-34s %9.2f'
              % (name, npar(net), qsd, cell, float(q.mean())))
        ok(np.isfinite(qsd), '%s 初始 Q 的 sd 有限' % name)
    # v7 的初始激活不应塌陷（relu 全死）或爆炸
    v7 = M.Critic(4, 6, arch='v7')
    st, qsd = hidden_stats(v7, (x, a))
    ok(all(sd > 1e-3 for sd, _ in st), 'v7 初始各层激活 sd > 1e-3（无塌陷）')
    ok(all(dead < 0.95 for _, dead in st), 'v7 初始各层死单元 < 95%（无整层死）')
    ok(1e-6 < qsd < 10.0, 'v7 初始 Q 的 sd 在 (1e-6, 10) 内（%0.4f）' % qsd)


def section_d():
    print('\n[D] infer_arch / feat_from_sd 往返（v7 不得与其它结构混淆）')
    cases = [
        ('v1', M.Critic(4, 6, hidden1=16, hidden2=100, arch='v1')),
        ('v1af', M.Critic(4, 6, hidden1=16, hidden2=100, arch='v1af')),
        ('v2', M.Critic(4, 6, hidden1=16, hidden2=100, arch='v2')),
        ('v3', M.Critic(4, 6, hidden1=16, hidden2=100, arch='v3')),
        ('v4', M.Critic(4, 6, hidden1=16, hidden2=100, arch='v4')),
        ('v5', M.Critic(4, 6, hidden1=16, hidden2=100, arch='v5')),
        ('v6', M.Critic(4, 6, arch='v6')),
        ('v7', M.Critic(4, 6, arch='v7')),
    ]
    for name, net in cases:
        sd = net.state_dict()
        got = M.infer_arch(sd, 4, 6)
        ok(got == name, 'infer_arch(%s sd) = %s' % (name, got))
    # L 层（na=1）也要认得出 v7
    ok(M.infer_arch(M.Critic(5, 1, arch='v7').state_dict(), 5, 1) == 'v7',
       'infer_arch(v7 L 层 ns=5, na=1) = v7')
    # feat_power 往返
    nf = M.Critic(4, 6, arch='v7', feat_power=True)
    ok(M.feat_from_sd(nf.state_dict(), 4, 6) is True, 'feat_from_sd(v7+feat) = True')
    ok(M.feat_from_sd(M.Critic(4, 6, arch='v7').state_dict(), 4, 6) is False,
       'feat_from_sd(v7 无 feat) = False')
    # build_critic_from_sd：v7 sd 自动重建并逐位还原
    built = M.build_critic_from_sd(nf.state_dict(), 4, 6)
    ok(built.arch == 'v7', 'build_critic_from_sd 重建出 v7')
    x, a = realistic_batch(4, 6, 32, seed=3)
    nf.eval(); built.eval()
    with torch.no_grad():
        ok(bool(torch.equal(nf([x, a]), built([x, a]))), '重建后的 v7 前向逐位一致')


def section_e():
    print('\n[E] 输出层 / bias 初始化接线')
    net = M.Critic(4, 6, arch='v7')
    ok(hinit._out_head(net) is net.out, '_out_head(v7) = out（不是 None）')
    holder = types.SimpleNamespace(
        critic_p=net, critic_target_p=M.Critic(4, 6, arch='v7'),
        critic_c=M.Critic(4, 6, arch='v7'), critic_target_c=M.Critic(4, 6, arch='v7'))
    done = hinit.init_critic_bias(holder, b_p=1.25, b_c=4.5, verbose=False)
    ok(len(done) == 4, 'init_critic_bias 覆盖 4 个网络（%s）' % done)
    ok(abs(float(holder.critic_p.out.bias) - 1.25) < 1e-6
       and abs(float(holder.critic_c.out.bias) - 4.5) < 1e-6,
       'bias 真的写进 out.bias（p=1.25, c=4.5）')
    ok(abs(float(holder.critic_target_p.out.bias) - 1.25) < 1e-6,
       'target 网络同步写入')


def section_f():
    print('\n[F] 前向耗时（CPU，batch=256，200 次，no_grad）')
    x, a = realistic_batch(4, 6, 256, seed=0)
    rows = [('v1 (16/100)', M.Critic(4, 6, hidden1=16, hidden2=100, arch='v1')),
            ('v1af (16/100)', M.Critic(4, 6, hidden1=16, hidden2=100, arch='v1af')),
            ('v6', M.Critic(4, 6, arch='v6')),
            ('v7', M.Critic(4, 6, arch='v7'))]
    base = None
    for name, net in rows:
        net.eval()
        with torch.no_grad():
            for _ in range(20):
                net([x, a])
            t0 = time.perf_counter()
            for _ in range(200):
                net([x, a])
            dt = (time.perf_counter() - t0) / 200 * 1e3
        base = dt if base is None else base
        print('  %-16s %8.3f ms/次  %5.2f× v1  %8d 参数'
              % (name, dt, dt / base, npar(net)))
        ok(np.isfinite(dt) and dt > 0, '%s 单次前向耗时有限' % name)


def section_g():
    """静态接线（不 import aifflor_main：它 import 期会拉起 FLORIS/MATLAB 依赖链）。"""
    print('\n[G] CLI / agent 接线（静态 AST / 源码检查，不 import aifflor_main）')
    import ast
    path = os.path.join(ROOT, 'aifflor', 'aifflor_main.py')
    with open(path, encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=path)
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
        ok('v7' in choices, '--critic_arch 的 choices 含 v7（%s）' % choices)
        ok(default == 'v6', '--critic_arch 默认仍是 v6（不改生产口径）')
    with open(os.path.join(ROOT, 'aifflor', 'aifflor_ddpg.py'), encoding='utf-8') as f:
        dsrc = f.read()
    ok("critic_cfg['arch']" in dsrc and "getattr(args, 'critic_arch'" in dsrc,
       'aifflor_ddpg 把 args.critic_arch 透传给 4 个 critic（+target）')
    ok("'v1af' if int(getattr(args, 'critic_action_first'" in dsrc,
       '--critic_action_first 仍优先映射到 v1af（不被 v7 抢走）')
    with open(os.path.join(ROOT, 'agentutil', 'critictest', 'train_c1.py'),
              encoding='utf-8') as f:
        tsrc = f.read()
    ok("'v7'" in tsrc, '离线 C1 基准 train_c1.py 支持 --arch v7')


def section_h():
    """aifflor 的真实 agent 接线：SP_TD3(critic_arch='v7') 构造 / 动作 / 存取往返。

    只用随机数据（不构建环境、不推进 FLORIS），与 `test_aifflor_scalar_wt.py` 的
    "轻量 args" 同口径。
    """
    print('\n[H] aifflor SP_TD3(critic_arch=v7) 端到端（随机数据，无 PER）')
    from types import SimpleNamespace
    from aifflor.aifflor_ddpg import SP_TD3

    args = SimpleNamespace(
        seed=0, cuda=False, hidden1=16, hidden2=100, init_w=3e-3,
        prate=1e-4, rate=3e-3, rmsize=256, bsize=16, tau=1e-3, discount=0.98,
        epsilon=4000, lam=1.0, lr_lam=0.0, cost_single_limit=1000.0,
        PER_alpha=0.6, PER_beta=0.4, PER_epsilon=0.01,
        beta_increment_per_sampling=0.0, nb_turbine=5,
        ou_theta=0.15, ou_mu=0.0, ou_sigma=0.2, max_episode_length=1440,
        critic_arch='v7', critic_act='relu', critic_out_init='3e3',
        critic_feat_power=0, critic_feat_power_k=1.357e-3, cost_clip=1,
        farm_level=0, scalar_wt=0, na_box=5,
        aux_model=0, critic_action_first=0, critic_bias_init='none',
        warmup_wt_sched='lhs',
    )
    torch.manual_seed(0)
    ag = SP_TD3(nb_states=4, nb_actions=6, args=args)
    names = ('critic_p', 'critic_target_p', 'critic_p2', 'critic_target_p2',
             'critic_c', 'critic_target_c', 'critic_c2', 'critic_target_c2')
    ok(all(getattr(ag, n).arch == 'v7' for n in names),
       '8 个 critic/target 全部是 v7')
    ok(npar(ag.critic_p) == 245501, 'critic_p 参数 = 245501（ns=4, na=6）')
    # 注：这里**不调用 update_policy()** —— 它要经过 PER_buffer 的 `np.array(obs, copy=False)`，
    #   该写法在 NumPy 2.x 下对"需要拷贝"的输入会抛错（与本结构无关的既有问题）。
    #   改为直接对 agent 里的 critic 做 1 步 Adam，验证优化器/反向真的接通。
    ag.is_training = True
    ag.epsilon = 0.0
    rng = np.random.RandomState(0)
    s = rng.rand(4).astype(np.float32)
    a = ag.select_action(s)
    ok(np.asarray(a).shape == (6,), 'select_action 返回 (6,) 动作')
    with torch.no_grad():
        det = ag._act(ag.actor, torch.as_tensor(s[None, :])).numpy().ravel()
    ok(det.shape == (6,), '_act（critic 评估点）形状 (6,)')
    xb = torch.rand(32, 4)
    ab = torch.rand(32, 6)
    w0 = ag.critic_p.h1.weight.detach().clone()
    opt = torch.optim.Adam(ag.critic_p.parameters(), lr=1e-3)
    loss = ((ag.critic_p([xb, ab]) - torch.rand(32, 1)) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    ok(not bool(torch.equal(w0, ag.critic_p.h1.weight)),
       'agent 内的 v7 critic_p 能被 Adam 更新（h1 权重变化）')
    with torch.no_grad():
        qp = ag.critic_p([xb, ab])
        qc = ag.critic_c([xb, ab])
    ok(bool(torch.isfinite(qp).all()) and bool(torch.isfinite(qc).all()),
       'profit/cost critic 输出有限')
    bad_net = [n for n in names
               if not all(bool(torch.isfinite(p).all()) for p in getattr(ag, n).parameters())]
    ok(not bad_net, '8 个网络参数全部有限（坏网络: %s）' % (bad_net or '无'))
    # save_model / load_weights 往返（v7 checkpoint 能存能读）
    # ⚠ 不用 tempfile.TemporaryDirectory：DSH 沙箱里系统临时目录的清理会 PermissionError，
    #   改用工作区内的临时目录（用完即删）。
    import shutil
    d = os.path.join(_HERE, '_tmp_v7_ckpt')
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)          # save_model 不会自己建目录
    try:
        ag.save_model(d)          # 只存 4 个 online critic（+actor），target 由 load_weights 同步
        ag2 = SP_TD3(nb_states=4, nb_actions=6, args=args)
        ag2.load_weights(d)
        online = names[::2]       # critic_p / critic_p2 / critic_c / critic_c2
        same = all(bool(torch.equal(getattr(ag, n).out.weight.nan_to_num(),
                                    getattr(ag2, n).out.weight.nan_to_num()))
                   for n in online)
        diff = ['%s:%.3g' % (n, float((getattr(ag, n).out.weight
                                       - getattr(ag2, n).out.weight).abs().max()))
                for n in online
                if not bool(torch.equal(getattr(ag, n).out.weight.nan_to_num(),
                                        getattr(ag2, n).out.weight.nan_to_num()))]
        synced = (bool(torch.equal(ag2.critic_target_p.out.weight,
                                   ag2.critic_p.out.weight))
                  and bool(torch.equal(ag2.critic_target_c.out.weight,
                                       ag2.critic_c.out.weight)))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    ok(same, 'save_model → load_weights 往返：4 个 online critic 权重逐位一致（差异: %s）'
       % (diff or '无'))
    ok(synced, 'load_weights 之后 target 被 hard_update 成 online（与训练口径一致）')


def main():
    print('=' * 96)
    print('v7 critic 结构契约（input=cat(state,action) → 400 → 300 → 400 → 1）')
    print('=' * 96)
    print('torch %s | cuda %s | V7_DIMS=%s'
          % (torch.__version__, torch.cuda.is_available(), M.V7_DIMS))
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
    section_f()
    section_g()
    section_h()
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
