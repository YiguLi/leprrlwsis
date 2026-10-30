# -*- coding: utf-8 -*-
"""`td3per` 的 **warmup7200** 契约测试（纯静态 / 纯算术，**不跑仿真、不需要 MATLAB**）。

口径
----
**warmup = 7200 步 = 1440 个块 = 5 个完整 episode/天**（单一真源
`hsrl_warmup.WARMUP_STEPS`，2026-10-23 由 1440 步改来）。`td3per` 与其余 8 个入口
共用 `hsrl_warmup.resolve_warmup` ⇒ **`--warmup` 不传时就是 7200**。

判据
----
  [A] 常量：`WARMUP_STEPS=7200`、`WARMUP_BSIZE_MULT=0`、7200/5=1440 块、7200/1440=5 天。
  [B] `resolve_warmup` 语义：不传 ⇒ 7200；显式 ⇒ 尊重；`--warmup_bsize_mult` 旧口径
      仍可用；显式优先于 mult；负值报错。
  [C] **td3per 的解析链**：复刻它的 `--warmup`（`default=None` 哨兵）+ `add_warmup_args`
      + `resolve_warmup` ⇒ 7200；并静态断言 main 的接线顺序与 `[warmup7200]` 自检行。
  [D] 缓冲/触发点不变量：1440 条 ≥ bsize(256)（首个 minibatch 不复制同一条）；
      `first_trigger_step(7200)=7204`（≡4 mod 5、≥warmup、差 <5）；`BiasInitTracker`
      在 warmup 期 `update_count!=0` 时**抛错**、=0 时不抛，并在 7204 把 bias 写到
      critic 输出层（v6 的 `out`）。
  [E] 1440 块的 **LHS 覆盖**：五维各 10 层、每层 ≈144 块（143~145）；覆盖 [0,0.5]
      两端；信号维 price_walk 方向（300 上调 / 980 下调）。
  [F] 主循环静态接线：`build_from_args` / `fill_bess_transition` / `check_residual_baseline` /
      `BiasInitTracker.from_args` / `step <= args.warmup`（warmup 动作）/
      `step > args.warmup`（零更新守卫）。

跑法::

    python agentutil/test_td3per_warmup7200.py

任一断言失败 → 退出码 1。
"""
import argparse
import contextlib
import io
import os
import re
import sys

import numpy as np

# ---- 路径自举（可被任意 CWD 调用）----------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import hsrl_warmup as hw                                # noqa: E402
import hsrl_agent_init as hinit                         # noqa: E402
import model as M                                       # noqa: E402

N_OK = 0


def ok(cond, msg):
    global N_OK
    if not cond:
        raise AssertionError('FAIL: ' + msg)
    N_OK += 1


MAIN = os.path.join(ROOT, 'td3per', 'td3per_main.py')
SRC = open(MAIN, encoding='utf-8').read()


# =============================================================================
# [A] 单一真源常量
# =============================================================================
def sec_a_constants():
    print('\n[A] 单一真源常量')
    ok(int(hw.WARMUP_STEPS) == 7200, 'WARMUP_STEPS = 7200 步')
    ok(float(hw.WARMUP_BSIZE_MULT) == 0.0,
       'WARMUP_BSIZE_MULT = 0（不绑 bsize ⇒ 默认走固定 7200）')
    ok(int(hw.WARMUP_STEPS) // 5 == 1440, '7200 步 = 1440 个块（H 每 5 步 1 条 transition）')
    ok(int(hw.WARMUP_STEPS) / 1440.0 == 5.0, '7200 步 = 5 个完整 episode/天')
    ok(abs(float(hw.SOC_START) - 0.25) < 1e-12,
       'episode 起始 SOC 固定 = SOC_START(0.25)（不再 U[0.5,5.5] 随机）')
    ok(np.allclose(hw.WT_PRIOR, (0.2, 0.2, 0.2, 0.2, 1.0 / 3.0)), '先验形状 WT_PRIOR')
    ok(float(hw.LHS_LO) == 0.0 and float(hw.LHS_HI) == 0.5,
       'LHS 覆盖的是完整学习带 [0, 0.5]（不是 [0.05,0.45]）')


# =============================================================================
# [B] resolve_warmup 语义
# =============================================================================
class _A(object):
    def __init__(self, bsize=256, warmup=None, mult=0.0):
        self.bsize, self.warmup, self.warmup_bsize_mult = bsize, warmup, mult


def sec_b_resolve():
    print('\n[B] resolve_warmup 语义')
    a = _A(); hw.resolve_warmup(a, verbose=False)
    ok(int(a.warmup) == 7200, '不传 --warmup ⇒ 7200（≠ bsize×mult）')
    b = _A(warmup=200); hw.resolve_warmup(b, verbose=False)
    ok(int(b.warmup) == 200, '显式 --warmup 200 ⇒ 200（复现历史，优先级最高）')
    c = _A(mult=4.0); hw.resolve_warmup(c, verbose=False)
    ok(int(c.warmup) == 4 * 256, '--warmup_bsize_mult 4 旧口径仍可用 ⇒ 1024')
    c2 = _A(warmup=900, mult=4.0); hw.resolve_warmup(c2, verbose=False)
    ok(int(c2.warmup) == 900, '显式值优先于 mult 口径')
    # bsize 很小/很大都不改变 7200（不挂钩）
    for bs in (32, 256, 1024):
        x = _A(bsize=bs); hw.resolve_warmup(x, verbose=False)
        ok(int(x.warmup) == 7200, 'bsize=%d ⇒ warmup 仍是 7200（不挂钩）' % bs)
    try:
        y = _A(warmup=-1); hw.resolve_warmup(y, verbose=False)
        ok(False, '负 warmup 必须报错')
    except ValueError:
        ok(True, '负 warmup 抛 ValueError')
    # 打印口径（verbose=True 时出现步数/块数与"零网络更新"）
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        hw.resolve_warmup(_A(), verbose=True)
    out = buf.getvalue()
    ok('7200' in out and '零网络更新' in out, 'resolve_warmup 打印 7200 与"零网络更新"口径')


# =============================================================================
# [C] td3per 的解析链（复刻 parser + 静态接线）
# =============================================================================
def td3per_args(argv=None):
    """复刻 `td3per_main` 的 warmup 相关 parser 片段（**不 import main**：它会启 MATLAB）。

    ⚠ 必须与 main 的原文一致 —— `sec_c_chain` 里有静态断言把两边钉在一起。
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--warmup', default=None, type=int)
    p.add_argument('--bsize', default=256, type=int)
    hw.add_warmup_args(p)
    d = p.parse_args(list(argv or []))
    hw.resolve_warmup(d, verbose=False)
    return d


def sec_c_chain():
    print('\n[C] td3per 的解析链：默认 7200')
    d = td3per_args([])
    ok(int(d.warmup) == 7200, 'td3per 空命令行的 warmup = 7200 步')
    ok(int(d.warmup) // 5 == 1440, '⇒ 1440 个块')
    ok(int(d.bsize) == 256 and int(d.warmup) // 5 >= int(d.bsize),
       '1440 条 ≥ bsize 256（首个 minibatch 不复制同一条 transition）')
    # 显式口径可复现历史
    ok(int(td3per_args(['--warmup', '1440']).warmup) == 1440,
       '--warmup 1440 ⇒ 288 块（旧口径）')
    ok(int(td3per_args(['--warmup', '200']).warmup) == 200, '--warmup 200 ⇒ 200 块')
    ok(int(td3per_args(['--warmup_bsize_mult', '4']).warmup) == 1024,
       '--warmup_bsize_mult 4 ⇒ 1024（旧"绑 bsize"口径）')
    # warmup 配套口径（td3per 与其余入口共用 hsrl_warmup 默认）
    ok(d.warmup_wt_sched == 'lhs', '风机动作 = LHS 全局覆盖（td3per 默认）')
    ok(d.warmup_sched == 'fixed', '上层调度 = fixed')
    ok(d.warmup_bess == 'rule', '下层 = 规则臂（Δ≡0，trackerr≡0）')
    ok(d.warmup_sig == 'price_walk', '第 6 维 = 价格条件游走')
    ok(abs(float(d.soc_reset_low) - 0.25) < 1e-12
       and abs(float(d.soc_reset_high) - 0.25) < 1e-12, 'episode 起点 SOC 固定 0.25')

    # ---- 静态：main 与上面的复刻一致 ----
    i0 = SRC.index("parser.add_argument('--warmup',")
    i1 = SRC.index('parser.add_argument(', i0 + 10)
    blk = SRC[i0:i1]
    ok('default=None' in blk,
       'main 的 --warmup 默认必须是 None（哨兵：写死 7200 会让 --warmup_bsize_mult 失效）')
    ok('7200' in blk, '--warmup 的 help 里写明默认 7200 步')
    ok('1440' in blk and '200' in blk, '--warmup 的 help 里给出复现历史的两个值（1440/200）')
    i_add = SRC.index('hwarm.add_warmup_args(parser)')
    i_res = SRC.index('hwarm.resolve_warmup(args)')
    ok(i_add < i_res, 'add_warmup_args 在 resolve_warmup 之前调用')
    ok(SRC.count('hwarm.resolve_warmup(args)') == 1, 'resolve_warmup 只调用一次')
    ok('[warmup7200]' in SRC, 'main 打印 [warmup7200] 自检行')
    ok('hinit.first_trigger_step(' in SRC, '自检行用 first_trigger_step 算 bias 触发步')


# =============================================================================
# [D] 缓冲 / 触发点 / 零更新断言
# =============================================================================
class _AgentNets(object):
    def __init__(self):
        self.update_count = 0
        self.critic_p = M.Critic(4, 6, hidden1=16, hidden2=100, arch='v6')
        self.critic_target_p = M.Critic(4, 6, hidden1=16, hidden2=100, arch='v6')


def sec_d_invariants():
    print('\n[D] 缓冲 / bias 触发点 / warmup 期零更新')
    trig = hinit.first_trigger_step(7200)
    ok(trig == 7204, 'first_trigger_step(7200) = 7204')
    ok(trig % 5 == 4, '触发步落在块末（≡4 mod 5）')
    ok(7200 <= trig < 7200 + 5, '触发步 = warmup 之后第一个块末')
    for w in (200, 1440, 7200, 7201):
        t = hinit.first_trigger_step(w)
        ok(t % 5 == 4 and w <= t < w + 5,
           'first_trigger_step(%d)=%d：块末且 ≥ warmup' % (w, t))
    # BiasInitTracker：warmup 期零更新（uc!=0 ⇒ 抛错）
    tr = hinit.BiasInitTracker(mode='estimate', warmup=7200, discount=0.95,
                               verbose=False, value_scale=1.0)
    ok(tr.trigger_step == 7204, 'tracker.trigger_step = 7204')
    ag0 = _AgentNets()
    ok(tr.on_block_end(100, 1.0, 0.1, ag0) is None, 'warmup 期（step=100）只收集样本，不应用')
    ok(len(tr.blocks) == 1 and not tr.applied, '样本数 +1 / 未应用')
    ag_bad = _AgentNets()
    ag_bad.update_count = 1
    try:
        tr.on_block_end(200, 1.0, 0.1, ag_bad)
        ok(False, 'warmup 期发生更新必须抛错')
    except AssertionError as e:
        ok('零网络更新' in str(e) or 'warmup' in str(e), 'warmup 期 update_count!=0 ⇒ AssertionError')
    # 7204（warmup 之后第一个块末）应用 bias，且写到 v6 的 out.bias
    tr2 = hinit.BiasInitTracker(mode='estimate', warmup=7200, discount=0.95,
                                verbose=False, value_scale=1.0, cost_scale=1.0)
    ag = _AgentNets()
    info = tr2.on_block_end(7204, 6.0, 0.5, ag)
    ok(info is not None and tr2.applied, 'step=7204 触发并应用 bias')
    ok(abs(float(ag.critic_p.out.bias.data) - float(info['b_p'])) < 1e-6,
       'b_p 写进 v6 critic 的输出层 out.bias（不是静默 no-op）')
    ok(abs(float(ag.critic_target_p.out.bias.data) - float(info['b_p'])) < 1e-6,
       'target 网络同步写入')
    ok(abs(float(info['b_p']) - 6.0 / 0.05) < 1e-6,
       'b_p = 块均值/(1−γ) = 6.0/0.05 = 120（value_scale=1）')
    # 触发后不再重复应用
    ok(tr2.on_block_end(7209, 1.0, 0.1, ag) is None, 'bias 只应用一次')


# =============================================================================
# [E] 1440 块的 LHS 覆盖
# =============================================================================
def sec_e_lhs():
    print('\n[E] 1440 块的 LHS 覆盖（warmup7200 的数据质量）')
    n = 7200 // 5
    sched = hw.WarmupSchedule(5, 6, seed=7, wt_sched='lhs', lhs_n_blocks=n)
    prices = np.array([300.0, 580.0, 980.0])[np.arange(n) % 3]
    A = np.array([sched.action(float(p)) for p in prices], dtype=float)
    ok(A.shape == (n, 6), '1440 块 × 6 维动作')
    ok(sched._lhs_n == n, 'LHS 确实跑了 1440 个块（不是旧的 288）')
    W = A[:, :5]
    ok(W.min() >= hw.LHS_LO - 1e-12 and W.max() <= hw.LHS_HI + 1e-12,
       '风机 5 维落在 [%.1f, %.1f]' % (hw.LHS_LO, hw.LHS_HI))
    ok(W.min() < 0.02 and W.max() > 0.48, 'LHS 铺到盒子两端（min<0.02, max>0.48）')
    bins = np.linspace(hw.LHS_LO, hw.LHS_HI, 11)
    for d in range(5):
        h = np.histogram(W[:, d], bins=bins)[0]
        ok(h.min() >= 141 and h.max() <= 147,
           'dim%d：10 层每层 ≈144 块（实测 [%d, %d]）' % (d, h.min(), h.max()))
    ok(np.abs(np.corrcoef(W.T)[~np.eye(5, dtype=bool)]).max() < 0.35,
       '五维之间基本独立（LHS 的意义）')
    # 信号维：price_walk 方向 + 范围
    ok(A[:, 5].min() >= hw.SIG_LO - 1e-12 and A[:, 5].max() <= hw.SIG_HI + 1e-12,
       '信号维落在 [%.1f, %.1f]' % (hw.SIG_LO, hw.SIG_HI))
    s2 = hw.WarmupSchedule(5, 6, seed=3, wt_sched='lhs')
    up = np.array([s2.action(300.0)[5] for _ in range(40)])
    dn = np.array([s2.action(980.0)[5] for _ in range(60)])
    ok(np.all(np.diff(up) >= -1e-12) and up[-1] > 0.0, '低谷 300 ⇒ 信号上调')
    ok(np.all(np.diff(dn) <= 1e-12) and dn[-1] < 0.0, '高峰 980 ⇒ 信号下调')
    # build_from_args 的块数与 args.warmup 同源
    class _HStub(object):
        nb_turbine, nb_actions = 5, 6
    ns = argparse.Namespace(**vars(td3per_args([])), seed=7, rbb_rated_power=0.5)
    sched_b, rule_b, stats_b = hw.build_from_args(ns, _HStub(), None)
    ok(sched_b._lhs_n == int(ns.warmup) // 5 == 1440,
       'build_from_args：LHS 块数 = warmup//5 = 1440')
    ok(stats_b is None, '无 agent_L ⇒ 无填入统计（td3per 有 L 时才会建）')


# =============================================================================
# [F] 主循环静态接线
# =============================================================================
def sec_f_wiring():
    print('\n[F] td3per_main 主循环接线（静态）')
    ok('hwarm.build_from_args(args, agent_H, agent_L)' in SRC, 'build_from_args 接线')
    ok('hinit.BiasInitTracker.from_args(args)' in SRC, 'BiasInitTracker.from_args 接线')
    ok('hwarm.check_residual_baseline(' in SRC, '残余动作基线自检')
    ok('hwarm.fill_bess_transition(' in SRC, 'warmup 期 L 层 transition 填入')
    ok(SRC.count('if step <= args.warmup:') >= 2,
       'warmup 动作分支（H 块首 + L 下层）存在（%d 处）'
       % SRC.count('if step <= args.warmup:'))
    ok(SRC.count('if step > args.warmup:') >= 2,
       '零更新守卫（H/L 更新段）存在（%d 处）' % SRC.count('if step > args.warmup:'))
    # 更新调用点确实在守卫内：`agent_H.update_policy()` 的缩进 > `if step > args.warmup:` 的缩进
    lines = SRC.splitlines()
    i_gate = max(i for i, l in enumerate(lines) if l.strip() == 'if step > args.warmup:')
    i_upd = [i for i, l in enumerate(lines) if l.strip() == 'agent_H.update_policy()']
    ok(i_upd and all(i > i_gate for i in i_upd), 'agent_H.update_policy 在零更新守卫之后')
    i_wm = [i for i, l in enumerate(lines) if 'warmup_sched.action(' in l]
    ok(i_wm and all(i < i_gate for i in i_wm), 'warmup 动作调度在守卫之前（warmup 期内）')
    # 更新调用点确实在缩进更深的守卫块内（不是同级的另一处）
    _gate_indent = len(lines[i_gate]) - len(lines[i_gate].lstrip())
    ok(all(len(lines[i]) - len(lines[i].lstrip()) > _gate_indent for i in i_upd),
       'update_policy 的缩进深于守卫（在 if 块内）')


def main():
    print('=' * 78)
    print('td3per warmup7200 契约测试（纯静态 / 纯算术）')
    print('=' * 78)
    sec_a_constants()
    sec_b_resolve()
    sec_c_chain()
    sec_d_invariants()
    sec_e_lhs()
    sec_f_wiring()
    print('\n' + '=' * 78)
    print('全部通过：%d 条断言' % N_OK)
    print('=' * 78)


if __name__ == '__main__':
    main()
