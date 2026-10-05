#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_yaw_env_contract.py —— P1 环境契约测试（只读，**不需 MATLAB**）。

`yawflor/README.md` §6 的 P1 验收判据（五条）：

① **契约同形**：6 维动作截断、`3.5 ≤ u ≤ 25` 守卫、`P_G_prev` 基准、obs 布局，
   以及三个 reset / step 接口的返回结构（与旧环境逐字同形）；
② **逐块预取路径 ≡ 直算路径（逐位）**：预取只是省 FLORIS 调用，不改变任何数值；
③ **确定性 / 无记忆**：同条件重复调用逐位相同，且与调用历史无关（test9 判据前提）；
④ **NaN 不传染**：全场同时大偏航（≥45°，`gauss` 挠曲的 `sqrt` 出 NaN）时仍返回有限值；
⑤ **旧环境默认路径逐位不变**（对 stub engine）：`WindBattery(..., physics=None)`
   仍只做 `matlab.double` 包装 + `WFSim_port` 调用，标称动作仍是 `0.3333`。

另附两条**时序证据**（README §0 的"调用预算可接受"在实现层面的复核）：
逐块预取 vs 全直算的单步耗时与 1440 步 episode 外推。

运行
----
    python agentutil/test_yaw_env_contract.py
    python agentutil/test_yaw_env_contract.py --json output/yawflor_env_contract.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

# ---- 路径 bootstrap（可被任意 CWD 调用）------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

os.environ.setdefault('MPLCONFIGDIR', os.path.join(ROOT, 'output', '_mplcache'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import env as ENV                                       # noqa: E402
from yawflor.yawflor_env import FlorisPhysicsYaw, WindBatteryYaw   # noqa: E402
from yawflor import yawflor_turbine as YT               # noqa: E402


class Report:
    def __init__(self):
        self.rows, self.fails = [], []

    def check(self, name, ok, detail=''):
        self.rows.append((name, bool(ok), detail))
        if not ok:
            self.fails.append(name)
        return bool(ok)

    def dump(self):
        w = max(len(r[0]) for r in self.rows) + 2
        print('| %-*s | %-6s | %s' % (w, '判据', '状态', '实测'))
        print('|%s|%s|%s' % ('-' * (w + 2), '-' * 8, '-' * 40))
        for name, ok, detail in self.rows:
            print('| %-*s | %-6s | %s' % (w, name, 'OK' if ok else '**FAIL**', detail))


# ===========================================================================
# 0. stub MATLAB engine（判据⑤；不启真 MATLAB）
# ===========================================================================
class StubEngine:
    """记录调用参数的假 `matlab.engine`：`out = 1e6·(0.001·u + Σact)`。"""

    def __init__(self):
        self.calls = []

    def WFSim_port(self, u, phi, act, phi2, want_full=False):
        # env.py::PowerOutput 实传 5 个实参（第 5 个 want_full=False）。收下并记录，
        # 以便下面把"生产调用形状"钉成判据 —— 本 stub 曾只收 4 个 ⇒ 判据⑤ 会走
        # env.py 的 except 分支后在 target_result['out'] 抛 UnboundLocalError。
        u_v = float(np.asarray(u).reshape(-1)[0])
        act_v = [float(x) for x in np.asarray(act).reshape(-1)]
        self.calls.append(dict(u=u_v, phi=float(np.asarray(phi).reshape(-1)[0]),
                               act=act_v, phi2=float(np.asarray(phi2).reshape(-1)[0]),
                               want_full=bool(want_full)))
        return {'out': 1e6 * (0.001 * u_v + float(np.sum(act_v)))}

    @staticmethod
    def expected(u, act):
        return 0.001 * float(u) + float(np.sum(np.asarray(act, dtype=float)[:5]))


def section(title, r):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)
    return r


# ===========================================================================
# 判据⑤：旧环境默认路径逐位不变
# ===========================================================================
def check_legacy_env(r):
    section('⑤ 旧环境默认路径（stub engine；physics=None）', r)
    stub = StubEngine()
    w = ENV.WindBattery(6.0, 5, False, stub)        # ← 与 9 个 main 完全同形的 4 参调用
    r.check('WindBattery(...) 4 参调用可用（physics 默认 None ⇒ MATLAB 路径）', True)

    # 标称动作：旧值 0.3333（逐位；不是 1/3）
    r.check('WheatBattery.nominal_action_value == 0.3333（旧标称动作）',
            ENV.WindBattery.nominal_action_value == 0.3333,
            repr(ENV.WindBattery.nominal_action_value))
    r.check('nominal() == np.ones(5)*0.3333（逐位）',
            np.array_equal(w.nominal(), np.ones(5) * 0.3333),
            str(w.nominal()))

    # PowerOutput：与"手写 matlab.double + WFSim_port"逐位同值
    u, act = 10.0, [0.2, 0.25, 0.3, 0.15, 0.4]
    got = w.PowerOutput(u, act)
    exp = stub.expected(u, act)
    r.check('PowerOutput(10, a) == stub 直算（逐位）', got == exp, '%.15g vs %.15g' % (got, exp))
    last = stub.calls[-1]
    r.check('调用形状：WFSim_port(u, 0.0, act(5 维), 0.0)',
            last['u'] == u and last['phi'] == 0.0 and last['phi2'] == 0.0
            and last['act'] == act, str(last))
    r.check('调用形状第 5 实参 = False（env.py 的 want_full=False，省 sol_array 回传）',
            last.get('want_full') is False, 'want_full=%r' % last.get('want_full'))

    # 守卫
    n0 = len(stub.calls)
    r.check('守卫：u=3.0 ⇒ 0.0 且不调用引擎',
            w.PowerOutput(3.0, act) == 0.0 and len(stub.calls) == n0, 'u=3.0')
    r.check('守卫：u=25.1 ⇒ 0.0 且不调用引擎',
            w.PowerOutput(25.1, act) == 0.0 and len(stub.calls) == n0, 'u=25.1')
    r.check('守卫边界：u=3.5 / u=25 均可用',
            w.PowerOutput(3.5, act) != 0.0 and w.PowerOutput(25.0, act) != 0.0, '')

    # 6/10 维动作 ⇒ 只取前 5 维（既不是"报错"也不是"隐式扩展"）
    w.PowerOutput(u, act)
    a5 = stub.calls[-1]['act']
    w.PowerOutput(u, list(act) + [0.5])
    a6 = stub.calls[-1]['act']
    w.PowerOutput(u, list(act) + [0.5] * 5)
    a10 = stub.calls[-1]['act']
    r.check('6 维/10 维动作 ⇒ 只取前 5 维（截断后下传）',
            a5 == a6 == a10 == act, '5维=%s 6维=%s 10维=%s' % (a5, a6, a10))

    # reset / TestReset / ScenarioReset：obs 布局 + P_G_prev 基准
    obs = w.reset()
    r.check('reset() obs = [pr, u, E, P_G_prev]（4 维，布局同形）',
            isinstance(obs, np.ndarray) and obs.shape == (4,),
            str(np.round(obs, 4)))
    r.check('reset()：P_G_prev == PowerOutput(u, nominal())（标称动作基准）',
            obs[3] == w.PowerOutput(w.u, w.nominal()), '%.15g' % float(obs[3]))

    obs = w.TestReset(0)
    r.check('TestReset(0) obs 布局同形 + P_G_prev 基准',
            obs.shape == (4,) and obs[3] == w.PowerOutput(w.u, w.nominal()),
            '%.15g' % float(obs[3]))
    obs = w.ScenarioReset([8.0, 9.0, 10.0], [300.0, 580.0, 980.0])
    r.check('ScenarioReset() obs 布局同形 + P_G_prev 基准',
            obs.shape == (4,) and obs[3] == w.PowerOutput(8.0, w.nominal()),
            '%.15g' % float(obs[3]))

    # step 返回结构（7 元组 + info 两键）
    w.reset()
    ptur = w.PowerOutput(w.u, w.nominal())
    out = w.step(np.array(list(w.nominal()) + [0.0]), np.zeros(1), ptur)
    r.check('step() 返回 7 元组且 info 含 P_F/P_G',
            len(out) == 7 and set(out[6]) == {'P_F', 'P_G'},
            'len=%d info=%s' % (len(out), sorted(out[6])))
    r.check('step()：P_F == |P_G − P_G_prev|（波动口径同形）',
            abs(out[6]['P_F'] - abs(out[6]['P_G'] - ptur)) < 1e-12,
            'P_F=%.6g' % out[6]['P_F'])
    return True


# ===========================================================================
# 判据①：新环境契约同形
# ===========================================================================
def check_shape(r, phys):
    section('① 新环境契约同形（obs/守卫/截断/P_G_prev/step 结构）', r)
    w = WindBatteryYaw(6.0, 5, False, None, physics=phys)

    r.check('WindBatteryYaw.nominal_action_value == 0.0（n=0 ⇒ γ=0°）',
            WindBatteryYaw.nominal_action_value == 0.0,
            repr(WindBatteryYaw.nominal_action_value))

    n0 = phys.n_direct
    r.check('守卫：u=3.0 ⇒ 0.0 且不调用 FLORIS',
            w.PowerOutput(3.0, np.zeros(5)) == 0.0 and phys.n_direct == n0, 'u=3.0')
    r.check('守卫：u=25.1 ⇒ 0.0 且不调用 FLORIS',
            w.PowerOutput(25.1, np.zeros(5)) == 0.0 and phys.n_direct == n0, 'u=25.1')

    p5 = w.PowerOutput(10.0, np.array([0.2] * 5))
    p6 = w.PowerOutput(10.0, np.array([0.2] * 5 + [0.5]))
    p10 = w.PowerOutput(10.0, np.array([0.2] * 5 + [0.5] * 5))
    r.check('6 维/10 维动作截断：第 6 维（SOC 信号）不影响出力',
            p5 == p6 == p10, '5维=%.12g 6维=%.12g 10维=%.12g' % (p5, p6, p10))

    obs = w.reset()
    r.check('reset() obs = [pr, u, E, P_G_prev]（4 维）',
            obs.shape == (4,), str(np.round(obs, 4)))
    r.check('reset()：P_G_prev == PowerOutput(u, 标称 n=0)',
            obs[3] == w.PowerOutput(w.u, w.nominal()), '%.15g' % float(obs[3]))
    r.check('标称动作逐位 = 0°：P_G_prev == PowerOutput(u, γ=0)',
            obs[3] == phys.power_deg(w.u, np.zeros(5)), '%.15g' % float(obs[3]))

    for rt in ('TestReset', 'ScenarioReset'):
        if rt == 'TestReset':
            o = w.TestReset(0)
        else:
            o = w.ScenarioReset([8.0, 9.0, 10.0], [300.0, 580.0, 980.0])
        r.check('%s() obs 布局同形 + P_G_prev 基准' % rt,
                o.shape == (4,) and o[3] == w.PowerOutput(o[1], w.nominal()),
                '%.15g' % float(o[3]))

    # 三个 step 接口的返回结构（7 元组 + info）
    w.reset()
    ptur = w.PowerOutput(w.u, w.nominal())
    for fn, args in (('step', (np.array([0.0] * 5 + [0.0]), np.zeros(1), ptur)),
                     ('TestStep', (np.array([0.0] * 5 + [0.0]), np.zeros(1), ptur)),
                     ('ScenarioStep', (np.array([0.0] * 5 + [0.0]), np.zeros(1), ptur))):
        w.ScenarioReset([8.0, 9.0, 10.0, 11.0], [300.0] * 4)
        out = getattr(w, fn)(*args)
        r.check('%s() 返回 7 元组 + info{P_F,P_G} + obs 4 维' % fn,
                len(out) == 7 and set(out[6]) == {'P_F', 'P_G'} and out[1].shape == (4,),
                'len=%d' % len(out))
    return True


# ===========================================================================
# 判据②③：预取 ≡ 直算（逐位）+ 确定性/无记忆
# ===========================================================================
def _drive(env, n_blocks, act_fn):
    """按**主循环的节奏**驱动 env：块首一次 PowerOutput + 5 次 step。

    返回所有 PowerOutput 结果的列表（块首值 + 每步 pw_tru2）。
    """
    vals = []
    obs = env.reset()
    for b in range(n_blocks):
        act = act_fn(b)
        action6 = np.concatenate([act, [0.0]])
        ptur = env.PowerOutput(obs[1], act)
        vals.append(float(ptur))
        for _ in range(5):
            pw2, obs, _aL, _pen, _d, _dL, _info = env.step(action6, np.zeros(1), ptur)
            vals.append(float(pw2))
            ptur = float(pw2)
    return vals


def check_prefetch_and_determinism(r):
    section('②③ 预取 ≡ 直算（逐位）/ 确定性 / 无记忆', r)
    phys_a = FlorisPhysicsYaw(cache_enabled=True)
    phys_b = FlorisPhysicsYaw(cache_enabled=False)
    env_a = WindBatteryYaw(6.0, 5, False, None, physics=phys_a)
    env_b = WindBatteryYaw(6.0, 5, False, None, physics=phys_b)

    act_fn = lambda b: 0.4 * np.array([np.sin(0.7 * b + i) for i in range(5)])
    va = _drive(env_a, 12, act_fn)
    vb = _drive(env_b, 12, act_fn)

    r.check('② 逐块预取序列 ≡ 全直算序列（逐位）', va == vb,
            'len=%d, max|Δ|=%.3e' % (len(va), max(abs(x - y) for x, y in zip(va, vb))))
    r.check('② 预取确实生效（批量调用 > 0 且直算被省下）',
            phys_a.n_batch >= 12 and phys_a.n_hit > 0 and phys_b.n_batch == 0,
            'A: 批量 %d / 命中 %d / 单条 %d ; B: 批量 %d / 单条 %d'
            % (phys_a.n_batch, phys_a.n_hit, phys_a.n_direct, phys_b.n_batch, phys_b.n_direct))

    # 单元级：同一批 (u, γ) 预取 vs 逐条直算
    phys_c = FlorisPhysicsYaw(cache_enabled=True)
    us = np.array([5.0, 7.5, 10.0, 12.5, 15.0, 9.0])
    g5 = np.array([12.0, -8.0, 0.0, 20.0, -25.0])
    phys_c.prefetch_deg(us, g5)
    got = [phys_c.power_deg(u, g5) for u in us]
    exp = [YT.farm_power(phys_c.model, u, g5) for u in us]
    r.check('② 预取查表 ≡ 单条直算（逐位，6 组 (u,γ)）', got == exp,
            'max|Δ|=%.3e' % max(abs(x - y) for x, y in zip(got, exp)))
    r.check('② 查表命中计数 == 6', phys_c.n_hit == 6, str(phys_c.n_hit))

    # ③ 确定性 / 无记忆
    e1 = FlorisPhysicsYaw()
    p1 = e1.power_deg(10.0, np.full(5, 10.0))
    p2 = e1.power_deg(10.0, np.full(5, 10.0))
    r.check('③ 同条件重复调用逐位相同', p1 == p2, '%.15g' % p1)

    e2 = FlorisPhysicsYaw()
    for k in range(8):                       # 先灌一段历史（含缓存/守卫路径）
        e2.power_deg(4.0 + k, np.full(5, -20.0 + 5 * k))
    p3 = e2.power_deg(10.0, np.full(5, 10.0))
    r.check('③ 无记忆：换一条历史路径后同点结果逐位相同（= 首次调用值）',
            p3 == p1, '%.15g vs %.15g' % (p3, p1))

    # 缓存开关本身不影响数值
    e3 = FlorisPhysicsYaw(cache_enabled=True)
    e4 = FlorisPhysicsYaw(cache_enabled=False)
    e3.prefetch_deg(np.array([10.0, 11.0]), np.full(5, 10.0))
    r.check('③ 缓存开/关同点逐位相同',
            e3.power_deg(10.0, np.full(5, 10.0)) ==
            e4.power_deg(10.0, np.full(5, 10.0)), '')
    return True


# ===========================================================================
# 判据④：NaN 不传染
# ===========================================================================
def check_nan_guard(r):
    section('④ NaN 守卫（全场大偏航；gauss 挠曲 sqrt 出 NaN）', r)
    phys = FlorisPhysicsYaw()
    for deg in (45.0, 47.0, 60.0, 89.0):
        p = phys.power_deg(10.0, np.full(5, deg))
        r.check('全场 γ=%.0f° ⇒ 有限值' % deg, bool(np.isfinite(p)),
                'P=%.6f MW（守卫触发 %d 次）' % (p, phys.n_nonfinite))
    r.check('NaN 守卫确实被触发过（否则该判据没测到东西）',
            phys.n_nonfinite > 0, str(phys.n_nonfinite))
    p_band = phys.power_deg(10.0, np.full(5, YT.GAMMA_MAX))
    r.check('带内边界 γ=+25° ⇒ 有限值且为正', bool(np.isfinite(p_band) and p_band > 0),
            'P=%.6f MW' % p_band)
    # 带外动作经 env 入口会被夹到带内 ⇒ 结构性不进入 NaN 区
    env = WindBatteryYaw(6.0, 5, False, None, physics=FlorisPhysicsYaw())
    r.check('env 入口把 n=5（γ=125°）夹回 +25° 带内',
            env.PowerOutput(10.0, np.full(5, 5.0)) == p_band, '')
    # 输运延迟默认必须关（开启会让 env 有记忆 ⇒ 破坏 test9 判据）
    import yawflor.yawflor_env as YE
    old = YT.YAW_TRANSPORT_DELAY
    try:
        YT.YAW_TRANSPORT_DELAY = True
        try:
            WindBatteryYaw(6.0, 5, False, None, physics=FlorisPhysicsYaw())
            raised = False
        except NotImplementedError:
            raised = True
    finally:
        YT.YAW_TRANSPORT_DELAY = old
    r.check('YAW_TRANSPORT_DELAY=True 被显式拒绝（不静默改变语义）', raised, '')
    return True


# ===========================================================================
# 附：时序证据（预取 vs 直算）
# ===========================================================================
def timing_evidence(r, n_blocks=12):
    section('附 时序证据：逐块预取 vs 全直算（%d 块 = %d 步）' % (n_blocks, n_blocks * 5), r)
    out = {}
    for tag, cache in (('prefetch', True), ('direct', False)):
        env = WindBatteryYaw(6.0, 5, False, None, physics=FlorisPhysicsYaw(cache_enabled=cache))
        t0 = time.perf_counter()
        _drive(env, n_blocks, lambda b: 0.3 * np.ones(5))
        out[tag] = time.perf_counter() - t0
    steps = n_blocks * 5
    r.check('预取比全直算快（本机实测）', out['prefetch'] < out['direct'],
            '预取 %.2f s / 直算 %.2f s（%.1f×）'
            % (out['prefetch'], out['direct'], out['direct'] / max(out['prefetch'], 1e-9)))
    print('  外推到 1440 步 episode：预取 ≈ %.2f s/episode，全直算 ≈ %.2f s/episode'
          % (out['prefetch'] / steps * 1440, out['direct'] / steps * 1440))
    return out


def main():
    ap = argparse.ArgumentParser(description='P1 yawflor 环境契约测试')
    ap.add_argument('--json', default=None)
    ap.add_argument('--no-timing', action='store_true', help='跳过期证据（省 ~10 s）')
    args = ap.parse_args()

    r = Report()
    print('FLORIS 后端：%s' % YT.build_model()[1]['floris_version'])

    check_legacy_env(r)
    check_shape(r, FlorisPhysicsYaw())
    check_prefetch_and_determinism(r)
    check_nan_guard(r)
    out = {} if args.no_timing else timing_evidence(r)

    r.dump()
    print()
    if r.fails:
        print('❌ P1 契约失败 %d 项：%s' % (len(r.fails), ', '.join(r.fails)))
    else:
        print('✅ P1 契约全部通过（%d 项）' % len(r.rows))
    if args.json:
        p = args.json if os.path.isabs(args.json) else os.path.join(ROOT, args.json)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w', encoding='utf-8') as fh:
            json.dump({'rows': [{'name': n, 'ok': o, 'detail': d} for n, o, d in r.rows],
                       'fails': r.fails, 'timing': out}, fh, ensure_ascii=False, indent=2)
        print('结果已落盘：%s' % p)
    return 1 if r.fails else 0


if __name__ == '__main__':
    sys.exit(main())
