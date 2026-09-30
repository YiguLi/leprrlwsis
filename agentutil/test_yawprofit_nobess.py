#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_yawprofit_nobess.py —— `yawprofit`（无储能偏航调度）的**契约测试**（只读, 不需 MATLAB）。

判据（"彻底抹去 BESS 在 state / action / reward 中的存在"的可执行定义）
----------------------------------------------------------------------
① **state 无 SOC**：观测 3 维 = `[pr, u, P_tur]`；前两维与 `WindBatteryYaw`
   逐位相同，第 3 维 = `WindBatteryYaw` 的 `P_G`（电池恒零功率时二者等价）。
② **步进等价**：同一动作序列下 `WindTurbineYaw` 的风速/电价/风机出力轨迹与
   `WindBatteryYaw(act_bat ≡ 0)` **逐位相同** —— 即"抹掉电池"只是删掉了恒零项，
   没有改变物理路径；且 yawflor 环境的 SOC 全程不动（证据的另一半）。
③ **action 只有偏航**：5 维、带 `[−1,+1]`、越界罚 = 100×Σ越界且不封顶；
   传 6 维（yawflor 的"5 偏航 + SOC 信号"）**必须报错**而不是静默截断。
④ **reward 无电池项**：`r1 = pr·P_tur·dt − penalty`，`c1 ≡ 0`；
   `details` 里**不存在** `bess_profit` / `degradation` / `soc_target` /
   `unfeasible_gap` / `soc_deviation` 这些键。
⑤ **电池通道结构性不存在**：`GetBatteryActions()` 抛 `NotImplementedError`。
⑥ **静态结构**：`yawprofit_main.py` / `yawprofit_myfunc.py` 的 AST 里
   **不出现** `RuleBasedBESS`/`rbb_system`/`soc_target`/`gap_penalty`/`agent_L`/
   `update_lam_ema`/`dual_update`/`hsrl_soctarget`/`matlab` 等标识符
   （AST 只看标识符 ⇒ 说明性 docstring 里提到这些名词不算违规）。
⑦ **观测维度**：`nb_states = 2 + PHASE_DIM[--phase_feat]`；`Normalize` 前两维与
   yawflor 逐位相同。
⑧ **九例测试无 BESS 且自洽**：零策略（n ≡ 0）下 `gain = profit − profit_nominal`
   必须为 0（参考臂与实跑走同一条物理路径），且 `Σr1 = profit`（罚为 0）。
⑨ **克隆的 agent 可用**：`SP_TD3(nb_states=4, nb_actions=5)` 可构造，且
   `random_action()` 落在偏航带 `[−1,+1]` 内（克隆源该分支给的是轴向因子带 `[0,0.5]`）。

运行
----
    python agentutil/test_yawprofit_nobess.py
"""
import ast
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

# ---- 路径 bootstrap（可被任意 CWD 调用）------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

os.environ.setdefault('MPLCONFIGDIR', os.path.join(ROOT, 'output', '_mplcache'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

# 终端编码守卫：Windows 下 stdout 被重定向/走管道时用的是 locale 编码（cp936），
# 而本项目文案含 '−'（U+2212）/'γ'/'⇒'/'≡' 等 GBK 没有的字符 ⇒ 会 UnicodeEncodeError
# **把测试直接打断**。这里只把不可编码字符降级为 '?'，不改编码、不吞其它错误。
try:
    sys.stdout.reconfigure(errors='replace')
except Exception:                                            # pragma: no cover
    pass

from yawflor.yawflor_env import FlorisPhysicsYaw, WindBatteryYaw          # noqa: E402
from yawprofit.yawprofit_env import WindTurbineYaw                        # noqa: E402
from yawprofit import yawprofit_myfunc as MF                              # noqa: E402
from yawflor import yawflor_myfunc as FL                                  # noqa: E402
from yawflor import yawflor_turbine as YT                                 # noqa: E402

MAIN_PY = os.path.join(ROOT, 'yawprofit', 'yawprofit_main.py')
MYFUNC_PY = os.path.join(ROOT, 'yawprofit', 'yawprofit_myfunc.py')


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
        print('\n| %-*s | %-6s | %s' % (w, '判据', '状态', '实测'))
        print('|%s|%s|%s' % ('-' * (w + 2), '-' * 8, '-' * 40))
        for name, ok, detail in self.rows:
            print('| %-*s | %-6s | %s' % (w, name, 'OK' if ok else '**FAIL**', detail))


def section(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


def _idents(path):
    """AST 里出现过的**标识符**集合（不含字符串/注释 ⇒ docstring 提到名词不算违规）。"""
    tree = ast.parse(open(path, encoding='utf-8').read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split('.')[0])
            for a in node.names:
                names.add(a.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


# ===========================================================================
# ① ② ⑤ 环境接口
# ===========================================================================
def test_env(r):
    section('① ② ⑤ 环境接口：观测无 SOC / 步进与"电池恒零功率"逐位等价 / 电池通道不存在')

    ph_wt = FlorisPhysicsYaw(nt=5)
    ph_fl = FlorisPhysicsYaw(nt=5)
    env_wt = WindTurbineYaw(6., 5, False, None, physics=ph_wt)
    env_fl = WindBatteryYaw(6., 5, False, None, physics=ph_fl)

    obs_wt = env_wt.reset()
    obs_fl = env_fl.reset()
    r.check('① 观测 3 维（无 SOC）', obs_wt.shape == (3,), 'shape={}'.format(obs_wt.shape))
    r.check('① [pr,u] 与 yawflor 逐位相同',
            np.array_equal(obs_wt[:2], obs_fl[:2]),
            'wt={} fl={}'.format(np.round(obs_wt[:2], 6), np.round(obs_fl[:2], 6)))
    r.check('① P_tur == yawflor 的 P_G（电池零功率）',
            obs_wt[2] == obs_fl[3], 'wt={} fl={}'.format(obs_wt[2], obs_fl[3]))

    # ---- 同步步进 30 步：块级动作序列（每 5 步换一次），yawflor 侧 act_bat=[0.0] ----
    rng = np.random.RandomState(7)
    maxdiff = 0.0
    soc_maxdev = 0.0
    act6 = np.zeros(6)
    pw_wt = obs_wt[2]
    pw_fl = obs_fl[3]
    for k in range(30):
        if k % 5 == 0:
            act6 = np.concatenate([rng.uniform(-1, 1, 5), [0.0]])
        pw2_wt, obs2_wt, info_wt = env_wt.step(act6[:5])
        pw2_fl, obs2_fl, _al, _pl, _d, _dl, info_fl = env_fl.step(act6, [0.0], pw_fl)
        maxdiff = max(maxdiff, float(np.max(np.abs(obs2_wt[:2] - obs2_fl[:2]))),
                      abs(pw2_wt - pw2_fl))
        soc_maxdev = max(soc_maxdev, abs(float(obs2_fl[2]) - 3.0))
        pw_wt, pw_fl = pw2_wt, pw2_fl
    r.check('② 30 步轨迹逐位相同（pr/u/P_tur）', maxdiff == 0.0, 'max|Δ|={:g}'.format(maxdiff))
    r.check('② yawflor 侧 SOC 全程不动（act_bat≡0）', soc_maxdev == 0.0,
            'max|E−3|={:g}'.format(soc_maxdev))
    r.check('② 本环境 info["P_G"] == 返回的 P_tur',
            bool(np.isclose(info_wt['P_G'], pw_wt, rtol=0, atol=0)), '')

    try:
        env_wt.GetBatteryActions(0.0, np.array([0.0]))
        ok, det = False, '未报错'
    except NotImplementedError as e:
        ok, det = True, type(e).__name__
    except Exception as e:                                       # pragma: no cover
        ok, det = False, '抛了 {}'.format(type(e).__name__)
    r.check('⑤ GetBatteryActions 抛 NotImplementedError', ok, det)


# ===========================================================================
# ③ 动作
# ===========================================================================
def test_action(r):
    section('③ 动作：只有偏航维 / 越界罚 / 6 维必须报错')

    a = np.zeros(5)
    applied, pen = MF.CheckAction(a)
    r.check('③ 带内动作原样返回且罚 = 0',
            np.array_equal(applied, a) and pen == 0.0, 'pen={}'.format(pen))

    raw = np.array([1.3, -1.4, 0.5, -0.2, 2.0])
    applied, pen = MF.CheckAction(raw)
    r.check('③ 夹到 [−1,+1]', np.array_equal(applied, np.clip(raw, -1, 1)),
            'applied={}'.format(applied))
    r.check('③ 罚 = 100×Σ越界', abs(pen - 100.0 * (0.3 + 0.4 + 1.0)) < 1e-12,
            'pen={} vs {}'.format(pen, 100.0 * 1.7))
    r.check('③ 罚不封顶（越界越深罚越大）',
            MF.CheckAction(np.array([5.0, 0, 0, 0, 0]))[1]
            > MF.CheckAction(np.array([2.0, 0, 0, 0, 0]))[1], '')

    try:
        MF.CheckAction(np.zeros(6))
        ok, det = False, '未报错（静默截断会掩盖接线错误）'
    except ValueError:
        ok, det = True, 'ValueError'
    r.check('③ 6 维动作（含 SOC 信号）报错', ok, det)

    # 与 yawflor 的 5 维分支逐位同口径（yawflor 也接受 5 维）
    raw = np.array([1.3, -1.4, 0.5, -0.2, 2.0])
    a_fl, p_fl = FL.CheckAction(raw)
    a_wp, p_wp = MF.CheckAction(raw)
    r.check('③ 与 yawflor 的 5 维 CheckAction 逐位相同',
            np.array_equal(a_fl, a_wp) and p_fl == p_wp,
            'pen fl={} wp={}'.format(p_fl, p_wp))


# ===========================================================================
# ④ 奖励
# ===========================================================================
def test_reward(r):
    section('④ 奖励：r1 = pr·P_tur·dt − penalty；c1 ≡ 0；details 无电池键')

    obs = np.array([580.0, 9.5, 3.2])
    r1, c1, d = MF.Cal_Rew_WT(obs, 4.0, penalty_NN_H=0.0)
    r.check('④ r1 = pr·P_tur/60', abs(r1 - 580.0 * 4.0 / 60) < 1e-12,
            'r1={:.6f} vs {:.6f}'.format(r1, 580.0 * 4.0 / 60))
    r.check('④ 越界罚线性扣除',
            abs(MF.Cal_Rew_WT(obs, 4.0, penalty_NN_H=13.0)[0] - (r1 - 13.0)) < 1e-12, '')
    r.check('④ c1 恒为 0', c1 == 0.0, 'c1={!r}'.format(c1))

    banned = ('bess_profit', 'degradation', 'soc_target', 'unfeasible_gap',
              'soc_deviation', 'soc_delta', 'pb_cmd', 'pbrule')
    present = [k for k in banned if k in d]
    r.check('④ details 不含任何电池键', not present, '命中={}'.format(present))

    c1s = [MF.Cal_Rew_WT(obs, p, penalty_NN_H=q)[1]
           for p in (0.0, 3.0, 10.0, -2.0) for q in (0.0, 5.0)]
    r.check('④ c1 在任意 pr/P_tur/罚下都恒 0', all(c == 0.0 for c in c1s), '')


# ===========================================================================
# ⑥ 静态结构
# ===========================================================================
def test_static(r):
    section('⑥ 静态结构：主循环/口径文件里不存在电池与对偶的标识符')

    banned = {'RuleBasedBESS', 'rbb_system', 'SocTracker', 'soc_track_slack',
              'soc_target', 'soc_target_from_signal', 'gap_penalty',
              'update_lam_ema', 'dual_update', 'agent_L', 'penalty_L',
              'applied_action_L', 'delta_L', 'hsrl_soctarget', 'rbb',
              'yawflor_dual', 'matlab', 'build_L_obs_distilled',
              'lower_cost_soc_track', 'lower_reward'}
    for path in (MAIN_PY, MYFUNC_PY):
        names = _idents(path)
        hit = sorted(banned & names)
        r.check('⑥ {} 无电池/对偶标识符'.format(os.path.basename(path)),
                not hit, '命中={}'.format(hit))


# ===========================================================================
# ⑦ 观测维度
# ===========================================================================
def test_obs_dim(r):
    section('⑦ 观测维度：2 维基 + 相位特征；与 yawflor 的基维前 2 维逐位相同')

    obs = np.array([980.0, 11.37, 7.5])
    n_wp = MF.Normalize(obs)
    n_fl = FL.Normalize(np.array([980.0, 11.37, 3.0, 7.5]))
    r.check('⑦ Normalize 是 2 维且前 2 维与 yawflor 逐位相同',
            n_wp.shape == (2,) and np.array_equal(n_wp, n_fl[:2]),
            'wp={} fl={}'.format(n_wp, n_fl))
    for mode in (0, 1, 2):
        h = MF.Normalize_H(obs, 600, mode)
        r.check('⑦ phase_feat={} ⇒ nb_states = 2 + {} = {}'.format(
            mode, MF.PHASE_DIM[mode], 2 + MF.PHASE_DIM[mode]),
            h.shape == (2 + MF.PHASE_DIM[mode],), 'shape={}'.format(h.shape))
    # 训练用的 4 维（phase_feat=1）里没有 SOC 的任何单调副本
    r.check('⑦ 归一化观测无 NaN/Inf', bool(np.isfinite(MF.Normalize_H(obs, 0, 1)).all()), '')


# ===========================================================================
# ⑧ 九例测试（零策略 ⇒ gain 必须为 0）
# ===========================================================================
def test_nine_case(r):
    section('⑧ 九例测试：只跑 H；零策略下 gain = 0（参考臂与实跑同路径）')

    class ZeroAgent:
        """n ≡ 0 的假上层（不需要真实网络）。"""
        is_training = True

        def select_action(self, s, decay_epsilon=True):
            return np.zeros(5)

        def reset(self, obs):
            pass

    env_test = WindTurbineYaw(6., 5, False, None, physics=FlorisPhysicsYaw(nt=5))
    ph_ref = FlorisPhysicsYaw(nt=5)
    cases, overall = MF.nine_case_test(ZeroAgent(), env_test, seed=42,
                                       phase_mode=1, physics_ref=ph_ref)
    r.check('⑧ 返回 9 条 case', len(cases) == 9, 'len={}'.format(len(cases)))
    r.check('⑧ gain == 0（零偏航 ⇒ 与无偏航参考臂同收益）',
            abs(float(overall['gain'])) < 1e-9,
            'gain={:.3e} profit={:.4f} nominal={:.4f}'.format(
                float(overall['gain']), float(overall['profit']),
                float(overall['profit_nominal'])))
    sum_r1 = sum(m['r1'] for m in cases) * 5
    r.check('⑧ Σr1 == profit（罚为 0 时逐位自洽）',
            abs(sum_r1 - float(overall['profit'])) < 1e-9,
            'Σr1={:.6f} profit={:.6f}'.format(sum_r1, float(overall['profit'])))
    r.check('⑧ 无偏航参考收益为正', float(overall['profit_nominal']) > 0,
            'nominal={:.3f} 元 / 45 步'.format(float(overall['profit_nominal'])))
    r.check('⑧ 零偏航 ⇒ 平均 |γ| = 0', abs(float(overall['yaw_deg_mean'])) < 1e-12,
            'yaw_deg_mean={:.3e}'.format(float(overall['yaw_deg_mean'])))
    r.check('⑧ c1 ≡ 0（九例表里也是）',
            all(m['c1'] == 0.0 for m in cases) and overall['c1'] == 0.0, '')
    r.check('⑧ 无参考臂时 profit_nominal = NaN（不编造参考值）',
            np.isnan(float(MF.nine_case_test(ZeroAgent(), env_test, seed=42,
                                             phase_mode=1)[1]['profit_nominal'])), '')


# ===========================================================================
# ⑨ 克隆的 agent 可用
# ===========================================================================
def test_agent(r):
    section('⑨ 克隆的 SP_TD3：nb_states=4 / nb_actions=5 可构造；随机动作在偏航带内')

    args = SimpleNamespace(
        seed=-1, cuda=False, hidden1=16, hidden2=100, init_w=3e-3,
        prate=1e-4, rate=1e-2, rmsize=1000, PER_alpha=0.6, PER_beta=0.4,
        PER_epsilon=0.01, beta_increment_per_sampling=3e-5, bsize=32,
        tau=1e-3, discount=0.98, epsilon=4000, nb_turbine=5,
        lam=0.0, lr_lam=0.1, cost_single_limit=1000., ou_theta=0.15,
        ou_sigma=0.2, ou_mu=0.0)
    from yawprofit.yawprofit_ddpg import SP_TD3
    agent = SP_TD3(nb_states=4, nb_actions=5, args=args, lam_max=None, lam_min=0.0,
                   ema_lam=True, wt_noise=0.2, nstep=5, target_agg='min')
    r.check('⑨ 可构造（nb_actions == nb_turbine == 5）', agent.nb_actions == 5, '')
    r.check('⑨ 动作幅度 = 0.5 = 偏航带半宽',
            bool(np.allclose(agent.action_amp, 0.5)), '{}'.format(agent.action_amp))
    acts = np.array([agent.random_action() for _ in range(200)])
    r.check('⑨ random_action 落在 [−1,+1]（不是克隆源的 [0,0.5]）',
            bool((acts >= -1.0).all() and (acts <= 1.0).all()
                 and acts.min() < -0.9 and acts.max() > 0.9),
            'min={:.3f} max={:.3f}'.format(acts.min(), acts.max()))
    s = np.zeros(4)
    agent.reset(s)
    a = agent.select_action(s)
    r.check('⑨ select_action 输出 5 维', np.asarray(a).reshape(-1).shape == (5,), '')
    r.check('⑨ 折扣/探索与 yawflor 同口径',
            agent.discount == 0.98 and abs(agent.depsilon - 1.0 / 4000) < 1e-15, '')


if __name__ == '__main__':
    rep = Report()
    test_env(rep)
    test_action(rep)
    test_reward(rep)
    test_static(rep)
    test_obs_dim(rep)
    test_nine_case(rep)
    test_agent(rep)
    section('汇总')
    rep.dump()
    if '--json' in sys.argv:
        out = sys.argv[sys.argv.index('--json') + 1]
        with open(out, 'w', encoding='utf-8') as f:
            json.dump([{'check': n, 'ok': ok, 'detail': d} for n, ok, d in rep.rows],
                      f, ensure_ascii=False, indent=2)
        print('json ->', out)
    if rep.fails:
        print('\n**FAIL** {} 项: {}'.format(len(rep.fails), rep.fails))
        sys.exit(1)
    print('\n全部通过 ({} 项)'.format(len(rep.rows)))
