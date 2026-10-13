"""
简单测试脚本：随机风速日 × 单模型

Usage:
    python test_simple.py
"""

import os, sys, argparse, numpy as np, pandas as pd
from copy import deepcopy
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from env import WindBattery
from util import *
import hsrl_soctarget as hst
from myfunc import Normalize, CheckAction, build_L_obs_distilled
from PER_ddpg import SP_DDPG
from prle.prle_ddpg import PRLE_DDPG
from prle.prle_myfunc import (Cal_Rew_PRRBB as Cal_Rew_PRLE,
                              build_L_obs_distilled as build_L_obs_PRLE)
from td3per.td3per_ddpg import SP_TD3 as SP_TD3_PER
from td3prrbb.td3prrbb_ddpg import SP_TD3 as SP_TD3_PRRBB
from td3prle.td3prle_ddpg import SP_TD3_PRLE
from td3prle.td3prle_myfunc import (Cal_Rew_PRRBB as Cal_Rew_TD3PRLE,
                                    build_L_obs_distilled as build_L_obs_TD3PRLE)
from prrbb.prrbb_rule_bess import RuleBasedBESS
from prrbb.prrbb_myfunc import Cal_Rew_PRRBB
from td3prrbb.td3prrbb_myfunc import (Cal_Rew_PRRBB as Cal_Rew_TD3PRRBB,
                                      build_L_obs_distilled as build_L_obs_TD3PRRBB)
from td3per.td3per_myfunc import (Cal_Rew_RB as Cal_Rew_TD3PER,
                                  build_L_obs_distilled as build_L_obs_TD3PER)
from torch.utils.tensorboard import SummaryWriter
import matlab, matlab.engine
import warnings
warnings.filterwarnings("ignore")

MAX_E, NB_TURBINE, TIME = 6.0, 5, 1.0 / 60
EPISODE_LENGTH = 1440


def main():
    print("=" * 55)
    print("  简单测试脚本")
    print("=" * 55)

    # ---- 参数 ----
    print()
    model_path = input("  模型路径: ").strip().strip('"')
    proj_type = input("  项目类型 (PER/PRRBB/PRLE/TD3PER/TD3PRRBB/TD3PRLE, 默认 PER): ").strip() or "PER"
    run_str = input("  测试序号 (run-x, 默认 1): ").strip()
    run_number = int(run_str) if run_str else 1
    n_str = input("  测试天数 (默认 5): ").strip()
    n_days = int(n_str) if n_str else 5
    seed_str = input("  随机种子 (默认 42): ").strip()
    seed = int(seed_str) if seed_str else 42
    label = input("  测试标签 (如 baseline, prle_v2): ").strip() or "test"

    output_dir = input(f"  输出目录 (默认 output/{label}-run-{run_number}): ").strip()
    if not output_dir:
        output_dir = f'output/{label}-run-{run_number}'
    os.makedirs(output_dir, exist_ok=True)

    # ---- 随机风速日 ----
    wind_df = pd.read_csv(os.path.join(_SCRIPT_DIR, 'FFC_speed_noise_1.csv'))
    n_cols = wind_df.shape[1]
    rng = np.random.RandomState(seed)
    day_positions = rng.choice(range(1, n_cols), size=n_days, replace=False)

    print(f"\n随机抽取 {n_days} 个风速日 (seed={seed}):")
    for i, pos in enumerate(day_positions):
        s = wind_df.iloc[:, pos].values.astype(float)
        print(f"  Day {i+1}: iloc[:, {pos:>3d}]  mean={s.mean():.1f}  min={s.min():.1f}  max={s.max():.1f}")

    # ---- Device ----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"\n使用 GPU: {torch.cuda.get_device_name(device)}")

    # ---- Agent ----
    ns = argparse.Namespace(
        hidden1=16, hidden2=100, init_w=0.003, rate=0.01, prate=0.0001,
        warmup=200, epsilon=50000,
        discount=0.98, bsize=32, rmsize=20000, window_length=1, tau=0.001,
        ou_theta=0.15, ou_sigma=0.2, ou_mu=0.0,
        nb_turbine=NB_TURBINE, lam=3.0, lr_lam=0.01, cost_single_limit=1000.0,
        PER_alpha=0.6, PER_beta=0.4, PER_epsilon=0.01,
        beta_increment_per_sampling=0.0, seed=seed, cuda=(device.type == 'cuda'),
        device=device,
    )

    if proj_type == 'PRLE':
        agent_H = PRLE_DDPG(nb_states=4, nb_actions=6, args=ns)
        agent_L = PRLE_DDPG(nb_states=5, nb_actions=1, args=ns, lam_max=100.0)
        agent_L.cost_limit = 10.0
    elif proj_type == 'TD3PRLE':
        agent_H = SP_TD3_PRLE(nb_states=4, nb_actions=6, args=ns)
        agent_L = SP_TD3_PRLE(nb_states=5, nb_actions=1, args=ns, lam_max=100.0)
        agent_L.cost_limit = 10.0
    elif proj_type == 'TD3PER':
        # ⚠️ 本分支仍是 td3per 改造【前】的口径 (H 4 维观测 + 影子价格下层信号),
        #    与改造后的 td3per 权重 (H 6 维相位观测 + SOC 目标下层) 不兼容。
        #    新 td3per 的单文件推理请用:
        #        python test_five_days.py --projects td3per --td3per <模型目录>
        agent_H = SP_TD3_PER(nb_states=4, nb_actions=6, args=ns)
        agent_L = SP_TD3_PER(nb_states=5, nb_actions=1, args=ns, lam_max=100.0)
        agent_L.cost_limit = 10.0
    elif proj_type == 'TD3PRRBB':
        agent_H = SP_TD3_PRRBB(nb_states=4, nb_actions=6, args=ns)
        agent_L = SP_TD3_PRRBB(nb_states=5, nb_actions=1, args=ns, lam_max=100.0)
        agent_L.cost_limit = 10.0
    else:
        agent_H = SP_DDPG(nb_states=4, nb_actions=6, args=ns)
        agent_L = SP_DDPG(nb_states=5, nb_actions=1, args=ns, lam_max=100.0)
        agent_L.cost_limit = 10.0

    agent_H.load_weights(model_path)
    agent_L.load_weights(model_path)
    print(f"模型加载完成: {model_path}")

    # ---- Env & RBB ----
    engine = matlab.engine.start_matlab()
    env = WindBattery(MAX_E, NB_TURBINE, upload=False, engine=engine)
    rbb = RuleBasedBESS(max_E=MAX_E, min_E_factor=0.0, rated_power=0.5,
                        eta_c=0.98, eta_d=0.98, time=TIME, fluctuation_limit=3.0)

    # ---- 测试 ----
    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    all_rows, all_steps = [], []

    for ep in range(n_days):
        wind_series = wind_df.iloc[:, day_positions[ep]]

        agent_H.is_training = False
        agent_L.is_training = False

        obs = deepcopy(env.reset())
        env.wind_day = wind_series.copy()
        env.count_wind = 0
        env.u = float(env.wind_day.iloc[0])
        obs[1] = env.u
        a_temp = np.ones(NB_TURBINE) * 0.3333
        env.P_G_prev = env.PowerOutput(env.u, a_temp)
        obs[-1] = env.P_G_prev

        norm_obs = Normalize(obs)
        agent_H.reset(norm_obs)

        ep_profit, ep_reward = 0.0, 0.0
        cvm = 0      # 违约次数: count of steps where P_F > 3 MW
        cpf = 0.0    # 违约功率绝对值之和: Σ max(P_F − 3, 0) [MW]
        episode_steps, pw_tru = 0, None
        # SOC 目标支: L 观测统一走各项目自己的 5 维构造器 (单一真源 hst.build_L_obs)
        soc_obs_builders = {'TD3PER': build_L_obs_TD3PER, 'TD3PRRBB': build_L_obs_TD3PRRBB,
                            'TD3PRLE': build_L_obs_TD3PRLE, 'PRLE': build_L_obs_PRLE}
        # TD3 全系列: 下层 = **残余动作** (actor 输出 Δ, 命令 = 规则基线 + Δ); PRLE 仍是绝对动作
        residual_proj = ('TD3PER', 'TD3PRRBB', 'TD3PRLE')

        while episode_steps < EPISODE_LENGTH:
            if episode_steps % 10 == 0:
                prCyan(f"step {episode_steps}")
            if episode_steps % 5 == 0:
                action_H = agent_H.select_action(norm_obs)
                action_H, penalty_NN_H = CheckAction(action_H)
                pw_tru = env.PowerOutput(obs[1], action_H)

            if proj_type in soc_obs_builders:
                # SOC 目标支 (TD3PER/TD3PRRBB/TD3PRLE/PRLE): [pg(t−1), ptur, SOC, target, n_rem]/刻度
                _, rbb_info_obs = rbb.compute_pbrule(
                    pb_original=0.0, ptur=float(pw_tru),
                    pg_prev=float(obs[-1]), E=float(obs[2]))
                norm_obs_L = soc_obs_builders[proj_type](
                    obs, pw_tru, action_H, rbb_info_obs, 5 - (episode_steps % 5))
            else:
                norm_obs_L = np.append(norm_obs, action_H[-1])
            action_L = agent_L.select_action(norm_obs_L)
            if proj_type in residual_proj:
                # 命令 = 规则基线 (单一真源 hst.rule_action) + Δ; RBB 仍收"命令值"⇒ 两段 clip 都计费
                pb_rule = hst.rule_action(
                    hst.soc_target_from_signal(action_H[-1]), float(obs[2]),
                    rbb_info_obs['pbrulemin'], rbb_info_obs['pbrulemax'],
                    5 - (episode_steps % 5))
                action_L = np.array([float(pb_rule) + float(action_L[0])])
            action_L, penalty_NN_L = CheckAction(action_L)

            action = np.concatenate((action_H, action_L))
            pbrule, rbb_info = rbb.compute_pbrule(
                pb_original=float(action[-1]), ptur=float(pw_tru),
                pg_prev=float(obs[-1]), E=float(obs[2]))

            action_env = action.copy()
            action_env[-1] = pbrule

            pw_tru2, obs2, _, penalty_L, _, _, info = \
                env.step(action_env[:6], action_env[6:], pw_tru)
            norm_obs2 = Normalize(obs2)

            P_F = info['P_F']  # 提前提取: reward 分支与 cvm 统计共用
            profit_step = (obs2[-1] * obs[0] - BESS_DEG_RMB_PER_MWH * max(-float(pbrule), 0.0)) * TIME

            if proj_type == 'PRRBB':
                # PRRBB: 影子电价支 (r2 含影子套利项)
                r1, c1, r2, c2, _ = Cal_Rew_PRRBB(
                    obs, obs2, action_env, rbb_info,
                    penalty_NN_H=penalty_NN_H, time=TIME, ptur=float(pw_tru))
            elif proj_type == 'TD3PRRBB':
                # TD3PRRBB: SOC 目标支 (r2 = −1·min(clip_amount,20), c2 = +1·trackerr [MW])
                r1, c1, r2, c2, _ = Cal_Rew_TD3PRRBB(
                    obs, obs2, action_env, rbb_info,
                    penalty_NN_H=penalty_NN_H, time=TIME, ptur=float(pw_tru),
                    n_rem=5 - (episode_steps % 5))
            elif proj_type == 'PRLE':
                r1, c1, r2, c2, _ = Cal_Rew_PRLE(
                    obs, obs2, action_env, rbb_info,
                    penalty_NN_H=penalty_NN_H, time=TIME, ptur=float(pw_tru),
                    n_rem=5 - (episode_steps % 5))
            elif proj_type == 'TD3PRLE':
                # TD3PRLE: 动态阈值 + W_GAP*unfeasible_gap 版
                r1, c1, r2, c2, _ = Cal_Rew_TD3PRLE(
                    obs, obs2, action_env, rbb_info,
                    penalty_NN_H=penalty_NN_H, time=TIME, ptur=float(pw_tru),
                    n_rem=5 - (episode_steps % 5))
            elif proj_type == 'TD3PER':
                # TD3PER: SOC 目标支 —— 直接调用项目 myfunc (此前手写影子公式, 已 stale)
                r1, c1, d = Cal_Rew_TD3PER(
                    obs, obs2, action_env, rbb_info,
                    penalty_NN_H=penalty_NN_H, time=TIME, ptur=float(pw_tru))
                # td3per 的 r2/c2 在它自己的 main 里算 (Cal_Rew_RB 只返回 r1/c1/details)
                r2 = hst.lower_reward(float(rbb_info['clip_amount']))
                c2, _c2_info = hst.lower_cost_soc_track(
                    d['soc_target'], float(obs[2]), float(obs2[2]),
                    rbb_info['pbrulemin'], rbb_info['pbrulemax'],
                    5 - (episode_steps % 5), return_info=True)
            else:  # PER (影子电价支)
                _, c = Cal_Rew(obs, obs2, action_env)
                r1 = obs2[-1] * obs[0] * TIME - penalty_NN_H - BESS_DEG_RMB_PER_MWH * max(-float(pbrule), 0.0) * TIME
                # c1 与训练口径一致: 越限惩罚 + 平滑项 + 50*unfeasible_gap (PER/TD3PER 相同)
                c1 = c + 0.1 * P_F + 50.0 * float(rbb_info['unfeasible_gap'])
                signal = action[5]
                shadow_profit = -signal * 2.0 * float(pbrule)
                soc = float(obs2[2])
                soc_dev = max(1.5 - soc, 0.0) + max(soc - 4.5, 0.0)
                r2 = 0.1 * shadow_profit - 1.0 * float(rbb_info['clip_amount'])
                c2 = 10.0 * soc_dev

            ep_profit += profit_step
            ep_reward += (r1 + r2)

            if P_F > 3.0:
                cvm += 1
                cpf += (P_F - 3.0)

            all_steps.append({
                'episode': ep + 1, 'step': episode_steps,
                'soc': float(obs2[2]), 'pf': float(P_F),
            })

            # Step-level TensorBoard (every 10 steps)
            if episode_steps % 10 == 0:
                gs = ep * EPISODE_LENGTH + episode_steps
                writer.add_scalar('step/soc', float(obs2[2]), gs)
                writer.add_scalar('step/pf', float(P_F), gs)

            norm_obs, obs, pw_tru = norm_obs2, obs2, pw_tru2
            episode_steps += 1

        print(f"  Day {ep+1:>3d}: profit={ep_profit:>10.2f}  "
              f"cvm={cvm:>5d}  cpf={cpf:>8.2f}")

        writer.add_scalar('episode/profit', ep_profit, ep)
        writer.add_scalar('episode/cvm', cvm, ep)
        writer.add_scalar('episode/cpf', cpf, ep)

        all_rows.append({
            'episode': ep + 1, 'wind_iloc': day_positions[ep],
            'episode_profit': ep_profit, 'cvm': cvm, 'cpf': cpf,
        })

    writer.close()
    engine.quit()

    pd.DataFrame(all_rows).to_csv(os.path.join(output_dir, 'test_simple.csv'), index=False)
    pd.DataFrame(all_steps).to_csv(os.path.join(output_dir, 'test_simple_steps.csv'), index=False)

    print(f"\n{'='*55}")
    print(f"  均值: profit={np.mean([r['episode_profit'] for r in all_rows]):.2f}  "
          f"cvm={np.mean([r['cvm'] for r in all_rows]):.1f}  "
          f"cpf={np.mean([r['cpf'] for r in all_rows]):.2f}")
    print(f"  CSV → {output_dir}/test_simple.csv")
    print(f"  TB  → tensorboard --logdir={output_dir}/tensorboard")


def Cal_Rew(observation, observation2, action_all, cost_single_cons=3.0):
    pr, p_g, p_g_last = observation[0], observation2[-1], observation[-1]
    P_B = action_all[-1]
    r = (p_g * pr - BESS_DEG_RMB_PER_MWH * max(-P_B, 0)) * TIME
    P_F = abs(p_g_last - p_g)
    c = 10.0 * (P_F - cost_single_cons) if P_F > cost_single_cons else 0.0
    return r, c


if __name__ == '__main__':
    main()
