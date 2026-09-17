"""
多项目对比测试脚本 (PER / PRRBB / PRLE)
========================================

功能:
  1. 文本方式输入各项目的模型路径，不输入则跳过
  2. 随机抽取风速天数，三个项目共享完全相同的风速序列（不使用固定随机序列）
  3. 电价按 env.CalPrice() 分布（峰谷平价）
  4. 每个项目使用各自的训练 reward 结构进行评估
  5. 输出 episode_reward / profit 到 CSV + TensorBoard

测试流程:
  每个 episode 对应随机抽取的一个风速日（1440分钟=1天），三个项目用同一日运行。
  使用 env.step() → 动态电价 CalPrice()。

Usage:
  python test_compare_projects.py
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from copy import deepcopy
import torch

# ---- Path setup ----
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from env import WindBattery
from util import *
from myfunc import Normalize, CheckAction
from memory import RingBuffer
from PER_ddpg import SP_DDPG
from prrbb.prrbb_rule_bess import RuleBasedBESS
from prle.prle_ddpg import PRLE_DDPG
from prrbb.prrbb_myfunc import Cal_Rew_PRRBB
from prle.prle_myfunc import get_wind_thresholds, set_wind_thresholds
from torch.utils.tensorboard import SummaryWriter

import matlab
import matlab.engine

import warnings
warnings.filterwarnings("ignore")

# ---- Constants ----
MAX_E = 6.0
NB_TURBINE = 5
TIME = 1.0 / 60
EPISODE_LENGTH = 1440  # 1 day
RBB_FLUCTUATION_LIMIT = 3.0
RBB_RATED_POWER = 10.0


def make_args(device):
    """Build a minimal args namespace for agent instantiation."""
    ns = argparse.Namespace()
    ns.hidden1 = 16
    ns.hidden2 = 100
    ns.rate = 0.01
    ns.prate = 0.0001
    ns.warmup = 200
    ns.discount = 0.98
    ns.bsize = 32
    ns.rmsize = 20000
    ns.window_length = 1
    ns.tau = 0.001
    ns.ou_theta = 0.15
    ns.ou_sigma = 0.2
    ns.ou_mu = 0.0
    ns.validate_episodes = 20
    ns.max_episode_length = 500
    ns.validate_steps = 2000
    ns.output = 'output'
    ns.debug = False
    ns.init_w = 0.003
    ns.train_iter_H = 200000
    ns.epsilon = 50000
    ns.seed = -1
    ns.resume = 'default'
    ns.cuda = (device.type == 'cuda')
    ns.lam = 3.0
    ns.lr_lam = 0.01
    ns.cost_single_limit = 1000.0
    ns.PER_alpha = 0.6
    ns.PER_beta = 0.4
    ns.PER_epsilon = 0.01
    ns.beta_increment_per_sampling = 0.0
    ns.nb_turbine = NB_TURBINE
    ns.rbb_fluctuation_limit = RBB_FLUCTUATION_LIMIT
    ns.rbb_rated_power = RBB_RATED_POWER
    ns.device = device
    return ns


# =====================================================================
# Cal_Rew (PER's training reward, from myfunc.py)
# =====================================================================

def Cal_Rew(observation, observation2, action_all, cost_single_cons=3.0):
    """PER training cost function. Returns (r, c)."""
    pr = observation[0]
    p_g = observation2[-1]
    p_g_last = observation[-1]
    P_B = action_all[-1]
    r = (p_g * pr - BESS_DEG_RMB_PER_MWH * max(-P_B, 0)) * TIME
    P_F = abs(p_g_last - p_g)
    if P_F > cost_single_cons:
        c = 10.0 * (P_F - cost_single_cons)
    else:
        c = 0.0
    return r, c


# =====================================================================
# Test one episode
# =====================================================================

def test_one_episode(env, agent_H, agent_L, rbb_system,
                     wind_series, project_type, episode_length=1440):
    """
    Run a single test episode using env.step() (dynamic pricing via CalPrice()).

    Each project uses its own training reward structure:
      - PER:    r1 = revenue - penalty_NN_H - 490·max(-pbrule,0)·time
                c1 = 10·max(P_F−3, 0)
                r2 = 0.1·shadow_profit − 1·clip_amount
                c2 = 10·soc_deviation
      - PRRBB:  Cal_Rew_PRRBB (segmented wind×price reward)
      - PRLE:   same as PRRBB

    Returns dict with per-episode metrics.
    """
    agent_H.is_training = False
    agent_L.is_training = False

    # ---- Start of episode: use env.reset() for price + battery init ----
    obs = deepcopy(env.reset())  # [pr, u, E, P_G]

    # Override wind_day with our shared random day
    env.wind_day = wind_series.copy()
    env.count_wind = 0
    env.u = float(env.wind_day.iloc[0])
    obs[1] = env.u

    # Fix P_G_prev: reset() computed it with fixed-sequence wind, recompute with shared wind
    a_temp = np.ones(env.num_turbine) * 0.3333
    temp = env.PowerOutput(env.u, a_temp)
    env.P_G_prev = temp
    obs[-1] = temp

    norm_obs = Normalize(obs)
    agent_H.reset(norm_obs)

    episode_steps = 0
    episode_reward = 0.0
    episode_r1 = 0.0
    episode_c1 = 0.0
    episode_r2 = 0.0
    episode_c2 = 0.0
    episode_profit = 0.0   # common metric: Σ(P_G·pr·dt − 490·max(−pb,0)·dt)
    cpf = 0                 # count of steps where |P_F| > 3 MW
    cvm = 0.0               # sum of excess: Σ max(|P_F| − 3, 0)  [MW]
    step_records = []       # per-step data: profit_step, soc, P_F
    pw_tru = None

    while episode_steps < episode_length:

        # ---- Upper Level (H) — every 5 steps ----
        if episode_steps % 5 == 0:
            sum_r1 = 0.0
            sum_c1 = 0.0
            action_H = agent_H.select_action(norm_obs)
            action_H, penalty_NN_H = CheckAction(action_H)
            pw_tru = env.PowerOutput(obs[1], action_H)

        # ---- Lower Level (L) — every step ----
        norm_obs_L = np.append(norm_obs, action_H[-1])
        action_L = agent_L.select_action(norm_obs_L)
        action_L, penalty_NN_L = CheckAction(action_L)

        action = np.concatenate((action_H, action_L))

        # ---- RBB clipping ----
        pb_original = float(action[-1])
        ptur = float(pw_tru)
        pg_prev = float(obs[-1])
        E_current = float(obs[2])

        pbrule, rbb_info = rbb_system.compute_pbrule(
            pb_original=pb_original,
            ptur=ptur,
            pg_prev=pg_prev,
            E=E_current,
        )

        # Apply hard clipping to env action
        action_env = action.copy()
        action_env[-1] = pbrule

        # ---- Environment step ----
        pw_tru2, obs2, action_L_env, penalty_L, done, done_L, info = \
            env.step(action_env[:6], action_env[6:], pw_tru)

        norm_obs2 = Normalize(obs2)

        # ================================================================
        # Project-specific reward computation
        # ================================================================
        if project_type in ('PRRBB', 'PRLE'):
            # ---- PRRBB / PRLE: segmented wind×price reward ----
            r1, c1, r2, c2, reward_details = Cal_Rew_PRRBB(
                obs, obs2, action_env, rbb_info,
                penalty_NN_H=penalty_NN_H, time=TIME,
                ptur=ptur,
            )
            # Common metric
            profit_step = (obs2[-1] * obs[0]
                           - BESS_DEG_RMB_PER_MWH * max(-float(pbrule), 0.0)) * TIME

        else:
            # ---- PER: flat revenue reward ----
            _, c = Cal_Rew(obs, obs2, action_env)

            # r1 = revenue - penalty_NN_H - degradation
            r1 = (obs2[-1] * obs[0] * TIME
                  - penalty_NN_H
                  - BESS_DEG_RMB_PER_MWH * max(-float(pbrule), 0.0) * TIME)
            c1 = c

            # r2 = shadow_profit - 1*clip_amount (pbrule 裁剪惩罚)
            signal = action[5]  # shadow price signal from H-agent
            shadow_price = signal * 2.0  # (-0.5, 0.5) → (-1, 1)
            shadow_profit = -shadow_price * float(pbrule)

            soc = float(obs2[2])
            soc_safe_min, soc_safe_max = 1.5, 4.5
            soc_deviation = 0.0
            if soc < soc_safe_min:
                soc_deviation = soc_safe_min - soc
            elif soc > soc_safe_max:
                soc_deviation = soc - soc_safe_max

            r2 = 0.1 * shadow_profit - 1.0 * float(rbb_info['clip_amount'])

            # c2 = SOC 越界项（拉格朗日约束, 系数 10/MWh = 原 1000 ÷100）
            c2 = 10.0 * soc_deviation

            # Common metric
            profit_step = (obs2[-1] * obs[0]
                           - BESS_DEG_RMB_PER_MWH * max(-float(pbrule), 0.0)) * TIME

        r = r1 + r2
        c = c1

        # ---- Accumulate ----
        sum_r1 += r1
        sum_c1 += c1
        episode_reward += r
        episode_r1 += r1
        episode_c1 += c1
        episode_r2 += r2
        episode_c2 += c2
        episode_profit += profit_step

        # Grid fluctuation tracking
        P_F = info['P_F']
        if P_F > 3.0:
            cpf += 1
            cvm += (P_F - 3.0)

        # Per-step record: profit_step, SOC, P_F
        step_records.append({
            'step': episode_steps,
            'profit_step': profit_step,
            'soc': float(obs2[2]),
            'pf': float(P_F),
        })

        # ---- Transition ----
        norm_obs = deepcopy(norm_obs2)
        obs = deepcopy(obs2)
        pw_tru = pw_tru2
        episode_steps += 1

    return {
        'episode_reward': episode_reward,
        'episode_r1': episode_r1,
        'episode_c1': episode_c1,
        'episode_r2': episode_r2,
        'episode_c2': episode_c2,
        'episode_profit': episode_profit,
        'cpf': cpf,
        'cvm': cvm,
        'step_records': step_records,   # per-step: profit_step, soc, pf
    }


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 65)
    print("  多项目对比测试: PER / PRRBB / PRLE")
    print("  共享随机风速日 + 动态电价(峰谷平)")
    print("=" * 65)

    # ---- 1. Model paths ----
    print()
    print("请输入各项目的模型路径（直接回车跳过该项目）:")
    print("-" * 55)
    per_path = input("  PER   模型路径: ").strip().strip('"')
    prrbb_path = input("  PRRBB 模型路径: ").strip().strip('"')
    prle_path = input("  PRLE  模型路径: ").strip().strip('"')

    projects = []
    if per_path:
        projects.append(('PER', per_path))
    if prrbb_path:
        projects.append(('PRRBB', prrbb_path))
    if prle_path:
        projects.append(('PRLE', prle_path))

    if not projects:
        print("\n[错误] 未提供任何模型路径，退出。")
        return

    print(f"\n将测试 {len(projects)} 个项目: {[p[0] for p in projects]}")

    # ---- 2. Test config ----
    print("-" * 55)
    num_ep_str = input("  测试 episode 数量（默认 10）: ").strip()
    num_episodes = int(num_ep_str) if num_ep_str else 10
    seed_str = input("  随机种子（默认 42）: ").strip()
    random_seed = int(seed_str) if seed_str else 42
    output_dir = input("  输出目录（默认 output/test_compare）: ").strip()
    if not output_dir:
        output_dir = 'output/test_compare'
    os.makedirs(output_dir, exist_ok=True)

    # ---- 3. Device ----
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        print(f"\n使用 GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("\n使用 CPU")

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(random_seed)

    # ---- 4. Generate shared random wind days ----
    # Read EXACTLY as env.py does: no index_col=0, so iloc[:,0]=Unnamed, iloc[:,1..]=wind data
    wind_df = pd.read_csv(os.path.join(_SCRIPT_DIR, 'FFC_speed_noise_1.csv'))
    n_cols = wind_df.shape[1]
    print(f"\n风速 CSV: {n_cols} 列 "
          f"(iloc[:,0]='Unnamed:0', iloc[:,1..{n_cols-1}]=风速数据)")

    # Sample iloc positions [1, n_cols) — same as env.reset()'s FIXED_DAY_SEQUENCE range
    rng = np.random.RandomState(random_seed)
    shared_day_positions = rng.choice(
        range(1, n_cols), size=num_episodes, replace=False
    )

    print(f"随机抽取 {num_episodes} 个共享风速日:")
    for i, pos in enumerate(shared_day_positions):
        series = wind_df.iloc[:, pos].values.astype(float)
        col_name = wind_df.columns[pos]
        print(f"  Ep {i+1:>3d}: iloc[:, {pos:>3d}] (col name=\"{col_name}\")  "
              f"mean={series.mean():.2f}, min={series.min():.2f}, max={series.max():.2f}")

    # ---- 5. MATLAB engine ----
    print("\n启动 MATLAB 引擎...")
    engine = matlab.engine.start_matlab()

    # ---- 6. Shared env & RBB system ----
    env = WindBattery(MAX_E, NB_TURBINE, upload=False, engine=engine)
    rbb_system = RuleBasedBESS(
        max_E=MAX_E,
        min_E_factor=0.0,
        rated_power=RBB_RATED_POWER,
        eta_c=0.98,
        eta_d=0.98,
        time=TIME,
        fluctuation_limit=RBB_FLUCTUATION_LIMIT,
    )

    args = make_args(device)

    # ---- 7. Load agents ----
    agents = {}
    for proj_name, proj_path in projects:
        print(f"\n加载 {proj_name} 模型: {proj_path}")
        if proj_name == 'PRLE':
            agent_H = PRLE_DDPG(nb_states=4, nb_actions=6, args=args)
            agent_L = PRLE_DDPG(nb_states=5, nb_actions=1, args=args, lam_max=100.0)
            agent_L.cost_limit = 10.0
        else:  # PER, PRRBB both use SP_DDPG
            agent_H = SP_DDPG(nb_states=4, nb_actions=6, args=args)
            agent_L = SP_DDPG(nb_states=5, nb_actions=1, args=args, lam_max=100.0)
            agent_L.cost_limit = 10.0

        agent_H.load_weights(proj_path)
        agent_L.load_weights(proj_path)
        agents[proj_name] = (agent_H, agent_L)
        print(f"  ✓ {proj_name} 模型加载完成")

    # ---- 8. Run tests ----
    print(f"\n{'='*65}")
    print(f"  开始测试 ({num_episodes} episodes × {len(projects)} 项目)")
    print(f"{'='*65}")

    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    all_results = []
    all_step_records = []   # per-step records across all episodes

    for ep_idx in range(num_episodes):
        day_pos = shared_day_positions[ep_idx]
        wind_series = wind_df.iloc[:, day_pos]

        print(f"\n--- Episode {ep_idx + 1}/{num_episodes} "
              f"(wind iloc[:, {day_pos}]) ---")

        ep_results = {}  # project_name → result dict (for computing profit_mape)

        for proj_name, (agent_H, agent_L) in agents.items():
            # Reset PRLE thresholds to default each episode
            if proj_name == 'PRLE':
                set_wind_thresholds(7.5, 8.5)

            result = test_one_episode(
                env, agent_H, agent_L, rbb_system,
                wind_series, proj_name, episode_length=EPISODE_LENGTH,
            )
            ep_results[proj_name] = result

            # Per-step records → flat table + TensorBoard
            step_data = result.pop('step_records')
            for sr in step_data:
                all_step_records.append({
                    'episode': ep_idx + 1,
                    'wind_iloc': day_pos,
                    'project': proj_name,
                    **sr,
                })
                # Per-step TensorBoard (every 10 steps to limit I/O)
                if sr['step'] % 10 == 0:
                    global_step = ep_idx * EPISODE_LENGTH + sr['step']
                    writer.add_scalar(f'{proj_name}/step_profit', sr['profit_step'], global_step)
                    writer.add_scalar(f'{proj_name}/step_soc', sr['soc'], global_step)
                    writer.add_scalar(f'{proj_name}/step_pf', sr['pf'], global_step)

            print(f"  {proj_name:>6s}: "
                  f"profit={result['episode_profit']:>10.2f}, "
                  f"cpf={result['cpf']:>5d}, "
                  f"cvm={result['cvm']:>8.2f}, "
                  f"reward={result['episode_reward']:>10.2f}")

        # ---- Compute profit_mape: (proj_profit - PER_profit) / |PER_profit| * 100% ----
        if 'PER' in ep_results:
            per_profit = ep_results['PER']['episode_profit']
            for proj_name in ['PRRBB', 'PRLE']:
                if proj_name in ep_results:
                    if abs(per_profit) > 1e-8:
                        profit_mape = (ep_results[proj_name]['episode_profit'] - per_profit) / abs(per_profit) * 100.0
                    else:
                        profit_mape = float('nan')
                    ep_results[proj_name]['profit_mape'] = profit_mape
                    print(f"           {proj_name} profit_mape (vs PER): {profit_mape:+.2f}%")
        # PER itself has no profit_mape
        for proj_name in ep_results:
            if 'profit_mape' not in ep_results[proj_name]:
                ep_results[proj_name]['profit_mape'] = float('nan')

        # ---- TensorBoard + CSV ----
        for proj_name, result in ep_results.items():
            tag = proj_name
            for key in ['episode_profit', 'cpf', 'cvm', 'profit_mape',
                        'episode_reward', 'episode_r1', 'episode_c1',
                        'episode_r2', 'episode_c2']:
                val = result.get(key, float('nan'))
                if not np.isnan(val):
                    writer.add_scalar(f'{tag}/{key}', val, ep_idx)

            all_results.append({
                'episode': ep_idx + 1,
                'wind_iloc': day_pos,
                'project': proj_name,
                **result,
            })

        # Incremental save (episode summary)
        df_ep = pd.DataFrame(all_results)
        df_ep.to_csv(os.path.join(output_dir, 'test_compare.csv'), index=False)
        # Incremental save (per-step)
        df_step = pd.DataFrame(all_step_records)
        df_step.to_csv(os.path.join(output_dir, 'test_compare_steps.csv'), index=False)

    # ---- 9. Summary ----
    writer.close()
    df = pd.DataFrame(all_results)

    print(f"\n{'='*65}")
    print(f"  汇总 (每项目 {num_episodes} episodes 均值)")
    print(f"{'='*65}")
    print(f"  {'Project':>8s}  {'profit_avg':>12s}  {'cpf_avg':>8s}  "
          f"{'cvm_avg':>10s}  {'profit_mape':>12s}")
    print(f"  {'-'*60}")
    for proj_name in [p[0] for p in projects]:
        sub = df[df['project'] == proj_name]
        profit_avg = sub['episode_profit'].mean()
        cpf_avg = sub['cpf'].mean()
        cvm_avg = sub['cvm'].mean()
        mape_avg = sub['profit_mape'].dropna()
        mape_str = f"{mape_avg.mean():+.2f}%" if len(mape_avg) > 0 else "N/A"
        print(f"  {proj_name:>8s}  {profit_avg:>12.2f}  {cpf_avg:>8.1f}  "
              f"{cvm_avg:>10.2f}  {mape_str:>12s}")

    print(f"\nEpisode 汇总 CSV: {os.path.join(output_dir, 'test_compare.csv')}")
    print(f"Per-step 明细 CSV: {os.path.join(output_dir, 'test_compare_steps.csv')}")
    print(f"TensorBoard:       {os.path.join(output_dir, 'tensorboard')}")
    print(f"\n运行:  tensorboard --logdir={os.path.join(output_dir, 'tensorboard')}")

    engine.quit()


if __name__ == '__main__':
    main()
