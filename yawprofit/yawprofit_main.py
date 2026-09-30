#!/usr/bin/env python3
"""
YAWPROFIT — **只调度 FLORIS 的 5-min 发电利润最大化智能体**（单层 TD3 + PER）
====================================================================================
本目录由 `yawflor` 派生（"从 yawflor 里改出来一个"），用来把"学习不稳定"从
**层次结构 / 电池耦合 / 约束对偶**里彻底隔离出来：只剩一个每 5 min 决策一次的
上层智能体，动作只有偏航，奖励只有发电利润。

口径（单一真源见 `yawprofit_myfunc.py` 顶部的差异表）
------------------------------------------------------
  * 被控对象 = `WindTurbineYaw`（无储能）：`P_G ≡ P_tur`；
  * **state**  = `[pr/1000, u/10]` + 24h 相位特征（`--phase_feat` 默认 1 ⇒ 4 维）
                 —— 没有 SOC、没有上一分钟功率、没有可行域边界；
  * **action** = 5 维无量纲偏航指令 `n ∈ [−1,+1]`（env 内 `γ = 25°·n`，块级 5 步固定）
                 —— 没有 yawflor 的第 6 维 SOC 目标信号；
  * **reward** = `r1 = pr·P_tur·dt − penalty_NN_H`（逐分钟）;  **c1 ≡ 0**
                 —— 没有 `bess_profit`、没有退化 490 元/MWh、没有波动约束、没有 gap 罚；
  * **cost 通道关闭**：`agent_H.lam = 0` ⇒ actor 目标精确退化为 `−Q_p`（纯利润最大化）。
    保留 `critic_c/critic_c2` 只是为了让本文件与 `yawflor_main` 的控制流同源
    （见 `yawprofit_ddpg.py` 的说明），它们回归到恒 0，不产生任何策略梯度。

与 yawflor 的**算法**相同之处（刻意保持，便于对照）
---------------------------------------------------
TD3（twin critics + 延迟策略更新 + target smoothing）、PER（`PER_buffer` 原样）、
n 步回报滑窗（`--nstep` 默认 5）、EMA 化的 `observe_jump` 块级接口、warmup 期的
`hsrl_warmup.WarmupSchedule`（偏航带 ±1、起点 0、每块 ±0.2）、critic 输出层 bias 的
价值基线初始化（`--critic_bias_init estimate`）、值函数回归空间缩放 `--value_scale`、
24h 相位特征与"电价时钟自检"、CSV 增量落盘、每 2000 步 checkpoint。

**刻意不搬**的东西（都在 yawflor 里为电池/约束而存在）
------------------------------------------------------
`RuleBasedBESS`（可行窗口 + 硬裁剪）、下层 agent、`hsrl_soctarget` 的 r2/c2、
`yawflor_dual` 的双通道对偶、块末 SOC gap 罚、`--viol_derate` 乘性越界罚
（yawflor 默认 0 ⇒ 等价于不存在）、6 维动作的信号维。
"""
import os
import sys
import argparse
from copy import deepcopy

import numpy as np
import torch

# Ensure both yawprofit/ and parent (HSRL) directories are on path
_YPDIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_YPDIR)
sys.path.insert(0, _YPDIR)
sys.path.insert(0, _ROOT)

# 终端编码守卫：Windows 下 stdout 被重定向/走管道时用的是 locale 编码（cp936），
# 而本项目文案含 '−'（U+2212）/'γ'/'⇒'/'≡' 等 GBK 没有的字符 ⇒ 会 UnicodeEncodeError
# **把训练直接打断**（日志重定向到文件就会踩到）。这里只把不可编码字符降级为 '?'，
# 不改编码、不吞其它错误；CSV/TensorBoard 落盘仍是 UTF-8。
try:
    sys.stdout.reconfigure(errors='replace')
except Exception:                                            # pragma: no cover
    pass

from yawprofit.yawprofit_env import FlorisPhysicsYaw, WindTurbineYaw   # noqa: E402
from yawprofit.yawprofit_ddpg import SP_TD3                            # noqa: E402
from yawprofit.yawprofit_myfunc import (                               # noqa: E402
    CheckAction, Normalize_H, Cal_Rew_WT, nine_case_test, violation_amount,
    PHASE_DIM, day_price_schedule, phase_minute_of_day, phase_daily_table,
    PENALTY_NN_H_COEF,
)
from yawflor import yawflor_turbine as yturb                           # noqa: E402
from util import *                                                     # noqa: E402,F401,F403

import hsrl_cost as hc          # 值函数回归空间缩放 (value_scale / cost_scale)  # noqa: E402
import hsrl_agent_init as hinit  # critic 输出层 bias 的价值基线               # noqa: E402
import hsrl_warmup as hwarm      # warmup 调度 + 偏航带单一真源                # noqa: E402

from torch.utils.tensorboard import SummaryWriter                      # noqa: E402

# 不显示warning
import warnings                                                        # noqa: E402
warnings.filterwarnings("ignore")


RUN_TAG = 'yawprofit'


def init_wt_action_bias(agent, lo=None, hi=None, verbose=True):
    """把 actor 输出层 **5 维偏航指令** 的 bias 初始化到"中性满带" `U(lo, hi)`。

    为什么必须做（与 yawflor 同因）：`model.py` 的 `Actor` 只把 `fc4.weight` 初始化成
    ±init_w(0.003)，`fc4.bias` 仍是 PyTorch 默认 U(±1/√fan_in)=U(±0.25)（fan_in=16）；
    偏航动作的语义是 `n = 0 ⇒ γ = 0°`（单机自损最小），而随机 bias 会让初始策略
    一上来就带一个非零偏航。这里与 `hsrl_warmup` 的 warmup 调度**同分布**
    （`[lo,hi] = [−1,+1]`），只编码**单机物理**，不预设"上游转向、下游归零"的风场最优形状
    —— 那是策略要学的函数。

    ⚠ yawprofit 的动作只有 5 维（没有 yawflor 的第 6 维 SOC 信号），
    因此这里把**全部** bias 维都当偏航维处理。
    """
    if lo is None:
        lo = hwarm.YAW_LO
    if hi is None:
        hi = hwarm.YAW_HI
    a = agent.actor
    if not hasattr(a, 'fc4'):
        return False
    nb = a.fc4.bias.data.shape[0]
    with torch.no_grad():
        b = a.fc4.bias.data
        b[:] = torch.tensor(np.random.uniform(lo, hi, nb), dtype=b.dtype)
        agent.actor_target.fc4.bias.data.copy_(b)
    if verbose:
        g = yturb.GAMMA_MAX * b.numpy()
        prGreen('init_wt_action_bias (yaw): n bias = {} (mean n = {:.3f} -> mean γ = {:.2f}°)'
                .format(np.round(b.numpy(), 3), float(b.mean()), float(g.mean())))
    return True


def train(num_iterations_H,
          agent_H,
          env,
          env_test,
          output,
          physics_ref=None,
          max_episode_length=None,
          debug=False,
          upload=False,
          use_csv=False):
    """单层训练循环：每 5 个环境步（= 5 min）上层决策一次，逐分钟累计发电收益。"""
    agent_H.is_training = True
    step = episode = episode_steps = 0
    observation = None
    time = 1.0 / 60
    # 每个 episode（= 1 天 = 1440 步）的累计量
    episode_reward = episode_profit = episode_penalty = 0.
    episode_energy = 0.
    episode_PF_sum = 0.
    episode_viol = 0
    episode_yaw_deg = 0.

    if use_csv:
        df_rows = []
        df_written = 0
        df_episode_rows = []
        df_episode_written = 0
        df_test_rows = []
        df_test_written = 0
        df_update_rows = []
        df_update_written = 0

    # ---- 24h 相位特征 (--phase_feat) ----
    phase_mode = int(getattr(args, 'phase_feat', 0))
    phase_table = phase_daily_table() if phase_mode >= 2 else None
    phase_sched = day_price_schedule()
    phase_mismatch = 0
    phase_checked = 0

    # ---- warmup 期上层确定性动作调度（偏航带单一真源 hsrl_warmup.YAW_*）----
    #   注意：yawprofit **没有** warmup 期的下层规则行为体（BESS 已抹去），
    #   所以这里直接构造 WarmupSchedule，不经 hsrl_warmup.build_from_args
    #   （后者会顺带造一个 WarmupBessRule —— 本臂用不到它）。
    warmup_sched = hwarm.WarmupSchedule(
        nb_turbine=int(args.nb_turbine), nb_actions=int(args.nb_turbine),
        start=float(args.warmup_wt_start), step=float(args.warmup_wt_step),
        lo=float(args.warmup_wt_lo), hi=float(args.warmup_wt_hi),
        seed=(int(args.seed) if int(args.seed) > 0 else None),
        enabled=(str(args.warmup_sched) != 'random'))

    # ---- critic 输出层 bias（价值基线）----
    bias_init = hinit.BiasInitTracker.from_args(args)
    bias_init.value_scale = float(getattr(agent_H, 'value_scale', hc.value_scale()))
    bias_init.discount = float(agent_H.discount)

    while step < num_iterations_H:
        prYellow('########### step:{} ###########'.format(step))
        if observation is None:
            observation = deepcopy(env.reset())          # [pr, u, P_tur]
            norm_observation = Normalize_H(observation, phase_minute_of_day(0),
                                           phase_mode, phase_table)
            agent_H.reset(norm_observation)

        # 相位自检: env 的电价必须与 day_price_schedule() 在 episode_steps%1440 处一致
        pm_now = phase_minute_of_day(episode_steps)
        if phase_mode > 0:
            if abs(float(observation[0]) - phase_sched[pm_now]) > 1e-6:
                phase_mismatch += 1
            phase_checked += 1

        # ================================================================
        # 块首 (每 5 步 = 5 min): 上层决策一次, 块内偏航保持不变
        # ================================================================
        if episode_steps % 5 == 0:
            sum_r1 = 0.0
            sum_pen = 0.0
            sum_profit = 0.0
            sum_PF = 0.0
            if step <= args.warmup:
                _wa = warmup_sched.action()          # None ⇒ 对照臂 (--warmup_sched random)
                applied_action_H = original_action_H = (
                    agent_H.random_action() if _wa is None else _wa)
                penalty_NN_H = 0.0                   # 调度动作在带内, 与 yawflor 同口径
            else:
                original_action_H = agent_H.select_action(norm_observation)
                applied_action_H, penalty_NN_H = CheckAction(original_action_H)
            # 越界量 (诊断; 罚 = PENALTY_NN_H_COEF × 本量)
            action_excess = violation_amount(original_action_H)
            norm_observation_H = norm_observation
            ptur = float(env.PowerOutput(observation[1], applied_action_H))

        prGreen('--H-- ori a_t = {} | γ = {}°'.format(
            round_list(original_action_H, 3),
            np.round(yturb.GAMMA_MAX * np.asarray(applied_action_H), 2)))

        # ================================================================
        # 逐步奖励: r1 = pr·P_tur·dt − penalty_NN_H;  c1 ≡ 0
        # ================================================================
        r1, c1_step, details = Cal_Rew_WT(observation, ptur,
                                          penalty_NN_H=penalty_NN_H, time=time)
        sum_r1 += r1
        sum_pen += penalty_NN_H
        sum_profit += details['turbine_profit']

        # env 前进 1 min（无电池: P_G ≡ P_tur）
        pw_tru2, observation2, info = env.step(applied_action_H)
        P_F = float(info['P_F'])
        sum_PF += P_F
        details['P_F'] = P_F
        norm_observation2 = Normalize_H(observation2,
                                        phase_minute_of_day(episode_steps + 1),
                                        phase_mode, phase_table)

        prGreen('r1 = {:.3f} (发电收益={:.3f}, 越界罚={:.1f}) | P_tur={:.3f} MW '
                'P_G={:.3f} MW P_F={:.3f} MW | SOC/电池项: 无'.format(
                    r1, details['turbine_profit'], penalty_NN_H, ptur,
                    info['P_G'], P_F))
        prGreen('price={} p_level={} wind={:.1f} w_level={}'.format(
            details['pr'], details['price_level'], details['u'], details['wind_level']))

        if max_episode_length and episode_steps >= max_episode_length - 1:
            done = True
        else:
            done = False

        # ================================================================
        # 块末: 上层块级 transition 入池 (n 步滑窗在 agent 内部完成)
        # ================================================================
        if episode_steps % 5 == 4:
            # critic bias: warmup 后首个块末一次性应用 (首次策略更新之前)
            bias_init.on_block_end(step, sum_r1, 0.0, agent_H)
            agent_H.observe_jump(norm_observation_H, original_action_H,
                                 sum_r1, 0.0, norm_observation2, done)

        # ================================================================
        # 更新 (每块 2 次, 与 yawflor H 层同频)
        # ================================================================
        if step > args.warmup and episode_steps % 5 == 4:
            for _ in range(2):
                agent_H.update_policy()
            if agent_H.record_update_stats and len(agent_H.update_stats) > 0:
                _stats = agent_H.update_stats
                _mean = {k: float(np.mean([s[k] for s in _stats])) for k in _stats[0]}
                agent_H.update_stats = []
                if upload:
                    for _k, _v in _mean.items():
                        writer.add_scalar('update/' + _k, _v, step)
                if use_csv:
                    df_update_rows.append({'step': step, **_mean})
                    df_update_written = flush_csv_append_batch(
                        '{}/update_stats_H.csv'.format(output),
                        df_update_rows, df_update_written)
                    df_update_rows = []

        # Test: 9-case 分段评测, 每 480 环境步一次 (固定 scenario seed=42)
        if step > args.warmup and episode_steps % 480 == 0:
            case_metrics, overall = nine_case_test(
                agent_H, env_test, seed=42,
                phase_mode=phase_mode, phase_table=phase_table,
                physics_ref=physics_ref)
            if use_csv:
                for m in case_metrics:
                    df_test_rows.append({'step': step,
                                         'wind_level': m['wind_level'],
                                         'price_level': m['price_level'],
                                         **_test_row(m)})
                df_test_rows.append({'step': step, 'wind_level': 'OVERALL',
                                     'price_level': 'OVERALL', **_test_row(overall)})
                df_test_written = flush_csv_append_batch(
                    '{}/df_test9.csv'.format(output), df_test_rows, df_test_written)
                df_test_rows = []
            if upload:
                for _k, _v in overall.items():
                    writer.add_scalar('test9/overall/' + _k, _v, step)
                for m in case_metrics:
                    tag = 'test9/{}_{}'.format(m['wind_level'], m['price_level'])
                    for _k, _v in m.items():
                        if _k in ('wind_level', 'price_level'):
                            continue
                        writer.add_scalar(tag + '/' + _k, _v, step)

        # ---- 逐步数据 (累积行, 每 50 步增量 flush) ----
        if use_csv:
            df_rows.append({
                'step': step,
                'observation': observation,
                'observation2': observation2,
                'phase_minute': pm_now,
                'phase_sin': float(np.sin(2 * np.pi * pm_now / 1440)),
                'phase_cos': float(np.cos(2 * np.pi * pm_now / 1440)),
                'phase_rest_frac': (float(phase_table[pm_now, 0]) if phase_table is not None else np.nan),
                'phase_rest_h': (float(phase_table[pm_now, 1]) * 8 if phase_table is not None else np.nan),
                'phase_next_pr': (float(phase_table[pm_now, 2]) * 1000 if phase_table is not None else np.nan),
                'original_action_H': original_action_H,
                'applied_action_H': applied_action_H,
                'action_excess': action_excess,
                'yaw_deg_mean': float(np.abs(yturb.GAMMA_MAX * np.asarray(applied_action_H)).mean()),
                'ptur': ptur,
                'r1': r1,
                'c1': c1_step,
                'profit_step': details['turbine_profit'],
                'penalty_NN_H': penalty_NN_H,
                'P_F': P_F,
                'pr': details['pr'],
                'u': details['u'],
                'wind_level': details['wind_level'],
                'price_level': details['price_level'],
                'lam': float(agent_H.lam),
                'value_scale': float(getattr(agent_H, 'value_scale', 1.0)),
            })
            if step % 50 == 0:
                df_written = flush_csv_append_batch(
                    '{}/df_{}.csv'.format(output, RUN_TAG), df_rows, df_written)
                df_rows = []

        # ---- Periodic checkpoint (every 2000 steps) ----
        if step > 0 and step % 2000 == 0:
            agent_H.save_checkpoint(output, step)
            prCyan('Checkpoint saved at step {}'.format(step))

        if upload:
            writer.add_scalar('r1', r1, step)
            writer.add_scalar('c1', c1_step, step)          # 恒 0 (无约束通道)
            writer.add_scalar('profit_step', details['turbine_profit'], step)
            writer.add_scalar('ptur', ptur, step)
            writer.add_scalar('P_F', P_F, step)
            writer.add_scalar('wind_speed', env.u, step)
            writer.add_scalar('price', details['pr'], step)
            writer.add_scalar('penalty_NN_H', penalty_NN_H, step)
            writer.add_scalar('action_excess', action_excess, step)
            writer.add_scalar('action/yaw_deg_mean',
                              float(np.abs(yturb.GAMMA_MAX * np.asarray(applied_action_H)).mean()),
                              step)
            writer.add_scalar('phase/hour', pm_now / 60.0, step)
            if phase_table is not None:
                writer.add_scalar('phase/rest_frac', float(phase_table[pm_now, 0]), step)
                writer.add_scalar('phase/rest_hours', float(phase_table[pm_now, 1]) * 8, step)
                writer.add_scalar('phase/next_price', float(phase_table[pm_now, 2]) * 1000, step)
            if phase_checked > 0:
                writer.add_scalar('phase/check_mismatch_frac',
                                  phase_mismatch / float(phase_checked), step)
            writer.add_scalar('seg/price_level',
                              {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(details['price_level'], 0), step)
            writer.add_scalar('seg/wind_level',
                              {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(details['wind_level'], 0), step)

            if episode_steps % 5 == 4:
                writer.add_scalar('sum_r1', sum_r1, step)
                writer.add_scalar('sum_penalty_NN_H', sum_pen, step)
                writer.add_scalar('sum_profit', sum_profit, step)
                writer.add_scalar('sum_P_F', sum_PF, step)
                for i in range(int(args.nb_turbine)):
                    writer.add_scalar('action/a_H_original_{}'.format(i),
                                      original_action_H[i], step)
                    writer.add_scalar('action/a_H_clipped_{}'.format(i),
                                      applied_action_H[i], step)

        # ---- 累计本 episode 的指标 ----
        step += 1
        episode_steps += 1
        episode_reward += r1
        episode_profit += details['turbine_profit']
        episode_penalty += penalty_NN_H
        episode_energy += ptur * time
        episode_PF_sum += P_F
        if P_F > 3.0 + 1e-4:
            episode_viol += 1
        episode_yaw_deg += float(np.abs(yturb.GAMMA_MAX * np.asarray(applied_action_H)).mean())

        # update env
        norm_observation = deepcopy(norm_observation2)
        observation = deepcopy(observation2)
        ptur = pw_tru2      # 下一分钟的 P_tur（块内动作不变, 但风速已前进 1 min）

        # ------------------------------ episode -----------------------------
        if done:
            agent_H.save_model(output)
            # ⚠ 分母是**步数**不是块数：`episode_yaw_deg` 每个环境步累加一次（块内 5 步相同），
            #   除以块数会把均值放大 5 倍（曾写成 n_blocks ⇒ 打印出 72° 这种不可能值）。
            n_steps = max(1, episode_steps)
            prGreen('##########{}: episode_reward:{:.1f} episode_profit:{:.1f} '
                    'penalty:{:.1f} energy:{:.2f} MWh P_F_mean:{:.3f} MW viol_steps:{} '
                    'yaw|γ|mean:{:.2f}° step:{}'.format(
                        episode, episode_reward, episode_profit, episode_penalty,
                        episode_energy, episode_PF_sum / n_steps,
                        episode_viol, episode_yaw_deg / n_steps, step))

            if upload:
                writer.add_scalar('episode/reward', episode_reward, episode)
                writer.add_scalar('episode/profit', episode_profit, episode)
                writer.add_scalar('episode/penalty_NN_H', episode_penalty, episode)
                writer.add_scalar('episode/energy_MWh', episode_energy, episode)
                writer.add_scalar('episode/P_F_mean', episode_PF_sum / n_steps, episode)
                writer.add_scalar('episode/viol_steps', episode_viol, episode)
                writer.add_scalar('episode/yaw_deg_mean', episode_yaw_deg / n_steps, episode)
                writer.add_scalar('wind_day', env.j, episode)

            if use_csv:
                df_episode_rows.append({
                    'episode': episode,
                    'episode_reward': episode_reward,
                    'episode_profit': episode_profit,
                    'episode_penalty_NN_H': episode_penalty,
                    'episode_energy_MWh': episode_energy,
                    'episode_PF_mean': episode_PF_sum / n_steps,
                    'episode_viol_steps': episode_viol,
                    'episode_yaw_deg_mean': episode_yaw_deg / n_steps,
                    'wind_day': env.j,
                })
                df_episode_written = flush_csv_append_batch(
                    '{}/df_episode_{}.csv'.format(output, RUN_TAG),
                    df_episode_rows, df_episode_written)
                df_episode_rows = []

            observation = None
            episode_steps = 0
            episode_reward = episode_profit = episode_penalty = 0.
            episode_energy = 0.
            episode_PF_sum = 0.
            episode_viol = 0
            episode_yaw_deg = 0.
            episode += 1

    if use_csv:
        flush_csv_append_batch('{}/df_{}.csv'.format(output, RUN_TAG), df_rows, df_written)
        flush_csv_append_batch('{}/df_episode_{}.csv'.format(output, RUN_TAG),
                               df_episode_rows, df_episode_written)
        flush_csv_append_batch('{}/df_test9.csv'.format(output), df_test_rows, df_test_written)
        flush_csv_append_batch('{}/update_stats_H.csv'.format(output),
                               df_update_rows, df_update_written)

    if phase_mode > 0:
        prGreen('phase check: mode={} dims={} checked={} mismatch={} ({:.4f}%)'.format(
            phase_mode, PHASE_DIM[phase_mode], phase_checked, phase_mismatch,
            100.0 * phase_mismatch / max(phase_checked, 1)))
        if phase_mismatch > 0:
            prRed('WARNING: ep 电价时钟与 day_price_schedule() 不一致 ({} 步) → 相位特征失效'
                  .format(phase_mismatch))


def _test_row(m):
    """9-case 评测结果 → CSV 行（**只保留 BESS-free 的列**, 不编造 SOC 列）。

    显式列出键名（不用 `m.get(k, nan)`）：少一个键就应该当场报错，
    而不是在 CSV 里悄悄写一列 NaN。
    """
    return {k: m[k] for k in
            ('r1', 'c1', 'profit', 'profit_nominal', 'gain', 'pf_viol',
             'P_F_mean', 'yaw_deg_mean', 'ptur_mean', 'penalty_NN_H')}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='YAWPROFIT — 只调度 FLORIS 的 5-min 发电利润最大化智能体 (单层 TD3+PER)')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default='YAWPROFIT', type=str,
                        help='run 身份 (决定日志目录 output/<env>-runN)')
    # 网络宽度 (与 yawflor/td3per 同格式 16/100; 改维度后旧 .pkl 不可复载)
    parser.add_argument('--hidden1', default=16, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=100, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.01, type=float, help='critic net learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate')
    parser.add_argument('--warmup', default=200, type=int,
                        help='time without training but only filling the replay memory (环境步)')
    # γ 以【块(5 min)】为单位生效 (observe_jump 聚合 5 步后 discount 只用一次)
    parser.add_argument('--discount', default=0.98, type=float, help='H 层块级折扣')
    parser.add_argument('--nstep', default=5, type=int,
                        help='H 层 n 步回报, 滑窗 stride-1 (默认 5; 1 = 1-step 历史口径)')
    parser.add_argument('--target_agg', default='min', type=str, choices=['min', 'mean'],
                        help='twin profit-critic 的 target 聚合 (min = TD3 原口径)')
    parser.add_argument('--bsize', default=128, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=20000, type=int,
                        help='H 层 memory 条数 (覆盖环境步 = rmsize x 5 x nstep)')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_iter_H', default=200000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=4000, type=int,
                        help='【探索噪声 ε 衰减所需的决策次数】每 5 个环境步决策一次 '
                             '⇒ 4000 次 = 20k 环境步')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--cuda', default=True, type=bool, help='Whether to use CUDA')
    parser.add_argument('--lam', default=0.0, type=float,
                        help='约束通道价格 —— yawprofit 的 c1 ≡ 0, 主循环强制 0 (纯利润最大化)')
    parser.add_argument('--lr_lam', default=0.1, type=float, help='(未使用: 无对偶上升)')
    parser.add_argument('--cost_single_limit', default=0.5, type=float, help='cost_bound for single step')

    parser.add_argument('--PER_alpha', default=0.6, type=float, help='priority exponent')
    parser.add_argument('--PER_beta', default=0.4, type=float, help='IS exponent (退火到 1)')
    parser.add_argument('--PER_epsilon', default=0.01, type=float, help='avoid zero priority')
    parser.add_argument('--beta_increment_per_sampling', default=3e-5, type=float,
                        help='β退火增量/次更新')
    parser.add_argument('--nb_turbine', default=5, type=int, help='number of turbines (= 动作维)')

    # ---- TD3 ----
    parser.add_argument('--policy_delay', default=2, type=int, help='delayed policy update interval')
    parser.add_argument('--policy_noise', default=0.1, type=float, help='target smoothing noise std')
    parser.add_argument('--noise_clip', default=0.3, type=float, help='target smoothing noise clip')
    parser.add_argument('--cost_critic_mode', default='max', type=str, choices=['min', 'max'],
                        help='cost twin-critic aggregation (c1 ≡ 0 ⇒ 无影响; 保持同源)')

    # ---- 探索噪声幅度 (逐维; 与 yawflor 前 5 维同口径) ----
    parser.add_argument('--wt_noise', default=hwarm.YAW_WT_NOISE, type=float,
                        help='偏航维探索噪声幅度 (±wt_noise*eps); 默认 0.2 = 满带宽 2 的 10%%')

    # ---- 24h 相位特征 ----
    parser.add_argument('--phase_feat', default=1, type=int, choices=[0, 1, 2],
                        help='观测的 24h 相位特征: 0=关(2维); 1=sin/cos(4维, 默认); '
                             '2=1+[段内剩余比例, 剩余小时/8, 下一段电价/1000](7维)')

    # ---- 输出层 bias 初始化 ----
    parser.add_argument('--wt_bias_init', default=1, type=int, choices=[0, 1],
                        help='1: 把 actor 输出层 5 维偏航 bias 初始化到中性满带 [−1,+1]; '
                             '0: 用 PyTorch 默认')

    parser.add_argument('--value_scale', default=hc.VALUE_SCALE, type=float,
                        help='r1 进入值函数回归空间的统一缩放 s (默认 1000)')
    parser.add_argument('--cost_scale', default=hc.COST_SCALE_TD3, type=float,
                        help='cost 通道缩放 (c1 ≡ 0 ⇒ 无影响; 保持同源)')
    parser.add_argument('--critic_bias_init', default='estimate', choices=['estimate', 'none'],
                        help='critic 输出层 bias 的价值基线初始化: estimate=用 warmup 观测的'
                             '块收益估计 (默认); none=不初始化(对照臂)')
    parser.add_argument('--critic_bias_p', default=-1.0, type=float,
                        help='显式指定 profit critic 基线 (>0 生效, 覆盖 estimate; 原始口径)')
    parser.add_argument('--critic_bias_c', default=-1.0, type=float,
                        help='显式指定 cost critic 基线 (>0 生效; c1 ≡ 0 ⇒ 建议默认)')
    parser.add_argument('--max_steps', default=0, type=int,
                        help='>0 时覆盖 train_iter_H, 用于短程 A/B 或 smoke')

    # warmup 期动作调度（偏航带；与 yawflor 同一组开关, 只是不挂 BESS 规则臂）
    parser.add_argument('--warmup_sched', default='fixed', choices=['fixed', 'random'],
                        help='warmup 期动作调度: fixed=偏航 5 维从 0 起每块随机 ±0.2 并 '
                             'clip 到 [−1,+1] (本方案); random=random_action() 对照臂')
    parser.add_argument('--warmup_wt_start', default=hwarm.YAW_START, type=float,
                        help='warmup 偏航动作起始值 (默认 0 = 单机最优点 γ=0°)')
    parser.add_argument('--warmup_wt_step', default=hwarm.YAW_STEP, type=float,
                        help='warmup 偏航动作每块随机步长 (默认 ±0.2 = 满带宽 2 的 10%%)')
    parser.add_argument('--warmup_wt_lo', default=hwarm.YAW_LO, type=float,
                        help='warmup 偏航动作 clip 下界 (默认 −1)')
    parser.add_argument('--warmup_wt_hi', default=hwarm.YAW_HI, type=float,
                        help='warmup 偏航动作 clip 上界 (默认 +1)')

    args = parser.parse_args()
    hc.set_value_scale(args.value_scale)
    hc.set_cost_scale(args.cost_scale)
    # ⚠ 这里**不**打印 hc.summary(): 那是一段 c1 = 波动当量的口径说明, 对
    #   "c1 ≡ 0" 的 yawprofit 不适用 (误读成"本臂也在管波动"是最坏的一种误导)。
    prCyan('[yawprofit] 值函数回归空间缩放: s_p={:g} (r1) / s_c={:g} '
           '(cost 通道恒 0 ⇒ 无影响)'.format(hc.value_scale(), hc.cost_scale()))
    args.output = get_output_folder(args.output, args.env)

    max_E = 6.          # 仅为与父类构造签名兼容; 本环境无储能, 该值不被使用
    upload = True
    use_csv = True
    args.max_episode_length = 1440            # 1d = 24h*60m = 1440 steps
    args.train_iter_H = 1000 * args.max_episode_length
    if getattr(args, 'max_steps', 0) and args.max_steps > 0:
        args.train_iter_H = int(args.max_steps)
    args.seed = np.random.randint(0, 10000)
    args.cost_single_limit = 1000.

    # Set device and random seeds
    if (args.cuda and torch.cuda.is_available()):
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        print("Using GPU:", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

    if upload:
        writer = SummaryWriter(args.output)
        writer.add_scalar('seed', args.seed)

    # 物理引擎 = FLORIS（纯 Python，不启 MATLAB）。三份实例各持一个 FlorisModel：
    #   env / env_test / physics_ref（九例测试的"无偏航"参考臂，独立缓存不干扰前两者）。
    yaw_physics = FlorisPhysicsYaw(nt=args.nb_turbine)
    yaw_physics_test = FlorisPhysicsYaw(nt=args.nb_turbine)
    yaw_physics_ref = FlorisPhysicsYaw(nt=args.nb_turbine)
    env = WindTurbineYaw(max_E, args.nb_turbine, upload, None, physics=yaw_physics)
    env_test = WindTurbineYaw(max_E, args.nb_turbine, upload, None, physics=yaw_physics_test)
    prCyan("口径: " + yturb.declared_scope())

    if args.seed > 0:
        np.random.seed(args.seed)

    # 上层: TD3 + PER（lam = 0 ⇒ 约束通道关闭 ⇒ actor 目标 = −Q_p）
    nb_states_H = 2 + PHASE_DIM[int(getattr(args, 'phase_feat', 0))]
    print('H nb_states = {} (2 维基观测 [pr/1000, u/10] + {} 相位维, --phase_feat={})'.format(
        nb_states_H, PHASE_DIM[int(getattr(args, 'phase_feat', 0))], args.phase_feat))
    agent_H = SP_TD3(nb_states=nb_states_H, nb_actions=int(args.nb_turbine), args=args,
                     lam_max=None, lam_min=0.0,
                     ema_lam=True, ema_alpha=0.02, ema_lam_lr=0.01,
                     record_update_stats=True,
                     policy_delay=args.policy_delay,
                     policy_noise=args.policy_noise,
                     noise_clip=args.noise_clip,
                     cost_critic_mode=args.cost_critic_mode,
                     wt_noise=args.wt_noise,
                     nstep=args.nstep, target_agg=args.target_agg)
    agent_H.lam = 0.0                       # c1 ≡ 0 ⇒ 约束通道关闭 (纯利润最大化)
    agent_H.value_scale = hc.value_scale()
    agent_H.cost_scale = hc.cost_scale()

    prCyan('[yawprofit] 口径: state=[pr/1000, u/10]+相位({}维) | action={}×n∈[−1,1] '
           '(γ=25°·n, 块级 5 min 固定) | r1 = pr·P_tur·dt − penalty_NN_H | c1 ≡ 0, λ=0 '
           '| 无 BESS/RBB/下层/对偶'.format(PHASE_DIM[int(args.phase_feat)], args.nb_turbine))
    print('[yawprofit] nstep={} target_agg={} (块级 γ={} ⇒ 自助深度 50→{:.1f}); '
          '复现 1-step 口径请传 --nstep 1'.format(
              args.nstep, args.target_agg, args.discount, 50.0 / max(1, args.nstep)))

    if args.wt_bias_init:
        init_wt_action_bias(agent_H, lo=hwarm.YAW_LO, hi=hwarm.YAW_HI)
    print('[yawprofit] 动作带罚系数 = {:.0f} 元/单位 (满带越界 = {:.0f} 元, 与 yawflor 等值); '
          '块级罚在块内 5 步各记一次 (与 yawflor 同记账口径)'.format(
              PENALTY_NN_H_COEF, PENALTY_NN_H_COEF * (hwarm.YAW_HI - hwarm.YAW_LO)))

    if args.mode == 'train':
        train(args.train_iter_H,
              agent_H,
              env,
              env_test,
              args.output,
              physics_ref=yaw_physics_ref,
              max_episode_length=args.max_episode_length,
              debug=args.debug,
              upload=upload,
              use_csv=use_csv)
        if upload:
            writer.close()
        prCyan('[yawprofit] 训练结束 —— FLORIS 调用统计:')
        prCyan(env.summary())
        prCyan(env_test.summary())
        prCyan(yaw_physics_ref.summary())
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
