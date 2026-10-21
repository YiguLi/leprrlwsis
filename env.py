from pickle import FALSE
import os
import numpy as np
import bisect
import pandas as pd

from util import *

# ---------------------------------------------------------------------------
# `matlab` 改成**惰性** import（yawflor/README.md §5 的最小改动①）
# ---------------------------------------------------------------------------
# 动机：新方法 `yawflor` 用 FLORIS 引擎（`yawflor/yawflor_env.py`），也不需要 MATLAB。
# 顶层 `import matlab` 会让"没有 MATLAB 的环境/受限沙箱"连 `import env` 都失败
# （例如只跑 agentutil 下的 FLORIS 契约测试时）。改成按需 import：
# 走旧 MATLAB 路径的**行为逐位不变**（第一次调用时 import，之后走模块缓存）。
_matlab_cache = None


def _get_matlab():
    """按需 import matlab（旧 MATLAB 后端专用）。"""
    global _matlab_cache
    if _matlab_cache is None:
        import matlab as _m
        _matlab_cache = _m
    return _matlab_cache


# Fixed random day sequence — permanently saved to file.
# Generated once with np.random.RandomState(42).randint(1, 200, size=199).
# All training runs share this identical sequence for fair comparison.
#: 风速池文件（可用 HSRL_WIND_CSV 覆盖；默认 = 生产口径的加噪池）
WIND_CSV = os.environ.get('HSRL_WIND_CSV', 'FFC_speed_noise_1.csv')

_DAY_SEQ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'fixed_day_sequence.csv')
FIXED_DAY_SEQUENCE = pd.read_csv(_DAY_SEQ_PATH, header=None).values.flatten().tolist()

class WindBattery:
    # ------------------------------------------------------------------
    # 标称动作（yawflor/README.md §5 的最小改动③）：**类属性单一真源**
    # ------------------------------------------------------------------
    # 旧环境 = 5 维轴向因子 `a` 的标称值 0.3333（单机 `C_p = 4a(1−a)²` 最优点）；
    # 新环境（`yawflor.yawflor_env.WindBatteryYaw`）覆写为 0.0 —— 那是**无量纲**
    # 偏航动作 `n`，`n = 0 ⇒ γ = 0°`（单机自损最小）。
    # `nominal_action` 非 None 时直接用它（可给完整向量）；否则用标量铺满 nt 维。
    nominal_action_value = 0.3333
    nominal_action = None

    def nominal(self):
        """标称动作向量（长度 = num_turbine）。"""
        if self.nominal_action is not None:
            return np.asarray(self.nominal_action, dtype=float).reshape(-1)[:self.num_turbine]
        return np.full(self.num_turbine, float(self.nominal_action_value), dtype=float)

    def __init__(self, max_E, num_turbine, upload, engine, physics=None):
        self.u = 10.
        self.episode_count = 0    # episode counter for fixed day sequence

        self.max_E = max_E
        self.engine = engine
        # `physics=None` ⇒ **现有 MATLAB 路径**（所有既有调用点行为逐位不变；
        # yawflor/README.md §5 的最小改动②）。非 None ⇒ 由该后端提供 PowerOutput
        # （接口 = `physics.PowerOutput(u, act)` → 风场总出力 [MW]）。
        self.physics = physics
        self.min_E = 0.0

        self.num_turbine = num_turbine

        self.eta_c = 0.98
        self.eta_d = 0.98

        self.time = 1 / 60  #TODO unit：hour
        self.upload = upload

        # electricity price
        self.count_price = 0
        self.price_list = [300, 580, 980]  # unit:RMB/Mwh
        self.pr = self.price_list[0]
        
        # wind_speed_2 (Sequence)
        self.wind_noise = pd.read_csv(WIND_CSV)
        self.wind_test_scenarios = pd.read_csv("test_scenarios.csv")
        self.count_wind = 0

        # 噪声
        self.is_noise = False
        self.gauss_noise = 0.5

        # 九例分段场景测试接口（45 步训练中测试）
        self.scenario_wind = None
        self.scenario_price = None
        self.scenario_idx = 0

        # 上一步电网功率，用于计算波动
        self.P_G_prev = 0.0

    def reset(self):
        # 1. reset电池
        self.E = 0.25
        
        # 2. reset风速（固定随机序列，保证跨项目可比较）
        idx = self.episode_count % len(FIXED_DAY_SEQUENCE)
        self.j = FIXED_DAY_SEQUENCE[idx]
        self.episode_count += 1
        # print(self.j)
        self.wind_day = self.wind_noise.iloc[:, self.j]
        self.count_wind = 0
        self.u = self.wind_day[self.count_wind]
        
        # 3. reset电价
        self.count_price = 0
        self.pr = self.CalPrice() #FIXME: price
        
        a_temp = self.nominal()
        temp = self.PowerOutput(self.u, a_temp)
        self.P_G_prev = temp  # 初始 P_G 作为波动计算的基准
        obs = np.array([self.pr, self.u, self.E, temp])
        return obs

    def TestReset(self,day=0):
        self.count_price = 0
        self.pr = self.CalPrice() #FIXME: price

        self.count_wind = 0
        if day == 0:
            self.u = self.wind_noise.iloc[self.count_wind, 1]# + self.is_noise*np.random.normal(0, self.gauss_noise) #TODO 噪声
        else:
            self.u = self.wind_test_scenarios.iloc[self.count_wind, day-1]# + self.is_noise*np.random.normal(0, self.gauss_noise) #TODO 噪声
        self.E = 5.

        # a_temp = np.array([0.1416, 0.1645, 0.1968, 0.2461, 0.3333])
        a_temp = self.nominal()
        temp = self.PowerOutput(self.u, a_temp)
        self.P_G_prev = temp  # 初始 P_G 作为波动计算的基准
        obs = np.array([self.pr, self.u, self.E, temp])
        return obs

    def PowerOutput(self, u0, act_wind):
        # 后端分派（README §5 最小改动②）：physics 非 None ⇒ 走新物理引擎（FLORIS），
        # 动作语义由后端定义（yawflor = 无量纲偏航 n）；None ⇒ 下面这条**逐位不变**
        # 的 MATLAB 路径。
        if self.physics is not None:
            return self.physics.PowerOutput(u0, act_wind)
        if 3.5 <= u0 <= 25:
            # 只取前 num_turbine 维: H 动作是 6 维 (5 台 + SOC 信号), 多传的维会经
            # `zeros(NT,1)+Control.c_prime` 隐式扩展成 NT×K 矩阵, 而 Actuator.m
            # 的线性索引只取第一列 ⇒ 只有 dim0 生效 (见 agentutil/probe_a_per_turbine.py)。
            act_wind = np.asarray(act_wind, dtype=float).reshape(-1)[:self.num_turbine]
            act_wind = np.clip(act_wind, 0., 0.5)
            try:
                u0 = float(u0)
                _ml = _get_matlab()          # 惰性 import（旧路径行为不变）
                # 第 5 个实参 want_full=False: 不要让 MATLAB 回传 5 步完整流场解
                # (sol_array)。本方法只读 'out'，实测净省 ~0.02-0.10 s/次 (2-8%)，
                # 另省掉每次调用在 Python 侧分配/回收那个嵌套 dict 的开销。
                # 数值逐位不变 —— 改前/改后 30 例直接调用 + 5 例经本方法的 A/B 逐位
                # 回归见 output/_wfsim_ab/。读 power_out.test 的诊断脚本不传该实参
                # ⇒ 默认 true，行为不变。
                target_result = self.engine.WFSim_port(_ml.double([u0]),
                                                _ml.double([0.0]),
                                                _ml.double([list(act_wind)]),
                                                _ml.double([0.0]),
                                                False)
            except:
                print(u0)   
                print(type(u0))

            output = target_result['out'] / 1000000.0
        else:
            output = 0.0
        return output

    #     return 10.

    def CalPrice(self):
        """
        计算当前时间点的电价
        基于一天24小时的时间段划分不同电价：
        - 0:00-7:00 和 23:00-24:00: 低谷电价 (300 RMB/MWh)
        - 11:00-17:00 和 21:00-23:00: 平峰电价 (580 RMB/MWh)
        - 其他时段: 高峰电价 (980 RMB/MWh)
        
        Returns:
            float: 当前时间点的电价 (RMB/MWh)
        """
        self.count_price += 1  # 时间计数器加1（每分钟计数一次）

        # --- 诊断开关：恒定电价（实验 1 的"固定电价"段；默认 None ⇒ 生产路径不变）---
        _cpr = getattr(self, 'diag_const_pr', None)
        if _cpr is not None:
            self.pr = float(_cpr)
            return self.pr

        # 判断当前时间点属于哪个时段
        if self.count_price in range(7 * 60) or self.count_price in range(23 * 60, 24 * 60):
            # 低谷时段：0:00-7:00 和 23:00-24:00
            self.pr = self.price_list[0]  # 300 RMB/MWh
        elif self.count_price in range(11 * 60, 17 * 60) or self.count_price in range(21 * 60, 23 * 60):
            # 平峰时段：11:00-17:00 和 21:00-23:00
            self.pr = self.price_list[1]  # 580 RMB/MWh
        else:
            # 高峰时段：其他时间
            self.pr = self.price_list[2]  # 980 RMB/MWh
            
        # 当时间计数器达到24小时（1440分钟）时重置
        if self.count_price == 24 * 60:
            self.count_price = -1
            
        return self.pr

    def CalWindSpeed(self):
        # ------给定风速序列--90---
        self.count_wind += 1
        if self.count_wind >= len(self.wind_day):
            self.count_wind = 0  # wrap around
        u1 = self.wind_day[self.count_wind]
        return u1

    # Test 风速序列
    def TestWindSpeed(self):
        self.count_wind += 1
        if self.count_wind >= len(self.wind_noise):
            self.count_wind = 0  # wrap around
        u = self.wind_noise.iloc[self.count_wind, 1] #+ self.is_noise*np.random.normal(0, self.gauss_noise)
        # prBlack('count = {}, u = {}'.format(self.count_wind, u))
        return u
 
    def GetBatteryActions(self, p_w, act_bat):
        P_C = max(0., act_bat[0]) # 充电>0
        P_D = abs(min(act_bat[0], 0.)) # 放电<0

        # ---- 更新 E_(t+1),得到applied [P_WB, P_BG]
        E_temp = self.E + (P_C * self.eta_c - P_D / self.eta_d) * self.time
        
        if E_temp > self.max_E: # 注意：假设没有接地
            penalty_L = 10 * (E_temp - self.max_E)
            P_C = (self.max_E - self.E) / self.eta_c/ self.time # 修正
            self.E = self.max_E

        elif E_temp < self.min_E:
            penalty_L = 10 * (self.min_E - E_temp) 
            P_D = (self.E - self.min_E) * self.eta_d / self.time  # 修正
            self.E = self.min_E
        else:
            penalty_L = 0
            self.E = E_temp
        applied_action_L = np.array([P_C - P_D])
        P_G = p_w - P_C + P_D # 注意：这里P_G可以为-
        return applied_action_L, penalty_L, P_G

    def step(self, applied_action_H, act_bat, pw_tru):
        # 2. 更新t+1电价
        self.pr = self.CalPrice() #FIXME: price
        # self.pr = 300.
        # 3. 更新t+1风速
        self.u = self.CalWindSpeed()
        # 1. t+1风机的总出力
        pw_tru2 = self.PowerOutput(self.u, applied_action_H[:self.num_turbine])
        # 4. 更新下层电池
        applied_action_L, penalty_L, P_G = self.GetBatteryActions(pw_tru, act_bat)

        # 下一个时刻的obs
        observation2_H = np.array([self.pr, self.u, self.E, P_G])

        # 电网功率波动 |P_G(t) - P_G(t-1)|
        P_F = abs(P_G - self.P_G_prev)
        self.P_G_prev = P_G

        done = done_L = False
        info = {'P_F': P_F, 'P_G': P_G}
        return pw_tru2, observation2_H, applied_action_L, penalty_L, done, done_L, info

    # 用作test的序列
    def TestStep(self, applied_action_H, act_bat, pw_tru):
        # 2. 更新t+1电价
        # self.pr = self.CalPrice()
        self.pr = 300.
        # prPurple('price = {}.'.format(self.pr))

        # 3. 更新t+1风速
        self.u = self.TestWindSpeed()
        # 1. 真实t风机的总出力
        pw_tru2 = self.PowerOutput(self.u, applied_action_H[:self.num_turbine]) #TODO
        # 4. 更新下层电池
        applied_action_L, penalty_L, P_G = self.GetBatteryActions(pw_tru, act_bat)

        # 下一个时刻的obs
        observation2_H = np.array([self.pr, self.u, self.E, P_G])

        # 电网功率波动 |P_G(t) - P_G(t-1)|
        P_F = abs(P_G - self.P_G_prev)
        self.P_G_prev = P_G

        done = done_L = False
        info = {'P_F': P_F, 'P_G': P_G}
        return pw_tru2, observation2_H, applied_action_L, penalty_L, done, done_L, info
    

    # ================================================================
    # 九例分段场景测试（训练中每 480 episode step 一次，45 步）
    # wind/price 完全由外部脚本序列给定，不走 CalPrice/CalWindSpeed。
    # ================================================================
    def ScenarioReset(self, wind_seq, price_seq):
        self.scenario_wind = wind_seq
        self.scenario_price = price_seq
        self.scenario_idx = 0
        self.E = 5.

        self.pr = price_seq[0]
        self.u = wind_seq[0]

        a_temp = self.nominal()
        temp = self.PowerOutput(self.u, a_temp)
        self.P_G_prev = temp  # 初始 P_G 作为波动计算的基准
        obs = np.array([self.pr, self.u, self.E, temp])
        return obs

    def ScenarioStep(self, applied_action_H, act_bat, pw_tru):
        # 前进到脚本序列的下一时刻
        self.scenario_idx += 1
        if self.scenario_idx >= len(self.scenario_wind):
            self.scenario_idx = len(self.scenario_wind) - 1  # 末尾保持
        self.pr = self.scenario_price[self.scenario_idx]
        self.u = self.scenario_wind[self.scenario_idx]

        # t+1 风机的总出力
        pw_tru2 = self.PowerOutput(self.u, applied_action_H[:self.num_turbine])
        # 更新下层电池
        applied_action_L, penalty_L, P_G = self.GetBatteryActions(pw_tru, act_bat)

        # 下一个时刻的obs
        observation2_H = np.array([self.pr, self.u, self.E, P_G])

        # 电网功率波动 |P_G(t) - P_G(t-1)|
        P_F = abs(P_G - self.P_G_prev)
        self.P_G_prev = P_G

        done = done_L = False
        info = {'P_F': P_F, 'P_G': P_G}
        return pw_tru2, observation2_H, applied_action_L, penalty_L, done, done_L, info
    