from pickle import FALSE
import numpy as np
import random
import bisect
import pandas as pd

import matlab

from util import *

class WindBattery:
    def __init__(self, max_E, num_turbine, upload, engine):
        self.u = 10.

        self.max_E = max_E
        self.engine = engine
        self.min_E = 0.2 * max_E

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
        self.wind_noise = pd.read_csv("FFC_speed_noise_1.csv")
        self.wind_test_scenarios = pd.read_csv("test_scenarios.csv")
        self.count_wind = 0

        # 噪声
        self.is_noise = False
        self.gauss_noise = 0.5

    def reset(self):
        # 1. reset电池
        self.E = 3.
        
        # 2. reset风速
        self.j = random.randint(1,self.wind_noise.shape[1]-1)
        # print(self.j)
        self.wind_day = self.wind_noise.iloc[:,self.j]
        self.count_wind = 0
        self.u = self.wind_day[self.count_wind]
        
        # 3. reset电价
        self.count_price = 0
        self.pr = self.CalPrice() #FIXME: price
        
        a_temp = np.ones(self.num_turbine) * 0.3333
        temp = self.PowerOutput(self.u, a_temp)
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
        a_temp = np.ones(self.num_turbine) * 0.3333
        temp = self.PowerOutput(self.u, a_temp)
        obs = np.array([self.pr, self.u, self.E, temp])
        return obs

    def PowerOutput(self, u0, act_wind):
        if 3.5 <= u0 <= 25:
            act_wind = np.clip(act_wind, 0., 0.5)
            act_wind = act_wind.astype(float)
            try:
                u0 = float(u0)
                target_result = self.engine.WFSim_port(matlab.double([u0]),
                                                matlab.double([0.0]),
                                                matlab.double([list(act_wind)]),
                                                matlab.double([0.0]))
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
        u1 = self.wind_day[self.count_wind]
        return u1
    
    # Test 风速序列
    def TestWindSpeed(self):
        self.count_wind += 1
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

        done = done_L = False
        info = None
        return pw_tru2, observation2_H, applied_action_L, penalty_L, done, done_L

    # 用作test的序列
    def TestStep(self, applied_action_H, act_bat, pw_tru):
        # 2. 更新t+1电价
        self.pr = self.CalPrice()
        # self.pr = 300.
        # prPurple('price = {}.'.format(self.pr))
        
        # 3. 更新t+1风速
        self.u = self.TestWindSpeed()
        # 1. 真实t风机的总出力
        pw_tru2 = self.PowerOutput(self.u, applied_action_H[:self.num_turbine]) #TODO
        # 4. 更新下层电池
        applied_action_L, penalty_L, P_G = self.GetBatteryActions(pw_tru, act_bat)

        # 下一个时刻的obs
        observation2_H = np.array([self.pr, self.u, self.E, P_G])

        done = done_L = False
        info = None
        return pw_tru2, observation2_H, applied_action_L, penalty_L, done, done_L
    