function power_out = WFSim_port(w_s, w_a, a, p, want_full)%wind_speed,wind_angle,c_prime,phi
%clear; clc; close all; %

% ---- 可选第 5 实参 want_full: 是否把 5 步完整流场解一并回传 (默认 true) ----
% 为什么要这个开关: sol_array 序列化跨引擎边界实测净代价 ~0.02-0.10 s/次
%   (min~mean, 占整次调用 2-8%; 机器争抢时偏高)。output/_wfsim_ab 的逐位 A/B:
%   同一批 30 例 (u,a) 只回 'out' vs 回 'out'+'test', 数值逐位相同 (0 项不一致)。
%   ⚠ 曾报 0.686 s / "32%" 是错的 —— 那次对比里"改前"那组含会话第一次调用
%   (冷启动 addpath + JIT), 差值被它污染; 用交错 + min 统计后就是上面这个量级。
% 默认 true ⇒ 读 power_out.test 的诊断脚本一字不改:
%   agentutil/probe_a_per_turbine.py / probe_a_farm_structure.py /
%   probe_engine_random_actions.py / probe_engine_newday_validate.py /
%   probe_engine_minute_compare.py / run_clip_check_wfsim.m
if nargin < 5 || isempty(want_full)
    want_full = true;
end

% ---- 路径守卫: WFSim_addpaths 幂等, 但每次调用都要重跑 addpath(genpath(libraries)) ----
% 实测单次 0.12 s (min) / 0.20 s (median), 占整次调用约 10-15%。
% persistent 在 `clear functions` / 引擎重启 / 本文件被编辑后自动失效
% ⇒ 那时会重新 addpath, 可解析状态与原实现一致, 数值不变。
persistent paths_ready

%wind
Wind.wind_speed = w_s;%m/s
Wind.wind_angle = w_a;%degree

%Control
% ---- 逐台轴向因子接线 (原实现只让 dim0 生效) ----
% `zeros(NT,1)+Control.c_prime` 是 NT×K 隐式扩展, 而 Actuator.m:55/:61 用
% `input.CT_prime(kk)` / `input.a(kk)` 线性索引只取**第一列** ⇒ K>1 时每台机组
% 都拿到 c_prime(a(1))。这里要求**恰好 NT 个**并转成 **列向量**, 使加法成为逐元素运算。
% 证据: agentutil/probe_a_per_turbine.py (a2/a4 组的逐台 CT/power 与 all-0.20 逐位相同)。
NT = 5;                     % windTunnel 场景机组数 (meshing.m:187 `Crx` 长度; WFSim.m:69 固定该场景)
% 维度守卫必须**对称**: 少传和多传都要报错。多传曾经是"静默只取前 NT 个"(实测
% `[0.2]*5+[0.5]*5` 与 `[0.2]*5` 的总功率逐位相同 3.717941 MW) —— 与本次修掉的
% "只有 dim0 生效"是同一类静默失效, 只是换了一层。策略侧的第 6 维 (SOC 信号)
% 由 env.py::PowerOutput 截到 num_turbine 后再下传, 不在这里兜。
if numel(a) ~= NT
    error(['WFSim_port: 需要恰好 %d 个轴向因子 (收到 %d 个)。' ...
           '策略动作是 6 维(5 台 + SOC 信号)时, 请先经 env.py::PowerOutput ' ...
           '截到 num_turbine 再调用。'], NT, numel(a));
end
Control.a = a(:);           % NT×1, 关键: 列向量才能逐台生效
Control.c_prime = 4*Control.a.*(1-Control.a).^2;   % C_p = 4a(1-a)^2 (沿用原口径)
Control.phi = p;%degree

if isempty(paths_ready)
    run('WFSim_addpaths.m');    % Add essential paths to MATLABs environment
    paths_ready = true;
end
%simulate by WFSim
sol_array = WFSim(Wind,Control);

power_out.out = sum(sol_array{1,5}.turbine.power);
if want_full
    power_out.test = sol_array;
end
%disp(['total power_out ' num2str(power_out)]);
end
