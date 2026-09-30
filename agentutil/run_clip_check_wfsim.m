% run_clip_check_wfsim.m —— 为 `agentutil/probe_clip_gap_fullrange.py` 生成 WFSim 参考功率
%
% 用途
%   读 `output/clip_check_plan.json`（36 档 × 10 个随机行为 = 360 点），
%   逐点调用 `WFSim_port`，把农场总有功与逐台功率存成
%   `output/clip_check_wfsim.json`。
%
% 用法（在仓库根目录的 MATLAB 里）
%   cd('C:\Users\SCP Site19\Desktop\llm enhanced wsis\leprrlwsis')
%   run_clip_check_wfsim
%
% 说明
%   · 需 JSON 读写；若无 `jsonencode`（R2016b 之前）请改用 `save(...,'-struct')`。
%   · 360 次 `WFSim_port` 调用按每次 ~2–5 s 估计需 **12–30 分钟**。
%   · 与 `agentutil/probe_engine_random_actions.run_wfsim` 同接口：
%       WFSim_port(u, wind_angle, a[1x5], phi)  ->  res.out / res.test{i}.turbine.power

clear; clc;
ROOT = pwd;
planFile = fullfile(ROOT, 'output', 'clip_check_plan.json');
outFile  = fullfile(ROOT, 'output', 'clip_check_wfsim.json');

assert(isfile(planFile), '找不到计划文件: %s', planFile);
raw  = fileread(planFile);
plan = jsondecode(raw);
us   = plan.u(:);
A    = plan.A;                 % n x 5
n    = numel(us);
NT   = plan.config.nt;
fprintf('计划: %d 点, u ∈ [%.1f, %.1f], NT=%d\n', n, min(us), max(us), NT);

p_farm = zeros(n, 1);
p_turb = nan(n, NT);
t0 = tic;
for i = 1:n
    res = WFSim_port(us(i), 0.0, A(i, :), 0.0);
    p_farm(i) = double(res.out) / 1e6;
    try
        tp = double(res.test{1}.turbine.power);
        p_turb(i, :) = reshape(tp, 1, []) / 1e6;
    catch
        % 逐台功率读不到就留 NaN（不影响农场总功率对比）
    end
    if mod(i, 20) == 1 || i == n
        fprintf('  %3d/%d  u=%6.2f  P=%9.6f MW  (%.0f s)\n', ...
                i, n, us(i), p_farm(i), toc(t0));
    end
end

obj = struct();
obj.config   = plan.config;
obj.u        = us;
obj.A        = A;
obj.p_wfsim  = p_farm;
obj.p_wfsim_turbine = p_turb;
obj.note     = 'WFSim reference for clip-gap check (360 points, 3-20 m/s, 0.5 m/s bins)';

fid = fopen(outFile, 'w');
fprintf(fid, '%s', jsonencode(obj));
fclose(fid);
fprintf('\n已写入 %s\n', outFile);
fprintf('下一步: 重跑 agentutil/probe_clip_gap_fullrange.py 以得到逐档对比。\n');
