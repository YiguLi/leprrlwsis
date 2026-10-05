function out = probe_wfsim_steady_steps(u_inf, a_in)
% probe_wfsim_steady_steps —— 量化 "WFSim_port 一次调用究竟推进了多少物理时间"
%
% 背景：`WFSim_port.m` 最终调用 `WFSim(Wind,Control)`，其内部是
%   `for sol.k = 1..Wp.sim.NN`（windTunnel 场景 NN=5，h=1 s）的时间推进循环。
%   本探针用 `want_full=true` 把这 5 步的完整解全部回传，逐步骤报告：
%     * 每步的时刻 time 与农场总功率 P
%     * 相邻两步的流场变化 max|Δu| / rms|Δu|
%     * 第 1 步相对"均匀来流初场"的变化（说明第 1 步是否已经把尾流场解出来）
%   ⇒ 用来判定这次调用是"稳态求解"还是"动态推进"。
%
% 用法（本机 F:\Matlab + F:\anaconda\envs\wsis）：
%   >> cd('<repo root>'); addpath(genpath('agentutil'));
%   >> out = probe_wfsim_steady_steps(10.0, [1/3 1/3 1/3 1/3 1/3]);
%
% 注意：本脚本**只读**引擎、不改任何项目文件；产物 = output/wfsim_steady_steps.mat
if nargin < 1 || isempty(u_inf); u_inf = 10.0; end
if nargin < 2 || isempty(a_in);  a_in  = ones(1,5)/3; end

r = WFSim_port(u_inf, 0.0, a_in, 0.0, true);
N = numel(r.test);

P = zeros(1,N); T = zeros(1,N); MU = zeros(1,N); MX = zeros(1,N); MN = zeros(1,N);
FS = zeros(1,N);                       % 每个 sol 内部的迭代次数（k=1 常见 >1）
U = cell(1,N);
for k = 1:N
    s = r.test{1,k};
    P(k)  = sum(s.turbine.power);
    T(k)  = s.time;
    U{k}  = s.u;
    MU(k) = mean(s.u(:)); MX(k) = max(s.u(:)); MN(k) = min(s.u(:));
    if isfield(s,'it'); FS(k) = s.it; end
end

DU = zeros(1,N-1); DRMS = zeros(1,N-1);
for k = 2:N
    d = U{k} - U{k-1};
    DU(k-1)   = max(abs(d(:)));
    DRMS(k-1) = sqrt(mean(d(:).^2));
end
D0    = max(abs(U{1}(:) - u_inf));
D0rms = sqrt(mean((U{1}(:) - u_inf).^2));

fprintf('\n==== probe_wfsim_steady_steps ====\n');
fprintf('u_inf = %.3f m/s | a = %s | phi = 0 | forcescale=1.5 powerscale=0.95\n', ...
        u_inf, mat2str(a_in, 4));
fprintf('%3s %10s %20s %14s %16s\n', 'k', 'time[s]', 'P_total[MW]', 'mean(u)[m/s]', 'min(u)[m/s]');
for k = 1:N
    fprintf('%3d %10.1f %20.9f %14.6f %16.6f\n', k, T(k), P(k)/1e6, MU(k), MN(k));
end
fprintf('\nPer-step field change (direct measure of "dynamic marching"):\n');
for k = 1:N-1
    fprintf('  k=%d -> k=%d : max|du| = %.3e m/s   rms|du| = %.3e m/s\n', ...
            k, k+1, DU(k), DRMS(k));
end
fprintf('\nStep 1 vs uniform initial field u=%.1f (what step 1 actually does):\n', u_inf);
fprintf('  max|du| = %.4f m/s   rms|du| = %.4f m/s   (relative %.3f%% / %.3f%%)\n', ...
        D0, D0rms, 100*D0/u_inf, 100*D0rms/u_inf);
fprintf('\nP(k=%d)/P(k=1) - 1 = %+.3e\n', N, P(N)/P(1)-1);
fprintf('max per-step |dP| for k>=2 = %.3e MW\n', max(abs(diff(P)))/1e6);

out = struct('u_inf',u_inf,'a',a_in,'N',N,'P',P,'T',T,'MU',MU,'MX',MX,'MN',MN, ...
             'DU',DU,'DRMS',DRMS,'D0',D0,'D0rms',D0rms);
if ~exist('output','dir'); mkdir('output'); end
savename = fullfile('output','wfsim_steady_steps.mat');
save(savename, '-struct', 'out');
fprintf('saved: %s\n', savename);
end
