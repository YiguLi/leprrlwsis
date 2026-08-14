function power_out = WFSim_port(w_s, w_a, a, p)%wind_speed,wind_angle,c_prime,phi
%clear; clc; close all; %

%wind
Wind.wind_speed = w_s;%m/s
Wind.wind_angle = w_a;%degree

%Control
Control.a = a;%degree
c_p = 4*Control.a.*(1-Control.a).^2;
Control.c_prime = c_p; %C_trust 
Control.phi = p;%degree

run('WFSim_addpaths.m');    % Add essential paths to MATLABs environment
%simulate by WFSim
sol_array = WFSim(Wind,Control);

power_out.out = sum(sol_array{1,5}.turbine.power);
power_out.test = sol_array;
%disp(['total power_out ' num2str(power_out)]);
end
