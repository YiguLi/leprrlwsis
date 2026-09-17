function f = myfun( x,u )
%FMINXY 此处显示有关此函数的摘要
%   此处显示详细说明

%para
Drotor      = 100;
powerscale  = 1; 
Ar = pi*(0.5*Drotor)^2;
Rho         = 1.20;

%PARK model
k = 0.08;

U = zeros(1,5);
U(1) = u;
u_decrease = 2*x(1)*(Drotor/(Drotor+2*k*5*Drotor))^2;
U(2) = U(1) * (1 - u_decrease);
u_decrease = 2*x(2)*(Drotor/(Drotor+2*k*5*Drotor))^2;
U(3) = U(2) * (1 - u_decrease);
u_decrease = 2*x(3)*(Drotor/(Drotor+2*k*5*Drotor))^2;
U(4) = U(3) * (1 - u_decrease);
u_decrease = 2*x(4)*(Drotor/(Drotor+2*k*5*Drotor))^2;
U(5) = U(4) * (1 - u_decrease);



CT = 4*x.*(1-x).^2;

%power output
f = -sum(powerscale*.5*Rho*Ar*CT.*(U.^3));

end

