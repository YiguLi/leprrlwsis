function output = solve_PARK(w_s, w_a)


u = w_s;
phi = w_a;
[x,fval,exitflag] = fmincon(@(x) myfun(x,u),[1/3 1/3 1/3 1/3 1/3],[],[],[],[],zeros(1,5),ones(1,5)/2);
fval = -fval;

%newx = [1 1 1 1]'*x;
%newx = reshape(newx,[9,1])

CT = 4*x.*(1-x).^2;

output.cp = CT;
output.result = fval;
output.a = x;
output.greedy = -myfun([1/3 1/3 1/3 1/3 1/3],u);


        
end


  
