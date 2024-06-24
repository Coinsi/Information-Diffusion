% x0 = [1 2 3 4];
% x = 2.1;
% dis = abs(x-x0)
% p1 = 1./dis
% p = p1./sum(p1);


T = readtable('Data.xlsx','VariableNamingRule','preserve');

t = T(:,13:16);

dis = 1./t(:,1:4).Variables/100

m = 5;

x0 = dis+m

syms x1 x2 x3 x4 m1 m2 m3;
e1 = m1+dis(1,:) == [x1 x2 x3 x4];
e2 = m2+dis(2,:)== [x1 x2 x3 x4];
e3 = m3+dis(3,:)== [x1 x2 x3 x4];


