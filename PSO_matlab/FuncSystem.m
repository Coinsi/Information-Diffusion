function [y1,y2,y3,y4] = FuncSystem(X_input,W1,W2,W3,W4)
%UNTITLED2 此处提供此函数的摘要
%   此处提供详细说明

% the first subsystem
X=X_input;
n=5;
dmu=1/(n-1);
X1=X(1:3);
sigma=0.05;
f1=zeros(n,1);
for i=1:n
    mu1=ones(length(X1),1)*(dmu*(i-1));
    f1(i)=exp(-(X1-mu1)'*(X1-mu1)/sigma);
end

y1=W1'*f1;

X2=X(4:6);
f2=zeros(n,1);
for i=1:n
    mu2=ones(length(X2),1)*(dmu*(i-1));
    f2(i)=exp(-(X2-mu2)'*(X2-mu2)/sigma);
end
y2=W2'*f2;

X3=[y1;X(7);X(8);y2];
f3=zeros(n,1);
for i=1:n
    mu3=ones(length(X3),1)*(dmu*(i-1));
    f3(i)=exp(-(X3-mu3)'*(X3-mu3)/sigma);
end
y3=W3'*f3;

X4=[y3;X(9);X(10)];
f4=zeros(n,1);
for i=1:n
    mu4=ones(length(X4),1)*(dmu*(i-1));
    f4(i)=exp(-(X4-mu4)'*(X4-mu4)/sigma);
end
y4=W4'*f4;


end