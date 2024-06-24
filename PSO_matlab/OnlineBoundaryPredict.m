%online predict boundary;
clear;
clc;

load Optimization_result2.mat
%for example 1
%x1 \in [0.4,0.8];
%x3 \in [0.8,0.95];


%case one 
%has one variable
t=0:0.01:5;

x1=0.1*sin(t)+0.2;
L=length(x1);
% x1=0.7;
x2=0.9;
x3=0.6;
x4=0.2;
x5=0.5;
x6=0.98;
x7=0.88;
x8=0.79;
x9=0.1;
x10=0.4;

yy1=zeros(L,1);
for i=1:L
    x1a=x1(i);
        Xj=[x1a;x2;x3;x4;x5;x6;x7;x8;x9;x10];
%         Phi=FuncFuzzy(W1,W2,q,dq,tau);
        [ya1,ya2,ya3,ya4]=FuncSystem(Xj,W1_NFS,W2_NFS,W3_NFS,W4_NFS);
        yy1(i)=ya4;

end


yy1_min=min(yy1);
yy1_max=max(yy1);

figure(11)
plot(t,x1,t,yy1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x5v=0.5*sin(t)+0.5;
L=length(x5v);
x1=0.7;
x2=0.9;
x3=0.6;
x4=0.2;
% x5=0.5;
x6=0.98;
x7=0.88;
x8=0.79;
x9=0.1;
x10=0.4;

yy5=zeros(L,1);
for i=1:L
    x5a=x5v(i);
   Xj=[x1;x2;x3;x4;x5a;x6;x7;x8;x9;x10];
   [ya1,ya2,ya3,ya4]=FuncSystem(Xj,W1_NFS,W2_NFS,W3_NFS,W4_NFS);
   yy5(i)=ya4;
end


% yy10_min=min(yy10);
% yy8_max=max(yy8);
% 
% figure(18)
% plot(t,x5,t,yy5);
%%%%%%%%%%%%%%%%%%%%%%%%
x9v=0.2*sin(t)+0.25;
L=length(x9v);
x1=0.7;
x2=0.9;
x3=0.6;
x4=0.2;
x5=0.5;
x6=0.98;
x7=0.88;
x8=0.79;
% x9=0.1;
x10=0.4;
yy9=zeros(L,1);
for i=1:L
    x9a=x9v(i);
        Xj=[x1;x2;x3;x4;x5;x6;x7;x8;x9a;x10];
%         Phi=FuncFuzzy(W1,W2,q,dq,tau);
        [ya1,ya2,ya3,ya4]=FuncSystem(Xj,W1_NFS,W2_NFS,W3_NFS,W4_NFS);
        yy9(i)=ya4;
end


% yy9_min=min(yy1);
% yy9_max=max(yy1);

figure(9)
subplot(121)
plot(t,x5v,t,yy5);
subplot(122)
plot(t,x9v,t,yy9);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Case two
%has two variables
NN=5000;
x1v=rand(NN,1)*0.3+0.3;
x9v=rand(NN,1)*0.3+0.05;
% n1=length(x1);
% n2=length(x9);
% A=zeros(n1*n2,3);
% x1=0.7;
x2=0.9;
x3=0.6;
x4=0.2;
x5=0.5;
x6=0.98;
x7=0.88;
x8=0.79;
% x9=0.1;
x10=0.4;
yy19=zeros(NN,1);
for i=1:NN
    x1a=x1v(i);
    x9a=x9v(i);

    Xj=[x1a;x2;x3;x4;x5;x6;x7;x8;x9a;x10];
%         Phi=FuncFuzzy(W1,W2,q,dq,tau);
        [ya1,ya2,ya3,ya4]=FuncSystem(Xj,W1_NFS,W2_NFS,W3_NFS,W4_NFS);
        yy19(i)=ya4;
end


x10v=rand(NN,1)*0.3+0.3;
% x9=rand(NN,1)*0.3+0.05;
% n1=length(x1);
% n2=length(x9);
% A=zeros(n1*n2,3);
x1=0.7;
x2=0.9;
x3=0.6;
x4=0.2;
x5=0.5;
x6=0.98;
x7=0.88;
x8=0.79;
% x9=0.1;
% x10=0.4;
yy20=zeros(NN,1);
for i=1:NN
    x10a=x10v(i);
    x9a=x9v(i);

    Xj=[x1;x2;x3;x4;x5;x6;x7;x8;x9a;x10a];
%         Phi=FuncFuzzy(W1,W2,q,dq,tau);
        [ya1,ya2,ya3,ya4]=FuncSystem(Xj,W1_NFS,W2_NFS,W3_NFS,W4_NFS);
        yy20(i)=ya4;
end


figure(21)
subplot(121)
plot3(x1v,x9v,yy19,'b.','MarkerSize',0.5);
xlabel('variable x1');ylabel('variable x9');zlabel('Assessment result')
subplot(122)
plot3(x9v,x10v,yy20,'b.','MarkerSize',0.5);
xlabel('variable x9');ylabel('variable x10');zlabel('Assessment result')

% for i=1:n1
%     x1a=x1(i);
%     for j=1:n2
%         x9a=x9(j);
%         Xj=[x1a;x2;x3;x4;x5;x6;x7;x8;x9a,x10];
% %         Phi=FuncFuzzy(W1,W2,q,dq,tau);
%         yy(i,j)=FuncSystem(Xj,W1_NFS,W2_NFS,W3_NFS,W4_NFS);
%         
%     end
% end
% 
% % A=[x1,x9,yy];
% plot3(x1,x9,yy,'b.','MarkerSize',0.5)
% [x,y]=meshgrid(x1,x9);
% z=yy;
% surf(x,y,z)