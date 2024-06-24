% PSO optimization weights

%load training data
%%particle swarm optimization
clc;
clear;
tic;
load Traindata2.mat
%200组训练，后50组测试
n=200;%number of sample
nt=50;
Xtrain_input=X_input(:,1:n);
Ytrain_output1=Y_output1(1:n);
Ytrain_output2=Y_output2(1:n);
Ytrain_output3=Y_output3(1:n);
Ytrain_output4=Y_output4(1:n);

%参数归一化


% load Data200.mat
%一些参数设置
% t=0:0.001:1;
% n=length(t);%样本数
% m=5;%神经元的个数，和Phi_2dof里的n1相等

popsize=200;
D=20;%需要确定权重参数w
maxgen=50;
Wmin=0.1;  
Wmax=0.9;
C1=0.5;
C2=0.5;

Xmin=0;%不确定性参数区间
Xmax=1;
Vmax = 0.5; %　最大飞行速度
Vmin = -0.5; %　最小飞行速度
%初始化PSO
Xinput=rand(popsize,D).*(Xmax-Xmin)+Xmin;
popx=Xinput;
popv= rand(popsize,D).*(Vmax-Vmin)+Vmin;
fitness1=zeros(popsize,1);
% fitness2=zeros(popsize,1);
fitness=zeros(popsize,1);
%计算fitness
for p=1:popsize
%     W=[popx(p,(1:3*m));popx(p,(3*m+1):6*m)];
    W=popx(p,:)';
    W1=W(1:5);W2=W(6:10);W3=W(11:15);W4=W(16:20);

    y_NNoutput1=zeros(n,1);y_NNoutput2=zeros(n,1);y_NNoutput3=zeros(n,1);y_NNoutput4=zeros(n,1);
    for i=1:n
        Xj=Xtrain_input(:,i);
%         Phi=FuncFuzzy(W1,W2,q,dq,tau);
       [ya1,ya2,ya3,ya4]=FuncSystem(Xj,W1,W2,W3,W4);
        y_NNoutput1(i)=ya1;
        y_NNoutput2(i)=ya2;
        y_NNoutput3(i)=ya3;
        y_NNoutput4(i)=ya4;
    end
%     Error=-Ytrain_output.*log(y_NNoutput);
    Error1=(y_NNoutput1-Ytrain_output1).^2;
    Error2=(y_NNoutput2-Ytrain_output2).^2;
    Error3=(y_NNoutput3-Ytrain_output3).^2;
    Error4=(y_NNoutput4-Ytrain_output4).^2;

%   fitness1(p)=sum(Error4);%A
    fitness1(p)=sum(Error2+Error4);%B
%     fitness1(p)=sum(Error1+Error2+Error3+Error4);%C

   fitness(p)=fitness1(p)/n;
end



%% %初始化个体最优和群体最优
[bestfitness,bestindex] = min(fitness);
zbest = popx(bestindex,:); %全局最佳
zbest1(1,:) = popx(bestindex,:);
gbest = popx; %个体最佳
fitnessgbest = fitness; %个体最佳适应度值
fitnesszbest = bestfitness; %全局最佳适应度值
yy=zeros(1,maxgen);

% WW=Wmin;
%% VI. 迭代寻优
for i = 1:maxgen
    % 更新个体的位置和速度
    WW=(maxgen-i)/(maxgen)*(Wmax-Wmin)+Wmin;
%         WW=(Wmax+Wmin)/2;
    for j=1:popsize
       
        popv(j,:) = WW*popv(j,:)+C1*rand*(gbest(j,:)-popx(j,:))+C2*rand*(zbest-popx(j,:)) ;
        popx(j,:) = popx(j,:)+popv(j,:) ;
        
        for kk=1:D
            if popx(j,kk) > Xmax
                popx(j,kk) = Xmax;
            end
            if popx(j,kk) < Xmin
                popx(j,kk) = Xmin;
            end
            if popv(j,kk) > Vmax
                popv(j,kk) = Vmax;
            end
            if popv(j,kk) < Vmin
                popv(j,kk) = Vmin;
            end
        end
        
    end
%%检查是否超出范围


%计算fitness
for p=1:popsize

    W=popx(p,:)';
    W1=W(1:5);W2=W(6:10);W3=W(11:15);W4=W(16:20);

     y_NNoutput1=zeros(n,1);y_NNoutput2=zeros(n,1);y_NNoutput3=zeros(n,1);y_NNoutput4=zeros(n,1);
    for k=1:n
        Xj=Xtrain_input(:,k);
%       Phi=FuncFuzzy(W1,W2,q,dq,tau);
        [ya1,ya2,ya3,ya4]=FuncSystem(Xj,W1,W2,W3,W4);
        y_NNoutput1(k)=ya1;
        y_NNoutput2(k)=ya2;
        y_NNoutput3(k)=ya3;
        y_NNoutput4(k)=ya4;
    end
%     Error=-Ytrain_output.*log(y_NNoutput);
    Error1=(y_NNoutput1-Ytrain_output1).^2;
    Error2=(y_NNoutput2-Ytrain_output2).^2;
    Error3=(y_NNoutput3-Ytrain_output3).^2;
    Error4=(y_NNoutput4-Ytrain_output4).^2;

%   fitness1(p)=sum(Error4);%A
    fitness1(p)=sum(Error2+Error4);%B
%     fitness1(p)=sum(Error1+Error2+Error3+Error4);%C


%    fitness(p)=(fitness1(p)+fitness2(p))/(2*n);
   fitness(p)=fitness1(p)/n;

end


for k = 1:popsize
% 个体最优更新
if fitness(k) < fitnessgbest(k)
gbest(k,:) = popx(k,:);
fitnessgbest(k) = fitness(k);
end
% 群体最优更新
if fitness(k) < fitnesszbest
zbest = popx(k,:);
fitnesszbest = fitness(k);
end
end
yy(i) = fitnesszbest;%适应度值收敛


zbest1(i,:)=zbest;%每次迭代的最佳结果
%calculate every subsystem output 
W1=zbest(1:5)';W2=zbest(6:10)';W3=zbest(11:15)';W4=zbest(16:20)';
for k=1:n
        Xj=Xtrain_input(:,k);
%         Phi=FuncFuzzy(W1,W2,q,dq,tau);
        [ya1,ya2,ya3,ya4]=FuncSystem(Xj,W1,W2,W3,W4);
        y_NNoutput1(k)=ya1;
        y_NNoutput2(k)=ya2;
        y_NNoutput3(k)=ya3;
        y_NNoutput4(k)=ya4;
end
 yError1=(y_NNoutput1-Ytrain_output1).^2;
 yError2=(y_NNoutput2-Ytrain_output2).^2;
 yError3=(y_NNoutput3-Ytrain_output3).^2;
 yError4=(y_NNoutput4-Ytrain_output4).^2;

 fitness_E1(i)=sum(Error1)/n;
 fitness_E2(i)=sum(Error2)/n;
 fitness_E3(i)=sum(Error3)/n;
 fitness_E4(i)=sum(Error4)/n;


end
min(yy)

W_NN=zbest';
W1_NFS=W_NN(1:5);
W2_NFS=W_NN(6:10);
W3_NFS=W_NN(11:15);
W4_NFS=W_NN(16:20);

toc;

T1=1:1:20;

figure(1)
plot(1:1:maxgen,yy);
xlabel('Iteration');ylabel('Fitness');

figure(2)
% subplot(221)
plot(T1,W_real,'--o',T1,zbestC,'--*');
xlabel('Number of variable');ylabel('Weight value');
legend('Optimal value','Optimized value');
% subplot(222)
% plot(T1,W2_real,'-o',T1,W2_NFS,'-*');
% subplot(223)
% plot(T1,W3_real,'-o',T1,W3_NFS,'-*');
% subplot(224)
% plot(T1,W4_real,'-o',T1,W4_NFS,'-*');



Xtest_input=X_input(:,((n+1):end));
Ytest_output1=Y_output1((n+1):end);
Ytest_output2=Y_output2((n+1):end);
Ytest_output3=Y_output3((n+1):end);
Ytest_output4=Y_output4((n+1):end);

    ytest_NNoutput1=zeros(nt,1);ytest_NNoutput2=zeros(nt,1);
    ytest_NNoutput3=zeros(nt,1);ytest_NNoutput4=zeros(nt,1);
    for i=1:length(Ytest_output1)
        Xj=Xtest_input(:,i);
%         Phi=FuncFuzzy(W1,W2,q,dq,tau);
        [ya1,ya2,ya3,ya4]=FuncSystem(Xj,W1_NFS,W2_NFS,W3_NFS,W4_NFS);
        ytest_NNoutput1(i)=ya1;
        ytest_NNoutput2(i)=ya2;
        ytest_NNoutput3(i)=ya3;
        ytest_NNoutput4(i)=ya4;

    end
T2=1:1:nt;

figure(3)
subplot(411)
plot(T2,Ytest_output1,'-*',T2,ytest_NNoutput1,'-o');
ylabel('Subsystem 1 output')
subplot(412)
plot(T2,Ytest_output2,'-*',T2,ytest_NNoutput2,'-o');
ylabel('Subsystem 2 output')
subplot(413)
plot(T2,Ytest_output3,'-*',T2,ytest_NNoutput3,'-o');
ylabel('Subsystem 3 output')
subplot(414)
plot(T2,Ytest_output4,'-*',T2,ytest_NNoutput4,'-o');
xlabel('Sample');ylabel('whole system output')
legend('Actural value','Predicted value');



%A:只有大系统输出已知；B：有一个小系统输出已知，C：系统全部输出已知
ng=1:1:maxgen;
figure(4)
subplot(221)
plot(ng,fitness_E1A,ng,fitness_E1B,ng,fitness_E1C);
subplot(222)
plot(ng,fitness_E2A,ng,fitness_E2B,ng,fitness_E2C);
subplot(223)
plot(ng,fitness_E3A,ng,fitness_E3B,ng,fitness_E3C);
subplot(224)
plot(ng,fitness_E4A,ng,fitness_E4B,ng,fitness_E4C);
legend('System','System with one subsystem','All system');


