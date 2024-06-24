clear;
clc;

%assume the weights of the system
W1_real=[0.71;0.29;0.8;0.5;0.4];
W2_real=[0.84;0.64;0.86;0.09;0.4];
W3_real=[0.95;0.7;0.6;0.05;0.3];
W4_real=[0.93;0.92;0.6;0.7;0.9];
%collecting data
N=250;%number of sample
X_input =rand(10,N);
Y_output1=zeros(N,1);Y_output2=zeros(N,1);Y_output3=zeros(N,1);Y_output4=zeros(N,1);
%calculate the response
for j=1:N
    Xj=X_input(:,j);
    [ya1,ya2,ya3,ya4]=FuncSystem(Xj,W1_real,W2_real,W3_real,W4_real);
    
    Y_output1(j)=ya1; Y_output2(j)=ya2; Y_output3(j)=ya3; Y_output4(j)=ya4;

end

figure(1)
subplot(221)
plot(Y_output1);
subplot(222)
plot(Y_output2);
subplot(223)
plot(Y_output3);
subplot(224)
plot(Y_output4);