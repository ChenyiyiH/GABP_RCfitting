%% 该代码为基于小波神经网络预测代码
function [errorsum_wavenn,R2_wavenn,MSE_wavenn,RMSE_wavenn]=wavenn(datatable,train_par)
%% 清空环境变量
% clc;
% clear;
% close all;
nntwarn off;


%% 网络参数配置
load(datatable);
input = data.in;   % 16000x3
output = data.out; % 16000x2

% 打乱数据
n = randperm(size(input, 1));
n_idx = size(input,1) * train_par;
idx = n_idx; % 数据集划分点

input_train = input(n(1:idx), :); % [样本数 x 特征数]
output_train = output(n(1:idx), :); % [样本数 x 输出数]
input_test = input(n(idx+1:end), :); % [样本数 x 特征数]
output_test = output(n(idx+1:end), :); % [样本数 x 输出数]

M=size(input,2); %输入节点个数
N=size(output,2); %输出节点个数

n=20; %隐形节点个数
lr1=0.1; %学习概率
lr2=0.01; %学习概率
maxgen=2000; %迭代次数

%权值初始化
Wjk=randn(n,M);Wjk_1=Wjk;Wjk_2=Wjk_1;
Wij=randn(N,n);Wij_1=Wij;Wij_2=Wij_1;
a=randn(1,n);a_1=a;a_2=a_1;
b=randn(1,n);b_1=b;b_2=b_1;

%节点初始化
y=zeros(1,N);
net=zeros(1,n);
net_ab=zeros(1,n);

%权值学习增量初始化
d_Wjk=zeros(n,M);
d_Wij=zeros(N,n);
d_a=zeros(1,n);
d_b=zeros(1,n);

%% 输入输出数据归一化
[inputn,inputps]=mapminmax(input');
[outputn,outputps]=mapminmax(output'); 
inputn=inputn';
outputn=outputn';

error=zeros(1,maxgen);
%% 网络训练
for i=1:maxgen
    
    %误差累计
    error(i)=0;
    
    % 循环训练
    for kk=1:size(input,1)
        x=inputn(kk,:);
        yqw=outputn(kk,:);
   
        for j=1:n
            for k=1:M
                net(j)=net(j)+Wjk(j,k)*x(k);
                net_ab(j)=(net(j)-b(j))/a(j);
            end
            temp=mymorlet(net_ab(j));
            for k=1:N
                y=y+Wij(k,j)*temp;   %小波函数
            end
        end
        
        %计算误差和
        error(i)=error(i)+sum(abs(yqw-y));
        
        %权值调整
        for j=1:n
            %计算d_Wij
            temp=mymorlet(net_ab(j));
            for k=1:N
                d_Wij(k,j)=d_Wij(k,j)-(yqw(k)-y(k))*temp;
            end
            %计算d_Wjk
            temp=d_mymorlet(net_ab(j));
            for k=1:M
                for l=1:N
                    d_Wjk(j,k)=d_Wjk(j,k)+(yqw(l)-y(l))*Wij(l,j) ;
                end
                d_Wjk(j,k)=-d_Wjk(j,k)*temp*x(k)/a(j);
            end
            %计算d_b
            for k=1:N
                d_b(j)=d_b(j)+(yqw(k)-y(k))*Wij(k,j);
            end
            d_b(j)=d_b(j)*temp/a(j);
            %计算d_a
            for k=1:N
                d_a(j)=d_a(j)+(yqw(k)-y(k))*Wij(k,j);
            end
            d_a(j)=d_a(j)*temp*((net(j)-b(j))/b(j))/a(j);
        end
        
        %权值参数更新      
        Wij=Wij-lr1*d_Wij;
        Wjk=Wjk-lr1*d_Wjk;
        b=b-lr2*d_b;
        a=a-lr2*d_a;
    
        d_Wjk=zeros(n,M);
        d_Wij=zeros(N,n);
        d_a=zeros(1,n);
        d_b=zeros(1,n);

        y=zeros(1,N);
        net=zeros(1,n);
        net_ab=zeros(1,n);
        
        Wjk_1=Wjk;Wjk_2=Wjk_1;
        Wij_1=Wij;Wij_2=Wij_1;
        a_1=a;a_2=a_1;
        b_1=b;b_2=b_1;
    end
end

%% 网络预测
%预测输入归一化
x=mapminmax('apply',input_test',inputps);
x=x';
% 预测输出维度修改
yuce = zeros(size(x,1), 2);

% 网络预测
for i = 1:size(x,1)
    x_test = x(i, :);

    % 清空临时变量
    y = zeros(1, N); % N 为输出节点个数，即2

    for j = 1:n
        for k = 1:M
            net(j) = net(j) + Wjk(j, k) * x_test(k);
            net_ab(j) = (net(j) - b(j)) / a(j);
        end
        temp = mymorlet(net_ab(j));
        for k = 1:N
            y(k) = y(k) + Wij(k, j) * temp; 
        end
    end

    % 存储预测结果
    yuce(i, :) = y;
    
    % 清理变量
    y = zeros(1, N);
    net = zeros(1, n);
    net_ab = zeros(1, n);
end

% 预测输出反归一化
ynn = mapminmax('reverse', yuce', outputps);
Y_sim_wavenn = ynn';

%% 计算预测误差
error_wavenn = Y_sim_wavenn - output_test;
errorsum_wavenn = sum(abs(error_wavenn));
% 评估模型性能
R_wavenn = corrcoef(output_test, Y_sim_wavenn);
R2_wavenn = R_wavenn(1, 2)^2;  % 决定系数 (R²)

MSE_wavenn = immse(output_test, double(Y_sim_wavenn));  % 均方误差
RMSE_wavenn = sqrt(MSE_wavenn);  % 均方根误差

% 打印结果
disp(['R² = ', num2str(R2_wavenn)]);
disp(['MSE = ', num2str(MSE_wavenn)]);
disp(['RMSE = ', num2str(RMSE_wavenn)]);

end



%%
function y = mymorlet(t)
% 小波函数
y = exp(-(t.^2)/2) .* cos(1.75*t);
end

function y = d_mymorlet(t)
% 小波偏导函数
y = -1.75 * sin(1.75 * t) .* exp(-(t.^2)/2) - t .* cos(1.75 * t) .* exp(-(t.^2)/2);
end
