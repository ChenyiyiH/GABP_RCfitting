%% 单层竞争神经网络预测 (SCNN)
function [errorsum_scnn,R2_scnn,MSE_scnn,RMSE_scnn,net_scnn]=SCNN(datatable,train_par)
%% 清空环境变量
% clc;
% clear;
% close all;
nntwarn off;
load(datatable);

%% 录入输入数据
% 载入数据并将数据分成训练和预测两类
input = data.in;   % 16000x3
output = data.out; % 16000x2
n = randperm(size(input, 1));  
n_idx = size(input,1) * train_par;
% 划分训练集和测试集
input_train = input(n(1:n_idx), :)'; % 3x13000
output_train = output(n(1:n_idx), :)'; % 2x13000
input_test = input(n(n_idx+1:end), :)'; % 3x3000
output_test = output(n(n_idx+1:end), :)'; % 2x3000

% 数据归一化
[inputn, inputps] = mapminmax(input_train); % 归一化到[0, 1]
[outputn, outputps] = mapminmax(output_train); % 归一化到[0, 1]

% 确保数据为双精度
inputn = double(inputn);
outputn = double(outputn);

% 转置后符合神经网络的输入格式
P = inputn; % 3x13000
T = outputn; % 2x13000

% 取输入元素的最大值和最小值Q
Q = minmax(P);

%% 网络建立和训练
% 利用newc()命令建立竞争网络
numClusters = 2; % 竞争层的神经元个数，也就是要分类的个数
learningRate = 0.1; % 学习速率
net_scnn = newc(Q, numClusters, learningRate);

% 初始化网络及设定网络参数
net_scnn = init(net_scnn);
net_scnn.trainParam.epochs = 200; % 设置训练周期
net_scnn.trainParam.lr = learningRate; % 设置学习率

% 训练网络
net_scnn = train(net_scnn, P, T); % 添加目标值 T

%% 网络的效果验证

% 将原数据带入，测试网络效果
a = sim(net_scnn, P); % 网络仿真
ac = vec2ind(a); % 将网络输出转化为下标向量

%% 网络作分类的预测
% 预测测试集
inputn_test = mapminmax('apply', input_test, inputps); % 归一化测试数据
inputn_test = double(inputn_test); % 确保数据为双精度

% 确保数据维度匹配
if size(inputn_test, 1) ~= size(P, 1)
    error('测试数据的特征数与训练数据的特征数不匹配');
end

% 进行预测
Y_SCNN = sim(net_scnn, inputn_test); % 预测

% 反归一化测试结果
Y_sim_scnn = mapminmax('reverse', Y_SCNN, outputps); % 反归一化

% 将网络输出转化为下标向量
yc = vec2ind(Y_SCNN); % 分类下标

% 计算预测误差
error_scnn = Y_sim_scnn - output_test;
errorsum_scnn = sum(abs(error_scnn)); 
%%
% 评估模型性能
R_scnn = corrcoef(output_test, Y_sim_scnn);
R2_scnn = R_scnn(1, 2)^2;  % 决定系数 (R²)

MSE_scnn = immse(output_test, double(Y_sim_scnn));  % 均方误差
RMSE_scnn = sqrt(MSE_scnn);  % 均方根误差

% 打印结果
disp(['R² = ', num2str(R2_scnn)]);
disp(['MSE = ', num2str(MSE_scnn)]);
disp(['RMSE = ', num2str(RMSE_scnn)]);
%%
% 如果需要查看预测结果，可以将其可视化
figure;
scrsz = get(0, 'ScreenSize');
set(gcf, 'Position', [scrsz(3)*1/4, scrsz(4)*1/6, scrsz(3)*4/5, scrsz(4)]*3/4);

% 选择第一个输出特征进行可视化（例如第一个特征）
subplot(2, 1, 1);
plot(output_test(1, :), 'o');
hold on;
plot(Y_sim_scnn(1, :), 'r*');
legend('真实数据', '预测数据');
title('测试集上的回归效果 (第一个输出特征)');
xlabel('样本');
ylabel('输出值');
grid on;

subplot(2, 1, 2);
plot(output_test(2, :), 'o');
hold on;
plot(Y_sim_scnn(2, :), 'r*');
legend('真实数据', '预测数据');
title('测试集上的回归效果 (第二个输出特征)');
xlabel('样本');
ylabel('输出值');
grid on;
end