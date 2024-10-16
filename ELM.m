%% 极限学习机在回归拟合问题中的应用研究
function [errorsum_elm,R2_elm,MSE_elm,RMSE_elm]=ELM(datatable,train_par)
%% 清空环境变量
% clc;
% clear;
% close all;
warning off;

% 载入数据
load(datatable);
input = data.in;   % 16000x3
output = data.out; % 16000x2
train_par = train_par/100;
% 打乱数据
n = randperm(size(input, 1));
n_idx = size(input,1) * train_par;
idx = n_idx; % 数据集划分点

% 划分训练集和测试集
input_train = input(n(1:idx), :)'; % 转置为 [特征数 x 样本数]
output_train = output(n(1:idx), :)'; % 转置为 [特征数 x 样本数]
input_test = input(n(idx+1:end), :)'; % 转置为 [特征数 x 样本数]
output_test = output(n(idx+1:end), :)'; % 转置为 [特征数 x 样本数]

% 训练集——13000个样本
P_train = input_train;
T_train = output_train;

% 测试集——3000个样本
P_test = input_test;
T_test = output_test;

%% 归一化
% 训练集
% [Pn_train, inputps] = mapminmax(P_train, -1, 1);
% Pn_test = mapminmax('apply', P_test, inputps);
Pn_train = P_train;
Pn_test = P_test;
% 测试集
[Tn_train, outputps] = mapminmax(T_train, -1, 1);
Tn_test = mapminmax('apply', T_test, outputps);

Tn_train = T_train;
Tn_test = T_test;
tic
%% ELM创建/训练
[IW, B, LW, TF, TYPE] = elmtrain(Pn_train, Tn_train, 20, 'sig', 0);

%% ELM仿真测试
Tn_sim = elmpredict(Pn_test, IW, B, LW, TF, TYPE);

% 反归一化
% T_sim = mapminmax('reverse', Tn_sim, outputps);
T_sim =Tn_sim;
toc
%% 结果对比
error_elm = T_sim - output_test;

% 计算总误差
errorsum_elm = sum(abs(error_elm));

result = [T_test' T_sim'];

% 均方误差
E = mse(T_sim - T_test);

% 决定系数计算
N = size(T_test, 2); % 样本数
y_mean = mean(T_test, 2); % 目标值均值
SS_tot = sum((T_test - y_mean).^2, 2); % 总平方和
SS_res = sum((T_test - T_sim).^2, 2); % 残差平方和
R2 = 1 - SS_res ./ SS_tot; % 决定系数

%% 绘图
% figure;
% plot(1:length(T_test), T_test, 'r*');
% hold on;
% plot(1:length(T_sim), T_sim, 'b:o');
% xlabel('测试集样本编号');
% ylabel('测试集输出');
% title('ELM测试集输出');
% legend('期望输出', '预测输出');
% 
% figure;
% plot(1:length(T_test), T_test - T_sim, 'r-*');
% xlabel('测试集样本编号');
% ylabel('绝对误差');
% title('ELM测试集预测误差');
%%
% 评估模型性能
R_elm = corrcoef(output_test, T_sim);
R2_elm = R_elm(1, 2)^2;  % 决定系数 (R²)

MSE_elm = immse(output_test, double(T_sim));  % 均方误差
RMSE_elm = sqrt(MSE_elm);  % 均方根误差

% 打印结果
disp(['R² = ', num2str(R2_elm)]);
disp(['MSE = ', num2str(MSE_elm)]);
disp(['RMSE = ', num2str(RMSE_elm)]);
end