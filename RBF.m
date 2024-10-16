%% RBF网络
% 严格径向基函数RBE
% 径向基函数RBF
function [errorsum_rbe,R2_rbe,MSE_rbe,RMSE_rbe,net_rbe]=RBF(datatable,train_par)
%% 清空环境变量（可根据需要选择是否使用）
% clc;
% clear;
% close all;
nntwarn off;
load(datatable);

%% 产生输入 输出数据并归一化
input = data.in;
output = data.out;
n = randperm(size(input, 1));
n_idx = size(input,1) * train_par;
idx = n_idx;% 数据集划分点

% 划分训练集和测试集
input_train = input(n(1:idx), :)';
output_train = output(n(1:idx), :)';
input_test = input(n(idx+1:end), :)';
output_test = output(n(idx+1:end), :)';

% 数据归一化
[inputn, inputps] = mapminmax(input_train);
[outputn, outputps] = mapminmax(output_train);

%% 网络建立和训练
% 使用 RBE 或 RBF 网络，二选一
net_rbe = newrbe(inputn, outputn);
% net_rbf = newrb(inputn, outputn,0.2,1,15);  % 可选网络构造方式

%% 网络的效果验证
%% RBE
% 归一化测试数据
inputn_test = mapminmax('apply', input_test, inputps);

% 仿真测试
Y_rbe = sim(net_rbe, inputn_test);

% 测试结果反归一化
Y_sim_rbe = mapminmax('reverse', Y_rbe, outputps);

% 计算预测误差
error_rbe = Y_sim_rbe - output_test;
errorsum_rbe = sum(abs(error_rbe));
% 评估模型性能
R_rbe = corrcoef(output_test, Y_sim_rbe);
R2_rbe = R_rbe(1, 2)^2;  % 决定系数 (R²)

MSE_rbe = immse(output_test, double(Y_sim_rbe));  % 均方误差
RMSE_rbe = sqrt(MSE_rbe);  % 均方根误差

% 打印结果
disp(['R² = ', num2str(R2_rbe)]);
disp(['MSE = ', num2str(MSE_rbe)]);
disp(['RMSE = ', num2str(RMSE_rbe)]);
%% RBF
% 仿真测试
% Y_rbf = sim(net_rbf, inputn_test);
% 
% % 测试结果反归一化
% Y_sim_rbf = mapminmax('reverse', Y_rbf, outputps);
% 
% % 计算预测误差
% error_rbf = Y_sim_rbf - output_test;
% 
% % 评估模型性能
% R_rbf = corrcoef(output_test, Y_sim_rbf);
% R2_rbf = R_rbf(1, 2)^2;  % 决定系数 (R²)
% 
% MSE_rbf = immse(output_test, double(Y_sim_rbf));  % 均方误差
% RMSE_rbf = sqrt(MSE_rbf);  % 均方根误差
% 
% % 打印结果
% disp(['R² = ', num2str(R2_rbf)]);
% disp(['MSE = ', num2str(MSE_rbf)]);
% disp(['RMSE = ', num2str(RMSE_rbf)]);
end