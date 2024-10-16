%% 概率神经网络的分类预测--基于PNN
function [errorsum_pnn,R2_pnn,MSE_pnn,RMSE_pnn,net_pnn]=PNN(datatable,train_par)
%% 清空环境变量
% clc;
% clear;
% close all;
nntwarn off;
load(datatable);

%% 数据载入
% 载入数据并将数据分成训练和测试集
input = data.in;   % 16000x3
output = data.out; % 16000x2

% 打乱数据
n = randperm(size(input, 1));
n_idx = size(input,1) * train_par;
% 划分训练集和测试集
input_train = input(n(1:n_idx), :)'; % 13000x3'
output_train = output(n(1:n_idx), :)'; % 13000x2'
input_test = input(n(n_idx+1:end), :)'; % 3000x3'
output_test = output(n(n_idx+1:end), :)'; % 3000x2'

% 数据归一化
[inputn, inputps] = mapminmax(input_train); % 归一化到[0, 1]
inputn_test = mapminmax('apply', input_test, inputps); % 归一化测试数据
[outputn, outputps] = mapminmax(output_train);
% % 转换目标数据
output_train_class = vec2ind(output_train'); % 将目标类别转为向量
output_test_class = vec2ind(output_test'); % 将测试目标类别转为向量
% 
% % 转换为PNN输入格式
t_train = full(ind2vec(output_train_class)); % 将训练标签转换为PNN的格式
t_test = full(ind2vec(output_test_class));   % 将测试标签转换为PNN的格式

%% 使用newpnn函数建立PNN
Spread = 0.01;
net_pnn = newpnn(inputn, outputn, Spread);

%% 训练数据回代 查看网络的分类效果
% Sim函数进行网络预测
Y_pnn_train = sim(net_pnn, inputn);

% 反归一化并转换为分类结果
Y_sim_pnn_train = vec2ind(Y_pnn_train);
Y_sim_pnn_train = mapminmax('reverse', Y_pnn_train, outputps);
%% 计算和显示训练集上的分类效果
% figure;
% subplot(1, 2, 1);
% plot(Y_sim_pnn_train, 'bo');
% hold on;
% plot(output_train_class, 'r*');
% title('PNN 网络训练后的效果');
% xlabel('样本编号');
% ylabel('分类结果');
% legend('预测结果', '实际结果');
% grid on;
% 
% % 计算训练误差
% train_errors = Y_sim_pnn_train ~= output_train_class;
% error_train = sum(train_errors) / length(train_errors);
% disp(['训练集准确率: ', num2str((1 - error_train) * 100), '%']);

%% 网络预测未知数据效果
Y_sim_pnn_test = sim(net_pnn, inputn_test);
Y_sim_pnn_test = mapminmax('reverse', Y_sim_pnn_test, outputps);
error_pnn = Y_sim_pnn_test - output_test;
% 计算测试误差
% test_errors = Y_sim_pnn_test ~= output_test_class;
% error_test = sum(test_errors) / length(test_errors);
% disp(['测试集准确率: ', num2str((1 - error_test) * 100), '%']);

%% 显示测试集分类结果
% figure;
% plot(Y_sim_pnn_test, 'b^');
% hold on;
% plot(output_test_class, 'r*');
% title('PNN 网络的预测效果');
% xlabel('预测样本编号');
% ylabel('分类结果');
% legend('预测结果', '实际结果');
% grid on;
% 
% % 计算和显示误差图
% errors = Y_sim_pnn_test - output_test_class;
% figure;
% plot(errors, 'k-');
% title('PNN 网络测试集误差图');
% xlabel('样本编号');
% ylabel('误差');
% grid on;
%%
% 评估模型性能
errorsum_pnn = sum(abs(error_pnn));
R_pnn = corrcoef(output_test, Y_sim_pnn_test);
R2_pnn = R_pnn(1, 2)^2;  % 决定系数 (R²)

MSE_pnn = immse(output_test, double(Y_sim_pnn_test));  % 均方误差
RMSE_pnn = sqrt(MSE_pnn);  % 均方根误差

% 打印结果
disp(['R² = ', num2str(R2_pnn)]);
disp(['MSE = ', num2str(MSE_pnn)]);
disp(['RMSE = ', num2str(RMSE_pnn)]);
end