%% 使用Libsvm进行回归的小例子
function [errorsum_svm,R2_svm,MSE_svm,RMSE_svm]=SVM(datatable,train_par)
% clc;
% clear;
% close all;
nntwarn off;
load(datatable);
%% 数据准备与归一化
% 下载输入输出数据
input = data.in;   % 16000x3
output = data.out; % 16000x2

% 随机排列数据索引
numSamples = size(input, 1);
n_idx = size(input,1) * train_par;
idx = n_idx; % 数据集划分点
randIdx = randperm(numSamples);

% 划分训练集和测试集
trainIdx = randIdx(1:idx);
testIdx = randIdx(idx+1:end);

input_train = input(trainIdx, :); % 13000x3
output_train = output(trainIdx, :); % 13000x2
input_test = input(testIdx, :); % 3000x3
output_test = output(testIdx, :); % 3000x2

% 确保数据类型为 double
input_train = double(input_train);
output_train = double(output_train);
input_test = double(input_test);
output_test = double(output_test);

% 数据归一化
[inputn, inputps] = mapminmax(input_train');
[outputn, outputps] = mapminmax(output_train');

% 确保数据类型为 double
inputn = double(inputn');
outputn = double(outputn');

%% 建立回归模型
% SVM回归模型不能直接用于多维输出，因此需要对每个输出特征建立独立的模型
numOutputFeatures = size(output_train, 2);
models = cell(numOutputFeatures, 1);

for i = 1:numOutputFeatures
    models{i} = fitrsvm(inputn, outputn(:, i), ...
        'Standardize', true, ...
        'KernelFunction', 'rbf', ...
        'BoxConstraint', 2.2, ...
        'KernelScale', 2.8, ...
        'Epsilon', 0.01);
end

%% 在测试集上进行预测
% 对测试集进行归一化
inputn_test = mapminmax('apply', input_test', inputps);

% 使用训练好的模型进行预测
predicted = zeros(size(input_test, 1), numOutputFeatures);
for i = 1:numOutputFeatures
    predicted(:, i) = predict(models{i}, inputn_test');
end

% 预测结果反归一化
Y_sim_svm = mapminmax('reverse', predicted', outputps)';

% 计算预测误差
error_svm = Y_sim_svm - output_test;
errorsum_svm = sum(abs(error_svm));
% 评估模型性能
R_svm = corrcoef(output_test, Y_sim_svm);
R2_svm = mean(R_svm(1, 2:end).^2);  % 决定系数 (R²) 对于多维输出取平均

MSE_svm = immse(output_test, double(Y_sim_svm));  % 均方误差
RMSE_svm = sqrt(MSE_svm);  % 均方根误差

% 打印结果
fprintf('R² = %.4f\n', R2_svm);
fprintf('MSE = %.4f\n', MSE_svm);
fprintf('RMSE = %.4f\n', RMSE_svm);

% 可视化测试集上的回归效果
% figure;
% scrsz = get(0, 'ScreenSize');
% set(gcf, 'Position', [scrsz(3)*1/4, scrsz(4)*1/6, scrsz(3)*4/5, scrsz(4)]*3/4);
% 
% % 选择一个输出特征进行可视化（例如第一个特征）
% plot(output_test(:, 1), 'o');
% hold on;
% plot(Y_sim_svm(:, 1), 'r*');
% legend('真实数据', '回归数据');
% title('测试集上的回归效果 (第一个输出特征)');
% xlabel('样本');
% ylabel('输出值');
% grid on;
end