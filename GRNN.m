%% 使用交叉验证的GRNN神经网络预测程序
function [errorsum_grnn,R2_grnn,MSE_grnn,RMSE_grnn,net_grnn]=GRNN(datatable,train_par)
%% 清空环境变量（可根据需要选择是否使用）
% clc;
% clear;
% close all;
nntwarn off;
load(datatable);

%% 载入数据
% 训练数据和预测数据提取及归一化
input = data.in;
output = data.out;
n = randperm(size(input, 1));  % 使用 size(input, 1) 代替硬编码的 16000
n_idx = size(input,1) * train_par/100;
% 划分训练集和测试集
input_train = input(n(1:n_idx), :)';
output_train = output(n(1:n_idx), :)';
input_test = input(n(n_idx+1:end), :)';  % 自动处理数据长度
output_test = output(n(n_idx+1:end), :)';

% 数据归一化
% [inputn, inputps] = mapminmax(input_train);
% [outputn, outputps] = mapminmax(output_train);
inputn =input_train;
outputn =output_train;
% 数据重整
p_train = input_train';
t_train = output_train';
p_test = input_test';
t_test = output_test';

%% 交叉验证

% 使用 crossvalind 生成交叉验证的折叠索引。
% 使用 4 折交叉验证来评估 GRNN 模型的性能。
% 对每一折，将数据划分为训练集和验证集。
% 对每个 spread 值（GRNN 的宽度参数），训练 GRNN 模型，并评估其在验证集上的性能。
% 选择 MSE（均方误差）最小的 spread 值作为最佳参数。

folds = 4;
indices = crossvalind('Kfold', length(p_train), folds);
mse_max = inf;  % 使用 inf 表述最大值，目标使其尽量低
desired_spread = [];
result_perfp = zeros(folds, ceil(20));  % 预分配结果矩阵
h = waitbar(0, '正在寻找最优化参数....');

for i = 1:folds
    test_idx = (indices == i);
    train_idx = ~test_idx;
    
    % 划分交叉验证数据
    p_cv_train = p_train(train_idx, :)';
    t_cv_train = t_train(train_idx, :)';
    p_cv_test = p_train(test_idx, :)';
    t_cv_test = t_train(test_idx, :)';
    
    % 数据归一化
    % [p_cv_train, minp, maxp, t_cv_train, mint, maxt] = premnmx(p_cv_train, t_cv_train);
    % p_cv_test = tramnmx(p_cv_test, minp, maxp);
    

    % GRNN训练与测试
    spreads = 0.1:0.1:2;
    perfp = zeros(size(spreads));
    
    for j = 1:length(spreads)
        spread = spreads(j);
        net_grnn = newgrnn(p_cv_train, t_cv_train, spread);
        test_Out = sim(net_grnn, p_cv_test);
        % test_Out = postmnmx(test_Out, mint, maxt);
        error = t_cv_test - test_Out;
        perfp(j) = mse(error);
        
        if perfp(j) < mse_max
            mse_max = perfp(j);
            desired_spread = spread;
            desired_input = p_cv_train;
            desired_output = t_cv_train;
        end
        
        waitbar((i - 1) * length(spreads) + j / (folds * length(spreads)), h);
        disp(['Fold ', num2str(i), ', Spread ', num2str(spread), ', MSE: ', num2str(perfp(j))]);
    end
    
    result_perfp(i, 1:length(spreads)) = perfp;
end

close(h);
disp(['最佳spread值为 ', num2str(desired_spread)]);

%% 采用最佳方法建立GRNN网络
% 使用最佳的 spread 值重新训练 GRNN 模型
net_grnn = newgrnn(desired_input, desired_output, desired_spread);
% p_test = tramnmx(p_test', minp, maxp);
p_test = p_test';
grnn_prediction_result = sim(net_grnn, p_test);
% grnn_prediction_result = postmnmx(grnn_prediction_result, mint, maxt);

% 评估模型
grnn_error = t_test' - grnn_prediction_result;
errorsum_grnn = sum(abs(grnn_error));
R_grnn = corrcoef(output_test, grnn_prediction_result);
R2_grnn = R_grnn(1, 2)^2;
MSE_grnn = immse(output_test, double(grnn_prediction_result));
RMSE_grnn = sqrt(MSE_grnn);

disp(['R² = ', num2str(R2_grnn)]);
disp(['MSE = ', num2str(MSE_grnn)]);
disp(['RMSE = ', num2str(RMSE_grnn)]);
end