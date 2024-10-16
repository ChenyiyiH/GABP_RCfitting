%% 清空环境变量（可根据需要选择是否使用）
function [R2_lstm,MSE_lstm,RMSE_lstm,net_lstm]=LSTM(datatable,train_par)
% clc;
% clear;
% close all;
nntwarn off;
%% 数据处理
% 划分数据集与测试集
load(datatable);
input = data.in;
output = data.out;
k = rand(1,size(input,1));
[~, n] = sort(k);
n_idx = size(input,1) * train_par/100;
% 训练数据和预测数据
input_train = input(n(1:n_idx), :)';
output_train = output(n(1:n_idx), :)';
input_test = input(n(n_idx+1:end),:)';
output_test = output(n(n_idx+1:end),:)';

% 样本输入输出数据归一化
[inputn, inputps] = mapminmax(input_train); % 训练输入归一化
[outputn, outputps] = mapminmax(output_train);% 训练输出归一化
inputn_test=mapminmax('apply',input_test,inputps); % 测试输入归一化
IN_train_1 =inputn;
OUT_train_1=outputn;
IN_test_1 =inputn_test;
OUT_test =output_test;

%% lstm
% 设置参数
inputSize = size(IN_train_1',2);
numHiddenUnits = 50;
numResponses = size(OUT_train_1',2);
layers = [ ...
                        sequenceInputLayer(inputSize)
                        lstmLayer(numHiddenUnits,...
                        'OutputMode','sequence',...
                        'StateActivationFunction','tanh',...
                        'GateActivationFunction','sigmoid')
                        fullyConnectedLayer(numResponses)
                        regressionLayer];
options = trainingOptions('adam', ...                        //训练网络的解决方法，自适应动量的随机优化方法,可选lbfgs，rmsprop，sgdm
                        'MaxEpochs',500, ...
                        'GradientThreshold',1e-5, ...
                        'InitialLearnRate',0.1, ...
                        'LearnRateSchedule','piecewise', ...
                        'LearnRateDropFactor',0.5, ...
                        'LearnRateDropPeriod',500, ...
                        'Plots', 'training-progress', ... 
                        'Verbose',1, ...
                        'CheckpointFrequency',10, ...
                        'MiniBatchSize',20);
% 创建网络
[net_lstm,info2] = trainNetwork(IN_train_1,OUT_train_1,layers,options);
 
% 仿真
Y_lstm= predict(net_lstm,IN_test_1) ;
 
% 测试结果反归一化
Y_sim_lstm=mapminmax('reverse',Y_lstm,outputps); 

%% 计算评价指标
%% 评估模型性能
% 决定系数 (R²)
R_lstm = corrcoef(output_test, Y_sim_lstm);
R2_lstm = R_lstm(1, 2)^2;

% 均方误差 (MSE) 和均方根误差 (RMSE)
MSE_lstm = immse(output_test, double(Y_sim_lstm));
RMSE_lstm = sqrt(MSE_lstm);

% 打印结果
fprintf('R² = %.4f\n', R2_lstm);
fprintf('MSE = %.4f\n', MSE_lstm);
fprintf('RMSE = %.4f\n', RMSE_lstm);
end