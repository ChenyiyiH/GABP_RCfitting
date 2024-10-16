%% 标准BP
% 基于BP神经网络的预测算法
function [errorsum_bps1,R2_bps1,MSE_bps1,RMSE_bps1,net_bp]=BP_single(datatable,train_par)
% 清空环境变量（根据需要）
% clc;
% clear;
% close all;
nntwarn off;
load(datatable);

%% 数据提取及归一化
% 加载输入和输出数据
inputn = data.in';
outputn = data.out';

%% BP网络训练
% 创建BP神经网络
% 初始化网络结构
trainFcn = 'trainrp';
hiddenLayerSizes = 4;% 隐含层神经元数量

% net_bp = feedforwardnet(hiddenLayerSize,'trainrp');
% net_bp = fitnet(hiddenLayerSize,'trainbr');
net_bp = newff(inputn, outputn, hiddenLayerSizes);
% 配置训练参数
net_bp.trainParam.epochs = 10000;    % 训练轮次
net_bp.trainParam.lr = 0.1;         % 学习率
net_bp.trainParam.goal = 1e-5;      % 目标误差
% net_bp.trainParam.mu = 0.5;
net_bp.input.processFcns = {'removeconstantrows','mapminmax'};
net_bp.output.processFcns = {'removeconstantrows','mapminmax'};
net_bp.trainFcn = trainFcn;
net_bp.layers{1}.transferFCn = 'logsig';
% net_bp.layers{2}.transferFCn = 'purelin';
net_bp.divideFcn = 'dividerand';  % Divide data randomly
net_bp.divideMode = 'sample';  % Divide up every sampl
net_bp.divideParam.trainRatio = train_par/100;
net_bp.divideParam.valRatio = 20/100;
net_bp.divideParam.testRatio = 10/100;

net_bp.performFcn = 'mse';  % 均方误差
net_bp.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};
% 训练网络
[net_bp,tr]= train(net_bp, inputn, outputn,'useGPU','no');

%% 网络预测
% 网络预测输出
x = in';
t = out';
y = net_bp(x);
e = gsubtract(t,y);
performance = perform(net_bp,t,y)

% 重新计算验证和测试性能
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net_bp,trainTargets,y)
valPerformance = perform(net_bp,valTargets,y)
testPerformance = perform(net_bp,testTargets,y)

%% 评估模型性能
% 决定系数 (R²)
error = t-y;
errorsum_bps1 = sum(abs(error));
R2_bps1 = corrcoef(t, y);
R2_bps1 = R2_bps1(1, 2)^2;

% 均方误差和均方根误差
MSE_bps1 = immse(t, double(y));
RMSE_bps1 = sqrt(MSE_bps1);

% 打印结果
fprintf('R² = %.4f\n', R2_bps1);
fprintf('MSE = %.4f\n', MSE_bps1);
fprintf('RMSE = %.4f\n', RMSE_bps1);
end