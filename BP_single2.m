% 清空环境变量（根据需要）
function [errorsum_bps2,R2_bps2,MSE_bps2,RMSE_bps2,net_bp]=BP_single2(datatable,train_par)
% clc;
% clear;
% close all;
nntwarn off;
load(datatable);
% 训练数据和预测数据提取及归一化
inputn = data.in';
outputn = data.out';


%% BP网络训练
% 创建BP网络，隐藏层有15个神经元
% 创建BP神经网络
% 初始化网络结构
trainFcn = 'trainrp';
hiddenLayerSizes = 10;% 隐含层神经元数量

net_bp = feedforwardnet(hiddenLayerSizes,trainFcn);
% 配置训练参数
net_bp.trainParam.epochs = 10000;    % 训练轮次
net_bp.trainParam.lr = 0.1;         % 学习率
net_bp.trainParam.goal = 1e-5;      % 目标误差
net_bp.input.processFcns = {'removeconstantrows','mapminmax'};
net_bp.output.processFcns = {'removeconstantrows','mapminmax'};
% net_bp.trainParam.mu = 0.5;
net_bp.trainFcn = trainFcn;
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
errorsum_bps2 = sum(abs(error));
R2_bps2 = corrcoef(t, y);
R2_bps2 = R2_bps2(1, 2)^2;

% 均方误差和均方根误差
MSE_bps2 = immse(t, double(y));
RMSE_bps2 = sqrt(MSE_bps2);

% 打印结果
fprintf('R² = %.4f\n', R2_bps2);
fprintf('MSE = %.4f\n', MSE_bps2);
fprintf('RMSE = %.4f\n', RMSE_bps2);
end