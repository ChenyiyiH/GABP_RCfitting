%% 基于双隐含层BP神经网络的预测代码
function [errorsum,R2_bpd1,MSE_bpd1,RMSE_bpd1,net_bpdou]=BP_double(datatable,train_par)
% 清空环境变量（可根据需要选择性使用）
% clc;
% clear;
% close all;

nntwarn off;
load(datatable);

%% 数据处理
% 下载输入输出数据

inputn = data.in';
outputn = data.out';


%% 创建和训练双隐含层BP神经网络
% 初始化网络结构
trainFcn = 'trainrp';
hiddenLayerSizes = [10 5];
net_bpdou = cascadeforwardnet(hiddenLayerSizes,trainFcn);

% 配置网络参数
net_bpdou.trainParam.epochs = 1000;
net_bpdou.input.processFcns = {'removeconstantrows','mapminmax'};
net_bpdou.output.processFcns = {'removeconstantrows','mapminmax'};
net_bpdou.trainParam.lr = 0.01;
net_bpdou.trainParam.goal = 1e-5; % 使用科学记数法简化
net_bpdou.divideFcn = 'dividerand';  % Divide data randomly
net_bpdou.divideMode = 'sample';  % Divide up every sample
net_bpdou.divideParam.trainRatio = train_par/100;
net_bpdou.divideParam.valRatio = 15/100;
net_bpdou.divideParam.testRatio = 15/100;

net_bpdou.performFcn = 'mse';  % 均方误差
net_bpdou.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};

% 网络训练
[net_bpdou,tr] = train(net_bpdou, inputn, outputn);

%% 网络预测
% 网络预测输出
x = in';
t = out';
y = net_bpdou(x);
e = gsubtract(t,y);
performance = perform(net_bpdou,t,y)

% 重新计算验证和测试性能
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net_bpdou,trainTargets,y)
valPerformance = perform(net_bpdou,valTargets,y)
testPerformance = perform(net_bpdou,testTargets,y)

%% 评估模型性能
% 决定系数 (R²)
error = t-y;
errorsum = sum(abs(t-y));
R2_bpd1 = corrcoef(t, y);
R2_bpd1 = R2_bpd1(1, 2)^2;

% 均方误差和均方根误差
MSE_bpd1 = immse(t, double(y));
RMSE_bpd1 = sqrt(MSE_bpd1);

% 打印结果
fprintf('R² = %.4f\n', R2_bpd1);
fprintf('MSE = %.4f\n', MSE_bpd1);
fprintf('RMSE = %.4f\n', RMSE_bpd1);

end