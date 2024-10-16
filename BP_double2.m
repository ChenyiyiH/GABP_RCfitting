% 清空环境变量
function [errorsum,R2_bpd2,MSE_bpd2,RMSE_bpd2,net_bpdou2]=BP_double2(datatable,train_par)
% clc;
% clear;
% close all;
nntwarn off;
load(datatable);

inputn = data.in';
outputn = data.out';

trainFcn = 'trainrp';
hiddenLayerSize = [15 10];
% BP网络训练
net_bpdou2 = feedforwardnet(hiddenLayerSize,trainFcn); % 双隐含层，分别有15和10个神经元
net_bpdou2.trainParam.epochs = 1000; % 训练次数
net_bpdou2.input.processFcns = {'removeconstantrows','mapminmax'};
net_bpdou2.output.processFcns = {'removeconstantrows','mapminmax'};
net_bpdou2.trainParam.lr = 0.1;
net_bpdou2.trainParam.goal = 1e-5; % 使用科学计数法更具语义
net_bpdou2.divideFcn = 'dividerand';  % Divide data randomly
net_bpdou2.divideMode = 'sample';  % Divide up every sample
net_bpdou2.divideParam.trainRatio = train_par/100;
net_bpdou2.divideParam.valRatio = 15/100;
net_bpdou2.divideParam.testRatio = 15/100;

net_bpdou2.performFcn = 'mse';  % 均方误差
net_bpdou2.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};

[net_bpdou2,tr]= train(net_bpdou2, inputn, outputn);

x = in';
t = out';
y = net_bpdou2(x);
e = gsubtract(t,y);
performance = perform(net_bpdou2,t,y)

% 重新计算验证和测试性能
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net_bpdou2,trainTargets,y)
valPerformance = perform(net_bpdou2,valTargets,y)
testPerformance = perform(net_bpdou2,testTargets,y)


%% 评估模型性能
% 决定系数 (R²)
error = t-y;
errorsum = sum(abs(t-y));
R2_bpd2 = corrcoef(t, y);
R2_bpd2 = R2_bpd2(1, 2)^2;

% 均方误差和均方根误差
MSE_bpd2 = immse(t, double(y));
RMSE_bpd2 = sqrt(MSE_bpd2);

% 打印结果
fprintf('R² = %.4f\n', R2_bpd2);
fprintf('MSE = %.4f\n', MSE_bpd2);
fprintf('RMSE = %.4f\n', RMSE_bpd2);

end