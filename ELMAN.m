%% 基于Elman神经网络的预测模型研究
function [errorsum_elman,R2_elman,MSE_elman,RMSE_elman,net_elman]=ELMAN(datatable,train_par)
%% 清空环境变量
% clc;
% clear;
% close all;
nntwarn off;
load(datatable);

%% 数据载入
% 载入数据并将数据分成训练和测试集
inputn = data.in';
outputn = data.out';


%% 选取训练数据和测试数据
% 训练数据输入
p_train = inputn;
% 训练数据输出
t_train = outputn;
% 测试数据输入
p_test = inputn;
% 测试数据输出
t_test = outputn;


%% 网络的建立和训练
% 不同隐藏层神经元个数的设置
% nn = [7, 11, 14, 18];%
nn= 15;

% 误差记录初始化
error = zeros(length(nn), size(t_test, 2));
trainFcn = 'trainrp';
for i = 1:length(nn)
    % 建立Elman神经网络 隐藏层为nn(i)个神经元
    % 这里创建的网络结构：输入层 - 隐藏层 - 输出层
    net_elman = newelm(minmax(p_train), [nn(i) size(t_train, 1)], {'tansig', 'purelin'});

    % 设置网络训练参数
    net_elman.trainParam.epochs = 2000;
    net_elman.trainParam.show = 10;
    net_elman.trainParam.goal = 1e-5;
    net_elman.input.processFcns = {'removeconstantrows','mapminmax'};
    net_elman.output.processFcns = {'removeconstantrows','mapminmax'};
    net_elman.trainFcn = trainFcn;
    net_elman.divideFcn = 'dividerand';  % Divide data randomly
    net_elman.divideMode = 'sample';  % Divide up every sampl
    net_elman.divideParam.trainRatio = train_par/100;
    net_elman.divideParam.valRatio = 20/100;
    net_elman.divideParam.testRatio = 10/100;
    % 初始化网络
    net_elman = init(net_elman);
    net_elman.performFcn = 'mse';  % 均方误差
    net_elman.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
        'plotregression', 'plotfit'};
    % Elman网络训练
    [net_elman,tr] = train(net_elman, p_train, t_train);

    % % 预测数据
    % y_elman = sim(net_elman, p_test);
    %
    % y_sim_elman = y_elman;
    %
    % % 计算均方根误差
    % error(i, :) = sqrt(mean((y_sim_elman - output_test').^2, 1)); % 计算均方根误差
end

%% 网络预测
% 网络预测输出
x = in';
t = out';
y = net_elman(x);
e = gsubtract(t,y);
performance = perform(net_elman,t,y)

% 重新计算验证和测试性能
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net_elman,trainTargets,y)
valPerformance = perform(net_elman,valTargets,y)
testPerformance = perform(net_elman,testTargets,y)

%% 评估模型性能
% 决定系数 (R²)
error = t-y;
errorsum_elman = sum(abs(error));
R2_elman = corrcoef(t, y);
R2_elman = R2_elman(1, 2)^2;

% 均方误差和均方根误差
MSE_elman = immse(t, double(y));
RMSE_elman = sqrt(MSE_elman);

% 打印结果
fprintf('R² = %.4f\n', R2_elman);
fprintf('MSE = %.4f\n', MSE_elman);
fprintf('RMSE = %.4f\n', RMSE_elman);
end