%% 基于遗传算法的神经网络预测代码
function [error_ga,R2_bpga,MSE_bpga,RMSE_bpga,net_gabp]=BP_GA(datatable,train_par)
% 清空环境变量（根据需要）
% clc;
% clear;
% close all;
nntwarn off;
load(datatable);
%% 网络结构建立
% 加载输入和输出数据

inputn = data.in';
outputn = data.out';

% 设置神经网络结构
inputnum = size(inputn, 1);
hiddennum = 15;
outputnum = size(inputn, 1);

% 创建BP网络
trainFcn = 'trainrp';
net_gabp = feedforwardnet(hiddennum,trainFcn);

% 配置网络
net_gabp = configure(net_gabp, inputn, outputn);
net_gabp.trainParam.epochs = 1000;
net_gabp.trainParam.lr = 0.01;


%% 遗传算法参数初始化
maxgen = 50;       % 进化代数（迭代次数）
sizepop = 10;     % 种群规模
pcross = 0.2;     % 交叉概率
pmutation = 0.1;  % 变异概率

% 节点总数和染色体长度
numsum = inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum;
lenchrom = ones(1, numsum);
bound = [-3 * ones(numsum, 1), 3 * ones(numsum, 1)];  % 数据范围

% 初始化种群
individuals = struct('fitness', zeros(1, sizepop), 'chrom', []);
avgfitness = [];
bestfitness = [];
bestchrom = [];

% 随机初始化种群
for i = 1:sizepop
    individuals.chrom(i, :) = Code(lenchrom, bound);
    x = individuals.chrom(i, :);
    individuals.fitness(i) = fun(x, inputnum, hiddennum, outputnum, net_gabp, inputn, outputn);
end

% 找到初始最优染色体
[bestfitness, bestindex] = min(individuals.fitness);
bestchrom = individuals.chrom(bestindex, :);
avgfitness = mean(individuals.fitness);

% 记录每一代的最优适应度和平均适应度
trace = [avgfitness, bestfitness];

%% 迭代进化
for i = 1:maxgen
    disp(['Iteration: ', num2str(i)]);
    
    % 选择
    individuals = select(individuals, sizepop);
    
    % 计算适应度
    for j = 1:sizepop
        x = individuals.chrom(j, :);
        individuals.fitness(j) = fun(x, inputnum, hiddennum, outputnum, net_gabp, inputn, outputn);
    end
    
    % 找到最优染色体
    [newbestfitness, newbestindex] = min(individuals.fitness);
    [worstfitness, worstindex] = max(individuals.fitness);
    
    % 更新最优染色体
    if bestfitness > newbestfitness
        bestfitness = newbestfitness;
        bestchrom = individuals.chrom(newbestindex, :);
    end
    individuals.chrom(worstindex, :) = bestchrom;
    individuals.fitness(worstindex) = bestfitness;
    
    % 交叉
    individuals.chrom = Cross(pcross, lenchrom, individuals.chrom, sizepop, bound);
    
    % 变异
    individuals.chrom = Mutation(pmutation, lenchrom, individuals.chrom, sizepop, i, maxgen, bound);
    
    % 计算适应度
    for j = 1:sizepop
        x = individuals.chrom(j, :);
        individuals.fitness(j) = fun(x, inputnum, hiddennum, outputnum, net_gabp, inputn, outputn);
    end
    
    % 更新最优染色体
    [newbestfitness, newbestindex] = min(individuals.fitness);
    [worstfitness, worstindex] = max(individuals.fitness);
    
    if bestfitness > newbestfitness
        bestfitness = newbestfitness;
        bestchrom = individuals.chrom(newbestindex, :);
    end
    individuals.chrom(worstindex, :) = bestchrom;
    individuals.fitness(worstindex) = bestfitness;
    
    avgfitness = mean(individuals.fitness);
    
    % 记录适应度
    trace = [trace; avgfitness, bestfitness];
end

%% 结果分析
% 绘制适应度曲线
figure('Color',[1 1 1]);
plot(trace(:, 2), 'b--');
title(['适应度曲线  终止代数 = ', num2str(maxgen)]);
xlabel('进化代数');
ylabel('适应度');
legend('最佳适应度');

% 打印适应度信息
disp('适应度    变量');

%% 应用最优染色体到网络
x = bestchrom;
w1 = x(1:inputnum * hiddennum);
B1 = x(inputnum * hiddennum + 1:inputnum * hiddennum + hiddennum);
w2 = x(inputnum * hiddennum + hiddennum + 1:inputnum * hiddennum + hiddennum + hiddennum * outputnum);
B2 = x(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1:end);

% 设置网络权重和偏置
net_gabp.iw{1, 1} = reshape(w1, hiddennum, inputnum);
net_gabp.lw{2, 1} = reshape(w2, outputnum, hiddennum);
net_gabp.b{1} = reshape(B1, hiddennum, 1);
net_gabp.b{2} = reshape(B2, outputnum, 1);

%% BP网络训练
% 配置训练参数
net_gabp.trainParam.epochs = 1000;
net_gabp.trainParam.lr = 0.1;
net_gabp.input.processFcns = {'removeconstantrows','mapminmax'};
net_gabp.output.processFcns = {'removeconstantrows','mapminmax'};
net_gabp.trainParam.goal = 1e-5; % 使用科学记数法简化
net_gabp.divideFcn = 'dividerand';  % Divide data randomly
net_gabp.divideMode = 'sample';  % Divide up every sample
net_gabp.divideParam.trainRatio = train_par/100;
net_gabp.divideParam.valRatio = 15/100;
net_gabp.divideParam.testRatio = 15/100;

net_gabp.performFcn = 'mse';  % 均方误差
net_gabp.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};
% 训练网络
[net_gabp, tr] = train(net_gabp, inputn, outputn);  % 训练网络

%% BP网络预测
% 网络预测输出
x = in';
t = out';
y = net_gabp(x);
e = gsubtract(t,y);
performance = perform(net_gabp,t,y)

% 重新计算验证和测试性能
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net_gabp,trainTargets,y)
valPerformance = perform(net_gabp,valTargets,y)
testPerformance = perform(net_gabp,testTargets,y)

%% 评估模型性能
% 决定系数 (R²)
error_ga = t-y;
errorsum = sum(abs(t-y));
R2_bpga = corrcoef(t, y);
R2_bpga = R2_bpga(1, 2)^2;

% 均方误差和均方根误差
MSE_bpga = immse(t, double(y));
RMSE_bpga = sqrt(MSE_bpga);

% 打印结果
fprintf('R² = %.4f\n', R2_bpga);
fprintf('MSE = %.4f\n', MSE_bpga);
fprintf('RMSE = %.4f\n', RMSE_bpga);
end