%% 神经网络变量筛选—基于MIV的神经网络变量筛选
function BP_miv(datatable)
%% 清空环境变量
% clc;
% clear;
% close all;
warning off;
load(datatable);
%% 数据载入
input = data.in;   % 16000x3
output = data.out; % 16000x2

% 打乱数据
n = randperm(size(input, 1));
idx = 19000; % 数据集划分点

% 划分训练集和测试集
input_train = input(n(1:idx), :)'; % 转置为 [特征数 x 样本数]
output_train = output(n(1:idx), :)'; % 转置为 [特征数 x 样本数]
input_test = input(n(idx+1:end), :)'; % 转置为 [特征数 x 样本数]
output_test = output(n(idx+1:end), :)'; % 转置为 [特征数 x 样本数]

% 数据归一化
[inputn, inputps] = mapminmax(input_train);
[outputn, outputps] = mapminmax(output_train);

% 设置网络输入输出值
p = inputn; % 训练数据输入 [特征数 x 样本数]
t = outputn; % 训练数据输出 [特征数 x 样本数]

%% 变量筛选 MIV算法的初步实现（增加或者减少自变量）

p = p'; % 转置为 [样本数 x 特征数]
[m, n] = size(p);
p_increase = cell(1, n);
p_decrease = cell(1, n);

for i = 1:n
    p_temp = p;
    p_temp(:, i) = p(:, i) * 1.1;
    p_increase{i} = p_temp;

    p_temp = p;
    p_temp(:, i) = p(:, i) * 0.9;
    p_decrease{i} = p_temp;
end

%% 利用原始数据训练神经网络
nntwarn off;
net_miv = feedforwardnet([8, 2], 'traingdm'); % 输出层节点数应与输出特征数相同

% 设置训练参数
net_miv.trainParam.show = 50;
net_miv.trainParam.lr = 0.05;
net_miv.trainParam.mc = 0.9;
net_miv.trainParam.epochs = 2000;

% 网络训练
net_miv = train(net_miv, p', t); % 确保p和t在训练时维度一致

%% 变量筛选 MIV算法的后续实现（差值计算）
result_in = cell(1, n);
result_de = cell(1, n);

for i = 1:n
    result_in{i} = sim(net_miv, p_increase{i}');
    result_de{i} = sim(net_miv, p_decrease{i}');
end

% 计算MIV
MIV = zeros(1, n);
for i = 1:n
    diff = mean(result_in{i} - result_de{i}, 2); % 计算每个特征的差异均值
    MIV(i) = mean(diff); % 对所有样本的差异均值取平均
end

% 显示MIV值
disp('MIV Values:');
disp(MIV);

end