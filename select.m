function ret = select(individuals, sizepop)
% 本函数对每一代种群中的染色体进行选择，以进行后面的交叉和变异
% individuals input  : 种群信息
% sizepop     input  : 种群规模
% ret         output : 经过选择后的种群

% 计算适应度的倒数并归一化
fitness1 = 10 ./ individuals.fitness;
sumfitness = sum(fitness1);
sumf = cumsum(fitness1) / sumfitness;  % 计算累计概率分布函数

% 选择个体
index = zeros(1, sizepop);  % 预分配索引数组
for i = 1:sizepop
    pick = rand;  % 随机选择一个值
    % 找到落在概率区间中的个体
    idx = find(sumf >= pick, 1, 'first');
    index(i) = idx;
end

% 从个体中选择对应的染色体
individuals.chrom = individuals.chrom(index, :);
individuals.fitness = individuals.fitness(index);
ret = individuals;
