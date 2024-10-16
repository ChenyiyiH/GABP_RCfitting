function ret = Cross(pcross, lenchrom, chrom, sizepop, bound)
% 本函数完成交叉操作
% pcross       input  : 交叉概率
% lenchrom     input  : 染色体的长度
% chrom        input  : 染色体群
% sizepop      input  : 种群规模
% bound        input  : 数据范围
% ret          output : 交叉后的染色体

% 确保交叉概率在合理范围内
if pcross < 0 || pcross > 1
    error('交叉概率 pcross 必须在 0 和 1 之间');
end

for i = 1:2:sizepop
    % 确保每次选择两个不同的个体进行交叉
    parent1 = i;
    parent2 = min(i+1, sizepop);  % 避免超出数组边界
    
    % 交叉概率决定是否进行交叉
    if rand > pcross
        continue;
    end
    
    % 随机选择交叉位置
    pos = randi(sum(lenchrom));
    
    % 交换两个染色体在交叉位置之前的部分
    temp1 = chrom(parent1, :);
    temp2 = chrom(parent2, :);
    
    chrom(parent1, 1:pos) = temp2(1:pos);
    chrom(parent2, 1:pos) = temp1(1:pos);
    
    % 检查交叉后的染色体是否在边界范围内
    if ~test(lenchrom, bound, chrom(parent1, :)) || ~test(lenchrom, bound, chrom(parent2, :))
        % 如果染色体不合法，恢复原状
        chrom(parent1, :) = temp1;
        chrom(parent2, :) = temp2;
    end
end

ret = chrom;
