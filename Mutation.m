function ret = Mutation(pmutation, lenchrom, chrom, sizepop, num, maxgen, bound)
% 本函数完成变异操作
% pmutation    input  : 变异概率
% lenchrom     input  : 染色体长度
% chrom        input  : 染色体群
% sizepop      input  : 种群规模
% num          input  : 当前迭代次数
% maxgen       input  : 最大迭代次数
% bound        input  : 变量的上下界
% ret          output : 变异后的染色体

for i = 1:sizepop
    % 随机选择是否进行变异
    if rand > pmutation
        continue;
    end
    
    % 随机选择变异位置
    pos = randi(sum(lenchrom)); % 选择变异位置，改用randi函数，避免浮点数误差
    
    % 变异幅度调整，增加边界保护
    fg = (rand * (1 - num / maxgen))^2;
    mutationAmount = (bound(pos, 2) - bound(pos, 1)) * fg;
    
    % 执行变异
    if rand > 0.5
        chrom(i, pos) = min(chrom(i, pos) + mutationAmount, bound(pos, 2));
    else
        chrom(i, pos) = max(chrom(i, pos) - mutationAmount, bound(pos, 1));
    end
    
    % 检查并修正超出边界的值
    chrom(i, pos) = max(min(chrom(i, pos), bound(pos, 2)), bound(pos, 1));
end

ret = chrom;

end
