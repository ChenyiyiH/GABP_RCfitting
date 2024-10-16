function ret = Code(lenchrom, bound)
% 本函数将变量编码成染色体，用于随机初始化一个种群
% lenchrom   input : 染色体长度
% bound      input : 变量的取值范围
% ret        output: 染色体的编码值

% 确保 lenchrom 和 bound 的维度一致
if length(lenchrom) ~= size(bound, 1)
    error('lenchrom 和 bound 的维度不匹配');
end

% 初始化染色体
ret = zeros(1, sum(lenchrom));

% 对每一个变量进行编码
startIndex = 1;
for i = 1:length(lenchrom)
    % 随机生成该变量的值
    pick = rand(1, lenchrom(i));
    % 将随机生成的值映射到指定的范围
    ret(startIndex:startIndex + lenchrom(i) - 1) = bound(i, 1) + (bound(i, 2) - bound(i, 1)) .* pick;
    startIndex = startIndex + lenchrom(i);
end

% 检查染色体的合法性
if ~test(lenchrom, bound, ret)
    error('生成的染色体不合法');
end

end
