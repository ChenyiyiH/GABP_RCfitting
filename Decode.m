function ret = Decode(lenchrom, bound, code, opts)
% 本函数对染色体进行解码
% lenchrom   input : 染色体长度
% bound      input : 变量取值范围
% code       input : 编码值
% opts       input : 解码方法标签
% ret        output: 染色体的解码值

switch opts
    case 'binary' % binary coding
        % Initialize
        data = zeros(1, length(lenchrom));
        
        % Decode binary
        for i = length(lenchrom):-1:1
            data(i) = bitand(code, 2^lenchrom(i) - 1); % Extract lower bits
            code = bitshift(code, -lenchrom(i)); % Right shift
        end
        
        % Convert binary to decimal and map to range
        ret = bound(:, 1)' + data ./ (2.^lenchrom - 1) .* (bound(:, 2) - bound(:, 1))';
        
    case 'grey' % grey coding
        % Convert grey code to binary code
        binaryCode = greyToBinary(code, lenchrom);
        
        % Decode binary code
        data = zeros(1, length(lenchrom));
        for i = length(lenchrom):-1:1
            data(i) = bitand(binaryCode, 2^lenchrom(i) - 1); % Extract lower bits
            binaryCode = bitshift(binaryCode, -lenchrom(i)); % Right shift
        end
        
        % Convert binary to decimal and map to range
        ret = bound(:, 1)' + data ./ (2.^lenchrom - 1) .* (bound(:, 2) - bound(:, 1))';
        
    case 'float' % float coding
        % Float coding is directly the encoded value
        ret = code;
        
    otherwise
        error('Unsupported decoding method: %s', opts);
end

end

function binaryCode = greyToBinary(greyCode, lenchrom)
% Convert grey code to binary code
% greyCode    input : Grey code value
% lenchrom     input : Length of each chromosome
% binaryCode   output: Binary code value

% Initialize binary code
binaryCode = greyCode;

% Process each chromosome length
for i = length(lenchrom):-1:1
    mask = 2^lenchrom(i) - 1;
    % Extract bits for current chromosome
    greySegment = bitand(binaryCode, mask);
    % Convert grey to binary
    binarySegment = greySegment;
    while mask > 1
        mask = bitshift(mask, -1);
        binarySegment = bitxor(binarySegment, bitshift(binarySegment, 1));
    end
    binaryCode = bitset(binaryCode, lenchrom(i), binarySegment);
end

end
