function Y = elmpredict(P, IW, B, LW, TF, TYPE)
% ELMPREDICT Simulate an Extreme Learning Machine (ELM)
% Syntax
% Y = elmpredict(P, IW, B, LW, TF, TYPE)
% Description
% Input
% P   - Input Matrix of Training Set  (R*Q)
% IW  - Input Weight Matrix (N*R)
% B   - Bias Matrix  (N*1)
% LW  - Layer Weight Matrix (N*S)
% TF  - Transfer Function:
%       'sig' for Sigmoidal function (default)
%       'sin' for Sine function
%       'hardlim' for Hardlim function
% TYPE - Regression (0, default) or Classification (1)
% Output
% Y   - Simulated Output Matrix (S*Q)

% Error checking
if nargin < 6
    error('ELM:Arguments', 'Not enough input arguments.');
end

% Calculate the hidden layer output matrix H
Q = size(P, 2); % Number of samples
BiasMatrix = repmat(B, 1, Q);
tempH = IW * P + BiasMatrix;

% Apply transfer function
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH)); % Sigmoid activation
    case 'sin'
        H = sin(tempH); % Sine activation
    case 'hardlim'
        H = hardlim(tempH); % Hardlim activation
    otherwise
        error('ELM:Arguments', 'Unknown transfer function type.');
end

% Calculate the simulated output
Y = (H' * LW)'; % Matrix multiplication to get the output

% Post-processing for classification
if TYPE == 1
    % Convert to class labels
    [~, classIndex] = max(Y, [], 1);
    Y = classIndex; % Vector of class indices
end
end
