function [IW, B, LW, TF, TYPE] = elmtrain(P, T, N, TF, TYPE)
% ELMTRAIN Create and Train an Extreme Learning Machine (ELM)
% Syntax
% [IW, B, LW, TF, TYPE] = elmtrain(P, T, N, TF, TYPE)
% Description
% Input
% P   - Input Matrix of Training Set  (R*Q)
% T   - Output Matrix of Training Set (S*Q)
% N   - Number of Hidden Neurons (default = Q)
% TF  - Transfer Function:
%       'sig' for Sigmoidal function (default)
%       'sin' for Sine function
%       'hardlim' for Hardlim function
% TYPE - Regression (0, default) or Classification (1)
% Output
% IW  - Input Weight Matrix (N*R)
% B   - Bias Matrix  (N*1)
% LW  - Layer Weight Matrix (N*S)

% Error checking
if nargin < 2
    error('ELM:Arguments', 'Not enough input arguments.');
end
if nargin < 3
    N = size(P, 2); % Default number of neurons is the number of samples
end
if nargin < 4
    TF = 'sig'; % Default transfer function is sigmoid
end
if nargin < 5
    TYPE = 0; % Default type is regression
end

% Verify dimensions
if size(P, 2) ~= size(T, 2)
    error('ELM:Arguments', 'The columns of P and T must be the same.');
end

% Extract dimensions
[R, Q] = size(P);
[S, ~] = size(T);

% Convert T to one-hot encoding if classification
if TYPE == 1
    T = ind2vec(T);
end

% Randomly generate input weight matrix and bias matrix
IW = rand(N, R) * 2 - 1; % [-1, 1] range
B = rand(N, 1);

% Calculate hidden layer output matrix
BiasMatrix = repmat(B, 1, Q);
tempH = IW * P + BiasMatrix;

% Apply transfer function
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH)); % Sigmoidal function
    case 'sin'
        H = sin(tempH); % Sine function
    case 'hardlim'
        H = hardlim(tempH); % Hardlim function
    otherwise
        error('ELM:Arguments', 'Unknown transfer function type.');
end

% Calculate the output weight matrix using pseudo-inverse
LW = pinv(H') * T';

% Optionally return transfer function and type
if nargout > 3
    varargout{1} = TF;
end
if nargout > 4
    varargout{2} = TYPE;
end
end
