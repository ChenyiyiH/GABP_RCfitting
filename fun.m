% function error = fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn)
% %该函数用来计算适应度值
% %x          input     个体
% %inputnum   input     输入层节点数
% %outputnum  input     隐含层节点数
% %net        input     网络
% %inputn     input     训练输入数据
% %outputn    input     训练输出数据
% 
% %error      output    个体适应度值
% 
% %提取
% %BP神经网络初始权值和阈值，x为个体
% w1=x(1:inputnum*hiddennum);
% B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
% w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
% B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);
% 
% 
% %网络进化参数
% net.trainParam.epochs=20;
% net.trainParam.lr=0.1;
% net.trainParam.goal=0.00001;
% net.trainParam.show=100;
% net.trainParam.showWindow=0;
% 
% %网络权值赋值
% net.iw{1,1}=reshape(w1,hiddennum,inputnum);
% net.lw{2,1}=reshape(w2,outputnum,hiddennum);
% net.b{1}=reshape(B1,hiddennum,1);
% net.b{2}=reshape(B2,outputnum,1);
% 
% %网络训练
% net=train(net,inputn,outputn);
% 
% an=sim(net,inputn);
% %预测误差和作为个体适应度值
% error=sum(abs(an(1,:)-outputn(1,:)));

%%
function error = fun(x, inputnum, hiddennum, outputnum, net, inputn, outputn)
    % 提取权重和偏置
    w1 = x(1:inputnum*hiddennum);
    B1 = x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
    w2 = x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
    B2 = x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:end);

    % 确保网络结构
    net.iw{1,1} = reshape(w1, hiddennum, inputnum);
    net.lw{2,1} = reshape(w2, outputnum, hiddennum);
    net.b{1} = reshape(B1, hiddennum, 1);
    net.b{2} = reshape(B2, outputnum, 1);  % Ensure B2 is a column vector with size [outputnum, 1]

    % 网络进化参数
    net.trainParam.epochs = 20;          % 训练的最大代数
    net.trainParam.lr = 0.1;             % 学习率
    net.trainParam.goal = 0.00001;       % 目标误差
    net.trainParam.show = 100;           % 显示训练进度的频率
    net.trainParam.showWindow = 0;       % 不显示训练窗口

    % 网络训练
    net = train(net, inputn, outputn);

    % 计算输出
    an = sim(net, inputn);

    % 计算误差
    % 计算均方误差作为适应度值
    error = sum((an - outputn).^2, 'all') / numel(outputn);
end

