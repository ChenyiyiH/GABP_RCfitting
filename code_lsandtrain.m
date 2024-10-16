%% 含有电容杂散电感与电阻杂散电感的数据集创建
clc
clear
close all
%% 定义系统初值
Ud = 2500;                  % 直流电压
Lc = 4e-7;                  % 母排杂散电感
Lr = 3e-7;                  % 电阻杂散电感
Ld = 2.5e-7;                % 二极管杂散电感
Ligct = 200e-9;             % IGCT杂散电感
L = 3.5e-6;                 % 缓冲回路电抗
C = 34e-6;                  % 缓冲回路电容
R = 0.65;                   % 缓冲回路电阻
% R = 0.72;                 % 缓冲回路电阻替换
It = 1750*1.414*1.016;      % 输出电流（10M）

% 试验数据
% Ud = 2458.58;               % 直流电压
% It = 2336;                  % 输出电流（10M）

% 程序选择 （1/0）
is_runrealslx = 0;          % 运行slx模拟仿真
is_runode = 0;              % 运行ode仿真
is_Ls = 0;                  % 计算杂散电感影响
choose_lr = 0;              % 是否含有Lr，与is_Ls相关

is_data = 0;                % 构造数据
choose = 0;                 % 是否构造数据集，与is_data相关
num_C = 30;                 % 电容数据长度
num_R = 50;                 % 电阻数据长度

is_train = 0;               % 训练
chose_model = 16;           % 选择模型1-16 % SCNN不太行 PNN一般 wavenn不行
train_par = 70;             % 训练集比例

is_getfun = 0;              % 获取模型数据，训练后
is_getsim = 0;

is_varify = 0;              % 验证数据，训练后
is_slx = 0;                 % simulink模型验证

is_rcut = 0;                % 绘制RC与峰值电压时间
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1：双隐层BP1 √          2：双隐层BP2  √      3：遗传算法优化BP  √          %
% 4：粒子群算法优化BP √   5：单隐层BP1  √      6：单隐层BP2   √              %
% 7：极限学习机ELM        8：ELMAN神经网络     9：GRNN      √                %
% 10：LSTM                11：SCNN             12：RBF（可选RBE）            %
% 13：PNN                 14：SVM              15：小波神经网络（波函数可选）%
% 16: 单隐层BP（稳定版）     (10-15未曾优化，慎用)                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 数据文件
date = 101624;                                             % M D Y
% 数据文件
dataname = sprintf("data_%u.mat", date);
% dataname = sprintf("data_%uuH.mat", L*1e6);               
% dataname = sprintf("data1_%uome_0920.mat", R*100);

% 载入数据
% load(dataname,'in','out');                            % 仅在不构造数据集训练时使用

%% 计算杂散电感的影响（[Lsc,Lsr]）
if is_Ls
    if choose_lr
        for i = 1:10
            Lc = 1e-7;
            for j = 1:10
                % 定义微分方程
                dydt_3 = @(t, Y) [Y(2);
                    Y(3);
                    -(((L*R*C+Lc*C*R)/(L*Lr*C+L*Lc*C+Lc*Lr*C)) * Y(3) +...
                    ((L+Lr)/(L*Lr*C+L*Lc*C+Lc*Lr*C)) * Y(2) + ...
                    (R/(L*Lr*C+L*Lc*C+Lc*Lr*C)) * Y(1) - (R/(L*Lr*C+L*Lc*C+Lc*Lr*C)) * Ud)];
                % 定义初值
                Y0 = [Ud; It / C; 0];
                % 解决常微分方程
                [t_3, Y_3] = ode45(dydt_3, [0 1e-4], Y0);
                [m,idx]=max(Y_3(:,1));
                % plot(t_3,Y_3(:, 1));
                % hold on
                tt = t_3(idx);
                Um(i,j) = m;
                tm(i,j) = tt;
                Lc = Lc + 1e-7;
            end
            Lr = Lr + 1e-7;
        end

        %% 绘图
        % 光滑度查看
        lr = 1e-7 : 1e-7 : 1.0e-6;
        lc = 1e-7 : 1e-7 : 1.0e-6;
        [lc,lr] = meshgrid(lc,lr);
        figure('Color',[1 1 1]);
        surf(lc,lr,Um);
        title('UM');
        xlabel('lc');
        ylabel('lr');
        zlabel('Um');
        figure('Color',[1 1 1]);
        surf(lr,lc,tm);
        s=surf(lr,lc,tm);
        title('tM');
        xlabel('lc');
        ylabel('lr');
        zlabel('tm');
        % s.EdgeColor = "none"

    elseif choose_lr == 0

        Lc = 1e-7;
        for i = 1:10
            % 定义微分方程
            dydt_3 = @(t, Y) [Y(2);
                Y(3);
                -((L + Lc) * R / (L * Lc) * Y(3) + (1 / (Lc * C)) * Y(2) + (R / (Lc * L * C)) * Y(1) - R * Ud / (Lc * L * C))];
            % 定义初值
            Y0 = [Ud; It / C; 0];
            % 解决常微分方程
            [t_3, Y_3] = ode45(dydt_3, [0 1e-4], Y0);
            [m,idx]=max(Y_3(:,1));
            % plot(t_3,Y_3(:, 1));
            % hold on
            tt = t_3(idx);
            Um(i) = m;
            tm(i) = tt;
            Lc = Lc+1e-7;
        end
        lc = 1e-7:1e-7:1.0e-6;
        figure('Color',[1 1 1]);
        lc_p = plot3(lc,Um,tm);
        lc_p.LineWidth = 2;
        xlabel('lc');
        ylabel('um');
        zlabel('tm');
    end
end
%% 数据集构造
if(is_data)
    if(choose == 0)
        %% 单个It（看光滑度）

        C_initial = 10e-6;
        C_step = 2e-6;
        C_end = 49e-6;
        R_initial = 0.3;
        R_step = 0.01;
        R_end = 0.89;

        % 计算C和R的范围
        C_values = C_initial:C_step:C_end;
        R_values = R_initial:R_step:R_end;
        num_C = length(C_values);
        num_R = length(R_values);

        % 预分配结果矩阵
        Um = zeros(num_R, num_C);
        tm = zeros(num_R, num_C);

        % 主循环计算
        for j = 1:num_R
            R = R_values(j);
            for l = 1:num_C
                C = C_values(l);

                % 定义微分方程
                dydt_3 = @(t, Y) [Y(2);
                    Y(3);
                    -((L + Lc) * R / (L * Lc) * Y(3) ...
                    + (1 / (Lc * C)) * Y(2) ...
                    + (R / (Lc * L * C)) * Y(1)...
                    - R * Ud / (Lc * L * C))];

                % 定义初值
                Y0 = [Ud; It / C; 0];

                % 解决常微分方程
                [t_3, Y_3] = ode45(dydt_3, [0 1e-4], Y0);

                % 查找最大值及其对应时间
                [m, idx] = max(Y_3(:,1));
                tt = t_3(idx);

                % 存储结果
                Um(j, l) = m - Ud;
                tm(j, l) = tt;
            end
        end

        % 绘制结果
        [R_grid, C_grid] = meshgrid(R_values, C_values);

        figure('Color',[1 1 1]);
        surf(R_grid, C_grid, Um');
        title('UM');
        xlabel('R');
        ylabel('C');
        zlabel('Um');

        figure('Color',[1 1 1]);
        surf(R_grid, C_grid, tm');
        title('TM');
        xlabel('R');
        ylabel('C');
        zlabel('tm');

    elseif(choose==1)
        %% 构造数据集
        % 计算循环次数
        % 初始化参数
        It = 2500;                                  % 数据集电流，后续使用需对电流进行线性变换
        Lc = Lc*2;                                  % Lc上下一共两组
        C_initial = 20e-6;                          % 电容初值
        C_step = 1e-6;                              % 电容步长
        C_end = C_initial + (num_C - 1) * C_step;   % num_C步，最后一个步长
        R_initial = 0.25;                           % 电阻初值
        R_step = 0.01;                              % 电阻步长
        R_end = R_initial + (num_R - 1) * R_step;   % num_R步，最后一个步长
        Um_cor = -19.5;                             % 峰值电压修正系数
        Tm_cor1 = 8.5;                              % 峰值时间修正系数（斜率）
        Tm_cor2 = 8.5;                              % 峰值时间修正系数（偏差）         
        rr = R_initial : R_step : R_end;            % 数据集R特征向量
        cc = C_initial : C_step : C_end;            % 数据集C特征向量
        [C_grid,R_grid] = meshgrid(cc,rr);
        % 预分配结果矩阵
        n_ori = zeros(num_C * num_R, 3);
        Um = zeros(num_C * num_R, 1);
        tm = zeros(num_C * num_R, 1);
        

        % 初始化计数器
        cnt = 1;

        % 主循环计算
        for j = 1:num_R
            R = R_initial + (j - 1) * R_step; % 更新R
            C = C_initial; % 每次内层循环开始时重置C
            for l = 1:num_C
                % 存储参数
                n_ori(cnt, :) = [R, C, It];

                % 定义微分方程 含有Lr
                % dydt_3 = @(t, Y) [Y(2);
                %     Y(3);
                %     -(((L*R*C+Lc*C*R)/(L*Lr*C+L*Lc*C+Lc*Lr*C)) * Y(3) +...
                %     ((L+Lr)/(L*Lr*C+L*Lc*C+Lc*Lr*C)) * Y(2) + ...
                %     (R/(L*Lr*C+L*Lc*C+Lc*Lr*C)) * Y(1) - (R/(L*Lr*C+L*Lc*C+Lc*Lr*C)) * Ud)];

                % 定义微分方程 含有Lc
                % dydt_3 = @(t, Y) [Y(2);
                %     Y(3);
                %     -((L + Lc) * R / (L * Lc) * Y(3) ...
                %     + (1 / (Lc * C)) * Y(2) ...
                %     + (R / (Lc * L * C)) * Y(1)...
                %     - R * Ud / (Lc * L * C))];

                %定义微分方程 含有Ld
                dydt_3 = @(t, Y)...
                    [Y(2);
                    Y(3);
                    -( ( (C*R*(Lc+L+Ld)) / ( C*(L*Lc+Lc*Ld+Lc*Lr+Lr*L+Lr*Ld) ) ) *Y(3) ...
                    + ((L+Ld+Lr)/( C*(L*Lc+Lc*Ld+Lc*Lr+Lr*L+Lr*Ld) )) * Y(2) ...
                    +(R/( C*(L*Lc+Lc*Ld+Lc*Lr+Lr*L+Lr*Ld) )) *Y(1) ...
                    -(R/( C*(L*Lc+Lc*Ld+Lc*Lr+Lr*L+Lr*Ld) ))* Ud...
                    )];


                % 定义初值
                Y0 = [Ud; It / C; 0];

                % 解决常微分方程
                [t_3, Y_3] = ode45(dydt_3, [0 1e-4], Y0);

                %0914
                % Y_3(:, 1) = Y_3(:, 1) + (-21)*Ld*1e7;
                % t_3 = (t_3 + 8.5*(1e-8)*Ld*1e7 + 2e-7)*1e6;

                %0920
                Y_3 = Y_3 + Um_cor*Ld*1e7;
                t_3 = (t_3 + Tm_cor1*(1e-8)*Ld*1e7 - Tm_cor2*1e-7)*1e6;

                % 找到最大值及其对应时间
                [m, idx] = max(Y_3(:, 1));
                tt = t_3(idx);

                % 存储结果
                Um(cnt) = m - Ud;
                tm(cnt) = tt;

                % 更新C
                C = C + C_step;
                cnt = cnt + 1;
            end
        end

        % 保存数据
        data.out = n_ori(:, 1:2);
        data.out(:,2) = data.out(:,2)*1e6;
        data.in = [Um, tm];
        in = data.in;
        out = data.out;
        savename = dataname;
        save(savename, 'data', 'in', 'out','R_grid','C_grid');

    end
end

%% 训练模型
if is_train

    model_functions = {@BP_double, @BP_double2, @BP_GA, @BP_PSO, @BP_single, ...
        @BP_single2, @ELM, @ELMAN, @GRNN, @LSTM, @SCNN, ...
        @RBF, @PNN, @SVM, @wavenn,@BP_single1};
    if chose_model >= 1 && chose_model <= numel(model_functions)
        if chose_model == 7||chose_model ==10
            [~,~,~,net] = model_functions{chose_model}(dataname,train_par);

        else
            [~,~,~,~,net] = model_functions{chose_model}(dataname,train_par);
        end
        % disp(results);
    end
end

%% 生成预测文件
if is_getsim
    gensim(net);
end
if is_getfun
    if chose_model~=7
        save_name = "predict"+num2str(chose_model)+".m";
        genFunction(net,save_name,'MatrixOnly','yes');
    elseif chose_model == 8
        gensim(net);
        % mcc -W lib:libpredict -T link:lib predict
    end
end
%% 验证
if is_varify
    if chose_model ~= 7
        load(dataname);
        switch chose_model
            case 1
                out_i = predict1(in');
            case 2
                out_i = predict2(in');
            case 3
                out_i = predict3(in');
            case 4
                out_i = predict4(in');
            case 5
                out_i = predict5(in');
            case 6
                out_i = predict6(in');        
            case 9
                out_i = predict9(in');
            case 16
                out_i = predict16(in');
        end

        out_i = out_i';
        error_i = (out-out_i)./out_i;
        error_i = abs(error_i).*100;
        error_1 = reshape(error_i(:,1),[num_R num_C]);
        error_2 = reshape(error_i(:,2),[num_R num_C]);
        figure('Color',[1 1 1]);
        surf(C_grid, R_grid, error_1);
        colorbar; % 显示颜色条
        colormap(jet); % 设置颜色映射
        figure('Color',[1 1 1]);
        surf(C_grid, R_grid, error_2);
        colorbar;
    end
end

if is_slx
    load(dataname);
    idx_rand = randi([1, num_C*num_R], 1, 1);
    sim('test_var.slx');
end

if is_rcut

    load(dataname);
    um = reshape(data.in(:,1),[num_C num_R]);
    tm = reshape(data.in(:,2),[num_C num_R]);
    figure('Color',[1 1 1]);
    surf(R_grid, C_grid, um');
    colorbar; % 显示颜色条
    colormap(jet); % 设置颜色映射
    figure('Color',[1 1 1]);
    surf(R_grid, C_grid, tm');
    colorbar;

end

if is_runrealslx
    Ud = 2500;                  % 直流电压
    Lc = 4e-7;                  % 母排杂散电感
    Lr = 3e-7;                  % 电阻杂散电感
    Ld = 2.5e-7;                % 二极管杂散电感
    Ligct = 200e-9;             % IGCT杂散电感
    L = 3.5e-6;                 % 缓冲回路电抗
    C = 34e-6;                  % 缓冲回路电容
    R = 0.65;                   % 缓冲回路电阻
    % R = 0.72;                 % 缓冲回路电阻替换
    It = 2310;                  % 输出电流（10M）
    Lc = Lc*2;
    sim('test_topo.slx');
end
if is_runode
    Ud = 2500;                  % 直流电压
    Lc = 4e-7;                  % 母排杂散电感
    Lr = 3e-7;                  % 电阻杂散电感
    Ld = 2.5e-7;                % 二极管杂散电感
    Ligct = 200e-9;             % IGCT杂散电感
    L = 3.5e-6;                 % 缓冲回路电抗
    C = 34e-6;                  % 缓冲回路电容
    R = 0.65;                   % 缓冲回路电阻
    % R = 0.72;                 % 缓冲回路电阻替换
    It = 2310;                  % 输出电流（10M）
    Lc = Lc*2;
    Um_cor = -19.5;             % 峰值电压修正系数
    Tm_cor1 = 7.5;              % 峰值时间修正系数（斜率）
    Tm_cor2 = 9e-7;             % 峰值时间修正系数（偏差）

    % dydt_3 = @(t, Y) [Y(2);
    %     Y(3);
    %     -((L + Lc) * R / (L * Lc) * Y(3) ...
    %     + (1 / (Lc * C)) * Y(2) ...
    %     + (R / (Lc * L * C)) * Y(1)...
    %     - R * Ud / (Lc * L * C))];

    % dydt_3 = @(t, Y)...
    %     [Y(2);
    %     Y(3);
    %     -(((L*R*C+Lc*C*R)/(L*Lr*C+L*Lc*C+Lc*Lr*C)) * Y(3) +...
    %     ((L+Lr)/(L*Lr*C+L*Lc*C+Lc*Lr*C)) * Y(2) + ...
    %     (R/(L*Lr*C+L*Lc*C+Lc*Lr*C)) * Y(1) - (R/(L*Lr*C+L*Lc*C+Lc*Lr*C)) * Ud)];

    dydt_3 = @(t, Y)...
        [Y(2);
        Y(3);
        -( ( (C*R*(Lc+L+Ld)) / ( C*(L*Lc+Lc*Ld+Lc*Lr+Lr*L+Lr*Ld) ) ) *Y(3) ...
        + ((L+Ld+Lr)/( C*(L*Lc+Lc*Ld+Lc*Lr+Lr*L+Lr*Ld) )) * Y(2) ...
        +(R/( C*(L*Lc+Lc*Ld+Lc*Lr+Lr*L+Lr*Ld) )) *Y(1) ...
        -(R/( C*(L*Lc+Lc*Ld+Lc*Lr+Lr*L+Lr*Ld) ))* Ud...
        )];
    % 定义初值
    Y0 = [Ud; It / C; 0 ];

    % dydt_3 = @(t,y)[y(2);
    %     y(3);
    %     y(4);
    %     -( ( (C*R*(Lc+L+Ld)) / (C*(L*Lc+Lc*Ld+Lc*Lr+L*Lr+Lr*Ld)) )* y(4) +...
    %     ( (Lr+L+Ld) / (C*(L*Lc+Lc*Ld+Lc*Lr+L*Lr+Lr*Ld)) ) * y(3) + ...
    %     ( R / (C*(L*Lc+Lc*Ld+Lc*Lr+L*Lr+Lr*Ld)) ) * y(2) + ...
    %     ( R / (C*(L*Lc+Lc*Ld+Lc*Lr+L*Lr+Lr*Ld)) ) * y(1) - ...
    %     ( R / (C*(L*Lc+Lc*Ld+Lc*Lr+L*Lr+Lr*Ld)) ) * Ud)];
    %
    % % 定义初值
    % Y0 = [Ud; It / C; 0 ; 0];

    % 解决常微分方程

    [t_3, Y_3] = ode45(dydt_3, [0 1e-4], Y0);
    Y_3 = Y_3 + (-19.5)*Ld*1e7;
    t_3 = t_3 + 7.5*(1e-8)*Ld*1e7-9e-7;
    plot(t_3,Y_3(:,1)-Ud);
    [max_val,max_idx] = max(Y_3(:,1));
    t_m = t_3(max_idx)
    m_v = max_val-Ud
end