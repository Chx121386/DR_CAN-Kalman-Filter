clc; clear; close all;

%% =========================
%  Step 1. 建立离散系统模型
%  对应推导：
%  x_{k+1} = A x_k + B u_k
%  y_k     = C x_k
% ==========================

dt = 0.1;   % 采样时间

A = [1 dt;
     0 1];

B = [0.5*dt^2;
     dt];

C = [1 0];   % 输出只取位置

nx = size(A,1);   % 状态维度 = 2
nu = size(B,2);   % 输入维度 = 1
ny = size(C,1);   % 输出维度 = 1

%% =========================
%  Step 2. 设置 MPC 参数
% ==========================

Np = 10;   % 预测时域

% 输出跟踪权重 Q
Qy = 10 * eye(Np*ny);

% 控制输入权重 R
Ru = 0.1 * eye(Np*nu);

% 输入约束
u_min = -2;
u_max =  2;

% 仿真步数
Nsim = 50;

% 参考输出：希望位置跟踪到 1
r_value = 1;

%% =========================
%  Step 3. 构造预测矩阵 F, G
%  对应推导：
%  Y_k = F x_k + G U_k
%
%  其中
%  F = [CA; CA^2; ...; CA^Np]
%
%  G = [CB      0   ...    0
%       CAB    CB   ...    0
%       ...
%       CA^(Np-1)B ...    CB]
% ==========================

F = zeros(Np*ny, nx);
G = zeros(Np*ny, Np*nu);

for i = 1:Np
    % F 的第 i 块：C*A^i
    F((i-1)*ny+1:i*ny, :) = C * (A^i);
    
    for j = 1:i
        % G 的 (i,j) 块：C*A^(i-j)*B
        G((i-1)*ny+1:i*ny, (j-1)*nu+1:j*nu) = C * (A^(i-j)) * B;
    end
end

%% =========================
%  Step 4. 初始化仿真
% ==========================

x = [0; 0];   % 初始状态：位置0，速度0

x_hist = zeros(nx, Nsim+1);
y_hist = zeros(ny, Nsim);
u_hist = zeros(nu, Nsim);

x_hist(:,1) = x;

%% =========================
%  Step 5. 开始滚动优化
%  对应推导：
%  J = (Y_k - R_k)'Q(Y_k - R_k) + U_k'R U_k
%
%  代入 Y_k = F x_k + G U_k
%
%  得到标准二次规划：
%  min 1/2 U' H U + f' U
% ==========================

options = optimoptions('quadprog', 'Display', 'off');

for k = 1:Nsim
    
    % ---------- 5.1 构造参考轨迹 R_k ----------
    % 对应推导里的 R_k
    Rk = r_value * ones(Np*ny, 1);
    
    % ---------- 5.2 当前时刻的线性项 ----------
    % Y_k = F x_k + G U_k
    % J = (F x + G U - R)'Q(F x + G U - R) + U'R U
    %
    % 展开后，写成 quadprog 形式：
    % min (1/2) U'HU + f'U
    %
    % H = 2*(G'QG + R)
    % f = 2*G'Q(Fx - R)
    
    H = 2 * (G' * Qy * G + Ru);
    f = 2 * G' * Qy * (F * x - Rk);
    
    % ---------- 5.3 输入约束 ----------
    % u_min <= u_i <= u_max
    %
    % 由于 U_k = [u_k; u_{k+1}; ... ; u_{k+Np-1}]
    % 所以直接给上下界即可
    lb = u_min * ones(Np*nu, 1);
    ub = u_max * ones(Np*nu, 1);
    
    % ---------- 5.4 求解优化问题 ----------
    U_opt = quadprog(H, f, [], [], [], [], lb, ub, [], options);
    
    % 若求解失败，做保护
    if isempty(U_opt)
        warning('quadprog 求解失败，控制输入置零');
        u = 0;
    else
        % ---------- 5.5 只取第一步控制 ----------
        % 对应 MPC 滚动优化思想
        u = U_opt(1:nu);
    end
    
    % ---------- 5.6 系统更新 ----------
    y = C * x;
    x = A * x + B * u;
    
    % ---------- 5.7 记录 ----------
    y_hist(:,k) = y;
    u_hist(:,k) = u;
    x_hist(:,k+1) = x;
end

%% =========================
%  Step 6. 结果绘图
% ==========================

t = 0:Nsim-1;
tx = 0:Nsim;

figure;
plot(t, y_hist, 'LineWidth', 1.8); hold on;
plot(t, r_value*ones(size(t)), '--', 'LineWidth', 1.5);
grid on;
xlabel('k');
ylabel('Position y_k');
legend('输出位置', '参考位置');
title('MPC位置跟踪效果');

figure;
plot(tx, x_hist(2,:), 'LineWidth', 1.8);
grid on;
xlabel('k');
ylabel('Velocity');
title('速度变化');

figure;
stairs(t, u_hist, 'LineWidth', 1.8);
grid on;
xlabel('k');
ylabel('Control input u_k');
title('控制输入');