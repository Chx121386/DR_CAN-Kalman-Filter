%% 单摆非线性系统 + EKF 扩展卡尔曼滤波
clear; clc; close all;

%% 1. 系统参数
g = 9.81;       % 重力加速度
L = 1.0;        % 摆长
dt = 0.01;      % 采样时间
N = 500;        % 总步数

% 噪声协方差 —— 常数矩阵
Q = diag([1e-3, 1e-2]);   % 过程噪声协方差
R = 1e-2;                 % 观测噪声协方差

%% 2. 生成仿真真实状态 + 观测
theta = zeros(1, N);
theta_dot = zeros(1, N);
theta(1) = 1.2;  % 初始大角度（强非线性）

for k = 2:N
    theta(k) = theta(k-1) + theta_dot(k-1)*dt;
    theta_dot(k) = theta_dot(k-1) - (g/L)*sin(theta(k-1))*dt;
end

% 非线性观测：z = L*sin(theta) + v
z = L*sin(theta) + sqrt(R)*randn(1, N);

%% 3. EKF 初始化
x_hat = zeros(2, N);    % 状态估计 [角度; 角速度]
x_hat(:,1) = [1.0; 0.0]; % 初始估计

P = eye(2);             % 初始协方差

%% 4. EKF 主循环
for k = 2:N
    % ===================== 预测步 Predict =====================
    x_prev = x_hat(:, k-1);
    th_prev = x_prev(1);
    dth_prev = x_prev(2);

    % 非线性状态预测
    x_pred = [
        th_prev + dth_prev * dt;
        dth_prev - (g/L)*sin(th_prev)*dt;
    ];

    % 状态雅可比 F —— 在 x_prev 处线性化
    F = [
        1, dt;
        -(g/L)*cos(th_prev)*dt, 1
    ];

    % 协方差预测
    P_pred = F * P * F' + Q;

    % ===================== 更新步 Update =====================
    th_pred = x_pred(1);

    % 非线性观测预测
    z_pred = L * sin(th_pred);

    % 观测雅可比 H —— 在 x_pred 处线性化
    H = [ L*cos(th_pred), 0 ];

    % 卡尔曼增益
    S = H * P_pred * H' + R;
    K = P_pred * H' / S;

    % 状态更新
    residual = z(k) - z_pred;
    x_hat(:,k) = x_pred + K * residual;

    % 协方差更新
    P = (eye(2) - K*H) * P_pred;
end

%% 5. 绘图
figure('Position',[100,100,800,600])

subplot(2,1,1)
plot(theta, 'k', 'LineWidth',1.5); hold on;
plot(x_hat(1,:), 'r--', 'LineWidth',1.5);
title('单摆角度 \theta');
legend('真实值','EKF估计'); grid on;

subplot(2,1,2)
plot(z, 'g.','MarkerSize',5); hold on;
plot(L*sin(x_hat(1,:)), 'r--','LineWidth',1.5);
title('观测值 vs EKF观测预测');
legend('观测 z','EKF预测'); grid on;
