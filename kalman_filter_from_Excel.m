function kalman_filter_from_Excel()
% 二维卡尔曼滤波示例（模块化版本）
% 说明：
% 1) 过程噪声 w ~ N(0, Q)
% 2) 观测噪声 v ~ N(0, R)
% 3) 将单步卡尔曼滤波封装为函数 kalman_filter_step
%
% 运行方式：
% >> kalman_filter_modular_gaussian

    clc; clear; close all;

    % ----------------------- 系统参数 -----------------------
    A = [1, 1;
         0, 1];                  % 状态转移矩阵

    H = [1, 0;
         0, 1];                  % 观测矩阵

    Q = [0.1, 0;
         0, 0.1];                % 过程噪声协方差

    R = [1, 0;
         0, 1];                  % 观测噪声协方差

    X0 = [0;
          1];                    % 初始状态 [position; speed]

    P0 = [1, 0;
          0, 1];                 % 初始估计协方差

    N = 30;                      % 仿真步数

    % 固定随机种子，便于复现实验结果
    rng(42);

    % ----------------------- 初始化 -----------------------
    X_true = X0;
    X_posterior = X0;
    P_posterior = P0;

    position_true = zeros(N,1);
    speed_true = zeros(N,1);

    position_measure = zeros(N,1);
    speed_measure = zeros(N,1);

    position_prior_est = zeros(N,1);
    speed_prior_est = zeros(N,1);

    position_posterior_est = zeros(N,1);
    speed_posterior_est = zeros(N,1);

    % 可选：记录卡尔曼增益
    K_history = zeros(2,2,N);

    % ----------------------- 主循环 -----------------------
    for k = 1:N
        % 1) 生成过程噪声 w ~ N(0, Q)
        w = generate_gaussian_noise(Q);

        % 2) 生成真实状态
        X_true = A * X_true + w;
        position_true(k) = X_true(1);
        speed_true(k) = X_true(2);

        % 3) 生成观测噪声 v ~ N(0, R)
        v = generate_gaussian_noise(R);

        % 4) 生成观测值
        Z_measure = H * X_true + v;
        position_measure(k) = Z_measure(1);
        speed_measure(k) = Z_measure(2);

        % 5) 单步卡尔曼滤波
        [X_prior, P_prior, K, X_posterior, P_posterior] = ...
            kalman_filter_step(A, H, Q, R, X_posterior, P_posterior, Z_measure);

        % 6) 保存结果
        position_prior_est(k) = X_prior(1);
        speed_prior_est(k) = X_prior(2);

        position_posterior_est(k) = X_posterior(1);
        speed_posterior_est(k) = X_posterior(2);

        K_history(:,:,k) = K;
    end

    k = 1:length(speed_true);

    plot_kalman_results(k, ...
    speed_true, speed_measure, speed_prior_est, speed_posterior_est, ...
    position_true, position_measure, position_prior_est, position_posterior_est);

    % ----------------------- 绘图 -----------------------
    function plot_kalman_results(k, ...
    speed_true, speed_measure, speed_prior_est, speed_posterior_est, ...
    position_true, position_measure, position_prior_est, position_posterior_est)

    % -------------------- 颜色配置 --------------------
    c_true      = [0.00, 0.45, 0.74];   % 蓝
    c_measure   = [0.85, 0.33, 0.10];   % 橙
    c_prior     = [0.93, 0.69, 0.13];   % 黄
    c_posterior = [0.49, 0.18, 0.56];   % 紫

    % -------------------- 画布 --------------------
    figure('Color', 'w', 'Position', [100, 100, 1280, 500]);
    t = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    % ==================== speed ====================
    nexttile;
    hold on;

    plot(k, speed_true, ...
        '-', 'Color', c_true, 'LineWidth', 2.0, ...
        'DisplayName', 'True');

    plot(k, speed_measure, ...
        '-o', 'Color', c_measure, 'LineWidth', 1.0, ...
        'MarkerSize', 4, 'MarkerFaceColor', c_measure, ...
        'DisplayName', 'Measurement');

    plot(k, speed_prior_est, ...
        '--', 'Color', c_prior, 'LineWidth', 1.6, ...
        'DisplayName', 'Prior');

    plot(k, speed_posterior_est, ...
        '-', 'Color', c_posterior, 'LineWidth', 2.2, ...
        'DisplayName', 'Posterior');

    title('Speed Estimation', 'FontWeight', 'bold', 'FontSize', 13);
    xlabel('Time Step k', 'FontSize', 11);
    ylabel('Speed', 'FontSize', 11);
    xlim([k(1), k(end)]);
    grid on;
    set(gca, ...
        'FontName', 'Times New Roman', ...
        'FontSize', 11, ...
        'LineWidth', 1, ...
        'GridAlpha', 0.18, ...
        'MinorGridAlpha', 0.08, ...
        'Box', 'off');
    legend('Location', 'northwest', 'Box', 'off');
    hold off;

    % ==================== position ====================
    nexttile;
    hold on;

    plot(k, position_true, ...
        '-', 'Color', c_true, 'LineWidth', 2.0, ...
        'DisplayName', 'True');

    plot(k, position_measure, ...
        '-o', 'Color', c_measure, 'LineWidth', 1.0, ...
        'MarkerSize', 4, 'MarkerFaceColor', c_measure, ...
        'DisplayName', 'Measurement');

    plot(k, position_prior_est, ...
        '--', 'Color', c_prior, 'LineWidth', 1.6, ...
        'DisplayName', 'Prior');

    plot(k, position_posterior_est, ...
        '-', 'Color', c_posterior, 'LineWidth', 2.2, ...
        'DisplayName', 'Posterior');

    title('Position Estimation', 'FontWeight', 'bold', 'FontSize', 13);
    xlabel('Time Step k', 'FontSize', 11);
    ylabel('Position', 'FontSize', 11);
    xlim([k(1), k(end)]);
    grid on;
    set(gca, ...
        'FontName', 'Times New Roman', ...
        'FontSize', 11, ...
        'LineWidth', 1, ...
        'GridAlpha', 0.18, ...
        'MinorGridAlpha', 0.08, ...
        'Box', 'off');
    legend('Location', 'northwest', 'Box', 'off');
    hold off;

    title(t, '2D Kalman Filter Result', ...
        'FontName', 'Times New Roman', ...
        'FontWeight', 'bold', ...
        'FontSize', 15);
end

    % ----------------------- 命令行输出 -----------------------
    disp('最后一步卡尔曼增益 K =');
    disp(K_history(:,:,end));
end


function [X_prior, P_prior, K, X_posterior, P_posterior] = kalman_filter_step(A, H, Q, R, X_posterior_prev, P_posterior_prev, Z_measure)
% 单步卡尔曼滤波函数
%
% 输入：
%   A, H, Q, R              系统矩阵
%   X_posterior_prev        上一时刻后验状态估计
%   P_posterior_prev        上一时刻后验协方差
%   Z_measure               当前时刻观测值
%
% 输出：
%   X_prior                 当前时刻先验估计
%   P_prior                 当前时刻先验协方差
%   K                       当前时刻卡尔曼增益
%   X_posterior             当前时刻后验估计
%   P_posterior             当前时刻后验协方差

    % 先验估计
    X_prior = A * X_posterior_prev;
    P_prior = A * P_posterior_prev * A' + Q;

    % 卡尔曼增益
    S = H * P_prior * H' + R;
    K = P_prior * H' / S;

    % 后验估计
    innovation = Z_measure - H * X_prior;
    X_posterior = X_prior + K * innovation;

    % 协方差更新
    I = eye(size(P_prior));
    P_posterior = (I - K * H) * P_prior;
end


function noise = generate_gaussian_noise(Cov)
% 按给定协方差矩阵生成零均值高斯噪声
% noise ~ N(0, Cov)
%
% 若 Cov 为对角阵，相当于每个维度独立高斯噪声；
% 若 Cov 含非对角元，则自动生成相关高斯噪声。

    n = size(Cov, 1);

    % 为了数值稳定性，做一次对称化
    Cov = (Cov + Cov.') / 2;

    % 使用 Cholesky 分解生成高斯噪声
    % 若 Cov 半正定但不是严格正定，可退化到 eig 方法
    [L, p] = chol(Cov, 'lower');
    if p == 0
        noise = L * randn(n, 1);
    else
        [V, D] = eig(Cov);
        D = diag(max(diag(D), 0));
        noise = V * sqrt(D) * randn(n, 1);
    end
end
