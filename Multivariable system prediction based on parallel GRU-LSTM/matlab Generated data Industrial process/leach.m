% 浸出系统算法主函数
function [X1, X11, C_s1, C_CN1] = leach(Q_s, Cw, Q_CN11, d, C_CN_IN, C_s_IN, C_O)
    % 调用single_leach函数
    [X11, C_s1, C_CN1] = single_leach(Q_s, Cw, Q_CN11, d, C_CN_IN, C_s_IN, C_O);
    
    % 计算剩余可溶性碳的比例
    X1 = (C_s_IN - C_s1) / C_s_IN;
end

% 单次LEACH算法的子函数
function [X, C_s, C_CN] = single_leach(Q_s, Cw, Q_CN, d, C_CN_IN, C_s_IN, C_O)
    % 常量和初始化
    rho_s = 2.8;
    rho_l = 1;
    V = 84.78;
    C_s_inf = 0.357 * (1 - 1.49 * exp(-1.76e-2 * d));
    Q_l = Q_s * (1 / Cw - 1);
    tau = V / (Q_s / rho_s + Q_l / rho_l);
    M_s = Q_s * tau;
    M_l = Q_l * tau;
    options = optimset('Display', 'off');

    % 使用fsolve和Balance_CN函数解决C_CN
    x0 = [C_CN_IN];
    [x] = fsolve(@Balance_CN, x0, options, Q_l, Q_CN, C_CN_IN, M_l, d);
    C_CN = x;
    clear x, x0;

    % 使用fsolve和Balance_Au函数解决C_s
    x0 = [C_s_IN];
    [x] = fsolve(@Balance_Au, x0, options, Q_s, C_s_IN, C_s_inf, C_CN, C_O, M_s, d);
    C_s = x;
    clear x, x0;

    % 计算剩余可溶性碳的比例
    X = (C_s_IN - C_s) / C_s_IN;
end

% 平衡碳和氮的函数
function F = Balance_CN(x, Q_l, Q_CN, C_CN_IN, M_l, d)
    % 碳和氮的平衡方程
    F = [Q_l * (C_CN_IN - x) + Q_CN - M_l * (1.69e-8 / (d^0.547 - 6.40) * x^9.18)];
end

% 平衡碳和金的函数
function F = Balance_Au(x, Q_s, C_s_IN, C_s_inf, C_CN, C_O, M_s, d)
    % 碳和金的平衡方程
    F = [Q_s * (C_s_IN - x) - M_s * ((1.13e-3 - 4.37e-11 * d^2.93) * (x - C_s_inf)^2.13 * C_CN^1.8 * C_O^0.228)];
end