% ����ϵͳ�㷨������
function [X1, X11, C_s1, C_CN1] = leach(Q_s, Cw, Q_CN11, d, C_CN_IN, C_s_IN, C_O)
    % ����single_leach����
    [X11, C_s1, C_CN1] = single_leach(Q_s, Cw, Q_CN11, d, C_CN_IN, C_s_IN, C_O);
    
    % ����ʣ�������̼�ı���
    X1 = (C_s_IN - C_s1) / C_s_IN;
end

% ����LEACH�㷨���Ӻ���
function [X, C_s, C_CN] = single_leach(Q_s, Cw, Q_CN, d, C_CN_IN, C_s_IN, C_O)
    % �����ͳ�ʼ��
    rho_s = 2.8;
    rho_l = 1;
    V = 84.78;
    C_s_inf = 0.357 * (1 - 1.49 * exp(-1.76e-2 * d));
    Q_l = Q_s * (1 / Cw - 1);
    tau = V / (Q_s / rho_s + Q_l / rho_l);
    M_s = Q_s * tau;
    M_l = Q_l * tau;
    options = optimset('Display', 'off');

    % ʹ��fsolve��Balance_CN�������C_CN
    x0 = [C_CN_IN];
    [x] = fsolve(@Balance_CN, x0, options, Q_l, Q_CN, C_CN_IN, M_l, d);
    C_CN = x;
    clear x, x0;

    % ʹ��fsolve��Balance_Au�������C_s
    x0 = [C_s_IN];
    [x] = fsolve(@Balance_Au, x0, options, Q_s, C_s_IN, C_s_inf, C_CN, C_O, M_s, d);
    C_s = x;
    clear x, x0;

    % ����ʣ�������̼�ı���
    X = (C_s_IN - C_s) / C_s_IN;
end

% ƽ��̼�͵��ĺ���
function F = Balance_CN(x, Q_l, Q_CN, C_CN_IN, M_l, d)
    % ̼�͵���ƽ�ⷽ��
    F = [Q_l * (C_CN_IN - x) + Q_CN - M_l * (1.69e-8 / (d^0.547 - 6.40) * x^9.18)];
end

% ƽ��̼�ͽ�ĺ���
function F = Balance_Au(x, Q_s, C_s_IN, C_s_inf, C_CN, C_O, M_s, d)
    % ̼�ͽ��ƽ�ⷽ��
    F = [Q_s * (C_s_IN - x) - M_s * ((1.13e-3 - 4.37e-11 * d^2.93) * (x - C_s_inf)^2.13 * C_CN^1.8 * C_O^0.228)];
end