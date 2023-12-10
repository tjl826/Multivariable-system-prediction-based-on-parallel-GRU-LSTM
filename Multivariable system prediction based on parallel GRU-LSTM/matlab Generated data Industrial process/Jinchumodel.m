
function [Y,Y11,C_s11,C_CN11]=Jinchumodel(Q_CN11,Q_s,Cw1,d,C_s_INau)

% %输入
% Q_CN11=265.84;%浸出第1槽氰化钠加入量(Kg/h)
% Q_CN12=205.5;
% Q_CN14=92.35;
% Q_CN16=11.36;
 
%%
 
% Q_s=21485;   %矿石流量  ((Kg/h))
% Cw1=0.3;      %调浆后矿浆浓度 Cw1 0.3
% d=80;         %矿石粒径  (um) 
% C_s_INau=40;  %矿石金品位(g/t)

Kau=1;
Q_CN11=Q_CN11*1e6; %浸出第1槽氰化钠加入量(Kg/h)
C_CN_IN=50;  %一浸前液体中氰根浓度
Qo11=10000;   %Qo11第1槽空气流量(m3/h)
a=10.3;
b=1.204e-004;
C_01=a-exp(-b*Qo11);

[Y,Y11,C_s11,C_CN11] = leach(Q_s,Cw1,Kau*Q_CN11,d,C_CN_IN,C_s_INau,C_01);

end



% Y;            % 浸出率(%)
% Y11;          %  浸出率
% C_s11;        %  浸出后固金品位（g/t） 
% C_CN11;       %  矿浆的中氰根离子浓度（g/t)



