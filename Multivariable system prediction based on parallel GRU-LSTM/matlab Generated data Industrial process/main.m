clear
clc
tic 
close all
L=11000;
rand('state',1) 
V1=0.003*randn(L);
rand('state',2) 
V2=0.2*randn(L);
rand('state',3) 
V3=500*randn(L); 

std1=200;
std2=2000;
std3=0.05;
std4=10;
std5=5;
for k=1:L
    rand('state',k) 
    U1=400+std1*(0.15*sin(0.11*k)+0.2*cos(0.08*k)+0.4*sin(0.19*k));
    rand('state',k+1) 
    U2=20000+std2*(0.3*sin(0.12*k)+0.2*cos(0.17*k)+0.12*sin(0.15*k));
    rand('state',k+2) 
    U3=0.3+std3*(0.2*sin(0.1*k)+0.25*cos(0.13*k)+0.16*sin(0.15*k));
    rand('state',k+3) 
    U4=70+std4*(0.2*sin(0.12*k)+0.15*cos(0.09*k)+0.05*sin(0.15*k));
    rand('state',k+4) 
    U5=60+std5*(0.2*sin(0.07*k)+0.15*cos(0.08*k)+0.11*sin(0.12*k));
    Xx(k,:)=[U1,U2,U3,U4,U5];
    
    [Y,Y1,Y2,Y3]=Jinchumodel(U1,U2,U3,U4,U5);
    Y=Y+V1(k);
    Y2=Y2+V2(k);
    Y3=Y3+V3(k);
    
    Yout(k,:)=[Y,Y2,Y3];
    DataSJ(k,:)=[U1,U2,U3,U4,U5,Y,Y2,Y3];
end

save DataSJ.mat DataSJ

figure(1)
subplot(2,3,1)
plot (Xx(:,1),'k','linewidth',1);
hold on
legend ('Flow rate of sodium cyanide')
xlabel('k','FontName','Times New Roman','FontSize',12)
ylabel('U1','FontName','Times New Roman','FontSize',12)
grid on


subplot(2,3,2)
plot (Xx(:,2),'b','linewidth',1);
hold on
legend ('Flow rate of ore pulp')
xlabel('k','FontName','Times New Roman','FontSize',12)
ylabel('U2','FontName','Times New Roman','FontSize',12)
grid on

subplot(2,3,3)
plot (Xx(:,3),'r','linewidth',1);
hold on
legend ('Ore pulp concentration')
xlabel('k','FontName','Times New Roman','FontSize',12)
ylabel('U3','FontName','Times New Roman','FontSize',12)
grid on


subplot(2,3,4)
plot (Xx(:,4),'g','linewidth',1);
hold on
legend ('Average diameter of ore particles')
xlabel('k','FontName','Times New Roman','FontSize',12)
ylabel('U4','FontName','Times New Roman','FontSize',12)
grid on


subplot(2,3,5)
plot (Xx(:,5),'m','linewidth',1);
hold on
legend ('Concentration of solid gold')
xlabel('k','FontName','Times New Roman','FontSize',12)
ylabel('U5','FontName','Times New Roman','FontSize',12)
grid on




figure(2)
subplot(3,1,1)
plot (Yout(:,1),'b','linewidth',1);
hold on
legend ('Leaching rate')
xlabel('k','FontName','Times New Roman','FontSize',12)
ylabel('Y_1','FontName','Times New Roman','FontSize',12)
grid on

subplot(3,1,2)
plot (Yout(:,2),'r','linewidth',1);
hold on
legend ('Solid gold concentration (g/t)')
xlabel('k','FontName','Times New Roman','FontSize',12)
ylabel('Y_2','FontName','Times New Roman','FontSize',12)
grid on

subplot(3,1,3)
plot (Yout(:,3),'g','linewidth',1);
hold on
legend ('Liquid hydrogen ion concentration (g/t)')
xlabel('k','FontName','Times New Roman','FontSize',12)
ylabel('Y_3','FontName','Times New Roman','FontSize',12)
grid on

toc 