clear all; close all; clc

ka = 0.00075;
kd = 0.00075;
kiN  = 0.006;
keN = 0.0002;
Vm = 7.0;
Km  = 350.0;
ktrlt = 1.0;
Vh  = 0.012;
kh = 700.0;
kv  = 8.5;
h  = 2.0;
ket = 0.00058;
xT = 500.0;

Vol = 1.0*10^(-18); NA = 6.02*10^(23); Cal = NA*Vol*10^(-6);
ka = ka/Cal;
xT = xT*Cal;
Vm = Vm*Cal;
Vh = Vh*Cal;
Km = Km*Cal;
kh = kh*Cal;
Km = Km/16.0;
Km0 = Km; Vh0 = Vh;
%%%% Parameters
VY = importdata('ysav.txt'); %%% IC
ome = 18.5;
R1 = 10; R2 = 80; delR = 7; nnl = length(R1:delR:R2); nj1 = 1;
LyaQ = zeros(nnl,nj1); LyaQ2 = zeros(nnl,nj1); LyaQ3 = zeros(nnl,nj1); LyaQ5 = zeros(nnl,nj1);
tic
plotyes = 1;


LL = 10000;
lenLL = length(2:100:LL);
ntest = 30;
tspan = [0 30000];
x = linspace(1/60.0,LL/60.,lenLL);
XX1 = zeros(lenLL,ntest);
XX2 = zeros(lenLL,ntest);
XX3 = zeros(lenLL,ntest);
XX4 = zeros(lenLL,ntest);
for j1 = 1:nj1
    Lya = [];
    for jn = R1:delR:R2
        jn
        if (j1 == 1)
            ETHmin = 0;
            ETHmax = 0;
        else
            ETHmin = -0.5*(j1-1);
            ETHmax = 0.5*(j1-1);
        end
        TNFmin = -0.1*jn;
        TNFmax = 0.1*jn;
        
        
        
        
        if (plotyes==1)
            figure
        end
        for test = 1:ntest
            %  figure
            z = [];
            y = zeros(8,1);
            y0 = [VY(1,:) VY(1,:)];
            y0(5) = y0(1)+0.1*randn;
            y0(6) = y0(2)+0.1*randn;
            y0(7) = y0(3)+0.1*randn;
            y0(8) = y0(4)+0.1*randn;
            %  y0 = [0 xT/2.0 100.0 10.0 0 xT/2.0 100.0 10.01];
            
            
            [t,y] = ode45(@(t,y) odefcn(t,y,ka,kd,kiN,keN,Vm,Km,ktrlt,Vh,kh,kv,h,ket,xT,Km0,Vh0,TNFmax,TNFmin,ETHmax,ETHmin,ome), tspan, y0);
            %             subplot(2,1,1)
            if (test < 4 && plotyes == 1)
                subplot(2,2,test)
                plot(t/60.0,y(:,3)); hold on
                plot(t/60.0,y(:,7)); hold on
            end
            %             figure(10)
            %             semilogy(t,abs(y(:,1)-y(:,5))); hold on;
            %  XX1(2:end,test) = 1./t(2:LL).*abs(y(2:LL,1)-y(2:LL,5))/(abs(y(1,1)-y(1,5)));
            XX1(:,test) = abs(y(2:100:LL,1)-y(2:100:LL,5))/mean(y(:,1));
            XX2(:,test) = abs(y(2:100:LL,2)-y(2:100:LL,6))/mean(y(:,2));
            XX3(:,test) = abs(y(2:100:LL,3)-y(2:100:LL,7))/mean(y(:,3));
            XX4(:,test) = abs(y(2:100:LL,4)-y(2:100:LL,8))/mean(y(:,4));
        end
        
        %
        
        %         for i = 1:100
        %             plot(x,log(XX1(:,i))); hold on;
        %             plot(x,log(XX2(:,i))); hold on;
        %             plot(x,log(XX3(:,i))); hold on;
        %             plot(x,log(XX4(:,i))); hold on;
        %         end
        if (plotyes == 1)
            subplot(2,2,4)
            plot(x,log(mean(XX1'))); hold on
            plot(x,log(mean(XX2'))); goodplot
            plot(x,log(mean(XX3'))); goodplot
            plot(x,log(mean(XX4'))); goodplot
             YT = mean([log(mean(XX1')) log(mean(XX2')) log(mean(XX3')) log(mean(XX4'))]')';
        plot(x,YT,'LineWidth',3); goodplot
        end
       
        mb = []; ma = [];
        bmax = 0;
        for i2 = 1:1
            NL = round(lenLL/2.0*i2);
            yn1 = log(mean(XX1')); yn1 = yn1(round(lenLL/20.0):NL)';
            yn2 = log(mean(XX2')); yn2 = yn2(round(lenLL/20.0):NL)';
            yn3 = log(mean(XX3')); yn3 = yn3(round(lenLL/20.0):NL)';
            yn4 = log(mean(XX4')); yn4 = yn4(round(lenLL/20.0):NL)';
            
            tx = x(round(lenLL/20.0):NL);
            %plot(tx,YT,'LineWidth',3); goodplot
            %plot(tx(1:end),yn(1:end),'r','LineWidth',3)
            XXn = zeros(length(tx),2); XXn(:,1) = 1; XXn(:,2) = tx';
            b1 = inv((XXn'*XXn))*XXn'*yn1;
            b2 = inv((XXn'*XXn))*XXn'*yn2;
            b3 = inv((XXn'*XXn))*XXn'*yn3;
            b4 = inv((XXn'*XXn))*XXn'*yn4;
            BB = inv((XXn'*XXn))*XXn'*YT;
            y11 = b1(1)+b1(2)*tx;
            y12 = b2(1)+b2(2)*tx;
            y13 = b3(1)+b3(2)*tx;
            y14 = b4(1)+b4(2)*tx;
            YF = BB(1)+BB(2)*tx;
%             plot(tx,y11,'LineWidth',3); goodplot
%             plot(tx,y12,'LineWidth',3); goodplot
%             plot(tx,y13,'LineWidth',3); goodplot
%             plot(tx,y14,'LineWidth',3); goodplot
          %  plot(tx,YF,'LineWidth',3); goodplot
           B = [b1(2) b2(2) b3(2) b4(2)];
            mb = [mb; b1(1)];
            A = [b1(1) b2(1) b3(1) b4(1)];
            ma = [ma; mean(A)];
            if (max(B)>bmax)
                bmax = max(B);
            end
        end
        % title(['Lya_',num2str(bmax),'_Val2_',num2str(mean(mb)),'Here'])
        yv = mean(ma)+mean(mb)*tx';
        Lya = [Lya; bmax mean(mb) mean(ma) yv(end) b1(2)];
        
        
    end
    LyaQ(:,j1) = Lya(:,1);
    LyaQ2(:,j1) = Lya(:,2);
    LyaQ3(:,j1) = Lya(:,4);
    LyaQ5(:,j1) = Lya(:,5);
end
toc

amp = linspace(R1,R2,(nnl))';
figure
for i = 1:nj1
    plot(amp,smooth(LyaQ(:,i)),'LineWidth',3); hold on; goodplot
end

figure
for i = 1:nj1
    plot(amp,(LyaQ2(:,i)),'LineWidth',3); hold on; goodplot
end

figure
for i = 1:nj1
    plot(amp,(smooth(LyaQ3(:,i))),'LineWidth',3); hold on; goodplot
end

figure
for i = 1:nj1
    plot(amp,((LyaQ5(:,i))),'LineWidth',3); hold on; goodplot
end

