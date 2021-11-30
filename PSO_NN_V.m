clc 
tic 
close all 
clear all 
rand('seed',45)
data = xlsread('Core & Logs_BED_modified.xlsx','Sheet1');  
input = data(:,[3 5 6 8 9]); 
target = data(:,2); 

%% scaling 0 to 1
xt=input';
yt=target';

[xt,ps]=mapminmax(xt,0,1);
[yt,ts]=mapminmax(yt,0,1);

xt=xt';
yt=yt';
%%
[trainInd,testInd,valInd]=dividerand(95,0.70,0.30,0.0);   

x_train=xt(trainInd,:);
x_test=xt(testInd,:);

y_train=yt(trainInd,:);
y_test=yt(testInd,:);


%%
inputs=x_train'; 
targets=y_train'; 

m=length(inputs(:,1)); 
o=length(targets(:,1)); 

n=6; 
net=feedforwardnet(n); 
net=configure(net,inputs,targets);
kk=m*n+n+n+o;
for j=1:kk 
    LB(1,j)=-1.5; 
    UB(1,j)=1.5; 
end
pop=10;
for i=1:pop 
    for j=1:kk 
        xx(i,j)=LB(1,j)+rand*(UB(1,j)-LB(1,j)); 
    end
end
maxrun=1;
for run=1:maxrun 
    fun=@(x) myfunc(x,n,m,o,net,inputs,targets); 
    x0=xx;          % pso initialization----------------------------------------------start 
    x=x0;           % initial population 
    v=0.1*x0;       % initial velocity 
    for i=1:pop 
        f0(i,1)=fun(x0(i,:)); 
    end
    [fmin0,index0]=min(f0);
    pbest=x0; % initial pbest 
    gbest=x0(index0,:); % initial gbest 
    % pso initialization------------------------------------------------end 
    % pso algorithm---------------------------------------------------start 
    c1=1.5; 
    c2=2.5; 
    ite=1; 
    maxite=1000; 
    tolerance=1; 
    while ite<=maxite && tolerance>10^-8 
        w=0.1+rand*0.4; % pso velocity updates 
        for i=1:pop 
            for j=1:kk 
                v(i,j)=w*v(i,j)+c1*rand*(pbest(i,j)-x(i,j))... 
                    +c2*rand*(gbest(1,j)-x(i,j)); 
            end
        end
        % pso position update 
        for i=1:pop 
            for j=1:kk 
                x(i,j)=x(i,j)+v(i,j); 
            end
        end
        % handling boundary violations 
        for i=1:pop 
            for j=1:kk 
                if x(i,j)<LB(j) 
                    x(i,j)=LB(j); 
                elseif x(i,j)>UB(j) 
                    x(i,j)=UB(j); 
                end
            end
        end
        % evaluating fitness 
        for i=1:pop 
            f(i,1)=fun(x(i,:)); 
        end
        
        % updating pbest and fitness 
        for i=1:pop 
            if f(i,1)<f0(i,1) 
                pbest(i,:)=x(i,:); 
                f0(i,1)=f(i,1); 
            end
        end
        [fmin,index]=min(f0);     % finding out the best particle 
        ffmin(ite,run)=fmin;  % storing best fitness 
        ffite(run)=ite; % storing iteration count % updating gbest and best fitness 
        if fmin<fmin0 
            gbest=pbest(index,:); 
            fmin0=fmin; 
        end
        
        %calculating tolerance 
        if ite>100; 
            tolerance=abs(ffmin(ite-100,run)-fmin0); 
        end
        % displaying iterative results 
        if ite==1            
            fprintf('Iteration Best particle Objective fun\n'); 
        end
        fprintf('%8g %8g %8.4f\n',ite,index,fmin0); 
        ite=ite+1; 
    end
    % pso algorithm-----------------------------------------------------end 
    xo=gbest; 
    fval=fun(xo); 
    xbest(run,:)=xo; 
    ybest(run,1)=fun(xo); 
    fprintf('****************************************\n'); 
    fprintf(' RUN fval ObFuVa\n'); 
    disp(fprintf('%6g %6g %8.4f %8.4f',run,fval,ybest(run,1))); 
end
toc 
% Final neural network model 
disp('Final nn model is net_f') 
net_f = feedforwardnet(n); 
net_f=configure(net_f,inputs,targets); 
[a b]=min(ybest); 
xo=xbest(b,:); 
k=0; 
for i=1:n 
    for j=1:m 
        k=k+1; 
        xi(i,j)=xo(k); 
    end
end
for i=1:n 
    k=k+1; 
    xl(i)=xo(k); 
    xb1(i,1)=xo(k+n); 
end
for i=1:o 
    k=k+1; 
    xb2(i,1)=xo(k); 
end
net_f.iw{1,1}=xi; 
net_f.lw{2,1}=xl; 
net_f.b{1,1}=xb1; 
net_f.b{2,1}=xb2; 
%Calculation of MSE
err=sum((net_f(inputs)-targets).^2)/length(net_f(inputs)) 
pred_tr=net_f(inputs);
pred_ts=net_f(x_test');

disp('Trained ANN net_f is ready for the use');

%% Rescaling
 y_train = mapminmax('reverse',y_train',ts);
 y_test= mapminmax('reverse',y_test',ts);
 pred_tr=mapminmax('reverse',pred_tr,ts);
 pred_ts=mapminmax('reverse',pred_ts,ts);

 %% RMSE and R2 score calculation
perf(1,1)=rmse(y_train,pred_tr);
perf(2,1)=rmse(y_test,pred_ts);
perf(1,2)=mae(y_train,pred_tr);
perf(2,2)=mae(y_test,pred_ts);
perf(1,3)=R2(y_train,pred_tr);
perf(2,3)=R2(y_test,pred_ts);
perf(1,4)=corr(y_train',pred_tr');
perf(2,4)=corr(y_test',pred_ts');
perf(1,5)=mape(y_train,pred_tr);
perf(2,5)=mape(y_test,pred_ts);
perf(1,6)=stde(y_train,pred_tr);
perf(2,6)=stde(y_test,pred_ts);
perf(1,7)=mre(y_train,pred_tr);
perf(2,7)=mre(y_test,pred_ts);
perf(1,8)=maxe(y_train,pred_tr);
perf(2,8)=maxe(y_test,pred_ts);


rowNm={'Train','Test'};
colNm={'RMSE','MAE','R2','corr','MAPE','STDE','MRE','MAXE'};
Table_1=array2table(perf,"RowNames",rowNm,"VariableNames",colNm);
%{
%% Regression ploting
figure(1)
scatter(y_train,pred_tr,'ro','filled');hold on;
scatter(y_test,pred_ts,'bo','filled');hold on;
plot(linspace(0,500,10),linspace(0,500,10),'black--');
xlim([0 500]);
ylim([0 500]);
legend('Train','Test','Location','northwest');
xlabel('observed','FontWeight','Bold');
ylabel('Predicted','FontWeight','Bold');
title('Vertical Permeability Prediction by PSO-NN','FontWeight','Bold');
exportgraphics(gcf,'Regression plot(V).jpg','Resolution',300);

%% sampling plot
figure(2)
plot(linspace(1,66,66),pred_tr,'b-','LineWidth',2);hold on;
plot(linspace(1,66,66),y_train,'blacko');hold on;
plot(linspace(66,95,29),pred_ts,'r-','LineWidth',2);hold on;
plot(linspace(66,95,29),y_test,'blacko');hold on;
ylim([0.001 800]);
view([90 90]);
set(gca,'YAxisLocation','right');
ylabel('Permeability(mD)','FontWeight','Bold');
xlabel('samples','FontWeight','Bold');
legend('Train','Observed','Test','Location','northeast');
set(gcf,'units','points','position',[150,100,200,500]);
%exportgraphics(gcf,'Sample plot_PSO(V).jpg','Resolution',300);

% fitness plot
%objective function plot
figure(3)
plot(linspace(1,1000,1000),ffmin,'-','color','Blue','linewidth',2);
ylabel('Best fitness','FontWeight','Bold');
xlabel('Iteration','FontWeight','Bold');
%}