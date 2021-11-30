data=xlsread('Core & Logs_BED_modified.xlsx','Sheet1');
x=data(:,[3 5 6 8 9]);
y=data(:,2);

rand('seed',20);

%% scaling 0 to 1
xt=x';
yt=y';

[xt,ps]=mapminmax(xt,0,1);
[yt,ts]=mapminmax(yt,0,1);

xt=xt';
yt=yt';

%%

[trainInd,testInd,~]=dividerand(95,0.70,0.30,0.0);

x_train=xt(trainInd,:);
x_test=xt(testInd,:);

y_train=yt(trainInd,:);
y_test=yt(testInd,:);

type = 'function estimation';
[gam,sig2] = tunelssvm({xt,yt,type,[],[],'RBF_kernel'},'simplex','leaveoneoutlssvm',{'mse'});

[alpha,b] = trainlssvm({xt,yt,type,gam,sig2,'RBF_kernel'});

%plotlssvm({xt,yt,type,gam,sig2,'RBF_kernel'},{alpha,b});

pred_tr= simlssvm({xt,yt,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},x_train);
pred_ts= simlssvm({xt,yt,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},x_test);

%% Rescaling
 y_train = mapminmax('reverse',y_train',ts);
 y_test= mapminmax('reverse',y_test',ts);
 pred_tr=mapminmax('reverse',pred_tr',ts);
 pred_ts=mapminmax('reverse',pred_ts',ts);
 
%% matrix
perf(1,1)=RMSE(y_train,pred_tr);
perf(2,1)=RMSE(y_test,pred_ts);
perf(1,3)=R2(y_train,pred_tr);
perf(2,3)=R2(y_test,pred_ts);
perf(1,2)=mae(y_train,pred_tr);
perf(2,2)=mae(y_test,pred_ts);
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
colNm={'RMSE','MAE','R2','CORR','MAPE','STDE','MRE','MAXE'};
Table_1=array2table(perf,"RowNames",rowNm,"VariableNames",colNm);

%{
%%
figure(1)
scatter(y_train, pred_tr,30,'bo','filled');
hold on;
scatter(y_test, pred_ts,30,'ro', 'filled');
hold on;
plot(linspace(0,1000,1000),linspace(0,1000,1000),'k--','LineWidth',2);
hold off;
xlim([0 500]);
ylim([0 500]);
legend('Train','Test','Location','northwest');
xlabel('observed','FontWeight','Bold');
ylabel('Predicted','FontWeight','Bold');
title('Vertical Permeability Prediction by LS-SVM','FontWeight','Bold');
text(350,100,'Train R^2=0.90','color','b','FontSize',10)
text(350,50,'Test R^2=0.77','color','b','FontSize',10)
%% sample plot
figure(2)
plot(linspace(1,66,66),pred_tr,'b-','LineWidth',2);hold on;
scatter(linspace(1,66,66),y_train','ko');hold on;
plot(linspace(66,95,29),pred_ts,'r-','LineWidth',2);hold on;
scatter(linspace(66,95,29),y_test','ko');hold on;
ylim([0.001 800]);
view([90 90]);
set(gca,'YAxisLocation','right');
ylabel('Permeability(mD)','FontWeight','Bold');
xlabel('samples','FontWeight','Bold');
legend('Train','Observed','Test','Location','northeast');
set(gcf,'units','points','position',[150,100,200,500]);
%exportgraphics(gcf,'Sample plot_PSO(H).jpg','Resolution',300);
%}
