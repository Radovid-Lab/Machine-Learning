dataset_1 = gauss([1000 1000],[0,0;10,10],cat(3,[1,0;0,1],[1,0;0,1]));
scatterd(dataset_1);
labelD1 = +dataset_1(1:1000,:);
labelL1=ones(1000,1);
labelD2 = +dataset_1(1001:end,:);
labelL2=-ones(1000,1);
data=[labelD1;labelD2];
label=[labelL1;labelL2];

% X = [randn(1000,2);randn(10000,2)+ repmat([3,2.5],10000,1)];
% labels = X(:,2)>0;
% labels=int8(labels);
% labels (find(labels==0))=-1;
% X_dataset = prdataset(X,labels);
% scatterd(X_dataset);
% data=X;
% label=labels;

figure(2);
%self learning
errSelf=[];
errSelf=[errSelf;[16,self(data,label,1000,0)]];
errSelf=[errSelf;[24,self(data,label,1000,8)]];
errSelf=[errSelf;[32,self(data,label,1000,16)]];
errSelf=[errSelf;[48,self(data,label,1000,32)]];
errSelf=[errSelf;[80,self(data,label,1000,64)]];
errSelf=[errSelf;[144,self(data,label,1000,128)]];
errSelf=[errSelf;[272,self(data,label,1000,256)]];
errSelf=[errSelf;[528,self(data,label,1000,512)]];
plot(log(errSelf(:,1)),errSelf(:,2),'o');
hold on;
plot(log(errSelf(:,1)),errSelf(:,2));
xlabel('log( number of training data )');
ylabel('avered error');

%supervised learning
errSuper=[];
errSuper=[errSuper;[16,supervised(data,label,1000)]];
errSuper=[errSuper;[24,supervised(data,label,1000)]];
errSuper=[errSuper;[32,supervised(data,label,1000)]];
errSuper=[errSuper;[48,supervised(data,label,1000)]];
errSuper=[errSuper;[80,supervised(data,label,1000)]];
errSuper=[errSuper;[144,supervised(data,label,1000)]];
errSuper=[errSuper;[272,supervised(data,label,1000)]];
errSuper=[errSuper;[528,supervised(data,label,1000)]];
plot(log(errSuper(:,1)),errSuper(:,2),'o');
hold on;
plot(log(errSuper(:,1)),errSuper(:,2));

% %supervised learning'
% errSuper=[];
% errSuper=[errSuper;[16,supervised2(data,label,1000,8)]];
% errSuper=[errSuper;[24,supervised2(data,label,1000,12)]];
% errSuper=[errSuper;[32,supervised2(data,label,1000,16)]];
% errSuper=[errSuper;[48,supervised2(data,label,1000,24)]];
% errSuper=[errSuper;[80,supervised2(data,label,1000,40)]];
% errSuper=[errSuper;[144,supervised2(data,label,1000,72)]];
% % errSuper=[errSuper;[272,supervised2(data,label,1000,136)]];
% % errSuper=[errSuper;[528,supervised2(data,label,1000,264)]];
% plot(log(errSuper(:,1)),errSuper(:,2),'o');
% hold on;
% plot(log(errSuper(:,1)),errSuper(:,2));



%co-training
errCo=[];
errCo=[errCo;[16,coTrain(data,label,1000,0)]];
errCo=[errCo;[24,coTrain(data,label,1000,8)]];
errCo=[errCo;[32,coTrain(data,label,1000,16)]];
errCo=[errCo;[48,coTrain(data,label,1000,32)]];
errCo=[errCo;[80,coTrain(data,label,1000,64)]];
errCo=[errCo;[144,coTrain(data,label,1000,128)]];
errCo=[errCo;[272,coTrain(data,label,1000,256)]];
errCo=[errCo;[528,coTrain(data,label,1000,512)]];

plot(log(errCo(:,1)),errCo(:,2),'o');
plot(log(errCo(:,1)),errCo(:,2));
legend('selt-training','selt-training','supervised','supervised','co-training','co-training');










