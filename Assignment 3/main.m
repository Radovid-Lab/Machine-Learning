data=dlmread('./twoGaussians.txt');
label=int8(data(:,12));
data=data(:,1:11);
%self learning
errSelf=[];
errSelf=[errSelf;[16,self(data,label,100,0)]];
errSelf=[errSelf;[24,self(data,label,100,8)]];
errSelf=[errSelf;[32,self(data,label,100,16)]];
errSelf=[errSelf;[48,self(data,label,100,32)]];
errSelf=[errSelf;[80,self(data,label,100,64)]];
errSelf=[errSelf;[144,self(data,label,100,128)]];
errSelf=[errSelf;[272,self(data,label,100,256)]];
errSelf=[errSelf;[528,self(data,label,100,512)]];
plot(log(errSelf(:,1)),errSelf(:,2),'o');
hold on;
plot(log(errSelf(:,1)),errSelf(:,2));
xlabel('log( number of training data )');
ylabel('avered error');

%supervised learning
errSuper=[];
errSuper=[errSuper;[16,supervised(data,label,100)]];
errSuper=[errSuper;[24,supervised(data,label,100)]];
errSuper=[errSuper;[32,supervised(data,label,100)]];
errSuper=[errSuper;[48,supervised(data,label,100)]];
errSuper=[errSuper;[80,supervised(data,label,100)]];
errSuper=[errSuper;[144,supervised(data,label,100)]];
errSuper=[errSuper;[272,supervised(data,label,100)]];
errSuper=[errSuper;[528,supervised(data,label,100)]];
plot(log(errSuper(:,1)),errSuper(:,2),'o');
hold on;
plot(log(errSuper(:,1)),errSuper(:,2));

% %supervised learning'
% errSuper=[];
% errSuper=[errSuper;[16,supervised2(data,label,100,8)]];
% errSuper=[errSuper;[24,supervised2(data,label,100,12)]];
% errSuper=[errSuper;[32,supervised2(data,label,100,16)]];
% errSuper=[errSuper;[48,supervised2(data,label,100,24)]];
% errSuper=[errSuper;[80,supervised2(data,label,100,40)]];
% errSuper=[errSuper;[144,supervised2(data,label,100,72)]];
% errSuper=[errSuper;[272,supervised2(data,label,100,136)]];
% errSuper=[errSuper;[528,supervised2(data,label,100,264)]];
% plot(log(errSuper(:,1)),errSuper(:,2),'o');
% hold on;
% plot(log(errSuper(:,1)),errSuper(:,2));



%co-training
errCo=[];
errCo=[errCo;[16,coTrain(data,label,100,0)]];
errCo=[errCo;[24,coTrain(data,label,100,8)]];
errCo=[errCo;[32,coTrain(data,label,100,16)]];
errCo=[errCo;[48,coTrain(data,label,100,32)]];
errCo=[errCo;[80,coTrain(data,label,100,64)]];
errCo=[errCo;[144,coTrain(data,label,100,128)]];
errCo=[errCo;[272,coTrain(data,label,100,256)]];
errCo=[errCo;[528,coTrain(data,label,100,512)]];

plot(log(errCo(:,1)),errCo(:,2),'o');
plot(log(errCo(:,1)),errCo(:,2));
legend('selt-training','selt-training','supervised','supervised','co-training','co-training');










