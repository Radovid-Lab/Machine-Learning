data=dlmread('fashion57_train.txt');
label=[ones(32,1);2*ones(28,1)];

%h
num=4;
h1=round(32*rand(num,1));
h2=round(28*rand(num,1))+33;
data=[data(h1,:);data(h2,:)];
label=[label(h1,:);label(h2,:)];
traindata=prdataset(data,label);

% D = (1/60)*ones(60,1);
% [error, beta,weight, ret]=adaBoost(traindata,D,30);
data=dlmread('fashion57_test.txt');
label=[ones(195,1);2*ones(205,1)];
testdata=prdataset(data,label);
% [Label error]=  adaPredict(testdata,beta,ret);




%train the classifier
error_train=ones(1,20);
error_test=ones(1,20);
for it=1:20
[predLab,beta,para] = qwe(traindata,it);
error_train(it) = sum(abs(predLab-(getlab(traindata))))/size(traindata,1);
% on test set
predLab_test = asd(beta,para,testdata);
error_test(it) = sum(abs(predLab_test-(getlab(testdata))))/size(testdata,1);
end