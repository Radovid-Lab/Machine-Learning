function [error, beta,H, ret] = adaBoost(dataset,D,T)
%adaBoost
%input: dataset: labeled data; D: distribution over dataset;
%T: number of iterations
data=getdata(dataset);
label=getlab(dataset);
[r c]=size(data);
weight=zeros(r,1);
%initialize weight
weight=D;
H=zeros(r,T);
%error for each iteration
error=zeros(T,1);
beta=zeros(T,1);
%returned values from weakLearner
ret=zeros(T,3);
for i=1:T
    p=weight/sum(weight);
    H(:,i)=p;
    ret(i,:)=decisionStu(dataset,p);
    [error(i) predict]=testerr(dataset,weight,ret(i,1),ret(i,2),ret(i,3));
    beta(i)=error(i)/(1-error(i));
    weight=weight.*(beta(i).^(1-abs(predict-label)));
end
end