function [ label,error ] = adaPredict(dataset,beta,ret)
%adaPredict: classify new unseen data
% Input: beta; ret: f,t,y; dataset
% Output: predicted label

%number of objects and the number of iterations
x=getdata(dataset);
r = size(x,1);
T = size(beta,1);
trueLab=getlab(dataset);
threshold = 0.5*sum(log(1./beta));
val = zeros(r,T);
for i=1:T
    f = ret(i,1);
    t = ret(i,2);
    y = ret(i,3);
    theta = ones(r,1)*t;
    if y==0
        val(:,i) =x(:,f)-t<=0;
    else
        val(:,i) =  x(:,f)-t>=0;
    end
    val(:,i) = val(:,i)*log(1/beta(i));
end
label = sum(val,2)>=threshold;
label = label+1;
error = sum(abs(label-trueLab)/r);
disp(error);
end