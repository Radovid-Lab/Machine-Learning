function [f,t,y] = decisionStu(data,weight)
%decisionStu: weak learner using decision stump method.
%input: x,y: dataset with label
%output: f,t,y

%read data.
lab=getlab(data);
if(ischar(lab))
    lab=str2num(lab);
end
x=getdata(data);
weight=weight/sum(weight);
%r objects with c dimensions.
[r,c]=size(x);
minError=inf;
for i=1:c
    for j=1:r
        %exhausive search on theta
        theta=ones(r,1)*x(j,i);
        %prediction based on threshold,method2 is to classify 
        %objects to class 2 if its smaller than threshold. method2
        %classifies objects smaller than threshold to class 1
        method1= x(:,i)<=theta;
        method2= x(:,i)>=theta;
        %error rate of method1 and method2
        err1=method1+1~=lab;
        err1=sum(weight'*err1);
        err2=method2+1~=lab;
        err2=sum(weight'*err2);
        %judge classification method
        if err1>err2
            err1=err2;
            sign=1;
        else
            sign=0;
        end
        %updata the optimal
        if err1<=minError
            y=sign;
            minError=err1;
            f=i;
            t=x(j);
        end
    end
end
disp(minError);
end

