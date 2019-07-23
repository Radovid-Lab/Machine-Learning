function [error,label] = testerr(testdata,weight,f,t,y)
%TESTERR Summary of this function goes here
%   Detailed explanation goes here
weight=weight/sum(weight);
data=getdata(testdata);
testlabel=getlab(testdata);
if(ischar(testlabel))
    testlabel=str2num(testlabel);
end
[objects features]=size(data);
label=zeros(objects,1);
if(y==1)
    for i=1:objects
        if(data(i,f)<t)
            label(i)=1;
        else
            label(i)=2;
        end
    end
else
        for i=1:objects
        if(data(i,f)>t)
            label(i)=1;
        else
            label(i)=2;
        end
        end
end
error = sum(weight'*abs(label-testlabel));
disp(error);
end

