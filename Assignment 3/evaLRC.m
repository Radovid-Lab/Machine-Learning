function [wrong,score] = evaLRC(x,y,w,b)
%EVALRC Summary of this function goes here
%   Detailed explanation goes here
[nrow,nlen]=size(x);
evay=x*w'+b;
evay=evay>0;
evay=int8(evay);
evay((find(evay==0)))=-1;
[wrong,useless]=size(find((y~=evay)==1));
wrong=wrong/nrow;
y=double(y);
score=(1/nrow)*sum(power((x*w'+b-y),2));
end

