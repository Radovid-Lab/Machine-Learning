function [w,b] = LRC(x,y)
y=  double(y);
[nrow,nlen]=size(x);
meanX=mean(x);
w=sum(y.*(x-meanX))./(sum(power(x,2))-(1/nrow)*power(sum(x),2));
b= (1/nrow)*sum(y-x*w');
end
