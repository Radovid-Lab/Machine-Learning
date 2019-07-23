function [outputArg1,outputArg2] = EMLDC(inputArg1,inputArg2)

function [tureE,error] = EM(data,label,round,num)
%EM Summary of this function goes here
%input: sl:labeled data su:unlabeled data num:number of unlabeled data
%label:label of data
i=1;

[w,b]=LRC(data,label);
score=evaLRC(data,label,w,b);
score;

error=0;
while i<round
    %select 8 labeled data from each class
    class1=find(label==1);
    [nrow,nlen]=size(class1);
    tmp=ceil(nrow*rand(8,1));
    ls1=data(class1(tmp),:);%labeled data
    ls1l=double(label(class1(tmp),:));
    class2=find(label==-1);
    [nrow,nlen]=size(class2);
    tmp=ceil(nrow*rand(8,1));
    ls2=data(class2(tmp),:);
    ls2l=double(label(class2(tmp),:));
    
    %select unlabeled data
    [nrow,nlen]=size(data);
    tmp=ceil(nrow*rand(num,1));
    us=data(tmp,:);%labeled data
    usl=double(label(tmp,:));
    
    %initialize parameters
    a=[0.5 0.5];
    u=[mean(ls1); mean(ls2)];
    sigma=[diag(cov(ls1))' ; diag(cov(ls2))' ];
    
    j=1;
    condition1=1;
    condition2=1;
    while condition1~=0||condition2~=0
        %E step:
        pos1=mvnpdf(us,u(1,:),sigma(1,:));
        pos2=mvnpdf(us,u(2,:),sigma(2,:));
        gamma1=a(1)*pos1./(a(1)*pos1+a(2)*pos2);
        gamma2=a(2)*pos2./(a(1)*pos1+a(2)*pos2);
        
        %M step:
        oldu=u;
        u(1,:)=(1/(sum(gamma1)+8))*(sum(gamma1.*us)+sum(ls1));
        u(2,:)=(1/(sum(gamma2)+8))*(sum(gamma2.*us)+sum(ls2));
        condition1=sum(sum(oldu-u));
        oldsigma=sigma;
        sigma(1,:)=(1/(sum(gamma1)+8))*(diag((us-u(1,:))'*(gamma1.*(us-u(1,:)))) + diag((ls1-u(1,:))'*(ls1-u(1,:))));
        sigma(2,:)=(1/(sum(gamma2)+8))*(diag((us-u(2,:))'*(gamma2.*(us-u(2,:)))) + diag((ls2-u(2,:))'*(ls2-u(2,:))));
        condition2=sum(sum(oldsigma-sigma));
        a(1)=1/(num+8)*(sum(gamma1)+8);
        a(2)=1/(num+8)*(sum(gamma2)+8);
        
        %assign class to unlabeled data
        p1=mvnpdf(us,u(1,:),sigma(1,:));
        p2=mvnpdf(us,u(2,:),sigma(2,:));
        lp=p1>p2;
        lp(find(lp==0))=-1;
        lrcdata=[ls1 ls1l;ls2 ls2l;us lp];
        
        %calculate line
        [w,b]=LRC(lrcdata(:,1:11),lrcdata(:,12));
        score=evaLRC(data,label,w,b);
        j=j+1;
    end
    error=error+score;
    i=i+1;
end
disp(error/round);
end





end

