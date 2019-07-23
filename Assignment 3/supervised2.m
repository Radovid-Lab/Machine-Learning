function [error] = supervised(odata,olabel,round,num)
error=0;
i=round;
while(i~=0)
    data=odata;
    label=olabel;
    class1=find(label==1);
    [nrow,nlen]=size(class1);
    tmp=randperm(nrow,num);
    ls1=data(class1(tmp),:);%labeled data
    data(class1(tmp),:)=[];
    ls1l=double(label(class1(tmp),:));
    label(class1(tmp),:)=[];
    class2=find(label==-1);
    [nrow,nlen]=size(class2);
    tmp=randperm(nrow,num);
    ls2=data(class2(tmp),:);
    data(class2(tmp),:)=[];
    ls2l=double(label(class2(tmp),:));
    label(class2(tmp),:)=[];
    labelD=[ls1;ls2];
    labelL=[ls1l;ls2l];
    
    %select test data
    [nrow,nlen]=size(data);
    tmp=randperm(nrow,200);
    ts=data(tmp,:);%unlabeled data
    tsl=double(label(tmp,:));
    
    [w,b]=LRC(labelD,labelL);
    [tmp,useless]=evaLRC(ts,tsl,w,b);
    error=error+tmp;
%     error=error+useless;
    i=i-1;
end
error=error/round;
end

