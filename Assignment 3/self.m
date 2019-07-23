function [error] = self(odata,olabel,round,num)
error=0;
i=0;
while i<round
    %select 8 labeled data from each class
    data=odata;
    label=olabel;
    class1=find(label==1);
    [nrow,nlen]=size(class1);
    tmp=randperm(nrow,8);
    ls1=data(class1(tmp),:);%labeled data
    data(class1(tmp),:)=[];
    ls1l=double(label(class1(tmp),:));
    label(class1(tmp),:)=[];
    class2=find(label==-1);
    [nrow,nlen]=size(class2);
    tmp=randperm(nrow,8);
    ls2=data(class2(tmp),:);
    data(class2(tmp),:)=[];
    ls2l=double(label(class2(tmp),:));
    label(class2(tmp),:)=[];
    labelD=[ls1;ls2];
    labelL=[ls1l;ls2l];
    
    %select unlabeled data
    [nrow,nlen]=size(data);
    tmp=randperm(nrow,num);
    us=data(tmp,:);%unlabeled data
    data(tmp,:)=[];
    usl=double(label(tmp,:));
    label(tmp,:)=[];
    [nrow,nlen]=size(us);
    
    %test set 200
    [trow,tlen]=size(data);
    tmp=randperm(trow,200);
    ts=data(tmp,:);
    tsl=double(label(tmp,:));
    [trow,tlen]=size(ts);
    
    %learn model based on labeled data
    [w,b]=LRC(labelD,labelL);
    [numError,score]=evaLRC(ts,tsl,w,b);
    while(nrow~=0)
        evay=abs(us*w'+b-sign(us*w'+b));
        addedData=find(evay==min(evay));
        labelD=[labelD;us(addedData,:)];
        labelL=[labelL;sign(us(addedData,:)*w'+b)];
        [rowAddData,lenAddData]=size(addedData);
        us(addedData,:)=[];
        usl(addedData,:)=[];
        nrow=nrow-rowAddData;
        [w,b]=LRC(labelD,labelL);
        [numError,score]=evaLRC(ts,tsl,w,b);
    end
    i=i+1;
    error=error+numError;
%     error=error+score;
end
error=error/round;
% disp('error is:');
% disp(error);
end

