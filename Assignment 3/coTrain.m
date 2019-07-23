function [error] = coTrain(odata,olabel,round,num)

i=1;
error=0;
while i~=round
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
    [urow,ulen]=size(us);
    
    %test set 200
    [trow,tlen]=size(data);
    tmp=randperm(trow,200);
    ts=data(tmp,:);
    tsl=double(label(tmp,:));
    [trow,tlen]=size(ts);
    
    
    %learn model based on labeled data
    [useless,nlen]=size(labelD);
    numf1=ceil(nlen/2);
    numf2=nlen-numf1;
    index=randperm(nlen,numf1);
    labelD1=labelD(:,index);
    us1=us(:,index);
    us21=us1;
    us3=us1;
    tmp2=labelD;
    tmp2(:,index)=[];
    tmp=us;
    tmp(:,index)=[];
    us2=tmp;
    us12=us2;
    us4=us2;
    labelD2=tmp2;
    labelL1=labelL;
    labelL2=labelL;
    [w1,b1]=LRC(labelD1,labelL1);
    [w2,b2]=LRC(labelD2,labelL2);
    j=0;
    q=0;
    while j~=num||q~=num
        %confidence 1
        evay1=abs(us1*w1'+b1-sign(us1*w1'+b1));
        addedData1=find(evay1==min(evay1));
        %confidence 2
        evay2=abs(us2*w2'+b2-sign(us2*w2'+b2));
        addedData2=find(evay2==min(evay2));
        %add label to 2
        if j~=num
            labelD2=[labelD2;us12(addedData1,:)];
            labelL2=[labelL2;sign(us1(addedData1,:)*w1'+b1)];
            [rowAddData1,lenAddData1]=size(addedData1);
            us1(addedData1,:)=[];
            us12(addedData1,:)=[];
            j=j+rowAddData1;
        end
        
        if q~=num
            labelD1=[labelD1;us21(addedData2,:)];
            labelL1=[labelL1;sign(us2(addedData2,:)*w2'+b2)];
            [rowAddData2,lenAddData2]=size(addedData2);
            us2(addedData2,:)=[];
            us21(addedData2,:)=[];
            q=q+rowAddData2;
        end
        %retrain c2
        [w2,b2]=LRC(labelD2,labelL2);
        %retrain c1
        [w1,b1]=LRC(labelD1,labelL1);
    end
    dataset1=[us3];
    dataset2=[us4];
    evaAfter1=sign(dataset1*w1'+b1);
    evaAfter2=sign(dataset2*w2'+b2);
    dataset3=[us3 us4];
    dataset3=dataset3(find((evaAfter1==evaAfter2)==1),:);
    label3=evaAfter1(find((evaAfter1==evaAfter2)==1),:);
    [w,b]=LRC([labelD;dataset3],[labelL;label3]);
    [err,score]=evaLRC(ts,tsl,w,b);
    error=error+err;
%     error=error+score;
    i=i+1;
end
error=error/round;
end


