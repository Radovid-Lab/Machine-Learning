function [error] = fisherT(bags)
% Trustworthy fisher
error=0;
for i =1:100
    instances=getident(bags,'milbag');
    origianlLabel=getlabels(bags);
    dataSetSize=length(unique(instances));
    trainingNum=randperm(dataSetSize,0.8*dataSetSize);
    trainingSet=[];
    testSet=[];
    for i = 1: length(instances)
        if ismember(instances(i),trainingNum)
            trainingSet=[trainingSet,i];
        else
            testSet=[testSet,i];
        end
    end
    trainingSet=bags(trainingSet,:);
    testSet=bags(testSet,:);
    fisherClassifier=fisherc(trainingSet);
    instances=getident(testSet,'milbag');
    data=testSet.data;
    num_images=length(unique(instances));
    prediction=[]
    original=[]
    start=1;
    len=0;
    origianlLabel=getlabels(testSet);
    for i=1:num_images
        currentBag=[];
        original=[original;origianlLabel(start)];
        current=instances(start+len);
        while start+len<=length(instances)&&instances(start+len)==current
            currentBag=[currentBag;data(start+len,:)];
            len=len+1;
        end
        result=labeld(currentBag,fisherClassifier);
        majority=combineinstlabels(result);
        prediction=[prediction;majority];
        start=start+len;
        len=0;
    end
    residual=original-prediction;
    disp('residual is 1:')
    disp(sum(residual==1));
    disp('residual is -1:')
    disp(sum(residual==-1));
    errorrate=sum(abs(residual))/length(original);
    error=error+errorrate;
end
error=error/100;