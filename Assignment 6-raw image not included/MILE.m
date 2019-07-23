function error = MILE(dataset,sigma)
error=0;
for time=1:10
    vectors=bagembed(dataset,sigma);
    labelList=getLabel(dataset)
    labelList=reshape(labelList,length(labelList),1);
    % 80%training set and 20% test set
    testnum=randperm(length(vectors),ceil(length(vectors)*0.2));
    testset=[]
    testlabel=[]
    trainingset=[]
    traininglabel=[]
    for i=1:size(vectors,1)
        if ismember(i,testnum)
            testset=[testset;vectors(i,:)];
            testlabel=[testlabel;labelList(i)];
        else
            trainingset=[trainingset;vectors(i,:)];
            traininglabel=[traininglabel;labelList(i)]
        end
    end
    prset=prdataset(trainingset,traininglabel);
    classifier=liknonc(prset);
    prediction=labeld(testset,classifier);
    residual=abs(prediction-testlabel);
    errorrate=sum(residual)/length(testlabel);
    error=error+errorrate;
end
error=error/10;
end