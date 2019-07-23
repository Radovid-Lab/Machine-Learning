function [errorrate] = fisher(bags)
% Now we are almost ready to classify images... First we have to train
% a classier; let's use a Fisher classier for this. Now apply the trained
% classier to each instance in a bag, classify the instances (using labeld),
% and combine the label outputs (using your combineinstlabels) to get
% a bag label.
% How many apple images are misclassied to be banana? And vice
% versa? Why is this error estimate not trustworthy? Estimate the clas-
% sication error in a trustworthy way!
instances=getident(bags,'milbag');
data=bags.data;
num_images=length(unique(instances));
fisherClassifier=fisherc(bags);
prediction=[]
original=[]
start=1;
len=0;
origianlLabel=getlabels(bags);
for i=1:num_images
    currentBag=[];
    original=[original;origianlLabel(start)];
    while start+len<=length(instances)&&instances(start+len)==i
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