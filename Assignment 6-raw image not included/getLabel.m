function labelList = getLabel(bags)
instances=getident(bags,'milbag');
data=bags.data;
num_images=length(unique(instances));
originalLabel=getlabels(bags);
labelList=zeros(1,length(unique(instances)));
for i = 1:length(instances)
    labelList(instances(i))=originalLabel(i);
end
end

