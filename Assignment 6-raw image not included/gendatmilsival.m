function [bagdataset] = gendatmilsival(dataset,width)
% Create a function gendatmilsival that creates a MIL dataset, by go-
% ing through all apple and banana-images, extracting the instances per
% image, and storing them in a Prtools dataset with bags2dataset. Note
% that, in addition to the class labels, also the bag identifiers are stored
% in the dataset. If you are interested, you can retrieve them using bagid
% = getident(a,'milbag')
bags_data = cell(0);
bags_label = [];
[nrow,ncol] = size(dataset);
len=0;
for j =1:nrow
    for i = 1:length(dataset(j,:))
        instances = extractInstances(dataset(j,i),width);
        bags_data{i+len,1} = instances;
    end
    len=len+length(dataset(j,:));
end
bags_label = []
for i =1:nrow
        bags_label=[bags_label;i*ones(length(dataset(i,:)),1)];
end
bagdataset = bags2dataset(bags_data,bags_label);
end