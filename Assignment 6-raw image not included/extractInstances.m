function [result] = extractInstances(raw_image,width)
% segments an image
% using the Mean Shift algorithm (using im meanshift), computes the
% average red, green and blue color per segment, and returns the resulting
% features in a small data matrix.
segmentation=im_meanshift(raw_image.data,width);
[w,h]=size(segmentation);
instances=unique(segmentation);
result=zeros(length(instances),3);
count=zeros(1,length(instances));
for i = 1:w
    for j =1:h
        NoInstance=segmentation(i,j);
        result(NoInstance,:)=result(NoInstance,:)+double(reshape(raw_image.data(i,j,:),1,3));
        count(NoInstance)=count(NoInstance)+1;
    end
end
for i = 1:length(instances)
    result(i,:)=result(i,:)/(count(i)*255);
end

