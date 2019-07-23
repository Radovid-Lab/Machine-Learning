function feature_vector = bagembed(bags,sigma)
instances=getident(bags,'milbag');
data=bags.data;
num_images=length(unique(instances));
feature_vector = [];
start=1;
len=0;
while start<length(instances)
    best = zeros(1,length(instances));
    len=0;
    current_vector= [];
    while start+len<=length(instances)&&instances(start)==instances(start+len)
        temp = exp((-1/sigma^2)*sum((data(start+len,:)-data(:,:)).^2,2));
        for i=1:length(best)
            if temp(i)>best(i)
                best(i)=temp(i);
            end
        end
        len=len+1;
    end
    if start+len-1<=length(instances)
        feature_vector= [feature_vector;reshape(best,1,length(best))];
    end
    start=start+len;
end
end