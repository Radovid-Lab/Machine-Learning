function majority = combineinstlabels(listLabel)
count=0;
majority=listLabel(1);
for i = 1:length(listLabel)
        if majority==listLabel(i)
            count=count+1;
        else
            count=count-1;
        end
        if count==0
            majority=listLabel(i);
        end
end