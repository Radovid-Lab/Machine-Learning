function prediction = predict(dataset, distance,bag_id,test_instance,k,R,C)
num_instances = length(getLabel(dataset));
instances = unique(getident(dataset,'milbag')),1);
distances = [];
for i = 1:num_instances
    res = HausdorffDist(test_instance,dataset(find(getident(dataset,'milbag') == instances(i)),:),k);
    distances = [distances res];
end
[~,max_id] = sort(distances,'ascend');
id_r = [];
for i = 1:R
    list_r_idx = find(getident(dataset,'milbag')==bag_id(max_id(i)));
    id_r = [id_r list_r_idx];
end
r_lab = dataset(id_r,:).labels;
c_lab = [];
for i=1:instances
    res = HausdorffDist(dataset(find(getident(dataset,'milbag')==instances(i)),:),test_instance,k);
    idx = find(test_instance==bag_id(i));
    distances = cat(2,distance(idx,:),res);
    [max_value,max_id] = sort(distances,'ascend');
    rank = find(max_value==res);
    if rank<=c
        id_r = find(getident(dataset,'milbag')==(instances(i)));
        ref = getlab(dataset(id_r(1),:));
        c_lab = [c_lab;ref];
    end
end
nb_ref=0;
nb_citer=0;
for i = 1:size(c_lab,1)
    if c_lab(i) == 1
        nb_ref = nb_ref+1;
    else
        nb_citer = nb_citer+1;
    end
end
for i = 1:size(r_lab,1)
    if r_lab(i) == 1
        nb_ref = nb_ref +1;
    else
        nb_citer = nb_citer+1;
    end
end
if nb_citer>nb_ref
    prediction = 2;
else
    prediction = 1;
end
end