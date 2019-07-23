dataset=gauss([50 50],[0 0;2 0]);
scatterd(dataset);
[objects features]=size(dataset);
weight=ones(objects,1);
[f t y]=decisionStu(dataset,weight);
testerr(dataset,weight,f,t,y);