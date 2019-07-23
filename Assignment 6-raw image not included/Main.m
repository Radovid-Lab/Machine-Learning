%file directory
if (exist('data_banana.mat','file')==2)
    load('data_banana.mat')
else
    dir_banana='./sival_apple_banana/banana'
    data_banana=imageRead(dir_banana)
    data_banana=reshape(data_banana,1,length(data_banana))
    save('data_banana','data_banana')
end
if(exist('data_apple.mat','file')==2)
    load('data_apple.mat')
else
    dir_apple='./sival_apple_banana/apple'
    save('data_apple','data_apple')
    data_apple=imageRead(dir_apple)
    reshape(data_apple,1,length(data_apple))
end

%selest proper width
% showImage(data_apple(1));
% showImage(data_banana(1));

%build bags
if (exist('bags.mat','file')==2)
    load('bags.mat')
else
    dataset=[data_apple;data_banana];
    bags=gendatmilsival(dataset,40);
end

%fisher-untrustworthy
errorrate=0;
for i=1:10
    errorrate=errorrate+fisher(bags);
end
errorFisher=errorrate/10;

%fisher-trustworthy
errorFisherT=fisherT(bags);


% %feature vecter calculation
% bagembed(bags,0.5);

%MILE
errorMILE=MILE(bags,0.2);

%show error rate
disp('Fisher untrustworthy');
disp(errorFisher);
disp('Fisher trustworthy');
disp(errorFisherT);
disp('MILE ');
disp(errorMILE);

