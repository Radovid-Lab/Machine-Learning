function showImage(image)
figure;
num=1;
for i = 10:10:120
    subplot(3,4,num);
    imshow(im_meanshift(image.data,i),[]);
    title('when width is '+ string(i));
    num=num+1;
end

