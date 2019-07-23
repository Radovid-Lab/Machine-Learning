function [S] = imageRead(path)
D = path;
S = dir(fullfile(D,'*.jpg')); % pattern to match filenames.
for k = 1:numel(S)
    F = fullfile(D,S(k).name);
    I = imread(F);
    imshow(I)
    S(k).data = I; % optional, save data.
end
close gcf;
end