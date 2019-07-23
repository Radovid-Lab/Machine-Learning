function [ feat,theta,y] = weightedWeakLearner( X,weight,lab)
%weakLearner: A very simple classifier
%  Input: dataset with labels
%  Find the optimal feature f and threshold theta
%  Output: optimal f, theta, and y.
if nargin < 3
    lab = getlab(X);
    X = getdata(X);
end
%%%%%%% for testing
X_min = min(X);  
X_max = max(X); 
%%%%%%%
[n,f] = size(X);
min_score = 10000000;
for i=1:f
    for j = X_min(i):1:X_max(i)
        sign = 0; %sign is: >
        %predict
        Theta = ones(n,1)*j;
        predict = X(:,i)-Theta<=0;
        predict1 = X(:,i)-Theta>=0;
        predict1 = predict1+1;
        predict = predict+1;
        score = weight'*abs(predict-lab);
        score1 = weight'*abs(predict1-lab);
        if score>score1
            score = score1;
            sign =1;
        end
        if score<min_score
            min_score = score;
            y = sign;
            feat = i;
            %theta = X(j);
            theta = j;
        end
    end
end
end