import pandas as pd
import dataLoading
import myutils

#read data
path="E:\Current Course\ml\Final\\train.csv"
train=dataLoading.readData(path)

#k-fold validation function
from sklearn.model_selection import KFold
def calcost(classifer,flag=0):
    #return cost and error rate #flag=0:normal =1:onehot
    kf = KFold(n_splits=10)
    count_cost=0
    count=0
    accuracy_count=0
    ratiogroup1=0
    ratiogroup2=0
    for trainIndex, testIndex in kf.split(train):
        trainpart=train.iloc[trainIndex,:]
        testpart=train.iloc[testIndex,:]
        count+=1
        trainpartF,trainpartL=dataLoading.divide(trainpart)
        cls=classifer.fit(trainpartF,trainpartL)
        testpartF,testpartL=dataLoading.divide(testpart)
        prediction=cls.predict(testpartF)
        prediction=pd.DataFrame(data=prediction)
        if flag==0:
            cost, p211, p212, p121, p122=myutils.cost(prediction, testpartF, testpartL)
        if flag==1:
            cost, p211, p212, p121, p122 = myutils.one_hot_cost(prediction, testpartF, testpartL)
        if(p212==0):
            print("p212 is 0")
            p212=1
        if (p122 == 0):
            print("p122 is 0")
            p122 = 1
        ratiogroup1+=p211/p212
        ratiogroup2+=p121/p122
        count_cost+=cost
        accuracy_count+=myutils.accuracy(prediction, testpartL)
    print(f"averaged cost is {count_cost/count}, accuracy is {accuracy_count/count}, ratio constraint 1 is {ratiogroup1/count}, ratio constraint 2 is {ratiogroup2/count}")
    return count_cost/count,accuracy_count/count,ratiogroup1/count,ratiogroup2/count

#pre-processing
##1.normalize(numerical data)
#normalizing features (each column)
import numpy as np
def normalize_feature(X_data):
    averages=np.average(X_data,axis=0)
    #calculate standard deviation
    X_data-=averages
    X2=np.square(X_data)
    var=np.average(X2,axis=0)
    sd=np.sqrt(var)
    #perform z-score transformation
    X_data/=sd
    return X_data

train.iloc[:,0:6]=normalize_feature(train.iloc[:,0:6])

path="E:\Current Course\ml\Final\\test.csv"
test=dataLoading.readData(path)
test.iloc[:,0:6]=normalize_feature(test.iloc[:,0:6])#normalize test set

def dummies(train,test):
    train=train.iloc[:,0:13]
    length=len(train)
    train=train.append(test)
    one_hot_df=pd.DataFrame()
    count=6
    col13=[]
    dic=[]
    for i in range(6,13):
        dfDummies = pd.get_dummies(train.iloc[:,i])
        for j in reversed(range(len(dfDummies.columns))):
            dfDummies = dfDummies.rename(columns={dfDummies.columns[j]: j+count})
            if i==12:
                col13.append(j + count)
        if i==12:
            dic.append(train.iloc[0, 12])
            for k in range(len(dfDummies.columns)):
                dic.append(dfDummies.iloc[0,k])
        one_hot_df=pd.concat([one_hot_df, dfDummies], axis=1)
        count+=len(dfDummies.columns)
    train=one_hot_df.iloc[:length,:]
    test=one_hot_df.iloc[length:,:]
    return train,test,col13,dic

##2.One-Hot endcoding
def oneHot(train,test):
    onehottrain,onehottest,col13,dic=dummies(train,test)
#column 14 has been deleted
    label=train.iloc[:,14].to_frame()
    train = train.drop(train.columns[list(range(6,15))], axis=1)
    train=pd.concat([train,onehottrain],axis=1)
    label=label.rename(columns={label.columns[0]: len(train.columns)})
    train=pd.concat([train,label],axis=1)

    test= test.drop(test.columns[list(range(6, 13))], axis=1)
    test = pd.concat([test, onehottest], axis=1)
    return train,test,col13,dic

train,test,col13,dic=oneHot(train,test)

# print(train.head())

#try different classifiers

##1.Gaussian naive bayes
from sklearn.naive_bayes import GaussianNB
classifer = GaussianNB()#assume gaussian distribution
calcost(classifer,1)
##2.Plain SVM
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
clf = SVC(C=1)
# calcost(clf)
calcost(clf,1)

clf = SVC(C=50,kernel='rbf')
calcost(clf,1)

clf = SVC(C=50,kernel='sigmoid')
calcost(clf,1)

##3.Quaratic discriminant analysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
clf = QuadraticDiscriminantAnalysis()
# calcost(clf)
calcost(clf,1)

##4.Linear discriminant analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()
# calcost(clf)
calcost(clf,1)

##5.Adaboost
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=1000, random_state=0,learning_rate=0.7)
calcost(clf,1)

#use svm as base classifier
clf = AdaBoostClassifier(SVC(probability=True,kernel='linear'),n_estimators=30,learning_rate=0.8, algorithm='SAMME')
calcost(clf)

trainpartF,trainpartL=dataLoading.divide(train)
clf = AdaBoostClassifier(n_estimators=1000, random_state=0,learning_rate=0.7)
cls=clf.fit(trainpartF,trainpartL)
testpartF,testpartL=dataLoading.divide(train)
prediction=cls.predict(testpartF)
prediction=pd.DataFrame(data=prediction)
cost, p211, p212, p121, p122 = myutils.one_hot_cost(prediction, testpartF, testpartL)
ratiogroup1 = p211 / p212
ratiogroup2 = p121 / p122
accuracy=myutils.accuracy(prediction, testpartL)
print(f"averaged cost is {cost}, accuracy is {accuracy}, ratio constraint 1 is {ratiogroup1}, ratio constraint 2 is {ratiogroup2}")
#predict on test.csv and save
prediction_testcsv=cls.predict(test)
pd.DataFrame(data=prediction_testcsv).to_csv('labels.csv',index=False,header=False)
print("labels.csv has been produced")

import matplotlib.pyplot as plt
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


print(train.head())

from sklearn.model_selection import learning_curve
plt=plot_learning_curve(clf, "learning curve", trainpartF, trainpartL, ylim=None, cv=None,n_jobs=None, train_sizes=np.linspace(.1, 1.0, 9))
plt.show()
train_sizes, train_scores, valid_scores = learning_curve(clf, trainpartF, trainpartL, train_sizes=[50, 80, 110], cv=5)

a=train[train.iloc[:,-1]==1]
a = a.drop(a.columns[-1], axis=1)
x=a.sample(n=1000)
x=np.mean(x)
b=train[train.iloc[:,-1]==2]
b = b.drop(b.columns[-1], axis=1)
y=b.sample(n=1000)
y=np.mean(y)
c=np.array([x.tolist(),y.tolist()])
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2,init=c)
# kmeans = KMeans(n_clusters=2)
possible=kmeans.fit(test).labels_
for i in range(len(possible)):
    if possible[i]==0:
        possible[i]=1
    else:
        possible[i]=2
possible=pd.DataFrame(data=possible)
cost, p211, p212, p121, p122 = myutils.one_hot_cost(prediction, test, possible)
print(f"averaged cost is {cost}, accuracy is {accuracy}, ratio constraint 1 is {ratiogroup1}, ratio constraint 2 is {ratiogroup2}")