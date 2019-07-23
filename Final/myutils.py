import pandas as pd

def one_hot_cost(prediction,testsetF,testsetL):
    p211=p212=p121=p122=0
    p11=p12=p21=p22=0
    for i in range(len(prediction)):
        if testsetF.iloc[i,101]==1 and testsetL.iloc[i,0]==1:
            if prediction.iloc[i, 0] == 2:
                p211+=1
            p11+=1
        if testsetF.iloc[i,102]==1 and testsetL.iloc[i,0]==1:
            if prediction.iloc[i, 0] == 2:
                p212+=1
            p12+=1

        if testsetF.iloc[i, 101] == 1 and testsetL.iloc[i, 0] == 2:
            if prediction.iloc[i, 0] == 1:
                p121 += 1
            p21+=1
        if testsetF.iloc[i, 102] == 1 and testsetL.iloc[i, 0] == 2:
            if prediction.iloc[i, 0] == 1:
                p122 += 1
            p22+=1

    cost=3*max(p211/p11,p212/p12)+max(p121/p21,p122/p22)
    return cost,p211/p11,p212/p12,p121/p21,p122/p22

def cost(prediction,testsetF,testsetL):
    p211=p212=p121=p122=0
    p11=p12=p21=p22=0
    for i in range(len(prediction)):
        if testsetF.iloc[i,12]==1 and testsetL.iloc[i,0]==1:
            if prediction.iloc[i, 0] == 2:
                p211+=1
            p11+=1
        if testsetF.iloc[i,12]==2 and testsetL.iloc[i,0]==1:
            if prediction.iloc[i, 0] == 2:
                p212+=1
            p12+=1

        if testsetF.iloc[i, 12] == 1 and testsetL.iloc[i, 0] == 2:
            if prediction.iloc[i, 0] == 1:
                p121 += 1
            p21+=1
        if testsetF.iloc[i, 12] == 2 and testsetL.iloc[i, 0] == 2:
            if prediction.iloc[i, 0] == 1:
                p122 += 1
            p22+=1

    cost=3*max(p211/p11,p212/p12)+max(p121/p21,p122/p22)
    return cost,p211/p11,p212/p12,p121/p21,p122/p22

def accuracy(prediction,truelabel):
    count=0
    for i in range(len(prediction)):
        if prediction.iloc[i, 0] == truelabel.iloc[i,0]:
            count+=1
    return count/len(prediction)

def crossValidation():
    return