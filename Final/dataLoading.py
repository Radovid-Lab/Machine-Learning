import os
import pandas as pd

def readData(path):
    try:
        train = pd.read_csv(path, header=None)
        print("reading csv")
    except:
        print("no such file")
    return train

def divide(train):
    train_raw = train.iloc[:, 0:len(train.columns)-1]
    label = train.iloc[:, len(train.columns)-1].to_frame()
    return train_raw,label