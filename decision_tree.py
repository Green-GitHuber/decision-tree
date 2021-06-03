import numpy as np
import pandas as pd
import matplotlib as plt
from math import log

class tree():
    def __init__(self,f=0,t=0,left=0,right=0):
        self.f=f
        self.t=t
        self.left=left
        self.right=right
        
    def fit(self,dataset):
        fmax,feamax,tmax,data1,data2=ent1(dataset)
        if(fmax>=0.01):
            self.f=feamax
            self.t=tmax
            self.left=tree()
            self.left.fit(data1)
            self.right=tree()
            self.right.fit(data2)
        else:
            a,b,c=leaf(dataset)
            self.f=a
            self.t=b
            self.left=c
            self.right=-1

def load_data():
    a=pd.read_csv("D:/iris.data")
    for i in range(a.shape[0]):
        if a.iloc[i,4]=='Iris-setosa':
            a.iloc[i,4]=0
        elif a.iloc[i,4]=='Iris-versicolor':
            a.iloc[i,4]=1
        else:
            a.iloc[i,4]=2
    return a

def ent(label):
    t=np.zeros(3)
    label=np.array(label).reshape(1,-1)
    length=label.shape[1]
    for i in range(length):
        t[label[0,i]]+=1
    s=t.sum()
    f=0
    for i in range(3):
        if t[i]==0:
            d=0
        else:
            d=t[i]/s*log(t[i]/s)
        f-=d
    return f

def ent1(dataset):
    feature_num=dataset.shape[1]-1
    feature_len=dataset.shape[0]
    fmax=-10000
    tmax=0
    feamax=-1
    Ent=ent(dataset.iloc[:,4])
    if Ent!=0:
        for j in range(feature_num):
            data=dataset.sort_values(by=str(j), ascending=True)
            for i in range(1,feature_len):
                f=Ent-i/feature_len*ent(data.iloc[:i,4])-(feature_len-i)/feature_len*ent(data.iloc[i:,4])
                if(f>fmax):
                    fmax=f
                    feamax=j
                    tmax=(data.iloc[i-1,j]+data.iloc[i,j])/2
                    data1=data.iloc[:i,:]
                    data2=data.iloc[i:,:]
        return fmax,feamax,tmax,data1,data2
    else:
        return 0,0,0,0,0

def leaf(dataset):
    t=np.zeros(3)
    label=np.array(dataset.iloc[:,4]).reshape(1,-1)
    length=label.shape[1]
    for i in range(length):
        t[label[0,i]]+=1
    return np.argmax(t),t.max()/t.sum(),length

def print_tree(tree):
    if type(tree.right)!=int:
        print(tree.f,tree.t)
        print_tree(tree.left)
        print_tree(tree.right)
    else:
        print(tree.f,tree.t,tree.left)

if __name__ == '__main__':
    data=load_data()
    a=tree()
    a.fit(data)
    print_tree(a)