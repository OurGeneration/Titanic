# coding: utf-8
import sys
import os
import csv
import random
from numpy import *
from sklearn import tree
from Titanic import loadData
from Titanic import shuffile_data
from sklearn.tree import DecisionTreeClassifier
def data_pre(dataSet):
    L = len(dataSet)
    print '数据集长度为：',L
    target = [0 for i in range(L)]
    datamat = [[0 for j in range(6)] for i in range(L)]
    for i in range(L):
        datamat[i][0] = int (dataSet[i][2])
        if dataSet[i][4] == 'female': #女性的特征为1，男性为0
            datamat[i][1] = 0
        else :
            datamat[i][1] = 1
        datamat[i][2] = int (dataSet[i][1]) #将年龄属性作为类别，存活作为属性,DT预测年龄
        datamat[i][3] = int (dataSet[i][6])
        datamat[i][4] = int (dataSet[i][7])
        if dataSet[i][11] == 'S':
            datamat[i][5] = 0
        if dataSet[i][11] == 'C':
            datamat[i][5] = 1
        if dataSet[i][11] == 'Q':
            datamat[i][5] = 2
        if dataSet[i][5] !='':
            age = float(dataSet[i][5])
            if(0<age and age<=16 ):
                target[i] = 0
            elif (16<age and age<=40 ):
                target[i] = 1
            elif (40<age and age<=99 ):
                target[i] = 2
    return datamat,target
def DT(train_set,test_set,target):
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(train_set, target)
    x = classifier.predict(test_set)
    write_csv(x,'result\\age.csv')
    return
def write_csv(x,filename):
    Pid = 1
    csvfile = file(filename, 'wb')
    writer = csv.writer(csvfile)
    writer.writerow(['id','age'])
    for i in x:
        if i == 0:
            writer.writerow([Pid,15])
        if i == 1:
            writer.writerow([Pid,30])
        if i == 2:
            writer.writerow([Pid,50])
        Pid += 1
    csvfile.close()
    return
if __name__ == '__main__':
    data_train,data_test,target,target0 = [],[],[],[]
    train = loadData('train\\train_remain.csv')
    test = loadData('train\\train_am.csv')
    #data_train,target = data_pre(train)
    #data_train,target = shuffile_data(data_train,target)

    #data_test,target0 = data_pre(test)
    #DT(data_train,data_test,target)