# coding: utf-8
import sys
import os
import random
import numpy
from numpy import *
import copy as copylist
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import csv
def loadData(filename):
    dataset = []
    csvfile = file(filename,'rb')
    reader = csv.reader(csvfile)
    row_num = 0
    for line in reader:
        if row_num == 0:
            row_num += 1
            continue
        dataset.append(line)
    csvfile.close()
    return dataset
#共Pclass，Sex, Age , SibSp, Parch, Embarked,儿童，母亲，名称，家庭数 10类特征。其中S计0，C记1，Q记2
#年龄离散化Age 以(0:16], (16:40],(40:99]离散化处理0,1,2

def preprocess(dataSet,sur_rate):

    L = len(dataSet)
    print '数据集长度为：',L
    target = [0 for i in range(L)]
    datamat = [[0 for j in range(10)] for i in range(L)]
    for i in range(L):
        datamat[i][0] = int (dataSet[i][2])
        if dataSet[i][4] == 'female': #女性的特征为1，男性为0
            datamat[i][1] = 0
        else :
            datamat[i][1] = 1
        if dataSet[i][5] == '':
            datamat[i][2] = 1
        age = float(dataSet[i][5])
        if(0<age and age<=16 ):
            datamat[i][2] = 0
        elif (16<age and age<=40 ):
            datamat[i][2] = 1
        elif (40<age and age<=99 ):
            datamat[i][2] = 2
        if (age<=16 and int(dataSet[i][7]) >0):
            datamat[i][6] = 1
        else :datamat[i][6] = 0
        title = get_title(dataSet[i][3])
        if (age>18 and int(dataSet[i][7]) >0 and dataSet[i][4] == 'female' and title != 'Miss'):
            datamat[i][7] = 1
        else :datamat[i][7] = 0

        datamat[i][3] = int (dataSet[i][6])
        datamat[i][4] = int (dataSet[i][7])
        if dataSet[i][11] == 'S':
            datamat[i][5] = 0
        if dataSet[i][11] == 'C':
            datamat[i][5] = 1
        if dataSet[i][11] == 'Q':
            datamat[i][5] = 2
        datamat[i][8] = replace_titles(dataSet[i][4],title)
        datamat[i][9] = int(dataSet[i][6]) + int(dataSet[i][7]) + 1
        '''if int(dataSet[i][2]) == 1:
            datamat[i][8] = sur_rate['c1']
        if int(dataSet[i][2]) == 2:
            datamat[i][8] = sur_rate['c2']
        if int(dataSet[i][2]) == 3:
            datamat[i][8] = sur_rate['c3']'''

        if dataSet[i][1] !='':
            target[i] = int(dataSet[i][1])
    return datamat,target
def shuffile_data(train_data,train_target):
    #洗牌
    r = random.randint(2147483647)
    random.seed(r)
    random.shuffle(train_data)
    random.seed(r)
    random.shuffle(train_target)
    print '洗牌完毕！'
    return train_data,train_target

def write_csv(x,filename): #写进csv文件
    Pid = 892
    csvfile = file(filename, 'wb')
    writer = csv.writer(csvfile)
    writer.writerow(['PassengerId','Survived'])
    for i in x:
        writer.writerow([Pid,i])
        Pid += 1
    csvfile.close()
    return
def get_title(str_title): #从姓名中获取标签
    start = str_title.index(',')
    end = str_title.index('.')
    title = str_title[start+2:end]
    return title
def replace_titles(s,title):#s为性别
    if title in ['Mr', 'Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 1#'Mr'
    elif title in ['Master']:
        return 4#'Master'
    elif title in ['Countess', 'Mme','Mrs']:
        return 2#'Mrs'
    elif title in ['Mlle', 'Ms','Miss']:
        return 3#'Miss'
    elif title == 'Dr':
        if s == 'male':
            return 1#'Mr'
        else:
            return 2#'Mrs'
    else:
        if s == 'male':
            return 4#'Master'
        else:
            return 3#'Miss'
def get_rate(dataSet):  #获得旅客等级的存活率
    Sur_rate = {'c1':0.0,'c2':0.0,'c3':0.0}
    rate1,rate2,rate3 = 0,0,0
    L = len(dataSet)
    for i in dataSet:
        if int(i[1]) == 1 and int(i[2]) == 3:
            rate3 += 1
        if int(i[1]) == 1 and int(i[2]) == 2:
            rate2 += 1
        if int(i[1]) == 1 and int(i[2]) == 1:
            rate1 += 1
    Sur_rate['c1'] = round(rate1*1.0/L,4)
    Sur_rate['c2'] = round(rate2*1.0/L,4)
    Sur_rate['c3'] = round(rate3*1.0/L,4)
    return Sur_rate

def LR_titanic(train_set,test_set,target):
    classifier = LogisticRegression(C=0.2, dual=False, fit_intercept=True,penalty='l2', tol=0.0001)
    classifier.fit(train_set,target)
    x = classifier.predict(test_set)
    write_csv(x,'result\LR.csv')
    return x
def RandomForest(train_set,test_set,target):
    classifier = RandomForestClassifier(n_estimators=12, max_depth=3)
    classifier.fit(train_set,target)
    x = classifier.predict(test_set)
    write_csv(x,'result\RF.csv')
    return x
def GBDT(train_set,test_set,target):
    classifier = GradientBoostingClassifier(n_estimators=11, subsample=0.5, max_depth=3)
    classifier.fit(train_set, target)
    x = classifier.predict(test_set)
    write_csv(x,'result\GBDT.csv')
    return
def DT(train_set,test_set,target):
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(train_set, target)
    x = classifier.predict(test_set)
    write_csv(x,'result\DT.csv')
    return
if __name__== '__main__':
    data_train,data_test,target,target0 = [],[],[],[]
    temp1,temp2,temp3,tar1,tar2,tar3 = [],[],[],[],[],[]
    traget_test = numpy.array([])
    sur_rate = {}
    train = loadData('train\\train.csv')
    test = loadData('test\\test.csv')
    sur_rate = get_rate(train)
    data_train,target = preprocess(train,sur_rate)
    temp1 = copylist.deepcopy(data_train)
    tar1 = copylist.deepcopy(target)
    temp2 = copylist.deepcopy(data_train)
    tar2 = copylist.deepcopy(target)
    temp3 = copylist.deepcopy(data_train)
    tar3 = copylist.deepcopy(target)
    temp1.extend(temp2)
    tar1.extend(tar2)
    temp1.extend(temp3)
    tar1.extend(tar3)

    data_train.extend(temp1)
    target.extend(tar1)
    data_train,target = shuffile_data(data_train,target)

    data_test,target0 = preprocess(test,sur_rate)

    LR_titanic(data_train,data_test,target)
    RandomForest(data_train,data_test,target)
    DT(data_train,data_test,target)
    GBDT(data_train,data_test,target)