#!usr/bin/python
#-*- coding: utf-8 -*-

#http://blog.csdn.net/u012162613/article/details/41978235

from numpy import *
import csv

def toInt(arr):
    arr = mat(arr)
    m,n = shape(arr)
    newArray = zeros((m,n))
    for i in range(m):
        for j in range(n):
            newArray[i,j] = int(arr[i,j])
    return newArray

def normalizing(arr):
    m,n=shape(arr)
    for i in range(m):
        for j in range(n):
            if arr[i,j] != 0:
                arr[i,j]=1
    return arr

def loadTrainData():
    l=[]
    with open('train.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    l=array(l)
    label=l[:,0]
    data = l[:,1:]
    return normalizing(toInt(data)),toInt(label)

def loadTestData():
    l=[]
    with open('test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    data = array(l)
    return normalizing(toInt(data))

def saveResult(result,csvName):
    with open(csvName,'wb') as myFile:
        myWriter = csv.writer(myFile)
        count = 1
        for i in result:
            tmp=[]
            tmp.append(i)
            myWriter.writerow(tmp)
            count += 1

from sklearn.neighbors import KNeighborsClassifier
def knnClassify(trainData, trainLabel, testData):
    knnClf = KNeighborsClassifier()
    knnClf.fit(trainData, ravel(trainLabel))        # ravel转换成行向量
    testLabel = knnClf.predict(testData)
    saveResult(testLabel,'sklearn_knn_result.csv')
    return testLabel

def digitRecognition():
    trainData, trainLabel = loadTrainData()
    testData = loadTestData()
    result = knnClassify(trainData,trainLabel,testData)
    print("Done!")

if __name__ == '__main__':
    digitRecognition()
