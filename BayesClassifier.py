#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 20:08:20 2019

@author: mayanktomar
"""

from NaiveBayes import NaiveBayes
#import math
#import numpy as np

import operator
import random

def preprocess(fileName, trainRatio):
    
    file = open(fileName, "r")
    dataSet = []
    trainDataSet = []
    testDataSet = []
    
    for line in file.readlines():
        elems = line.strip().split(',')
        dataSet.append(elems)
    
    dataSize = len(dataSet)
    trainIndex = random.sample(range(0, dataSize), int(dataSize * trainRatio))
    
    for i in range(dataSize):
        if i in trainIndex:
            trainDataSet.append(dataSet[i])
        else:
            testDataSet.append(dataSet[i])
#    trainDataSet = dataSet[0:int(len(dataSet)*trainRatio)]
#    testDataSet = dataSet[int(len(dataSet)*trainRatio) :]
    return [trainDataSet, testDataSet]

def train(trainData):
    return NaiveBayes(trainData)    

def calcAccuracy(testData, result):
    totalCorrect = 0
    for i in range(len(testData)):
        if testData[i][-1] == result[i]:
            totalCorrect += 1
    return totalCorrect/len(testData)

def createBayesModel(fileName, fileRatio):
    
    accuracy = {}
    
    for i in range(len(fileRatio)):
        [trainData, testData] = preprocess(fileName, fileRatio[i])
        classifier = train(trainData)
        #classifier.laplacianSmooth(True)
        result = classifier.predict(testData)
        accuracy[fileRatio[i]] = calcAccuracy(testData, result)
    
    accuracy['Best'] = max(accuracy.items(), key=operator.itemgetter(1))
    
    return accuracy
    
def main():
    Files = ["./2019S1-proj1-data/anneal.csv", "./2019S1-proj1-data/breast-cancer.csv", "./2019S1-proj1-data/car.csv",
                "./2019S1-proj1-data/cmc.csv", "./2019S1-proj1-data/hepatitis.csv", "./2019S1-proj1-data/hypothyroid.csv",
                "./2019S1-proj1-data/mushroom.csv", "./2019S1-proj1-data/nursery.csv"]
    for fileName in Files:
        ratio = [0.5, 0.6, 0.7, 0.8, 0.9]
        accuracy = createBayesModel(fileName, ratio)  
        print(fileName)
        print('Accuracy: ', accuracy)
    
if __name__ == "__main__":
    main()
    