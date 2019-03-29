#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 20:08:20 2019

@author: mayanktomar
"""

from NaiveBayes import NaiveBayes
#import math
#import numpy as np

import random

# Global variables
itrPerRatio = 1

def randomAccessPreprocess(file, trainRatio):
    
    dataSet = []
    trainDataSet = []
    testDataSet = []
    knownResult = []
    
    for line in file.readlines():
        elems = line.strip().split(',')
        dataSet.append(elems)
    
    dataSize = len(dataSet)
    testIndex = random.sample(range(0, dataSize), int(dataSize * (1-trainRatio)))
    
    for i in range(dataSize):
        if i in testIndex:
            testDataSet.append(dataSet[i][:-1])
            knownResult.append(dataSet[i][-1])
        else:
            trainDataSet.append(dataSet[i])
    
#    trainDataSet = dataSet[0:int(len(dataSet)*trainRatio)]
#    testDataSet = dataSet[int(len(dataSet)*trainRatio) :]
    
    return [dataSet, trainDataSet, testDataSet, knownResult]

def createBayesModel(fileName, fileRatio):
    print('___________', fileName.split('/')[-1], '_____________')
    nbClassifier = NaiveBayes()
    
    file = open(fileName, "r")
    [dataSet, trainData, testData, knownResult] = randomAccessPreprocess(file, fileRatio)
    
    # Train the classifier
    nbClassifier.train(trainData)
    # Predict the results
    predictedValues = nbClassifier.predict(testData)
  
    #Evaluate Results
    metrics = nbClassifier.evaluate(predictedValues, knownResult)

        
#    print('Accuracy(%s): %f' % (fileName.split('/')[-1] ,(count / len(testData))))
        
def main():
    Files = ["./2019S1-proj1-data/anneal.csv", "./2019S1-proj1-data/breast-cancer.csv", "./2019S1-proj1-data/car.csv",
                "./2019S1-proj1-data/cmc.csv", "./2019S1-proj1-data/hepatitis.csv", "./2019S1-proj1-data/hypothyroid.csv",
                "./2019S1-proj1-data/mushroom.csv", "./2019S1-proj1-data/nursery.csv", "./2019S1-proj1-data/primary-tumor.csv"]
#    Files = ["./2019S1-proj1-data/car.csv"]
    for fileName in Files:
#        ratio = [0.5, 0.6, 0.7, 0.8, 0.9]
        ratio = 0.8
        createBayesModel(fileName, ratio)


if __name__ == "__main__":
    main()
    