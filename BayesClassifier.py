#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 20:08:20 2019

@author: mayanktomar
"""

from NaiveBayes import NaiveBayes
#import math
#import numpy as np


def preprocess(fileName, trainRatio):
    file = open(fileName, "r")
    dataSet = []
    for line in file.readlines():
        elems = line.strip().split(',')
        dataSet.append(elems)
    
    trainDataSet = dataSet[0:int(len(dataSet)*trainRatio)]
    testDataSet = dataSet[int(len(dataSet)*trainRatio) :]
    return [trainDataSet, testDataSet]

def train(trainData):
    return NaiveBayes(trainData)    
    
def main():
    fileName = "./2019S1-proj1-data/playGolf.csv"
    [trainData, testData] = preprocess(fileName, 0.8)
    
    classifier = train(trainData)
    
    classifier.predict(testData)
    
if __name__ == "__main__":
    main()
    