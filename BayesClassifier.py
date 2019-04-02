#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 20:08:20 2019

@author: mayanktomar
"""
import operator
from NaiveBayes import NaiveBayes, convertIntoFreqTable, createFeatureFreqTable, intersectionOfLists
import math as m
import numpy as np

# Global variables
Files = ["./2019S1-proj1-data/anneal.csv", "./2019S1-proj1-data/breast-cancer.csv", "./2019S1-proj1-data/car.csv",
                "./2019S1-proj1-data/cmc.csv", "./2019S1-proj1-data/hepatitis.csv", "./2019S1-proj1-data/hypothyroid.csv",
                "./2019S1-proj1-data/mushroom.csv", "./2019S1-proj1-data/nursery.csv", "./2019S1-proj1-data/primary-tumor.csv"]
'''
Files = ["./2019S1-proj1-data/car.csv"]
'''
FOLDS = 10

def main():    
    for fileName in Files:
        evaluateNBayesModel(fileName)
        

def evaluateNBayesModel(fileName):
    print('\n_________________________', fileName.split('/')[-1], '___________________________')
    
    file = open(fileName, "r")
    [dataSet, dataSet_X, dataSet_Y] = preprocess(file)
    
    # Traning and predicting the Naive Bayes with same data
    badTrainingNB(dataSet, dataSet_X, dataSet_Y)
    
    # Evaluating the data set with cross validation
    crossValidationEval(dataSet, dataSet_X, dataSet_Y)
    
    # Calculating the information gain and best choice for root attribute
    [informationGain, bestChoice] = info_gain(dataSet_X, dataSet_Y)
    
    print ('INFORMATION GAIN\n',informationGain)
    print ('The best choice for root node is attribute no. %d with the information gain of %f ' % (bestChoice[0], bestChoice[1]))

###############################################################################
################ PREPROCESSING THE FILE AND CREATING THE DATASET ##############
###############################################################################
def preprocess(file):
    dataSet = []
    testDataSet_X = []
    testDataSet_Y = []
    
    for line in file.readlines():
        elems = line.strip().split(',')
        dataSet.append(elems)
        testDataSet_X.append(elems[:-1])
        testDataSet_Y.append(elems[-1])
        
    return [dataSet, testDataSet_X, testDataSet_Y]

###############################################################################
################ EVALUATING BAD TRANING AND TESTING DATA ######################
###############################################################################
def badTrainingNB(dataSet, dataSet_X, dataSet_Y):
    
    nbClassifier = NaiveBayes()
    
    # Train the classifier
    nbClassifier.train(dataSet)
    
    # Predict the results
    predictedValues = nbClassifier.predict(dataSet_X)
  
    #Evaluate Results
    metrics = nbClassifier.evaluate(predictedValues, dataSet_Y)
    
    printMetrics(metrics, 'BAD TRAINING AND TESTING METRICS')
    
    del nbClassifier


###############################################################################
################ EVALUATING CROSS VALIDATION ##################################
###############################################################################
def crossValidationEval(dataSet, dataSet_X, dataSet_Y):
    print ('CROSS VALIDATION EVALUATION')
    
    nbClassifier = NaiveBayes()
    allMetrics = []
    
    # Producing the evaluation metrics for all the folds
    for fld in range(FOLDS):
        
        # Preparing train and test dataset for each hold-out/fold
        [trainData, testData, knownResults] = prepareCrossValidData(fld, 
        dataSet, dataSet_X, dataSet_Y)
            
        # Train the classifier
        nbClassifier.train(trainData)
        
        # Predict the results
        predictedValues = nbClassifier.predict(testData)
      
        #Evaluate Results
        metrics = nbClassifier.evaluate(predictedValues, knownResults)
        
        msg = 'Testing set %d:' % (fld+1)
        printMetrics(metrics, msg)
        
        allMetrics.append(metrics)
    
    avgMetrics = averageAllMetrics(allMetrics)    
    printMetrics(avgMetrics, 'AVERAGE OF ALL THE FOLDS')    
    del nbClassifier


def prepareCrossValidData(fld, dataSet, dataSet_X, dataSet_Y):
    
    trainData = []
    testData = []
    knownResults = []
    testSize = int(len(dataSet)/FOLDS)
    startIndex = fld*testSize    
    dataSize = len(dataSet_Y)
    
    if fld < (FOLDS-1):
        testIndex = list(range(startIndex, startIndex + testSize))
    else:
        testIndex = list(range(startIndex, dataSize))
        
    for i in range(dataSize):
        if i in testIndex:
            testData.append(dataSet_X[i])
            knownResults.append(dataSet_Y[i])
        else:
            trainData.append(dataSet[i])
    return [trainData, testData, knownResults]

def averageAllMetrics(allMetrics):
    
    sumAvgMetric= {'weightedAvg':[0.0,0.0,0.0], 'FScoreMacroAvg':0.0, 'accuracy':0.0}
    
    for metric in allMetrics:
        sumAvgMetric['weightedAvg'] = [sumAvgMetric['weightedAvg'][i] + metric['weightedAvg'][i] 
        for i in range(len(sumAvgMetric['weightedAvg']))]
        sumAvgMetric['FScoreMacroAvg'] += metric['FScoreMacroAvg']
        sumAvgMetric['accuracy'] += metric['accuracy']
    
    sumAvgMetric['weightedAvg'] = [sumAvgMetric['weightedAvg'][i]/FOLDS for i in range(len(sumAvgMetric['weightedAvg']))]
    sumAvgMetric['FScoreMacroAvg'] /= FOLDS
    sumAvgMetric['accuracy'] /= FOLDS

    return sumAvgMetric

def printMetrics(metrics, msg):
    print (msg)
    for metric in metrics.keys():
        print(metric, ': ', metrics[metric])
    print()

###############################################################################
################ CALCULATIONS FOR INFORMATION GAIN ############################
###############################################################################
def info_gain(attributes, label):
    
    labelFreq = convertIntoFreqTable(label)
    totalSize = len(label)
    
    # Calculating label entropy
    labelEnpy = 0.0
    for key in labelFreq.keys():
        labelProb = len(labelFreq[key]) / totalSize
        labelEnpy += -1*labelProb * m.log2(labelProb)

    features = createFeatureFreqTable(attributes)
    
    mean_info = []
    information_gain = {}
    attIndex = 1
    
    # Calculating the Entropy, mean-info and info-gain of all the features
    for attr in features:
        meanInfo = 0.0
        
        for att in attr.keys():
            entropy = 0.0
            
            for labl in labelFreq.keys():
                attrProb = intersectionOfLists(labelFreq[labl], attr[att]) / len(attr[att])
                if(attrProb != 0):
                    entropy += -1*attrProb * m.log2(attrProb)
                    
            meanInfo += entropy * len(attr[att])/totalSize
            
        mean_info.append(meanInfo)    
        information_gain[attIndex] = np.around((labelEnpy - meanInfo), decimals=5)
        attIndex += 1
        
    return [information_gain, max(information_gain.items(), key=operator.itemgetter(1))]

'''
QUESTION1:
  
    
QUESTION4:

* Reusing the same data for both training and testing 
is a bad idea because we need to know how
the method will work on data it wasn't trained on.    

'''

if __name__ == "__main__":
    main()