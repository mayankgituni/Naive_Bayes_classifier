#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Fri Mar 22 20:08:20 2019

@author: Mayank Tomar

--------------------------------------------------------------------------------------
QUESTION 1:
    
Yes, there is an observable correlation between the information-gain(IG) and the 
F-score. Since IG is a measure of how much information a feature contains about 
the class, the higher the IG, the more effectively the Naive Bayes classifier 
predicts the label. To examine, let's calculate the average IG of attributes 
and the F-score for each .csv file and plot it on a scatter plot as shown in 
the 'Figure-1' of the output. The higher IG of attributes indicates there are 
low mean-information and entropy per attribute. The lower entropy of the attribute 
shows that most of the probability mass is assigned to a single label. This makes 
the event more predictable; it also contributes to a higher value of posterior 
probability which helps the classifier predict an event with higher certainty 
and accuracy.

For examples:
    1. 'hypothyroid.csv' has the lowest (F-score=0.4877 and avg-Info-Gain=0.002),
    2. 'car.csv' has a mediocre (F-score=0.7223 and avg-Info-Gain=0.1144),
    3. 'mushroom.csv' has the highest (F-score=0.9865 and avg-Info-Gain=0.1996). 

Except a few exceptions such as:
    1. 'anneal.csv' has a high F-score=0.98344 and low avg-Info-Gain=0.08822
    2. 'primary-tumor.csv' has a Low F-score=0.6585 and high avg-info-Gain=0.08822

The exceptions can be explained using additional constraints i.e. number of 
attributes and number of distinct labels. This means the dataset with a lower 
number of attributes and high IG is not very effective when predicting many 
distinct labels. A better strategy would be to calculate
"Scaled-Info-Gain = avg-Info-Gain * no. of Attributes / no. of distinct labels". 
When we plot the scaled-IG Vs F-score(Figure-2), we can observe that the 
datasets (Including exceptional) exhibits the expected relationship. Therefore, it 
can be concluded that the information-gain is directly proportional to the F-score. 
-------------------------------------------------------------------------------------
    
QUESTION 4:

When we compare the effectiveness of the model on the same training/testing 
dataset Vs cross-validation evaluation strategy, we can observe the accuracy, 
F-score and other metrics of the model substantially decreasing. The changes 
can be validated by the following examples when both the strategies are applied 
to the model:

1. The F-score of 'primary-tumor.csv' changed from 0.65845 to 0.42892 
when evaluated under cross-validation.
2. Similarly, F-score of ‘car.csv’ changed from 0.72252 to 0.599255

The dramatic change in the results occurs due to overfitting of the data. 
Reusing the same data for both training and testing is a bad idea because the 
classifier is prone to create a bias towards the results that it has seen 
during the training. This makes the prediction of the labels seemingly more 
accurate but makes the evaluation process unreliable. A better evaluation 
strategy would be to use cross-validation which can provide a meticulous 
estimation of the accuracy. In the example above, the evaluation metrics were 
calculated by splitting the dataset into smaller chunks(Folds=10) which are 
used to train, test and evaluate until all the datasets have participated in 
the testing cycle. The rigorous process leaves less room for the classifier 
to create biases towards labels and making the accuracy seemingly less accurate 
but more reliable. Hence, the estimation of the effectiveness of a model can 
seem to change from more accurate to less accurate.
------------------------------------------------------------------------------------
'''

import operator
from NaiveBayes import NaiveBayes, convertIntoFreqTable, createFeatureFreqTable, intersectionOfLists
import math as m
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2

# Global variables
Files = ["./2019S1-proj1-data/anneal.csv", "./2019S1-proj1-data/breast-cancer.csv", "./2019S1-proj1-data/car.csv",
                "./2019S1-proj1-data/cmc.csv", "./2019S1-proj1-data/hepatitis.csv", "./2019S1-proj1-data/hypothyroid.csv",
                "./2019S1-proj1-data/mushroom.csv", "./2019S1-proj1-data/nursery.csv", "./2019S1-proj1-data/primary-tumor.csv"]
'''
Files = ["./2019S1-proj1-data/playGolf.csv"]
'''
FOLDS = 10
infoGain = []
FScore = []
ScaledInfoGain = []

def main():    
    for fileName in Files:
        evaluateNBayesModel(fileName)
    
    print('\nF1-score: ', FScore)    
    plt1.scatter(infoGain, FScore)
    plt1.title('Figure-1: Info-Gain Vs F1-Score')
    plt1.xlabel('Avg. Info-Gain')
    plt1.ylabel('FScore')
    plt1.grid(True)
    plt1.show()

    print('Avg. InfoGain: ', infoGain)
    
    plt2.scatter(ScaledInfoGain, FScore)
    plt2.title('Figure-2: Scaled Info-Gain Vs F1-Score')
    plt2.xlabel('Scaled Info-Gain')
    plt2.ylabel('FScore')
    plt2.grid(True)
    plt2.show()
    print('Scaled Info-Gain', ScaledInfoGain)

def evaluateNBayesModel(fileName):
    print('\n_________________________', fileName.split('/')[-1], '___________________________')
    
    file = open(fileName, "r")
    [dataSet, dataSet_X, dataSet_Y] = preprocess(file)
    
    global attrNo    
    attrNo = len(dataSet_X[0])
    # Traning and predicting the Naive Bayes with same data
    badTrainingNB(dataSet, dataSet_X, dataSet_Y)
    
    # Evaluating the data set with cross validation
    crossValidationEval(dataSet, dataSet_X, dataSet_Y)

    # Calculating the information gain and best choice for root attribute
    [informationGainDist, bestChoice] = info_gain(dataSet_X, dataSet_Y)
    
    print('No. of Attributes: ', attrNo,'\nNo. of labels: ', labelNo)
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
    FScore.append(metrics['FScoreMacroAvg'])
    
    printMetrics(metrics, 'SAME TRAINING AND TESTING METRICS')
    
    del nbClassifier


###############################################################################
################ EVALUATING CROSS VALIDATION ##################################
###############################################################################
def crossValidationEval(dataSet, dataSet_X, dataSet_Y):
    
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
        
        allMetrics.append(metrics)
    
    avgMetrics = averageAllMetrics(allMetrics)    
    printMetrics(avgMetrics, 'CROSS VALIDATION EVALUATION')    
    del nbClassifier

# PREPARE THE DATA FOR CROSS VALIDATION EVALUATION STRATEGY
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

# AVERAGE OF ALL THE ITERATIONS OF RESULTS FROM CROSS VALIDATION METHOD
def averageAllMetrics(allMetrics):
    
    sumAvgMetric= {'weightedAvg':[0.0,0.0,0.0], 'FScoreMacroAvg':0.0, 'accuracy':0.0}
    for metric in allMetrics:
        sumAvgMetric['weightedAvg'] = [sumAvgMetric['weightedAvg'][i] + metric['weightedAvg'][i] 
        for i in range(len(sumAvgMetric['weightedAvg']))]
        
        if not m.isnan(metric['FScoreMacroAvg']):
            sumAvgMetric['FScoreMacroAvg'] += metric['FScoreMacroAvg']
        
        if not m.isnan(metric['accuracy']):
            sumAvgMetric['accuracy'] += metric['accuracy']
    
    sumAvgMetric['weightedAvg'] = [sumAvgMetric['weightedAvg'][i]/FOLDS for i in range(len(sumAvgMetric['weightedAvg']))]
    sumAvgMetric['FScoreMacroAvg'] /= FOLDS
    sumAvgMetric['accuracy'] /= FOLDS
    
    return sumAvgMetric

# A METHOD TO PRINT MATRICS WITH BETTER STRUCTURE
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
    totalSize = calcSize(labelFreq)
    labelEnpy = 0.0    

    # Calculating label entropy H(R)
    for key in labelFreq.keys():
        labelProb = len(labelFreq[key]) / totalSize
        labelEnpy += -1*labelProb * m.log2(labelProb)
    
    mean_info = []
    infoGainList = []
    information_gain = {}
    attIndex = 1
    features = createFeatureFreqTable(attributes)
    
    # Calculating the Entropy, mean-info and info-gain of all the features
    for attr in features:
        meanInfo = 0.0
        featureSize = calcSize(attr)

        for att in attr.keys():
            entropy = 0.0
            
            for labl in labelFreq.keys():
                attrProb = intersectionOfLists(labelFreq[labl], attr[att]) / len(attr[att])
                
                if(attrProb != 0):
                    entropy += -1*attrProb * m.log2(attrProb)
                   
            meanInfo += entropy * len(attr[att])/featureSize
            
        mean_info.append(meanInfo)    
        information_gain[attIndex] = np.around((labelEnpy - meanInfo), decimals=5)
        infoGainList.append(labelEnpy - meanInfo)
        attIndex += 1
    
    global labelNo
    labelNo = len(labelFreq.keys())
    
    infoGainDistribution = createClassDistribution(infoGainList)
    
    return [infoGainDistribution, max(information_gain.items(), key=operator.itemgetter(1))]

def calcSize(attributes):
    size = 0
    for attr in attributes:
        size += len(attributes[attr])

    return size
# CREATING THE CLASS DISTRIBUTION (MEAN, MEDIAN, STDEV) OF INFO-GAIN
def createClassDistribution(infoGainList):
    infoGainList.sort()
    
    # Calculating the distribution of class info gain
    distribution = {}
    distribution['mean'] = np.around(stats.mean(infoGainList), decimals=5)
    distribution['median'] = np.around(stats.median(infoGainList), decimals=5)
    distribution['stdev'] = np.around(stats.stdev(infoGainList), decimals=5)
    infoGain.append(distribution['mean'])
    
    # Calculating the relative Information gain
    ScaledInfoGain.append(distribution['mean']* attrNo /labelNo)
    
    return distribution

# THE MAIN FUNCTION
if __name__ == "__main__":
    main()