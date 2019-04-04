#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 20:05:07 2019

@author: Mayank Tomar

Note: The following implementation of the Naive Bayes is NOT created with a
efficiency in the mind. This class was created for an educational purpose and
to study various aspects of the Naive Bayes model and algorythms and methods. 
Therefore, one may find the implementation of the class to be long and ineffiecient.
"""

import operator
import numpy as np
import math
np.seterr(all='ignore')

###############################################################################
###############################################################################
################ NAIVE BAYES CLASS TO CREATE A CLASSIFIER #####################
###############################################################################
###############################################################################
class NaiveBayes:
    
    def __del__(self):
        pass
    
    def train(self, dataSet):
        self.dataSize = len(dataSet)
        self.epsilon = 0.1/self.dataSize
        self.parseDataSet(dataSet)
        
# PREPARE THE DATA AND CREATE FREQUENCY MATRIX WHEN THE CLASS IS CREATED  
    def parseDataSet(self, dataSet):
        
        dataX = []
        dataY = []
        
        for line in dataSet:
            dataX.append(line[:-1])
            dataY.append(line[-1:][0])
            
        self.features = createFeatureFreqTable(dataX)
        self.labels = convertIntoFreqTable(dataY) 
        

#  CALCULATE PRIOR PROBABULITY P(LABELS)
    def calculatePriorProb(self, label):
        return len(self.labels[label]) / self.dataSize

#  CALCULATE POSTERIOR PROBABILITY P(FEATURES | LABEL)
    def calculatePosteriorProb(self, feature, label):
        
        prob = 1
        
        labelcount = len(self.labels[label])
        
        for index in range(len(feature)):
            if ((feature[index] != '?') and (feature[index] in self.features[index])):
                intersectionCount = intersectionOfLists(self.labels[label], 
                                                        self.features[index][feature[index]])
                
                # Apply epsilon smoothing
                if (intersectionCount == 0):
                    prob *= self.epsilon
                else:
                    prob *= float(intersectionCount / labelcount)
                    
        return prob

# PREDICT THE LABELS AND RETURN THE RESULT
    def predict(self, testDataSet):
        
        result = []
        
        for line in testDataSet:
            probability = {}
            
            for label in self.labels.keys():
                postProb = self.calculatePosteriorProb(line, label)
                probability[label] = self.calculatePriorProb(label) * postProb
                
            result.append(findMaxValueKey(probability))
           
        return result

# EVALUATES THE MODEL AND CALCULATES THE PRECISION, RECALL, F-SCORE, ACCURACY
    def evaluate(self, predicted, actual):
        
        precision = []
        recall = [] 
        fScore = []
        
        cMatrix = createConfusionMatrix(predicted, actual)
        classSize = len(cMatrix)

        # Calculate precision
        TPFP = np.sum(cMatrix, axis=0)
        TPFN = np.sum(cMatrix, axis=1)
        
        for i in range(classSize):
            
            if TPFP[i] == 0:
                preci = 0.0
            else:
                preci = cMatrix[i][i]/TPFP[i]
            
            if TPFN[i] == 0:
                rcal = 0.0
            else:
                rcal = cMatrix[i][i]/TPFN[i]                
            
            precision.append(np.around(preci, decimals=5))
            recall.append(np.around(rcal, decimals=5))
            
            if preci+rcal != 0:
                fScr = 2*preci*rcal/(preci+rcal)
            else:
                fScr = 0.0
            
            if math.isnan(fScr):
                fScr = 0.0
            
            fScore.append(np.around(fScr, decimals=5))
            
        metrics = {'weightedAvg':[]}
        metrics['weightedAvg'].append(np.around(np.sum(precision)/classSize, decimals=5))
        metrics['weightedAvg'].append(np.around(np.sum(recall)/classSize, decimals=5))
        metrics['weightedAvg'].append(np.around(np.sum(fScore)/classSize, decimals=5))
        
        metrics['FScoreMacroAvg'] = np.around(2 * metrics['weightedAvg'][0] * metrics['weightedAvg'][1]/(metrics['weightedAvg'][0] + metrics['weightedAvg'][1]), decimals=5)
        metrics['accuracy'] = np.around(np.sum(cMatrix.diagonal()) / np.sum(cMatrix), decimals=5)
       
        return metrics
    
###############################################################################
###############################################################################
################ HELPER FUNCTIONS FOR THE NAIVE BAYES CLASS ###################
###############################################################################
###############################################################################
def intersectionOfLists(list1, list2):
    return len(set(list1) & set(list2))

# CREATE THE FREQUENCY TABLE AND RETURN AS A DICT.
def convertIntoFreqTable(dataSet):

    freqTable = {}
    
    for i in range(len(dataSet)):
        if dataSet[i] not in freqTable:
            freqTable[dataSet[i]] = []
            
        freqTable[dataSet[i]].append(i)
        
        # removing empty('?') values from the list
        if '?' in freqTable.keys():
            del freqTable['?']
            
    return freqTable

#  CREATE FREQUENCY TABLES FOR FEATURES FOR LATER CALCULATIONS
def createFeatureFreqTable(dataSet):
    
    features = []
    featureData = []
    featuresCount = len(dataSet[0])
    
    for i in range(featuresCount):
        featureData.append([])
    
    for line in dataSet:
        for i in range(featuresCount):
            featureData[i].append(line[i])
    
    for i in range(featuresCount):
        features.append(convertIntoFreqTable(featureData[i]))
    
    return features

# CREATE THE CONFUSION MATRIX FOR THE EVALUATION PROCESS
def createConfusionMatrix(predicted, actual):
    classIndex = convertIntoFreqTable(actual)
    
    index = 0
    for key in classIndex.keys():
        classIndex[key] = index
        index += 1
        
    classSize = len(classIndex)
    confusionMatrix = np.zeros(shape=(classSize, classSize))
    
    size = len(actual)
    for i in range(size):
        if predicted[i] in classIndex.keys() and actual[i] in classIndex.keys():
            pIndex = classIndex[predicted[i]]
            aIndex = classIndex[actual[i]]
            confusionMatrix[aIndex][pIndex] += 1
    
    return confusionMatrix

def findMaxValueKey(dictinary):
        return max(dictinary.items(), key=operator.itemgetter(1))[0]