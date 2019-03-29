#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 20:05:07 2019

@author: mayanktomar
"""

import operator
import numpy as np
from decimal import Decimal, getcontext
getcontext().prec = 5
###############################################################################
###############################################################################
################ HELPER FUNCTIONS FOR THE NAIVE BAYES CLASS ###################
###############################################################################
###############################################################################
def intersectionOfLists(list1, list2):
    return len(set(list1) & set(list2))

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

def createConfusionMatrix(predicted, actual):
    classIndex = convertIntoFreqTable(predicted)
    
    index = 0
    for key in classIndex.keys():
        classIndex[key] = index
        index += 1
        
    print(classIndex)
    classSize = len(classIndex)
    confusionMatrix = np.zeros(shape=(classSize, classSize))
    
    size = len(predicted)
    for i in range(size):
        if predicted[i] in classIndex.keys() and actual[i] in classIndex.keys():
            pIndex = classIndex[predicted[i]]
            aIndex = classIndex[actual[i]]
            confusionMatrix[aIndex][pIndex] += 1
    
    return confusionMatrix

def findMaxValueKey(dictinary):
        return max(dictinary.items(), key=operator.itemgetter(1))[0]

###############################################################################
###############################################################################
################ NAIVE BAYES CLASS TO CREATE A CLASSIFIER #####################
###############################################################################
###############################################################################
class NaiveBayes:
    
    def train(self, dataSet):
        self.dataSize = len(dataSet)
        self.epsilon = 0.1/self.dataSize
        self.parseDataSet(dataSet)
        
    def parseDataSet(self, dataSet):
        
        dataX = []
        dataY = []
        
        for line in dataSet:
            dataX.append(line[:-1])
            dataY.append(line[-1:][0])
            
        self.createFeatureFreqTable(dataX)
        self.createLabelFreqTable(dataY)
    
#  CREATE FREQUENCY TABLES FOR FEATURES
    def createFeatureFreqTable(self, dataSet):
        
        self.features = []
        featureData = []
        featuresCount = len(dataSet[0])
        
        for i in range(featuresCount):
            featureData.append([])
            
        for line in dataSet:
            for i in range(featuresCount):
                featureData[i].append(line[i])
                
        for i in range(featuresCount):
            self.features.append(convertIntoFreqTable(featureData[i]))
            
#  CREATE FREQUENCY TABLES FOR LABLES
    def createLabelFreqTable(self, dataSet):
        self.labels = convertIntoFreqTable(dataSet)     

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
    
    def predict(self, testDataSet):
        
        result = []
        
        for line in testDataSet:
            probability = {}
            
            for label in self.labels.keys():
                postProb = self.calculatePosteriorProb(line, label)
                probability[label] = self.calculatePriorProb(label) * postProb
                
            result.append(findMaxValueKey(probability))
           
        return result
    
    def evaluate(self, predicted, actual):
        
        metrics = {'precision':[], 'recall':[], 'fScore':[], 'weightedAvg':[]}
        cMatrix = createConfusionMatrix(predicted, actual)
        classSize = len(cMatrix)
        print(cMatrix)
        # Calculate precision
        TPFP = np.sum(cMatrix, axis=0)
        TPFN = np.sum(cMatrix, axis=1)
        
        for i in range(classSize):
            precision = cMatrix[i][i]/TPFP[i]
            recall = cMatrix[i][i]/TPFN[i]
            metrics['precision'].append(np.around(precision, decimals=5))
            metrics['recall'].append(np.around(recall, decimals=5))
            metrics['fScore'].append(np.around(2*precision*recall/(precision+recall),
                   decimals=5))
        
        metrics['weightedAvg'].append(np.around(np.sum(metrics['precision'])/classSize, decimals=5))
        metrics['weightedAvg'].append(np.around(np.sum(metrics['recall'])/classSize, decimals=5))
        metrics['weightedAvg'].append(np.around(np.sum(metrics['fScore'])/classSize, decimals=5))
        
        metrics['macroAvg'] = np.around(2 * metrics['weightedAvg'][0] * metrics['weightedAvg'][1]/(metrics['weightedAvg'][0] + metrics['weightedAvg'][1]), decimals=5)
        metrics['accuracy'] = np.around(np.sum(cMatrix.diagonal()) / np.sum(cMatrix), decimals=5)
        print(metrics)
        return metrics
    