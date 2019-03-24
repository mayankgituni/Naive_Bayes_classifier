#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 20:05:07 2019

@author: mayanktomar
"""

import operator

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

def findMaxValueKey(dictinary):
        return max(dictinary.items(), key=operator.itemgetter(1))[0]

###############################################################################
###############################################################################
################ NAIVE BAYES CLASS TO CREATE A CLASSIFIER #####################
###############################################################################
###############################################################################
class NaiveBayes:
    def __init__(self, dataSet):
        self.lSmoothing = False
        self.laplacianUsed = False
        self.dataSize = len(dataSet)
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
        featureData = [[]*len(dataSet[0])]
        featuresCount = len(dataSet[0])
        
        for i in range(featuresCount-1):
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
        epsilon = 0.1 / self.dataSize
        labelcount = len(self.labels[label])
        
        for index in range(len(feature)):
            if ((feature[index] != '?') ):
                if (feature[index] in self.features[index]):
                    intersectionCount = intersectionOfLists(self.labels[label], 
                                                            self.features[index][feature[index]])
                else:
                    intersectionCount = 0.0
                        
                p = float(intersectionCount / labelcount)
                
                # Apply epsilon smoothing
                if ((p == 0.0) and (not self.lSmoothing)):
                    prob *= epsilon
                elif ((p == 0.0) and self.lSmoothing):
                    break
                else:
                    prob *= p   
        
        # Apply laplacian smoothing
        if(prob == 0.0 and self.lSmoothing):
            prob = self.laplacianSmoothing(feature, label)
            self.laplacianUsed = True
            
        return prob 
    
# Calculating the probability of features using laplacian smoothing
    def laplacianSmoothing(self, feature, label):
        
        prob = 1
        labelcount = len(self.labels[label])
        
        for index in range(len(feature)):
            if (feature[index] != '?'):
                if (feature[index] in self.features[index]):
                    intersectionCount = intersectionOfLists(self.labels[label], 
                                                        self.features[index][feature[index]])
                else:
                    intersectionCount = 0.0
                
                p = float((intersectionCount+0.5) / (labelcount + len(self.features[index])))    
                prob *= p
                
        return prob

# Turn on/off the laplacian smoothing option
    def laplacianSmooth(self, switch):
        self.lSmoothing = switch
    
    def predict(self, dataSet):
        
        result = []
        
        for line in dataSet:
            line = line[:-1]
            probability = {}
            
            for label in self.labels.keys():
                if(self.laplacianUsed):
                    postProb = self.laplacianSmoothing(line, label)
                else:
                    postProb = self.calculatePosteriorProb(line, label)
                                    
                probability[label] = self.calculatePriorProb(label) * postProb
                
            self.laplacianUsed = False
            result.append(findMaxValueKey(probability))
           
        return result