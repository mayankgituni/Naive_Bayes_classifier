#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 20:05:07 2019

@author: mayanktomar
"""

import operator


class NaiveBayes:
    def __init__(self, dataSet):
        
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
    
    def convertIntoFreqTable(self, dataSet):
        
        freqTable = {}
        
        for i in range(self.dataSize):
            if dataSet[i] not in freqTable:
                freqTable[dataSet[i]] = []
                
            freqTable[dataSet[i]].append(i)
            
            # removing empty('?') values from the list
            if '?' in freqTable.keys():
                del freqTable['?']
                
        return freqTable
    
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
            self.features.append(self.convertIntoFreqTable(featureData[i]))

    
    def createLabelFreqTable(self, dataSet):
        self.labels = self.convertIntoFreqTable(dataSet)        
    
    def calculatePriorProb(self, label):
        return len(self.labels[label]) / self.dataSize
    
    def intersectionOfLists(self, list1, list2):
        return len(set(list1) & set(list2))
    
    def calculatePosteriorProb(self, feature, label):
        
        prob = 1
        
        for index in range(len(feature)):
            intersectionCount = self.intersectionOfLists(self.labels[label], self.features[index][feature[index]])
            labelcount = len(self.labels[label])
            prob *= float(intersectionCount / labelcount)
        
        if(prob == 0.0):
            print('Smoothing needed')
            pass
            #prob = self.epsilonSmoothing(feature, label)
            
        return prob
    
    def epsilonSmoothing(self, feature, label):
        pass
    
    def findMaxProb(self, probability):
        return max(probability.items(), key=operator.itemgetter(1))[0]
    
    def predict(self, dataSet):
        
        result = []
        
        for line in dataSet:
            line = line[:-1]
            probability = {}
            
            for label in self.labels.keys():
                probability[label] = self.calculatePriorProb(label) * self.calculatePosteriorProb(line, label)
            print("Prob: ", probability)
            result.append(self.findMaxProb(probability))
        print('Result: ', result)
        return result
