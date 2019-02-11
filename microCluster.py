#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 17:46:51 2017

@author: anr.putina
"""

import math
import numpy as np
import datetime

def computeReductionFactor(lamb, steps):
    return math.pow(2, -lamb * steps)

class MicroCluster():
    def __init__(self, currenttimestamp, lamb, clusterNumber):

        self.dimensions = None
                
        self.creationTimeStamp = currenttimestamp
        self.lamb = lamb

        self.reductionFactor = computeReductionFactor(self.lamb, 1)
        self.clusterNumber = clusterNumber
                        
    def insertSample(self, sample, timestamp):

        if self.dimensions == None:

            if isinstance(sample.value, type(list)):
                self.dimensions = len(sample.value)
            elif isinstance(sample.value, float):
                self.dimensions = 1
            else:
                sys.exit('Error instance sample.value type')

            ### incremental parameteres ###
            self.N = 0
            self.weight = 0
            self.LS = np.zeros(self.dimensions)
            self.SS = np.zeros(self.dimensions)
            self.center = np.zeros(self.dimensions)
            self.radius = 0 

        self.N += 1
        self.updateRealTimeWeight()
        self.updateRealTimeLSandSS(sample)
        
    def updateRealTimeWeight(self):
        
        self.weight *= self.reductionFactor
        self.weight += 1
        
    def updateRealTimeLSandSS(self, sample):
        self.LS = np.multiply(self.LS, self.reductionFactor)
        self.SS = np.multiply(self.SS, self.reductionFactor)
                
        self.LS = self.LS + sample.value
        self.SS = self.SS + sample.value**2

        self.center = np.divide(self.LS, float(self.weight))

        LSd = np.power(self.center, 2)
        SSd = np.divide(self.SS, float(self.weight))

        maxRad = np.nanmax(np.sqrt(SSd.astype(float)-LSd.astype(float)))
        # maxRad = np.nanmax(np.lib.scimath.sqrt(SSd-LSd))
        self.radius = maxRad        

    def noNewSamples(self):
        self.LS = np.multiply(self.LS, self.reductionFactor)
        self.SS = np.multiply(self.SS, self.reductionFactor)
        self.weight = np.multiply(self.weight, self.reductionFactor)
                
    def getCenter(self):
        return self.center

    def getRadius(self):
        return self.radius