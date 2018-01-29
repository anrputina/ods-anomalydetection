#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 17:19:09 2017

@author: anr.putina
"""

import sys
sys.path.append('./../')

import time

import pandas as pd

from sample import Sample
from DenStream import DenStream
from statistics import Statistics

from groundTruth import groundTruth

def normalize_matrix(df):
    return (df - df.mean())/df.std()

nodes = ['leaf1', 'leaf2', 'leaf3', 'leaf5', 'leaf6', 'leaf7', 'leaf8',\
         'spine1', 'spine2', 'spine3', 'spine4']

nodes = ['leaf2']
resultSimulation = {}

for node in nodes:

    df = pd.read_csv('Data/'+node+'clear_bgp.csv').dropna()\
        .drop('Unnamed: 0', axis=1)

    deleteFeatureList = []
    
    for columnName in df.columns:
        if 'rate' in columnName:            
            deleteFeatureList.append(columnName)
        if 'load' in columnName:            
            deleteFeatureList.append(columnName)

            
    df = df.drop(deleteFeatureList, axis=1)

    times = df['time']
    df = df.drop(['time'], axis=1)

    dfNormalized = normalize_matrix(df).dropna(axis=1)

    sampleSkip = 35
    bufferDf = dfNormalized[0:sampleSkip]
    testDf = dfNormalized[sampleSkip:]

    den = DenStream(lamb=0.03, epsilon='auto', beta=0.03, mu='auto', startingBuffer=bufferDf, tp=12)
    den.runInitialization()

    outputCurrentNode = []
    startingSimulation = time.time()
    for sampleNumber in range(len(testDf)):
        sample = testDf.iloc[sampleNumber]
        result = den.runOnNewSample(Sample(sample.values, times.iloc[sampleNumber]))
        outputCurrentNode.append(result)
    ### END SIMULATION ###
    print time.time() - startingSimulation

    df['result'] = [False] * sampleSkip + outputCurrentNode

    truth = groundTruth()
    truth.simulationBGP_CLEAR_Second_DATASET()
    truth.simulationBGP_CLEAR2_CLEAR()

    statistics = Statistics(node, truth)
    resultSimulation[node] = statistics.getNodeResult(df, times, kMAX=5)

statistics.getPrecisionRecallFalseRate(resultSimulation, kMAX=5, plot=True)
statistics.getDelay(resultSimulation, kMAX=5, plot=True, samplingRate=4)
