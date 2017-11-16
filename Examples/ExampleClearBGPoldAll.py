#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 17:19:09 2017

@author: anr.putina
"""

import sys
sys.path.append('./../')

import time
import numpy as np
import pandas as pd

from multiprocessing import Process

from sample import Sample
from DenStream import DenStream
from statistics import Statistics

from groundTruth import groundTruth

from visualization import Visualization
def normalize_matrix(df):
    return (df - df.mean())/df.std()

#simulationDataset = 'PortFlap'
simulationDataset = 'BGP_CLEAR'
simulationDataset = 'BGP_CLEAR2'

nodes = ['leaf1', 'leaf2', 'leaf3', 'leaf5', 'leaf6', 'leaf7', 'leaf8',\
         'spine1', 'spine2', 'spine3', 'spine4']

#nodes = ['leaf2']

def worker(beta):

    lambdas = np.arange(0.02, 0.03, 0.01)

    results = []
    for lamb in lambdas:

        resultSimulation = {}

        for simulationDataset in ['BGP_CLEAR', 'BGP_CLEAR2']:
        
            for node in nodes:
        
                if simulationDataset == 'PortFlap':
                    
                    df = pd.read_csv('Data/'+node+'base_no_traffic.csv').dropna()\
                            .drop('Unnamed: 0', axis=1)\
                            .drop('MgmtEth0/RP0/CPU0/0reliability', axis=1)\
                            .drop('MgmtEth0/RP0/CPU0/0packets-sent', axis=1)
                            
                elif simulationDataset == 'BGP_CLEAR':
                    ### DATA PLANE
                    df = pd.read_csv('Data/'+node+'_clearbgp.csv').dropna()\
                            .drop('Unnamed: 0', axis=1)\
                            .drop('MgmtEth0/RP0/CPU0/0packets-sent', axis=1)\
                #            .drop('HundredGigE0/0/0/30packets-sent', axis=1)   
                elif simulationDataset == 'BGP_CLEAR2':
                    ### DATA PLANE
                    df = pd.read_csv('Data/'+node+'bgp_clear_NoDataPlane.csv').dropna()\
                            .drop('Unnamed: 0', axis=1)\
        
                else:
                    print 'CHECK DATASET'
        
                df = df[:500]
        
                df['time'] = df['time'] / 1000000    
                times = df['time']
                df = df.drop(['time'], axis=1)
        
                featuresDrop = []
                for feature in df.columns:
                    if 'MgmtEth' in feature:
                        featuresDrop.append(feature)
                    if 'reliability' in feature:
                        featuresDrop.append(feature)
        
                df = df.drop(featuresDrop, axis=1)
        
                if 'cluster' in df:    
                    df = df.drop('cluster', axis=1)
        
        
                dfNormalized = normalize_matrix(df).dropna(axis=1)
        
                sampleSkip = 41
                bufferDf = dfNormalized[0:sampleSkip]
                testDf = dfNormalized[sampleSkip:]
        
                den = DenStream(lamb=lamb, epsilon='auto', beta=beta, mu='auto', startingBuffer=bufferDf, tp=12)
                den.runInitialization()
        
                outputNEW = []
                startingSimulation = time.time()
                for sampleNumber in range(len(testDf)):
                    sample = testDf.iloc[sampleNumber]
                    result = den.runOnNewSample(Sample(sample.values, times.iloc[sampleNumber]))
                    outputNEW.append(result)
                ### END SIMULATION ###
                print time.time() - startingSimulation
        
                df['result'] = [False] * sampleSkip + outputNEW
                
                truth = groundTruth()
                if simulationDataset == 'PortFlap':
                    truth.simulationPORT_FLAP()
                    truth.simulationPORT_FLAP_CLEAR()
                elif simulationDataset =='BGP_CLEAR':
                    truth.simulationBGP_CLEAR3_TwoMin()
                    truth.simulationBGP_CLEAR_CLEAR()
                elif simulationDataset =='BGP_CLEAR2':
                    truth.simulationBGP_CLEAR_Second_DATASET()
                    truth.simulationBGP_CLEAR2_CLEAR()
        
        #        visual=Visualization()
        #        visual.plotResults(df, times, truth)
        
                statistics = Statistics(node, truth)
                resultSimulation[node+simulationDataset] = statistics.getNodeResult(df, times, kMAX=5)

        statistics.getPrecisionRecallFalseRate(resultSimulation, kMAX=5, plot=True)
        statistics.getDelay(resultSimulation, kMAX=5, plot=True)
        
        #    result = {
        #        'PRF': statistics.getPrecisionRecallFalseRate(resultSimulation, kMAX=5, plot=False),
        #        'Delay': statistics.getDelay(resultSimulation, kMAX=5, plot=False)
        #    }
        
#        results.append(statistics.getPrecisionRecallFalseRate(resultSimulation, kMAX=5, plot=True))

    

#    filename = 'beta'+str(beta)
#    df = pd.DataFrame(results)
#    df.to_csv(filename+'.csv')
    
#    filename = 'beta'+str(beta)
#    import json
#    with open('Results/'+filename+'.txt', 'w') as outfile:
#       json.dump({beta:results}, outfile)
       
       
    return results

if __name__ == '__main__':
    
    betas = np.arange(0.02, 0.03, 0.01)
    worker(0.02)
#    for beta in betas:
#
#        p = Process(target=worker, args=(beta,))
#        p.start()
        


# #features = ['Precision, Recall']
# #Ks = [2, 3]
# #
# #arrays = np.zeros((len(features), len(Ks)))
# #
# Precision2 = []
# Precision3 = []

# Recall2 = []
# Recall3 = []
# for result in results:
    
#     Precision2.append(result['PRF']['Precision'][2])
#     Precision3.append(result['PRF']['Precision'][3])
#     Recall2.append(result['PRF']['Recall'][2])
#     Recall3.append(result['PRF']['Recall'][3])

# import matplotlib.pyplot as plt
# #
# #fig, ax = plt.subplots()
# ##
# ##ax.plot(lambdas, Precision2, label='Precision2')
# ##ax.plot(lambdas, Precision3, label='Precision3')
# ##ax.plot(lambdas, Recall2, label='Recall2')
# ##ax.plot(lambdas, Recall3, label='Recall3')
# ##
# ##ax.legend()
# #
# fig, ax = plt.subplots()

# ax.plot(betas, Precision2, label='Precision2')
# ax.plot(betas, Precision3, label='Precision3')
# ax.plot(betas, Recall2, label='Recall2')
# ax.plot(betas, Recall3, label='Recall3')

# ax.legend()
# #
