# -*- coding: utf-8 -*-

"""
Main module.
"""

simulation = True
debug = False

import sys
import copy
import math
import time
import numpy as np

class Sample():
    
    """
    Each record of the stream has to be declared as a `Sample` class.

    :param value: the `values` of the current sample. 
    :param timestamp: the `timestamp` of current sample.

    """

    def __init__(self, value, timestamp: int):
        self.value = value
        self.timestamp = 0
        self.realTimestamp = timestamp
        
    def getValue(self):
        """
        :return: :attr:`value`
        """
        return self.value
    
    def setTimestamp(self, timestamp: int):
        """
        :set: :attr:`timestamp`
        """
        self.timestamp = timestamp
        
    def setRealTimestamp(self, timestamp):
        self.realTimestamp = timestamp

    def setMicroClusterNumber(self, microClusterNumber: int):
        """
        Assign to each sample the microClusterNumber in which was merged.

        :set: :attr:`microClusterNumber`
        """
        self.microClusterNumber = microClusterNumber

def computeReductionFactor(lamb, steps):
    return math.pow(2, -lamb * steps)

class MicroCluster():

    """
    Micro-Cluster class

    :param currenttimestamp: the `timestamp` in which the cluster is created.
    :param lamb: the `lamb` parameter used as decay factor.
    :param clusterNumber: the `number` of the micro-cluster.

    """

    def __init__(self, currenttimestamp, lamb, clusterNumber):

        self.dimensions = None
                
        self.creationTimeStamp = currenttimestamp
        self.lamb = lamb
        self.weight = 0

        self.reductionFactor = computeReductionFactor(self.lamb, 1)
        self.clusterNumber = clusterNumber
                        
    def insertSample(self, sample, timestamp=0):

        """
        Adds a sample to a micro-cluster. Updates the variables of the micro-cluster with :meth:`updateRealTimeWeight` and :meth:`updateRealTimeLSandSS`

        :param sample: the `sample` object 
        :param timestamp: deprecated, not needed anymore. Will be removed in the next versions.
        """

        if self.dimensions == None:

            if isinstance(sample.value, type(list)):
                self.dimensions = len(sample.value)
            elif isinstance(sample.value, float):
                self.dimensions = 1
            elif isinstance(sample.value, np.ndarray):
                self.dimensions = len(sample.value)
            else:
                print ('Received {}'.format(sample.value))
                print ('Type {}'.format(type(sample.value)))
                sys.exit('Error instance sample.value type')

            """
            incremental parameteres
            """
            self.N = 0
            self.weight = 0
            self.LS = np.zeros(self.dimensions)
            self.SS = np.zeros(self.dimensions)
            self.center = np.zeros(self.dimensions)
            self.radius = 0

            self.rLS = 0
            self.rSS = 0
            self.r_std = 0
            self.r_mean = 0

            self.M2 = 0

        self.N += 1
        self.updateRealTimeWeight()
        self.updateRealTimeLSandSS(sample)
        self.updateIncrementalRadiusTreshold()

    def updateIncrementalRadiusTreshold(self):
        """
        Doc update incremental radius 
        """

        n1 = self.N - 1
        n = self.N - 1 + 1
        delta = self.radius - self.r_mean
        delta_n = delta / n
        term1 = delta*delta_n*n1
        self.r_mean += delta_n
        self.M2 = self.M2 + term1 
        self.r_std = np.sqrt((self.M2)/(n-1))
        
    def updateRealTimeWeight(self):

        """
        Updates the Weight of the micro-cluster by the fading factor and increases it by 1. 
        """
        
        self.weight *= self.reductionFactor
        self.weight += 1
        
    def updateRealTimeLSandSS(self, sample):
        """
        Updates the `Weighted Linear Sum` (WLS), the `Weighted Squared Sum` (WSS), the `center` and the `radius` of the micro-cluster when a new sample is merged. 

        :param sample: the `sample` to merge into the micro-cluster.
        """

        sample = np.array(sample.value)

        self.LS = np.multiply(self.LS, self.reductionFactor)
        self.SS = np.multiply(self.SS, self.reductionFactor)
                
        self.LS = self.LS + sample
        self.SS = self.SS + np.power(sample, 2)

        self.center = np.divide(self.LS, float(self.weight))

        LSd = np.power(self.center, 2)
        SSd = np.divide(self.SS, float(self.weight))

        maxRad = np.nanmax(np.sqrt((SSd.astype(float)-LSd.astype(float))))

        self.radius = maxRad      

    def noNewSamples(self):
        """
        Updates the `Weighted Linear Sum` (WLS), the `Weighted Squared Sum` (WSS) and the weight of the micro-cluster when no new samples are merged.
        """
        self.LS = np.multiply(self.LS, self.reductionFactor)
        self.SS = np.multiply(self.SS, self.reductionFactor)
        self.weight *= self.reductionFactor
                
    def getCenter(self):
        """
        :return: the `center` of the micro-cluster.
        """
        return self.center

    def getRadius(self):
        """
        :return: the `radius` of the micro-cluster.
        """
        return self.radius

class Cluster():

    """
    Cluster class. Contains the list of the micro-cluster and the number of micro-clusters.
    """

    def __init__(self):
        self.clusters = []
        self.N = 0
        
    def insert(self, mc):
        """
        Inserts a micro-cluster into the cluster

        :param mc: the `micro-cluster` to be added to the list of the micro-clusters that make up the cluster.

        Increases the counter of micro-clusters into the cluster by 1.
        """
        self.clusters.append(mc)
        self.N += 1
        
    def show(self):
        print ('Number of Clusters: ' + str(len(self.clusters)))
        print ('-----')
        
        for cluster in self.clusters:
            print ('Cluster #'+str(self.clusters.index(cluster)))
            print ('Samples: '+str(cluster.N))
            print ('Weight: '+str(cluster.weight))
            print ('Creation Time: '+str(cluster.creationTimeStamp))
            print ('LastEdit Time: '+str(cluster.lastEditTimeStamp))

class OADDS():
    
    """
    OADDS class. 

    :param lamb: the `lambda` parameter - fading factor
    :param epsilon: the `epsilon` parameter 
    :param beta: the `beta` parameter
    :param mu: the `mu` parameter
    :param numberInitialSamples: samples to use as initial buffer
    :param startgingBuffer: initial `buffer` on which apply DBScan or use it as unique class.
    :param tp: frequency at which to apply the pruning strategy and remove old micro-clusters.
    """

    def __init__(self, lamb=0.1, epsilon='dynamic', minPts=1, beta=0.4, mu='auto',\
                numberInitialSamples=None, startingBuffer=None, tp=60, k_std = 3):
        self.lamb = lamb
        self.minPts = minPts
        self.beta = beta
        self.numberInitialSamples = numberInitialSamples
        self.buffer = startingBuffer
        self.tp = tp
        self.k_std = k_std

        self.exportVariables = False

        if debug:
            self.updated_epsilons = []
            self.nearest_distances = []

        ### Check input type: epsilon ### 
        if isinstance(epsilon, int) or isinstance(epsilon, float):
            self.epsilon = epsilon
        elif isinstance(epsilon, str) or isinstance(epsilon, unicode):
            self.epsilon = epsilon

        else:
            sys.exit('Error in parameter: epsilon')

        ### Check input type: mu ###
        if isinstance(mu, int) or isinstance(mu, float):
            self.mu = mu
        elif isinstance(mu, str) or isinstance(mu, unicode):
            if mu == 'auto':
                self.mu = 'auto'
        else:
            sys.exit('Error in parameter: mu')

        ### Running parameters ###
        self.inizialized = False

        ### Real timestamp or steps ###          
        if simulation:
            self.currentTimestamp = 0
        else:
            self.currentTimestamp = time.time()
        
    def resetLearningImpl(self):
        
        """
        Initializes two empty `Cluster` as a p-micro-cluter list and o-micro-cluster list.
        
        If `mu` is `auto` computes the value
        """

        if simulation:
            self.currentTimestamp = 0
        else:
            self.currentTimestamp = time.time()

        self.inizialized = False
        
        self.pMicroCluster = Cluster()
        self.oMicroCluster = Cluster()
                        
        if isinstance(self.mu, str):
            if self.mu == 'auto':
                self.mu = (1/(1-math.pow(2, -self.lamb)))

        if isinstance(self.tp, str):
            """
            Set TP
            """
            if self.tp == 'auto':
                self.tp = int(-1/self.lamb*math.log2(self.beta)) + 1


        self.th_beta_mu = self.beta * self.mu
                                
    def initWithoutDBScan(self):

        """
        Produces a micro-cluster merging all the samples passed into the initial buffer

        If `epsilon` is auto computes `epsilon` as the maxium radius obtained from these initial samples.
        """
        
        sample = Sample(self.buffer[0], 0)
        sample.setTimestamp(1)
        
        mc = MicroCluster(1, self.lamb, self.pMicroCluster.N + 1)
        
        maxEpsilon = 0

        for sampleNumber in range(0, len(self.buffer)):
            sample = Sample(self.buffer[sampleNumber], sampleNumber)
            sample.setTimestamp(sampleNumber+1)
            mc.insertSample(sample, self.currentTimestamp)

            if debug:
                self.updated_epsilons.append(mc.r_mean + mc.r_std*self.k_std)

            if mc.radius > maxEpsilon:
                maxEpsilon = mc.radius
            
        self.pMicroCluster.insert(mc)

        if isinstance(self.epsilon, str):

            if self.epsilon == 'dynamic':
                try:
                    self.epsilon = self.pMicroCluster.clusters[0].r_mean + self.pMicroCluster.clusters[0].r_std*self.k_std
                except:
                    sys.exit('Unkown epsilon')
        
    def nearestCluster (self, sample, timestamp, kind):
        minDist = 0.0
        minCluster = None
        
        if kind == 'cluster':
            clusterList = self.pMicroCluster.clusters
        elif kind == 'outlier':
            clusterList = self.oMicroCluster.clusters
        else:
            sys.exit('Error in choosing kind nearestCluster type: if pMicroCluster or oMicroCluster')
        
        for cluster in clusterList:
            
            if (minCluster == None):
                minCluster = cluster
                minDist = np.linalg.norm(sample.value - cluster.center)
              
            dist = np.linalg.norm(sample.value - cluster.center)
            dist -= cluster.radius

            if (dist < minDist):
                minDist = dist
                minCluster = cluster

        if kind == 'cluster':
            self.dist_nearest_pmc = minDist
            self.dist_nearest_omc = 0
        if kind == 'outlier':
            self.dist_nearest_omc = minDist
            self.dist_nearest_pmc = 0

        if debug:
            self.nearest_distances.append(minDist)
                
        return minCluster
       
    def updateAll(self, mc):
        
        if len(self.pMicroCluster.clusters) > 0:

            try:
                mc_start = self.pMicroCluster.clusters[0]
                max_update = mc_start.r_mean + mc_start.r_std*self.k_std
                max_w = mc_start.weight
            except Exception as e:
                max_update = self.epsilon

            for cluster in self.pMicroCluster.clusters:
                
                if (cluster != mc):
                    cluster.noNewSamples()
                    
                if cluster.weight > max_w:
                    max_w = cluster.weight
                    max_update = cluster.r_mean + cluster.r_std*self.k_std

            self.epsilon = copy.deepcopy(max_update)

        else:
            pass

        for cluster in self.oMicroCluster.clusters:
            
            if (cluster != mc):
                cluster.noNewSamples()

        if debug:
            self.updated_epsilons.append(self.epsilon)

    def runInitialization(self):
        """
        Initializes the variables of the main algorithm with the methods :meth:`resetLearningImpl` and :meth:`initWithoutDBScan`
        """
        self.resetLearningImpl()
        self.initWithoutDBScan()
        self.inizialized = True
    
    def runOnNewSample(self, sample):

        """
        Performs the basic DenStream procedure for merging new samples.

            * Try to merge the sample to the closest core-micro-cluster (or)
            * Try to merge the sample to the closest outlier-micro-cluster (or)
            * Generate new outlier-micro-cluster by the sample

        :param sample: the new available `sample` in the stream

        :return: ``False`` if the sample is merged to an existing core-micro-cluster otherwise ``True`` meaning "anomalous" sample.  

        """

        if simulation:
            self.currentTimestamp += 1
            sample.setTimestamp(self.currentTimestamp)
        else:
            self.currentTimestamp = time.time()

        ### INITIALIZATION PHASE ###
        if not self.inizialized:
            self.buffer.append(sample)
            if (len(self.buffer) >= self.numberInitialSamples):
                self.resetLearningImpl()
                self.initialDBScanSciLearn()
                self.inizialized = True

        ### MERGING PHASE ###
        else:
            merged = False
            TrueOutlier = True
            returnOutlier = True
            score_ = np.inf

            if len(self.pMicroCluster.clusters) != 0:
                
                closestMicroCluster = self.nearestCluster(sample, self.currentTimestamp, kind='cluster')
                            
                backupClosestCluster = copy.deepcopy(closestMicroCluster)
                backupClosestCluster.insertSample(sample, self.currentTimestamp)
                score_ = backupClosestCluster.radius

                if (backupClosestCluster.radius <= self.epsilon):

                    self.pMicroCluster.clusters.pop(self.pMicroCluster.clusters.index(closestMicroCluster))
                    self.pMicroCluster.insert(backupClosestCluster)

                    """
                    Removing 1 from N since we are only changing old pMC with backup of pMC
                    """
                    self.pMicroCluster.N = copy.deepcopy(self.pMicroCluster.N - 1)
                    sample.setMicroClusterNumber(backupClosestCluster.clusterNumber)
                    merged=True
                    TrueOutlier=False
                    returnOutlier=False
                    self.updateAll(backupClosestCluster)
                    
            if not merged and len(self.oMicroCluster.clusters) != 0:
                
                closestMicroCluster = self.nearestCluster(sample, self.currentTimestamp, kind='outlier')
            
                backupClosestCluster = copy.deepcopy(closestMicroCluster)
                backupClosestCluster.insertSample(sample, self.currentTimestamp)

                if (backupClosestCluster.radius <= self.epsilon):
                    merged = True
                    sample.setMicroClusterNumber(copy.deepcopy(backupClosestCluster.clusterNumber))
                    self.oMicroCluster.clusters.pop(self.oMicroCluster.clusters.index(closestMicroCluster))
                    if (backupClosestCluster.weight > self.th_beta_mu):                  
                        backupClosestCluster.clusterNumber = copy.deepcopy(self.pMicroCluster.N +1)
                        self.pMicroCluster.insert(backupClosestCluster)
                    else:
                        self.oMicroCluster.insert(backupClosestCluster)
                        self.oMicroCluster.N -= 1
                    self.updateAll(backupClosestCluster)
                                            
            if not merged:
                newOutlierMicroCluster = MicroCluster(self.currentTimestamp, self.lamb, 0)
                newOutlierMicroCluster.insertSample(sample, self.currentTimestamp)

                if TrueOutlier:
                    newOutlierMicroCluster.clusterNumber = self.oMicroCluster.N + 1
                    self.oMicroCluster.insert(newOutlierMicroCluster)
                    sample.setMicroClusterNumber(newOutlierMicroCluster.clusterNumber)
                    self.updateAll(newOutlierMicroCluster)
                else:
                    newOutlierMicroCluster.clusterNumber = self.pMicroCluster.N + 1
                    self.pMicroCluster.insert(newOutlierMicroCluster)
                    sample.setMicroClusterNumber(newOutlierMicroCluster.clusterNumber)
                    self.updateAll(newOutlierMicroCluster)
                    returnOutlier = False
                
            if self.currentTimestamp % self.tp == 0:
                            
                for cluster in self.pMicroCluster.clusters:

                    if cluster.weight < self.th_beta_mu:                        
                        self.pMicroCluster.clusters.pop(self.pMicroCluster.clusters.index(cluster))
                        
                for cluster in self.oMicroCluster.clusters:
                    
                    creationTimestamp = cluster.creationTimeStamp
                        
                    xs1 = math.pow(2, -self.lamb * (self.currentTimestamp - creationTimestamp + self.tp)) - 1
                    xs2 = math.pow(2, -self.lamb * self.tp) - 1
                    xsi = xs1 / xs2

                    if cluster.weight < xsi:
                        
                        self.oMicroCluster.clusters.pop(self.oMicroCluster.clusters.index(cluster))
                        
            if self.exportVariables:

                record = {
                    'pMicroClusters': len(self.pMicroCluster.clusters),
                    'oMicroClusters': len(self.oMicroCluster.clusters),
                    'pmc': copy.deepcopy(self.pMicroCluster.clusters),
                    'omc':  copy.deepcopy(self.oMicroCluster.clusters),    
                    'dist_nearest_pmc': self.dist_nearest_pmc,
                    'dist_nearest_omc': self.dist_nearest_omc,                
                    'result': returnOutlier,
                    'sample': sample
                }

                return record

            else:
                return returnOutlier, score_