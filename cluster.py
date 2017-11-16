#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 17:48:58 2017

@author: anr.putina
"""

class Cluster():
    def __init__(self):
        self.clusters = []
        
    def insert(self, mc):
        self.clusters.append(mc)
        
    def show(self):
        print 'Number of Clusters: ' + str(len(self.clusters))
        print ('-----')
        
        for cluster in self.clusters:
            print 'Cluster #'+str(self.clusters.index(cluster))
            print 'Samples: '+str(cluster.N)
            print 'Weight: '+str(cluster.weight)
            print 'Creation Time: '+str(cluster.creationTimeStamp)
            print 'LastEdit Time: '+str(cluster.lastEditTimeStamp)