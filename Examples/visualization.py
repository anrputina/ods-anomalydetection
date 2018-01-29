#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 10:35:07 2017

@author: anr.putina
"""
import numpy
import matplotlib
import matplotlib.pylab as plt
import matplotlib.pylab as pl
import matplotlib.ticker as mtick
import time as tm


fontlabel = 25
matplotlib.rcParams['xtick.labelsize'] = fontlabel 
matplotlib.rcParams['ytick.labelsize'] = fontlabel 
matplotlib.rc('axes', edgecolor='black')
matplotlib.rcParams['legend.numpoints'] = 1
matplotlib.rcParams['legend.handlelength'] = 1

class Visualization():
    
    def __init__(self):
        pass
    
    def plotOutliers(self, features, df, outliers, merged, truth, stepsTrain=10):
        
        plotNumber = 0
        fig, ax = plt.subplots(len(features), sharex=True)

        for feature in features:
            
            ax[plotNumber].plot(df[feature].values)
            
            for outlier in outliers:
                ax[plotNumber].plot(stepsTrain+outlier['time'],
                                    df[feature].iloc[stepsTrain+outlier['time']],
                                    marker='o',
                                    color='r')
                
            for mergePoint in merged:
                ax[plotNumber].plot(stepsTrain+mergePoint,\
                                    df[feature].iloc[stepsTrain+mergePoint],\
                                    marker='x',\
                                    color='g')
                            
            for event in truth.events:
                if event['type'] == 'single':
                    ax[plotNumber].axvspan(event['startIndex'], event['endIndex'], alpha=0.5, color='red')
            
            if 'free-application-memory' in feature:
                ax[plotNumber].set_ylabel('free memory')
            elif 'vrf__update-messages-received' in feature:
                ax[plotNumber].set_ylabel('update msg rx')
            else:
                ax[plotNumber].set_ylabel(feature)    
            plotNumber += 1

        ax[len(features)-1].set_xlabel('Simulation Step')
  
    def plotOutliersfromDF(self, df, times, truth):
        
        fig, ax = plt.subplots()
        
        outliers = df[df['result']==True]
        
        ax.plot(times, df['5paths-count'])
        
        for outlier in outliers.iterrows():
            ax.plot(times.loc[outlier[0]], outlier[1]['5paths-count'], color='r', marker='o')
        
        for event in truth.events:
            if event['type'] == 'single':
                ax.axvspan(event['startTime'], event['endTime'], alpha=0.5, color='red')

    
    def plotOutliersInteractive(self, features, df, outliers, merged, truth, stepsTrain=10):
        
        plotNumber = 0
        fig, ax = pl.subplots(len(features), sharex=True)
#        ax[0].set_title('Nome Grafico')

        for feature in features:
            
            ax[plotNumber].plot(df[feature].values)
            
            for outlier in outliers:
                ax[plotNumber].plot(stepsTrain+outlier['time'] - 51,
                                    df[feature].iloc[stepsTrain+outlier['time'] - 51],
                                    marker='o',
                                    color='r')
                
            for mergePoint in merged:
                ax[plotNumber].plot(stepsTrain+mergePoint - 51,\
                                    df[feature].iloc[stepsTrain+mergePoint] - 51,\
                                    marker='x',\
                                    color='g')
                            
            for event in truth.events:
                if event['type'] == 'single':
                    ax[plotNumber].axvspan(event['startIndex'], event['endIndex'], alpha=0.5, color='red')
            
            if 'free-application-memory' in feature:
                ax[plotNumber].set_ylabel('free memory')
            elif 'vrf__update-messages-received' in feature:
                ax[plotNumber].set_ylabel('update msg rx')
            else:
                ax[plotNumber].set_ylabel(feature)    
            plotNumber += 1

        ax[len(features)-1].set_xlabel('Simulation Step', fontsize=fontlabel)
     
    def plotOutliersInteractiveTimestamp(self, features, df, outliers, merged, truth, stepsTrain=10, time=0):
        
        plotNumber = 0
        fig, ax = pl.subplots(len(features), sharex=True, figsize=(13, 5))
#        ax[0].set_title('Nome Grafico')

        for feature in features:
            
            ax[plotNumber].plot(time, df[feature])
            
            for outlier in outliers:
                outlierPlot, = ax[plotNumber].plot(time.iloc[stepsTrain+outlier['time']],
                                    df[feature].iloc[stepsTrain+outlier['time']],
                                    marker='o',
                                    markersize=6,
                                    color='r')
                
            for mergePoint in merged:
                mergedPlot, = ax[plotNumber].plot(time.iloc[stepsTrain+mergePoint['time']],\
                                    df[feature].loc[stepsTrain+mergePoint['time']],\
                                    marker='o',\
                                    markersize=6,\
                                    color='g')
                            
            for event in truth.events:
                if event['type'] == 'single':
                    ax[plotNumber].axvspan(event['startTime'], event['endTime'], alpha=0.5, color='rosybrown')
            
            if 'free-application-memory' in feature:
                ax[plotNumber].set_ylabel('free memory', fontsize=fontlabel)
            elif 'vrf__update-messages-received' in feature:
                ax[plotNumber].set_ylabel('update msg rx', fontsize=fontlabel)
            elif 'packets-sent' in feature:
                ax[plotNumber].set_ylabel('Packets-tx int0', fontsize=fontlabel)
            elif 'packets-received' in feature:
                ax[plotNumber].set_ylabel('Packets-rx int0', fontsize=fontlabel)
            else:
                ax[plotNumber].set_ylabel(feature, fontsize=fontlabel)  
                
            ax[plotNumber].grid(c='gray')
            plotNumber += 1

        # Shrink current axis's height by 10% on the bottom
        box = ax[len(features)-1].get_position()
        ax[0].set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
                
        lgnd = ax[0].legend([mergedPlot, outlierPlot], ["Normal", "Abnormal"], loc='upper center', bbox_to_anchor=(0.5, +1.3),
                  fancybox=False, shadow=False, ncol=5, fontsize=fontlabel, markerscale=3., frameon=False)

        ax[len(features)-1].set_xlim((time.iloc[0], time.iloc[-1]))
        ax[len(features)-1].set_xlabel('Time', fontsize=fontlabel, labelpad=20)

        plt.gcf().autofmt_xdate()

        plt.gca().xaxis.set_major_locator(mtick.FixedLocator(time.values))
        plt.gca().xaxis.set_major_formatter(
            mtick.FuncFormatter(lambda pos,_: tm.strftime("%H:%M",tm.localtime(int(pos/1000.)))))
#        plt.tight_layout()
        ax[len(features)-1].locator_params(nbins=13, axis='x')
        
        plt.setp(ax[len(features)-1].get_xticklabels()[0], visible=False)
    
    def plotOutliersInteractiveTimestamp2(self, features, df, outliers, merged, truth, stepsTrain=10, time=0):
        
        plotNumber = 0
        fig, ax = pl.subplots(len(features), sharex=True, figsize=(13, 5))
#        ax[0].set_title('Nome Grafico')

        for feature in features:
            
            ax.plot(time, df[feature])
            
            for outlier in outliers:
                outlierPlot, = ax.plot(time.iloc[stepsTrain+outlier['time']],
                                    df[feature].iloc[stepsTrain+outlier['time']],
                                    marker='o',
                                    markersize=6,
                                    color='r')
                
            for mergePoint in merged:
                mergedPlot, = ax.plot(time.iloc[stepsTrain+mergePoint['time']],\
                                    df[feature].loc[stepsTrain+mergePoint['time']],\
                                    marker='o',\
                                    markersize=6,\
                                    color='g')
                            
            for event in truth.events:
                if event['type'] == 'single':
                    ax.axvspan(event['startTime'], event['endTime'], alpha=0.5, color='rosybrown')
            
            if 'free-application-memory' in feature:
                ax.set_ylabel('free memory', fontsize=fontlabel)
            elif 'vrf__update-messages-received' in feature:
                ax.set_ylabel('update msg rx', fontsize=fontlabel)
            elif 'packets-sent' in feature:
                ax.set_ylabel('Packets-tx int0', fontsize=fontlabel)
            elif 'packets-received' in feature:
                ax.set_ylabel('Packets-rx int0', fontsize=fontlabel)
            else:
                ax.set_ylabel(feature, fontsize=fontlabel)  
                
            ax.grid(c='gray')
            plotNumber += 1

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
                
        lgnd = ax.legend([mergedPlot, outlierPlot], ["Normal", "Abnormal"], loc='upper center', bbox_to_anchor=(0.5, +1.2),
                  fancybox=False, shadow=False, ncol=5, fontsize=fontlabel, markerscale=3., frameon=False)

        ax.set_xlim((time.iloc[0], time.iloc[-1]))
#        ax.set_xlabel('Time', fontsize=fontlabel, labelpad=20)

        plt.gcf().autofmt_xdate()

        plt.gca().xaxis.set_major_locator(mtick.FixedLocator(time.values))
        plt.gca().xaxis.set_major_formatter(
            mtick.FuncFormatter(lambda pos,_: tm.strftime("%H:%M",tm.localtime(int(pos/1000.)))))
#        plt.tight_layout()
        ax.locator_params(nbins=13, axis='x')
        
        plt.setp(ax.get_xticklabels()[0], visible=False)  
        plt.show()
    

    def plotResults(self, df, times, truth):

        fig, ax = plt.subplots()

        ax.plot(times, df['paths-count'])
        ax.plot(times.loc[df['result']==False], df['paths-count'][df['result']==False], color='green', marker='x', linewidth=0)
        ax.plot(times.loc[df['result']==True], df['paths-count'][df['result']==True], color='red', marker='o', linewidth=0)

        for event in truth.events:
            if event['type'] == 'single':
                ax.axvspan(event['startTime'], event['endTime'], alpha=0.5, color='rosybrown')

    def plotOutliersByStep(self, df, features, outliers, merged, truth, stepsTrain=10, time=0):
                
#        mergedPlot = None
#        outlierPlot = None
        
        plotNumber = 0
        fig, ax = plt.subplots(len(features), sharex=True)
        
        for feature in features:
            
            ax[plotNumber].plot(df[feature])
            
            for outlier in outliers:
                outlierPlot, = ax[plotNumber].plot(outlier,
                                      df[feature].loc[outlier],
                                      marker='o',
#                                      markersize=10,
                                      color='r')
            
            for mergedPoint in merged:
                mergedPlot, = ax[plotNumber].plot(mergedPoint,
                                      df[feature].loc[mergedPoint],
                                      marker='x', 
                                      color='g')
            
#            for event in truth.events:
#                if event['type'] == 'single':
#                    if 'startidx' in event:
#                        ax[plotNumber].axvspan(event['startidx'], event['endidx'], alpha=0.5, color='red')
#            
            if 'free-application-memory' in feature:
                ax[plotNumber].set_ylabel('free memory [GB]', fontsize=fontlabel)
            elif 'vrf__update-messages-received' in feature:
                ax[plotNumber].set_ylabel('update msg rx', fontsize=fontlabel)
            elif 'packets-sent' in feature:
                ax[plotNumber].set_ylabel('Packets-tx int0', fontsize=fontlabel)
            elif 'packets-received' in feature:
                ax[plotNumber].set_ylabel('Packets-rx int0', fontsize=fontlabel)
            else:
                ax[plotNumber].set_ylabel(feature, fontsize=fontlabel)  
        
            ax[plotNumber].grid(c='gray')
            plotNumber += 1    
            
        # Shrink current axis's height by 10% on the bottom
        box = ax[len(features)-1].get_position()
        ax[len(features)-1].set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
                
        lgnd = ax[len(features)-1].legend([mergedPlot, outlierPlot], ["Normal", "Abnormal"], loc='upper center', bbox_to_anchor=(0.5, -0.35),
                  fancybox=False, shadow=False, ncol=5, fontsize=fontlabel+10, markerscale=3., frameon=False)

        
        ax[len(features)-1].set_xlabel('Simulation Step', fontsize=fontlabel)
        ax[len(features)-1].set_xlim((0, len(df)))


    def plotDBScanClusters(self, df, features):
        
        def get_cmap(n, name='hsv'):
            '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
            RGB color; the keyword argument name must be a standard mpl colormap name.'''
            return plt.cm.get_cmap(name, n)
        
#        cmap = get_cmap(len(df['cluster'].unique()), name='gist_earth')
        cmap = get_cmap(len(df['cluster'].unique()), name='Dark2')

        labels = []
        plotNumber = 0
        fig, ax = plt.subplots(len(features), sharex=True)
        
        for feature in features:
            ax[plotNumber].plot(df[feature])
            
            for clusterID in df['cluster'].unique():
                if clusterID != -1:
                    ax[plotNumber].scatter(df[df['cluster']==clusterID].index, df[df['cluster']==clusterID][feature], marker='x', c=cmap(clusterID))
                else:
                    pass
            clusterID = -1
            ax[plotNumber].scatter(df[df['cluster']==clusterID].index, df[df['cluster']==clusterID][feature], marker='o', color='r')
                            
            if 'free-application-memory' in feature:
                ax[plotNumber].set_ylabel('free memory [GB]', fontsize=fontlabel)
            elif 'vrf__update-messages-received' in feature:
                ax[plotNumber].set_ylabel('update msg rx', fontsize=fontlabel)
            elif 'packets-sent' in feature:
                ax[plotNumber].set_ylabel('Packets-tx int0', fontsize=fontlabel)
            elif 'packets-received' in feature:
                ax[plotNumber].set_ylabel('Packets-rx int0', fontsize=fontlabel)
            else:
                ax[plotNumber].set_ylabel(feature, fontsize=fontlabel)  
            
            ax[plotNumber].grid(c='gray')
            plotNumber += 1
        
        handles, labels = ax[plotNumber-1].get_legend_handles_labels()
        display=[1, 3, 4, 5, 7, 8, 10, 12]
        
        # Shrink current axis's height by 10% on the bottom
        box = ax[len(features)-1].get_position()
        ax[len(features)-1].set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
                
        lgnd = ax[len(features)-1].legend([handle for i, handle in enumerate(handles) if i in display],\
                                          ['cluster1', 'cluster2', 'cluster3', 'cluster4', 'cluster5', 'cluster6', 'cluster7', 'Abnormal'],\
                                          loc='upper center', bbox_to_anchor=(0.5, -0.10),\
                                          fancybox=False, shadow=False, ncol=4, fontsize=fontlabel+10,\
                                          markerscale=2., frameon=False)
            
        ax[len(features)-1].set_xlabel('Simulation Step', fontsize=fontlabel)
        ax[len(features)-1].set_xlim((0, len(df)))

    def plotOutliersInteractiveTimestampNoTruth(self, features, df, outliers, merged, stepsTrain=10, time=0):
        
        plotNumber = 0
        fig, ax = pl.subplots(len(features), sharex=True)
#        ax[0].set_title('Nome Grafico')


        for feature in features:
            
            ax[plotNumber].plot(time, df[feature])
            
            for outlier in outliers:
                ax[plotNumber].plot(time.iloc[stepsTrain+outlier['time']],
                                    df[feature].iloc[stepsTrain+outlier['time']],
                                    marker='o',
                                    color='r')
                
            for mergePoint in merged:
                ax[plotNumber].plot(time.iloc[stepsTrain+mergePoint['time']],\
                                    df[feature].loc[stepsTrain+mergePoint['time']],\
                                    marker='x',\
                                    color='g')

            if 'free-application-memory' in feature:
                ax[plotNumber].set_ylabel('free memory [GB]')
            elif 'vrf__update-messages-received' in feature:
                ax[plotNumber].set_ylabel('update msg rx')
            else:
                ax[plotNumber].set_ylabel(feature)    
            plotNumber += 1

        ax[len(features)-1].set_xlabel('Timestamp')
        ax[len(features)-1].set_xlim((0, len(features)))
#    def plotSingleOutlier(self, feature, df, outliers, mergeg, truth, stepsTrain=10):
#        
#        fig, ax = plt.subplots()
#        
#        ax.plot(df[feature].values)
#        
#        for outlier in outliers:
#            ax.plot(stepsTrain + outlier['time'],
#                    df[feature].iloc[stepsTrain+outlier['time']],
#                    marker='o',
#                    color='r')
#            
#        for mergePoint in merged:
#            ax[plotNumber].plot(stepsTrain+mergePoint,\
#                                df[feature].iloc[stepsTrain+mergePoint],\
#                                marker='x',\
#                                color='g')
#            
#        for event in truth.events:
#            ax[plotNumber].axvspan(event['startIndex'], event['endIndex'], alpha=0.5, color='red')
#            
#        
#            
#            

    def plotHistTimeDifference(self, dfs, cumulative=False, bins=15, normed=True, scale='linear'):
        
        if len(dfs) == 1:
            
            fig, ax = plt.subplots()
            ax.hist(dfs[0]['time'].diff().dropna())
            
        if len(dfs) > 1:
            
            fig, ax = plt.subplots()
            
            alphaChannel = 1
            label = 'leaf1'
            for df in dfs:
                ax.hist(df['time'].diff().dropna(), alpha=alphaChannel, bins=bins, normed=normed, cumulative=cumulative, label=label)
                alphaChannel = 0.5
                label='leaf8'
            
            ax.set_yscale(scale)
            
            ax.legend()
            
            ax.set_xlabel('sample rate [s]')
            ax.grid()
                        
    def plotTimes(self, dfs):
        
        fig, ax = plt.subplots()
        
        label = 'leaf1'
        for df in dfs:
            
            ax.plot(df['time'], label=label)
            label = 'leaf8'
            
            
        ax.set_xlabel('sample number')
        ax.set_ylabel('timestamp')
        ax.legend(loc=4)
        ax.grid()
            
            
            
            
    def plotProbabilityDetection(self, results):
        
        colors = ['brown', 'blue', 'darkgreen', 'red', 'black']
        xticks = [0, 1, 2]
        labels = [0, 1, 2]

        fig, ax = plt.subplots()
        
        counter = 0
        for k in results:
            if counter == 6:
                ax.plot(k, label='K = ' + str(counter+1), color = colors[counter], linestyle= '--', linewidth=2)
            else:
                ax.plot(k, label='K = ' + str(counter+1), color = colors[counter], linewidth=1.5)
            counter += 1
            
        ax.grid(c='gray')
        ax.legend()
        ax.set_xlabel('#hops', fontsize=fontlabel)
        ax.set_ylabel('probability detection', fontsize=fontlabel)
        plt.xticks(xticks, labels)

        legend = plt.legend(loc=3, fontsize=fontlabel)
        legend.get_frame().set_alpha(0.5)
    
    def plotProbabilityDetectionBar(self, results, confInterval):
        
#        colors = ['brown', 'blue', 'darkgreen', 'red', 'black']
#        xticks = [0, 1, 2, 3, 4]
#        labels = [1, 2, 3, 4, 5]

        n_groups = 5
        index = numpy.arange(n_groups)
        bar_width = 0.2
        opacity = 0.7

        fig, ax = plt.subplots(figsize=(13,8))
        
        rects2 = plt.bar(index-bar_width, results[:,0], bar_width,
                 alpha=opacity,
                 color='lightslategrey',
                 yerr = confInterval['hop0'],
                 ecolor = 'black',
                 label='Node')
        
        rects3 = plt.bar(index, results[:,1], bar_width,
                 alpha=opacity,
                 yerr = confInterval['hop1'],
                 ecolor = 'black',
                 color='cadetblue',
                 label='1 Hop Away')
        
        rects1 = plt.bar(index + bar_width, results[:,2], bar_width,
                 alpha=opacity,
                 yerr = confInterval['hop2'],
                 ecolor = 'black',
                 color='seagreen',
                 label='2 Hops Away')
            
        plt.xlabel('K', fontsize=fontlabel)
        plt.ylabel('Detection Probability', fontsize=fontlabel)
#        plt.title('Scores by group and gender')
        plt.xticks(index + bar_width/2, ('1', '2', '3', '4', '5'))
        ax.set_xlim((-2*bar_width, index[4]+3*bar_width))
        ax.set_ylim((0,1))
        ax.grid(c='gray')

        legend = plt.legend(loc=1, fontsize=fontlabel)
        legend.get_frame().set_alpha(0.5)        
#        plt.tight_layout()
#        plt.show()
        
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        
        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
                  fancybox=False, shadow=False, ncol=5, fontsize=fontlabel, frameon=False)
        
        
        
    
    
    def plotDelays(self, results):
        
        colors = ['brown', 'blue', 'darkgreen', 'red', 'black']
        xticks = [0, 1, 2]
        labels = [0, 1, 2]

        fig, ax = plt.subplots()
        
        counter = 0
        for k in results:
            if counter == 6:
                ax.plot(k, label='K = ' + str(counter+1), color = colors[counter], linestyle= '--', linewidth=2)
            else:
                ax.plot(k, label='K = ' + str(counter+1), color = colors[counter], linewidth=1.5)
            counter += 1
            
        ax.grid(c='gray')
        ax.legend()
        ax.set_xlabel('#hops', fontsize=fontlabel)
        ax.set_ylabel('probability detection', fontsize=fontlabel)
        plt.xticks(xticks, labels)

        legend = plt.legend(loc=1, fontsize=fontlabel)
        legend.get_frame().set_alpha(0.5)
    
    def plotRecallvsK(self, results, plots):
        
        labels = {0: {'name': 'Hop 0',
                      'color': 'blue',
                      'entity': 'single'}, 
                  1: {'name': 'Hop 1', 
                      'color': 'darkgreen',
                      'entity': 'single'}  ,
                  2: {'name': 'Hop 2', 
                      'color': 'sienna',
                      'entity': 'single'},
                  3: {'name': 'Hop 0', 
                      'color': 'darkgoldenrod',
                      'entity': 'multiple'},
                  4: {'name': 'Hop 1', 
                      'color': 'darkslategray',
                      'entity': 'multiple'},
                  5: {'name': 'Hop 2', 
                      'color': 'navy',
                      'entity': 'multiple'},
                  6: {'name': 'Recall', 
                      'color': 'black',
                      'entity': 'single'}
                }
        

        
        fig, ax = plt.subplots()
        
        for column in range(results.shape[1]):
            
            if column in plots:
            
                ax.plot(results[:, column], label='p '+labels[column]['name']+' '+labels[column]['entity'], color=labels[column]['color'])
                
                ax.grid(c='gray')
                ax.set_xlabel('#K', fontsize=fontlabel)
                ax.set_ylabel('probability detection', fontsize=fontlabel)
            
        xticks = [0, 1, 2, 3, 4]
        labels = [1, 2, 3, 4, 5]
            
        ax.legend()
        plt.xticks(xticks, labels)

        legend = plt.legend(loc=1, fontsize=fontlabel)
        legend.get_frame().set_alpha(0.5)
            
            
    def plotRecallPrecisionvsK(self, results, falseRate, precision):
        
        linewidth = 2
        fig, ax = plt.subplots()
                
        ax.plot(results[:, 6], label='Recall', color='black', linewidth=linewidth)
        ax.plot(precision, label='Precision', color='firebrick', linewidth=linewidth)
        ax.plot(falseRate, label='False Positive Rate', color='grey', linewidth=linewidth)

        ax.grid(c='gray')
        ax.set_xlabel('#K', fontsize=fontlabel)
        ax.set_ylabel('probability', fontsize=fontlabel)
        
        xticks = [0, 1, 2, 3, 4]
        labels = [1, 2, 3, 4, 5]
            
        ax.legend()
        plt.xticks(xticks, labels)

        legend = plt.legend(loc=1, fontsize=fontlabel)
        legend.get_frame().set_alpha(0.5)
        
    def plotDelayvsK(self, results):
        
        linewidth = 2
        fig, ax = plt.subplots()
        
        ax.plot(results[:,0], label='Hop0', color='black', linewidth=linewidth)
        ax.plot(results[:,1], label='Hop1', color='green', linewidth=linewidth)
        ax.plot(results[:,2], label='Hop3', color='blue', linewidth=linewidth)

#        ax.plot(precision, label='Precision', color='firebrick', linewidth=linewidth)
#        ax.plot(falseRate, label='False Positive Rate', color='grey', linewidth=linewidth)

        ax.grid(c='gray')
        ax.set_xlabel('#K', fontsize=fontlabel)
        ax.set_ylabel('delay', fontsize=fontlabel)
        
        xticks = [0, 1, 2, 3, 4]
        labels = [1, 2, 3, 4, 5]
            
        ax.legend()
        plt.xticks(xticks, labels)

        legend = plt.legend(loc=1, fontsize=fontlabel)
        legend.get_frame().set_alpha(0.5)
        
               
    def barRecallPrecisionvsK(self, results, falseRate, precision):
            
        n_groups = 5

        recall = results[:, 6]
        
        fig, ax = plt.subplots()
        index = numpy.arange(n_groups)
        bar_width = 0.35
        opacity = 0.7
        
#        rects1 = plt.bar(index, falseRate, bar_width,
#                 alpha=opacity,
#                 color='b',
#                 label='False Rate')
    
        rects2 = plt.bar(index , precision, bar_width,
                         alpha=opacity,
                         color='goldenrod',
                         label='Precision')

        rects3 = plt.bar(index + bar_width, recall, bar_width,
                 alpha=opacity,
                 color='darkorange',
                 label='Recall')
            
        plt.xlabel('K', fontsize=fontlabel)
        plt.ylabel('Scores', fontsize=fontlabel)
#        plt.title('Scores by group and gender')
        plt.xticks(index + bar_width, ('1', '2', '3', '4', '5'))
        ax.set_xlim((-bar_width, index[4]+3*bar_width))
        ax.set_ylim((0,1.2))
        ax.grid(c='gray')

        legend = plt.legend(loc=1, fontsize=fontlabel)
        legend.get_frame().set_alpha(0.5)        
#        plt.tight_layout()
#        plt.show()

    def barRecallPrecisionvsK2(self, recall, falseRate, precision, errorsRecall, errorprecision, errorFalse):
            
        n_groups = len(recall)

        fig, ax = plt.subplots(figsize=(10,6))
        index = numpy.arange(n_groups)
        bar_width = 0.2
        opacity = 0.7
        
        rects2 = plt.bar(index-bar_width, precision, bar_width,
                 alpha=opacity,
                 color='goldenrod',
                 yerr = errorprecision,
                 ecolor = 'black',
                 label='Precision')
        
        rects3 = plt.bar(index, recall, bar_width,
                 alpha=opacity,
                 color='darkorange',
                 yerr = errorsRecall,
                 ecolor = 'black',
                 label='Recall')
        
        rects1 = plt.bar(index + bar_width, falseRate, bar_width,
                 alpha=opacity,
                 yerr = errorFalse,
                 ecolor = 'black',
                 color='firebrick',
                 label='False Rate')
            
        plt.xlabel('K', fontsize=fontlabel)
        plt.ylabel('Scores', fontsize=fontlabel)
        plt.xticks(index + bar_width/2, ('1', '2', '3', '4', '5'))
        ax.set_xlim((-2*bar_width, index[len(index)-1]+3*bar_width))
        ax.set_ylim((0,1))
        ax.grid(c='gray')

#        legend = plt.legend(fontsize=fontlabel)
#        legend.get_frame().set_alpha(0.5)  
        
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        
        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  fancybox=False, shadow=False, ncol=5, fontsize=fontlabel, frameon=False)
        
    def plotDelay(self, delays, trunc='no'):
        
        fig, ax = plt.subplots()
        
        if trunc == 'no':
            ax.plot(delays[0]/4)
            ax.plot(delays[1]/4)
            ax.plot(delays[2]/4)
#            ax.plot(delays[3]/4)
#            ax.plot(delays[4]/4)
        else:
            ax.plot(numpy.trunc(delays[0]/4))
            ax.plot(numpy.trunc(delays[1]/4))
            ax.plot(numpy.trunc(delays[2]/4))
#            ax.plot(numpy.trunc(delays[3]/4))
#            ax.plot(numpy.trunc(delays[4]/4))
            
        ax.grid(c='gray')
#        ax.set_ylim(0,10)
        
        ax.set_xlabel('#hop', fontsize=fontlabel)
        ax.set_ylabel('delay', fontsize=fontlabel)
        
        xticks = [0, 1, 2]
        labels = [0, 1, 2]
            
        ax.legend()
        plt.xticks(xticks, labels)
        
    def plotBarDelay(self, delays, confInterval, trunc='no'):
        
        n_groups = 5
        index = numpy.arange(n_groups)
        bar_width = 0.2
        opacity = 0.7

        fig, ax = plt.subplots(figsize=(10,6))
        
        rects2 = plt.bar(index-bar_width, delays[:,0], bar_width,
                 alpha=opacity,
                 color='lightslategrey',
                 yerr = confInterval['hop0'],
                 ecolor = 'black',
                 label='Node')
        
        rects3 = plt.bar(index, delays[:,1], bar_width,
                 alpha=opacity,
                 color='cadetblue',
                 yerr = confInterval['hop1'],
                 ecolor = 'black',
                 label='1 Hop')
        
        rects1 = plt.bar(index + bar_width, delays[:,2], bar_width,
                 alpha=opacity,
                 color='seagreen',
                 yerr = confInterval['hop2'],
                 ecolor = 'black',
                 label='2 Hops')
            
        plt.xlabel('K', fontsize=fontlabel)
        plt.ylabel('Delay: samples', fontsize=fontlabel)
#        plt.title('Scores by group and gender')
        plt.xticks(index + bar_width/2, ('1', '2', '3', '4', '5'))
        ax.set_xlim((-2*bar_width, index[4]+3*bar_width))
#        ax.set_ylim((0,5dd))
        ax.grid(c='gray')

        legend = plt.legend(loc=1, fontsize=fontlabel)
        legend.get_frame().set_alpha(0.5)        
#        plt.tight_layout()
#        plt.show()
        
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        
        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  fancybox=False, shadow=False, ncol=5, fontsize=fontlabel, frameon=False)
        
        
        