#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 10:14:24 2017

@author: anr.putina
"""

class groundTruth():
    
    def __init__(self):
        self.events = []
        self.clears = []
        pass
    
    def setEvents(self, events):
        pass
    
    def addEvent(self, event):
        self.events.append(event)
        
    def addClear(self, clear):
        self.clears.append(clear)
        
    def simulationBGP_CLEAR(self):
        eventRecord = {
            'name': 'event1',
            'startTime': 1498754415519000000,
            'startIndex': 76,
            'endTime': 1498754521434000000,
            'endIndex': 100,
            'node': 'leaf1',
            'type': 'single'
        }
        
        self.addEvent(eventRecord)
        
        eventRecord = {
            'name': 'event2',
            'startTime': 1498754656637000000,
            'startIndex': 129,
            'endTime': 1498754760063000000,
            'endIndex': 154,
            'node': 'spine4',
            'type': 'single'
        }
        
        self.addEvent(eventRecord)
        
        eventRecord = {
            'name': 'event3',
            'startTime': 1498754885121000000,
            'startIndex': 183,
            'endTime': 1498755000770000000,
            'endIndex': 208,
            'node': 'leaf8',
            'type': 'single'

        }
        
        self.addEvent(eventRecord)
        
        eventRecord = {
            'name': 'event4',
            'startTime': 1498755124567000000,
            'startIndex': 232,
            'endTime': 1498755304432000000,
            'endIndex': 276,
            'node': 'spine2',
            'type': 'single'
        }
        
        self.addEvent(eventRecord)
        
        eventRecord = {
            'name': 'event5',
            'startTime': 1498755424734000000,
            'startIndex': 303,
            'endTime': 1498755601181000000,
            'endIndex': 343,
            'node': 'leaf2',
            'type': 'single'

        }
        
        self.addEvent(eventRecord)
        
        eventRecord = {
            'name': 'event5',
            'startTime': 1498755419850000000,
            'startIndex': 304,
            'endTime': 1498755601181000000,
            'endIndex': 343,
            'node': 'leaf6',
            'type': 'multiple'
        }
        
        self.addEvent(eventRecord)
        
        eventRecord = {
            'name': 'event6',
            'startTime': 1498755721694000000,
            'startIndex': 366,
            'endTime': 1498755900264000000,
            'endIndex': 410,
            'node': 'spine1',
            'type': 'single'
        }

        self.addEvent(eventRecord)
        
        eventRecord = {
            'name': 'event6',
            'startTime': 1498755722500000000,
            'startIndex': 366,
            'endTime': 1498755900264000000,
            'endIndex': 410,
            'node': 'spine3',
            'type': 'multiple'
        }

        self.addEvent(eventRecord)
        
        eventRecord = {
            'name': 'event7',
            'startTime': 1498756020858000000,
            'startIndex': 437,
            'endTime': 1498756199112000000,
            'endIndex': 477,
            'node': 'spine1',
            'type': 'single'
        }
        
        self.addEvent(eventRecord)
        
        eventRecord = {
            'name': 'event7',
            'startTime': 1498756020858000000,
            'startIndex': 438,
            'endTime': 1498756199112000000,
            'endIndex': 477,
            'node': 'spine2',
            'type': 'multiple'
        }
        
        self.addEvent(eventRecord)
        
        eventRecord = {
            'name': 'event7',
            'startTime': 1498756044535000000,
            'startIndex': 438,
            'endTime': 1498756199112000000,
            'endIndex': 477,
            'node': 'spine3',
            'type': 'multiple'
        }
        self.addEvent(eventRecord)

    def simulationBGP_CLEAR2(self):
        eventRecord = {
            'name': 'event1',
            'startTime': 1498754400000,
            'endTime': 1498754520000,
            'node': 'leaf1',
            'type': 'single'
        }
        
        self.addEvent(eventRecord)
        
        eventRecord = {
            'name': 'event2',
            'startTime': 1498754640000,
            'endTime': 1498754760000,
            'node': 'spine4',
            'type': 'single'
        }
        
        self.addEvent(eventRecord)
        
        eventRecord = {
            'name': 'event3',
            'startTime': 1498754880000,
            'endTime': 1498755000000,
            'node': 'leaf8',
            'type': 'single'

        }
        
        self.addEvent(eventRecord)
        
        eventRecord = {
            'name': 'event4',
            'startTime': 1498755120000,
            'endTime': 1498755300000,
            'node': 'spine2',
            'type': 'single'
        }
        
        self.addEvent(eventRecord)
        
        eventRecord = {
            'name': 'event5',
            'startTime': 1498755420000,
            'endTime': 1498755600000,
            'node': 'leaf2',
            'type': 'single'

        }
        
        self.addEvent(eventRecord)
        
        eventRecord = {
            'name': 'event5',
            'startTime': 1498755420000,
            'endTime': 1498755600000,
            'node': 'leaf6',
            'type': 'multiple'
        }
        
        self.addEvent(eventRecord)
        
        eventRecord = {
            'name': 'event6',
            'startTime': 1498755720000,
            'endTime': 1498755900000,
            'node': 'spine1',
            'type': 'single'
        }

        self.addEvent(eventRecord)
        
        eventRecord = {
            'name': 'event6',
            'startTime': 1498755720000,
            'endTime': 1498755900000,
            'node': 'spine3',
            'type': 'multiple'
        }

        self.addEvent(eventRecord)
        
        eventRecord = {
            'name': 'event7',
            'startTime': 1498756020000,
            'endTime': 1498756200000,
            'node': 'spine1',
            'type': 'single'
        }
        
        self.addEvent(eventRecord)
        
        eventRecord = {
            'name': 'event7',
            'startTime': 1498756020000,
            'endTime': 1498756200000,
            'node': 'spine2',
            'type': 'multiple'
        }
        
        self.addEvent(eventRecord)
        
        eventRecord = {
            'name': 'event7',
            'startTime': 1498756020000,
            'endTime': 1498756200000,
            'node': 'spine3',
            'type': 'multiple'
        }
        
        self.addEvent(eventRecord)
        
        
    def simulationBGP_CLEAR3(self):
        
        ##indexStart : 76
        eventRecord = {
            'name': 'event1',
            'startTime': 1498754415519,
            'endTime': 1498754580441,
            'startidx': 76,
            'endidx': 100,
            'node': 'leaf1',
            'type': 'single',
            'entity': 'single'
        }
        
        self.addEvent(eventRecord)
        
        ### indexStart : 129
        eventRecord = {
            'name': 'event2',
            'startTime': 1498754656637,
            'endTime': 1498754820000,
            'node': 'spine4',
            'type': 'single',
            'entity': 'single'

        }
        self.addEvent(eventRecord)
        
        ### indexStart : 183
        eventRecord = {
            'name': 'event3',
            'startTime': 1498754885121,
            'endTime': 1498755060000,
            'node': 'leaf8',
            'type': 'single',
            'entity': 'single'


        }
        self.addEvent(eventRecord)
        
        ### indexStart : 232
        eventRecord = {
            'name': 'event4',
            'startTime': 1498755124567,
            'endTime': 1498755300000,
            'node': 'spine2',
            'type': 'single',
            'entity': 'single'

        }
        self.addEvent(eventRecord)
        
        ### indexStart : 303 
        eventRecord = {
            'name': 'event5leaf2',
            'startTime': 1498755424540,
            'endTime': 1498755600000,
            'node': 'leaf2',
            'type': 'single',
            'entity': 'multiple'


        }
        self.addEvent(eventRecord)
        
        ### indexStart : 304
        eventRecord = {
            'name': 'event5leaf6',
            'startTime': 1498755429032,
            'endTime': 1498755600000,
            'node': 'leaf6',
            'type': 'multiple',
            'entity': 'multiple'

        }
        self.addEvent(eventRecord)
        
        ### indexStart : 366
        eventRecord = {
            'name': 'event6spine1',
            'startTime': 1498755727878,
            'endTime': 1498755900000,
            'node': 'spine1',
            'type': 'single',
            'entity': 'multiple'

        }
        self.addEvent(eventRecord)
        
        ### indexStart : 368
        eventRecord = {
            'name': 'event6spine3',
            'startTime': 1498755731460,
            'endTime': 1498755900000,
            'node': 'spine3',
            'type': 'multiple',
            'entity': 'multiple'

        }
        self.addEvent(eventRecord)

        ### indexStart : 437        
        eventRecord = {
            'name': 'event7spine1',
            'startTime': 1498756046457,
            'endTime': 1498756200000,
            'node': 'spine1',
            'type': 'single',
            'entity': 'multiple'

        }
        self.addEvent(eventRecord)
        
        ### indexStart : 437
        eventRecord = {
            'name': 'event7spine2',
            'startTime': 1498756045516,
            'endTime': 1498756200000,
            'node': 'spine2',
            'type': 'multiple',
            'entity': 'multiple'

        }
        self.addEvent(eventRecord)
        
        ### indexStart : 439
        eventRecord = {
            'name': 'event7spine3',
            'startTime': 1498756044535,
            'endTime': 1498756200000,
            'node': 'spine3',
            'type': 'multiple',
            'entity': 'multiple'
        }
        
        self.addEvent(eventRecord)
        
    def simulationBGP_CLEAR3_TwoMin(self):
        
        ##indexStart : 76
        eventRecord = {
            'name': 'event1',
            'startTime': 1498754415519,
            'endTime': 1498754535519,
            'startidx': 76,
            'endidx': 100,
            'node': 'leaf1',
            'type': 'single',
            'entity': 'single'
        }
        
        self.addEvent(eventRecord)
        
        ### indexStart : 129
        eventRecord = {
            'name': 'event2',
            'startTime': 1498754656637,
            'endTime': 1498754776637,
            'node': 'spine4',
            'type': 'single',
            'entity': 'single'

        }
        self.addEvent(eventRecord)
        
        ### indexStart : 183
        eventRecord = {
            'name': 'event3',
            'startTime': 1498754885121,
            'endTime': 1498755005121,
            'node': 'leaf8',
            'type': 'single',
            'entity': 'single'


        }
        self.addEvent(eventRecord)
        
        ### indexStart : 232
        eventRecord = {
            'name': 'event4',
            'startTime': 1498755124567,
            'endTime': 1498755244567,
            'node': 'spine2',
            'type': 'single',
            'entity': 'single'

        }
        self.addEvent(eventRecord)
        
        ### indexStart : 303 
        eventRecord = {
            'name': 'event5leaf2',
            'startTime': 1498755424540,
            'endTime': 1498755600000,
            'node': 'leaf2',
            'type': 'single',
            'entity': 'multiple'


        }
        self.addEvent(eventRecord)
        
        ### indexStart : 304
        eventRecord = {
            'name': 'event5leaf6',
            'startTime': 1498755429032,
            'endTime': 1498755600000,
            'node': 'leaf6',
            'type': 'multiple',
            'entity': 'multiple'

        }
        self.addEvent(eventRecord)
        
        ### indexStart : 366
        eventRecord = {
            'name': 'event6spine1',
            'startTime': 1498755727878,
            'endTime': 1498755900000,
            'node': 'spine1',
            'type': 'single',
            'entity': 'multiple'

        }
        self.addEvent(eventRecord)
        
        ### indexStart : 368
        eventRecord = {
            'name': 'event6spine3',
            'startTime': 1498755731460,
            'endTime': 1498755900000,
            'node': 'spine3',
            'type': 'multiple',
            'entity': 'multiple'

        }
        self.addEvent(eventRecord)

        ### indexStart : 437        
        eventRecord = {
            'name': 'event7spine1',
            'startTime': 1498756046457,
            'endTime': 1498756200000,
            'node': 'spine1',
            'type': 'single',
            'entity': 'multiple'

        }
        self.addEvent(eventRecord)
        
        ### indexStart : 437
        eventRecord = {
            'name': 'event7spine2',
            'startTime': 1498756045516,
            'endTime': 1498756200000,
            'node': 'spine2',
            'type': 'multiple',
            'entity': 'multiple'

        }
        self.addEvent(eventRecord)
        
        ### indexStart : 439
        eventRecord = {
            'name': 'event7spine3',
            'startTime': 1498756044535,
            'endTime': 1498756200000,
            'node': 'spine3',
            'type': 'multiple',
            'entity': 'multiple'
        }
        
        self.addEvent(eventRecord)
        
    def simulationBGP_CLEAR_CLEAR(self):
        
        ##indexStart : 76
        eventRecord = {
            'name': 'clear1',
            'startTime': 1498754580442,
            'endTime': 1498754645636
        }
        
        self.addClear(eventRecord)
        ### OK
                
        ##indexStart : 76
        eventRecord = {
            'name': 'clear2',
            'startTime': 1498754820001,
            'endTime': 1498754885122
        }
        
        self.addClear(eventRecord)
        ### OK
                
        eventRecord = {
            'name': 'clear3',
            'startTime': 1498755060001,
            'endTime': 1498755124566
        }
        
        self.addClear(eventRecord)
        ###OK
                
        eventRecord = {
            'name': 'clear4',
            'startTime': 1498755300001,
            'endTime': 1498755424530
        }
        
        self.addClear(eventRecord)
        ### OK
        
        eventRecord = {
            'name': 'clear5',
            'startTime': 1498755600001,
            'endTime': 1498755727877
        }
        self.addClear(eventRecord)
        ### OK
        
        eventRecord = {
            'name': 'clear6',
            'startTime': 1498755900001,
            'endTime': 1498756044534
        }
        self.addClear(eventRecord)
        ### OK

#        eventRecord = {
#            'name': 'clear7',
#            'startTime': 1498756200000,
#            'endTime': 1598756200000
#        }
#        self.addClear(eventRecord)
#        ### OK
        
    def simulationPORT_FLAP(self):
        
        ##indexStart : 64
        eventRecord = {
            'name': 'event1',
            'startTime': 1504286090625,
            'endTime': 1504286362296,
            'node': 'spine2',
            'interface': 8,
            'neighbor': 'leaf3',
            'type': 'single',
            'entity': 'single'

        }
        
        self.addEvent(eventRecord)
        
        ##indexStart : 130
        eventRecord = {
            'name': 'event2',
            'startTime': 1504286455946,
            'endTime': 1504286753902,
            'node': 'leaf3',
            'interface': 8,
            'neighbor': 'spine2',
            'type': 'single',
            'entity': 'single'

        }
        
        self.addEvent(eventRecord)
        
        ##indexStart : 212
        eventRecord = {
            'name': 'event3',
            'startTime': 1504286868536,
            'endTime': 1504287184286,
            'node': 'spine2',
            'interface': 8,
            'neighbor': 'leaf3',
            'type': 'single',
            'entity': 'single'

        }
        
        self.addEvent(eventRecord)

        ##indexStart : 274
        eventRecord = {
            'name': 'event4',
            'startTime': 1504287248681,
            'endTime': 1504287531916,
            'node': 'leaf3',
            'interface': 8,
            'neighbor': 'spine2',
            'type': 'single',
            'entity': 'single'

        }
        
        self.addEvent(eventRecord)    
        
        ##indexStart : 380
        eventRecord = {
            'name': 'event5',
            'startTime': 1504287831774,
            'endTime': 1504288112632,
            'node': 'leaf3',
            'interface': 8,
            'neighbor': 'rswA5',
            'type': 'single',
            'entity': 'single'

        }
        
        self.addEvent(eventRecord)  
        
        ##indexStart : 461
        eventRecord = {
            'name': 'event6',
            'startTime': 1504288281689,
            'endTime': 1504288718480,
            'node': 'rswA5',
            'interface': 8,
            'neighbor': 'leaf3',
            'type': 'single',
            'entity': 'single'

        }
        
        self.addEvent(eventRecord)  
    
    def simulationPORT_FLAP_CLEAR(self):
        
        eventRecord = {
            'name': 'clear1',
            'startTime': 1504286362297,
            'endTime': 1504286455945
        }
        
        self.addClear(eventRecord)
        
        ##indexStart : 76
        eventRecord = {
            'name': 'clear2',
            'startTime': 1504286753903,
            'endTime': 1504286868535
        }
        self.addClear(eventRecord)
        
        eventRecord = {
            'name': 'clear3',
            'startTime': 1504287184287,
            'endTime': 1504287248680
        }
        
        self.addClear(eventRecord)
        
        eventRecord = {
            'name': 'clear4',
            'startTime': 1504287531918,
            'endTime': 1504287831775
        }
        
        self.addClear(eventRecord)
        
        eventRecord = {
            'name': 'clear5',
            'startTime': 1504288112633,
            'endTime': 1504288281690
        }
        self.addClear(eventRecord)
        
#        eventRecord = {
#            'name': 'clear6',
#            'startTime': 1504288718480,
#            'endTime': 1550000000000
#        }
#        self.addClear(eventRecord)

    def simulationBGP_CLEAR_Second_DATASET(self):
        
        ##indexStart : 76
        eventRecord = {
            'name': 'event1',
            'startTime': 1501867182617,
            'endTime': 1501867403375,
            'startidx': 39,
            'endidx': 39+50,
            'node': 'leaf1',
            'type': 'single',
            'entity': 'single',
            'ONLINE' : False,
            'endSent': False
        }
        
        self.addEvent(eventRecord)
        
        ### indexStart : 129
        eventRecord = {
            'name': 'event2',
            'startTime': 1501867481598,
            'endTime': 1501867701677,
            'node': 'leaf6',
            'type': 'single',
            'entity': 'single',
            'ONLINE' : False,
            'endSent': False

        }
        self.addEvent(eventRecord)
        
        ### indexStart : 183
        eventRecord = {
            'name': 'event3',
            'startTime': 1501867783946,
            'endTime': 1501868006351,
            'node': 'spine2',
            'type': 'single',
            'entity': 'single',
            'ONLINE' : False,
            'endSent': False

        }
        self.addEvent(eventRecord)
#        
        ### indexStart : 232
        eventRecord = {
            'name': 'event4',
            'startTime': 1501868091453,
            'endTime': 1501868312576,
            'node': 'leaf2',
            'type': 'single',
            'entity': 'single',
            'ONLINE' : False,
            'endSent': False

        }
        self.addEvent(eventRecord)
#        
        ### indexStart : 303 
        eventRecord = {
            'name': 'event5leaf2',
            'startTime': 1501868124311,
            'endTime': 1501868343454,
            'node': 'leaf8',
            'type': 'single',
            'entity': 'single',
            'ONLINE' : False,
            'endSent': False

        }
        self.addEvent(eventRecord)
#        
        ### indexStart : 304
        eventRecord = {
            'name': 'event5leaf6',
            'startTime': 1501868384250,
            'endTime': 1501868606662,
            'node': 'spine1',
            'type': 'single',
            'entity': 'single',
            'ONLINE' : False,
            'endSent': False

        }
        self.addEvent(eventRecord)
#        
        ### indexStart : 366
        eventRecord = {
            'name': 'event6spine1',
            'startTime': 1501868428974,
            'endTime': 1501868650995,
            'node': 'spine2',
            'type': 'single',
            'entity': 'single',
            'ONLINE' : False,
            'endSent': False
        }
        self.addEvent(eventRecord)
#        
        ### indexStart : 368
        eventRecord = {
            'name': 'event6spine3',
            'startTime': 1501868492556,
            'endTime': 1501868714299,
            'node': 'spine3',
            'type': 'single',
            'entity': 'single',
            'ONLINE' : False,
            'endSent': False
        }
        self.addEvent(eventRecord)

    def simulationBGP_CLEAR2_CLEAR(self):
        
        eventRecord = {
            'name': 'clear1',
            'startTime': 1501867403376,
            'endTime': 1501867481597
        }
        
        self.addClear(eventRecord)
        
        ##indexStart : 76
        eventRecord = {
            'name': 'clear2',
            'startTime': 1501867701677,
            'endTime': 1501867783946
        }
        self.addClear(eventRecord)
        
        eventRecord = {
            'name': 'clear3',
            'startTime': 1501868006351,
            'endTime': 1501868091453
        }
        
        self.addClear(eventRecord)
        
        eventRecord = {
            'name': 'clear4',
            'startTime': 1501868343454,
            'endTime': 1501868384250
        }
        
        self.addClear(eventRecord)
        
        eventRecord = {
            'name': 'clear5',
            'startTime': 1504288281690,
            'endTime': 159428828169
        }
        self.addClear(eventRecord)
        
#        eventRecord = {
#            'name': 'clear6',
#            'startTime': 1504288718480,
#            'endTime': 1550000000000
#        }
#        self.addClear(eventRecord)