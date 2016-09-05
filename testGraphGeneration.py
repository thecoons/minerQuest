#!/usr/bin/python3.5
#-*- coding: utf-8 -*-
from graphFactory.utils import *
from visualisationGraph.utils import *

from graphFactory.DataEngine import GraphConnexBinEngine

gcbe = GraphConnexBinEngine()
print(gcbe.engine_type)
print(gcbe.generate_element(10,0.3,True))

# SESSION1 = 'RDM'
# SESSION2 = 'NormalDistrib'
# SESSION3 = 'ErdosRenyi'

# graphFactoryPlanar(10000,28,SESSION1)
# graphFactoryPlanarNormalDistribution(10000,28,SESSION2,0.08,0.03)
# graphFactoryPlanarErdosRenyiGenration(50000,28,SESSION3,0.15)

# datarowsFactory(SESSION1)
# datarowsFactory(SESSION2)
# datarowsFactory(SESSION3)

# input_data1 = pd.read_csv('Datarows/'+SESSION1+'/data_train.csv')
# input_data2 = pd.read_csv('Datarows/'+SESSION2+'/data_train.csv')
# input_data3 = pd.read_csv('Datarows/'+SESSION3+'/data_train.csv')

# gridVisualisationData(input_data1,SESSION1)
# gridVisualisationData(input_data2,SESSION2)
# gridVisualisationData(input_data3,SESSION3)
