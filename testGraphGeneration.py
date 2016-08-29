#!/usr/bin/python3.5
#-*- coding: utf-8 -*-
from graphFactory.utils import *
from visualisationGraph.utils import *

SESSION1 = 'RDM'
SESSION2 = 'NormalDistrib'
SESSION3 = 'ErdosRenyi'
#
# graphFactoryPlanar(1000,28,SESSION1)
# graphFactoryPlanarNormalDistribution(1000,28,SESSION2,0.08,0.03)
# graphFactoryPlanarErdosRenyiGenration(1000,28,SESSION3,0.15)
#
# datarowsFactory(SESSION1)
# datarowsFactory(SESSION2)
# datarowsFactory(SESSION3)

input_data1 = pd.read_csv('Datarows/'+SESSION1+'/data_train.csv')
input_data2 = pd.read_csv('Datarows/'+SESSION2+'/data_train.csv')
input_data3 = pd.read_csv('Datarows/'+SESSION3+'/data_train.csv')
# print(input_data.head())
# sns.set(style="ticks")
# fig = sns.jointplot(x='785', y='786', kind="hex",stat_func=None, data=input_data, color="#4CB391")
# fig.savefig("testout.png")
# print(input_data)
# input_data = input_data.iloc[:,-4:]
# input_data = input_data.astype(np.float)
gridVisualisationData(input_data1,SESSION1)
gridVisualisationData(input_data2,SESSION2)
gridVisualisationData(input_data3,SESSION3)
