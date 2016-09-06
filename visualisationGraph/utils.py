#!/usr/bin/python3.5
#-*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

from networkx.drawing.nx_agraph import graphviz_layout
from scipy import stats
from scipy.stats import kendalltau

def gridVisualisationData(input_data,type_graph):
    g = sns.PairGrid(input_data,vars=['nb_node','nb_edge','nb_connex'],hue='Value')
    g.map_diag(plt.hist)
    g.map_offdiag(plt.scatter)
    g.savefig("Datarows/"+type_graph+"/datarows_grid.png")
    plt.clf()

# Export un graph et indique sa planarité#
def exportGraph(mat, name, path=''):
    G = nx.from_numpy_matrix(mat)
    pos = graphviz_layout(G)
    nx.draw(G,pos)
    plt.savefig(path+'img_'+name+'.png')
    plt.clf()

def exportAccuracyLearning(train_acu,valid_acu,x_range,path=''):
    plt.plot(x_range, train_acu,'-b', label='Training')
    plt.plot(x_range, valid_acu,'-g', label='Validation')
    plt.legend(loc='lower right', frameon=False)
    plt.ylim(ymax = 1.1, ymin = 0.7)
    plt.ylabel('accuracy')
    plt.xlabel('step')
    plt.savefig('Datarows/'+path+'/accuracyTrainingValidation.png')
    plt.clf()
