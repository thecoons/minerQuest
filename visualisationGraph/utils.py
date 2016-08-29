#!/usr/bin/python3.5
#-*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy import stats
from scipy.stats import kendalltau

def gridVisualisationData(input_data,type_graph):
    g = sns.PairGrid(input_data,vars=['nb_node','nb_edge','nb_connex'],hue='Value')
    g.map_diag(plt.hist)
    g.map_offdiag(plt.scatter)
    g.savefig("Datarows/"+type_graph+"/datarows_grid.png")
