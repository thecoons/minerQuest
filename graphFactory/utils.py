#!/usr/bin/python3.5
#-*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import planarity as pl
import pandas as pd

import csv
import os


# Génére une Datarow brute Graph Planar (random density) #
def graphFactoryPlanar(nb_graph, size_graph, graphtype, section='all'):
    cpt = 0
    while cpt <= nb_graph:
        m = np.random.random_sample(1)
        G = nx.gnm_random_graph(size_graph,edgeForDensity(size_graph,m))
        if graphToCSV(G,graphtype,section,pl.is_planar(G)):
            cpt+=1
            if cpt%10 == 0:
                print(str(cpt)+'/'+str(nb_graph)+' '+str(100*cpt/nb_graph)+'%')

#Génére une Datarow brute Graph Planar (normal distribution)
def graphFactoryPlanarNormalDistribution(nb_graph,size_graph,graphtype,location,spread,section='all'):
    cpt=0
    while cpt <= nb_graph:
        rdm_density = np.random.normal(location,spread)
        G = nx.gnm_random_graph(size_graph,edgeForDensity(size_graph,rdm_density))
        if graphToCSV(G,graphtype,section,pl.is_planar(G)):
            cpt+=1
            if cpt%10 == 0:
                print(str(cpt)+'/'+str(nb_graph)+' '+str(100*cpt/nb_graph)+'%')

def graphFactoryPlanarErdosRenyiGenration(nb_graph,size_graph,graphtype,edgeProba,section='all'):
    cpt=0
    while cpt <= nb_graph:
        G = nx.gnp_random_graph(size_graph,edgeProba)
        if graphToCSV(G,graphtype,section,pl.is_planar(G)):
            cpt+=1
            if cpt%10 == 0:
                print(str(cpt)+'/'+str(nb_graph)+' '+str(100*cpt/nb_graph)+'%')


# Transforme une Datarows brute en Datarows Treain/Test/Valid #
def datarowsFactory(graphtype):
    data_false = pd.read_csv('Datarows/'+graphtype+'/all_false.csv')
    data_true = pd.read_csv('Datarows/'+graphtype+'/all_true.csv')

    print('data_false({0[0]},{0[1]})'.format(data_false.shape))
    # print(data_false.head())
    print('data_true({0[0]},{0[1]})'.format(data_true.shape))
    # print(data_true.head())
    ind_train = int((data_true.shape[0]*67)/100)
    ind_test = int((data_true.shape[0]*22)/100+ind_train)
    print('Ind #1 :'+str(ind_train),'Ind #2 :'+str(ind_test))

    graph_true_train = data_true.iloc[:ind_train,:]
    graph_true_train.columns = [x for x in range(graph_true_train.shape[1])]
    graph_true_test = data_true.iloc[ind_train:ind_test,:]
    graph_true_test.columns = [x for x in range(graph_true_test.shape[1])]
    graph_true_valid = data_true.iloc[ind_test:,:]
    graph_true_valid.columns = [x for x in range(graph_true_valid.shape[1])]

    print('graph_true_train({0[0]},{0[1]})'.format(graph_true_train.shape))
    print('graph_true_test({0[0]},{0[1]})'.format(graph_true_test.shape))
    print('graph_true_valid({0[0]},{0[1]})'.format(graph_true_valid.shape))

    graph_false_train = data_false.iloc[:ind_train,:]
    graph_false_train.columns = [x for x in range(graph_false_train.shape[1])]
    graph_false_test = data_false.iloc[ind_train:ind_test,:]
    graph_false_test.columns = [x for x in range(graph_false_test.shape[1])]
    graph_false_valid = data_false.iloc[ind_test:,:]
    graph_false_valid.columns = [x for x in range(graph_false_valid.shape[1])]

    print('graph_false_train({0[0]},{0[1]})'.format(graph_false_train.shape))
    print('graph_false_test({0[0]},{0[1]})'.format(graph_false_test.shape))
    print('graph_false_valid({0[0]},{0[1]})'.format(graph_false_valid.shape))


    graph_train = pd.concat([graph_true_train,graph_false_train])
    graph_test = pd.concat([graph_true_test,graph_false_test])
    graph_valid = pd.concat([graph_true_valid,graph_false_valid])

    print('graph_train({0[0]},{0[1]})'.format(graph_train.shape))
    print('graph_test({0[0]},{0[1]})'.format(graph_test.shape))
    print('graph_valid({0[0]},{0[1]})'.format(graph_valid.shape))
    # print(graph_train.head())

    ind_node = ['#'+str(i) for i in range(784)]
    meta = ['Value','nb_node','nb_edge','nb_connex']

    graph_train.columns = graph_test.columns = graph_valid.columns = ind_node + meta


    graph_train.to_csv('Datarows/'+graphtype+'/data_train.csv',index=False)
    print('Datarows/'+graphtype+'/data_train.csv write!')
    graph_test.to_csv('Datarows/'+graphtype+'/data_test.csv',index=False)
    print('Datarows/'+graphtype+'/data_test.csv write!')
    graph_valid.to_csv('Datarows/'+graphtype+'/data_valid.csv',index=False)
    print('Datarows/'+graphtype+'/data_valid.csv write!')



# Evalue le nombre d'arrête d'un graph pour densité donnée #
def edgeForDensity(n,density):
    return (n*(n-1)*density)/2

# Export un graph en Datarows CSV #
def graphToCSV(G,graphtype, section, test):
    directory = "Datarows/"+graphtype+"/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    writer_true = csv.writer(open(directory+section+"_true.csv", "a"))
    writer_false = csv.writer(open(directory+section+"_false.csv", "a"))
    A = nx.to_numpy_matrix(G)
    A = np.reshape(A, -1)
    arrGraph = np.squeeze(np.asarray(A))

    nb_nodes = 0
    for node in nx.nodes_iter(G):
        if len(G.neighbors(node))>0:
            nb_nodes += 1

    meta_info = [test,nb_nodes,G.number_of_edges(),nx.number_connected_components(G)]
    # On garde la même taille d'élemt de valeur de vérité #
    if test:
        if os.path.getsize(directory+section+"_true.csv") <= os.path.getsize(directory+section+"_false.csv"):
            writer_true.writerow(np.append(arrGraph, meta_info))
            return True
        else:
            return False
    else:
        if os.path.getsize(directory+section+"_false.csv") <= os.path.getsize(directory+section+"_true.csv"):
            writer_false.writerow(np.append(arrGraph, meta_info))
            return True
        else:
            return False
