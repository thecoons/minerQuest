import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import itertools as it
import numpy as np
import networkx as nx
import planarity as pl
from tensorflow.examples.tutorials.mnist import input_data
from pylab import *
import csv
import os
import random, string

def randomword(length):
    return ''.join(random.choice(string.lowercase) for i in range(length))

def exportImgTop(img, name, path=''):
    # (784) => (28,28)
    one_image_entry = img.reshape(28, 28)
    one_image_half = np.triu(one_image_entry, 1)
    one_image = np.maximum(one_image_half, one_image_half.transpose())
    plt.matshow(one_image, cmap=plt.cm.gray)
    savefig(path+name+'.png')
    plt.clf()

def exportImgDown(img, name, path=''):
    # (784) => (28,28)
    one_image_entry = img.reshape(28, 28)
    one_image_half = np.tril(one_image_entry, 1)
    one_image = np.maximum(one_image_half, one_image_half.transpose())
    plt.matshow(one_image, cmap=plt.cm.gray)
    savefig(path+name+'.png')
    plt.clf()

def exportImg(img, name, path=''):
    # (784) => (28,28)
    one_image = img.reshape(28,28)
    plt.matshow(one_image, cmap=plt.cm.gray)
    savefig(path+name+'.png')
    plt.clf()

def exportImageAsGraph(img, name, path=''):
    one_image_entry = img.reshape(28, 28)
    one_image_top = np.triu(one_image_entry, 1)
    # one_image_bot = np.tril(one_image_entry,1)
    one_image_top_full = np.maximum(one_image_top, one_image_top.transpose())
    G = nx.from_numpy_matrix(one_image_top_full)
    # outdeg = G.out_degree()
    to_remove = [n for n in G.nodes_iter() if G.degree(n)==0]
    # for n in G.nodes_iter():
    #     if(G.degree(n)==0):

    G.remove_nodes_from(to_remove)
    nx.draw_circular(G)
    plt.savefig('img_rdm_graph_'+name+'.png')
    plt.clf()

def exportRandomGraph(name, path=''):
    G = nx.newman_watts_strogatz_graph(6,2,0.5)
    nx.draw(G)
    plt.savefig('img_rdm_generate_graph_'+name+'.png')
    plt.clf()
    print(pl.is_planar(G))

def exportGraph(G, name, path=''):
    nx.draw(G)
    plt.savefig('img_'+name+'.png')
    plt.clf()
    print(pl.is_planar(G))

def graphToCSV(G,graphtype, section, test):
    directory = "Datarows/"+graphtype+"/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    writer_true = csv.writer(open(directory+section+"_true.csv", "ab"))
    writer_false = csv.writer(open(directory+section+"_false.csv", "ab"))
    A = nx.to_numpy_matrix(G)
    A = np.reshape(A, -1)
    arrGraph = np.squeeze(np.asarray(A))
    if test:
        if os.path.getsize(directory+section+"_true.csv") <= os.path.getsize(directory+section+"_false.csv"):
            writer_true.writerow(np.append(arrGraph, test))
            return True
        else:
            return False
    else:
        if os.path.getsize(directory+section+"_false.csv") <= os.path.getsize(directory+section+"_true.csv"):
            writer_false.writerow(np.append(arrGraph, test))
            return True
        else:
            return False

def graphFactoryPlanar(nb_graph, size_graph, graphtype, section='all'):
    cpt = 0
    while cpt <= nb_graph:
        m = np.random.random_sample(1)
        G = nx.gnm_random_graph(size_graph,edgeForDensity(size_graph,m))
        if graphToCSV(G,graphtype,section,pl.is_planar(G)):
            cpt+=1
        if cpt%10 == 0:
            print(str(cpt)+'/'+str(nb_graph)+' '+str(100*cpt/nb_graph)+'%')

        # print(nx.density(G),m,edgeForDensity(size_graph,m),pl.is_planar(G))
        # exportAsGraph(G,randomword(2)+"_"+str(size_graph)+"_"+str(nx.density(G)))

def datarowsFactory(graphtype):
    data_false = pd.read_csv('Datarows/'+graphtype+'/all_false.csv')
    data_true = pd.read_csv('Datarows/'+graphtype+'/all_true.csv')

    print('data_false({0[0]},{0[1]})'.format(data_false.shape))
    # print(data_false.head())
    print('data_true({0[0]},{0[1]})'.format(data_true.shape))
    # print(data_true.head())
    ind_train = (data_true.shape[0]*67)/100
    ind_test = (data_true.shape[0]*22)/100+ind_train
    print(str(ind_train),str(ind_test))

    graph_true_train = data_true.iloc[:ind_train,:]
    graph_true_train.columns = [x for x in xrange(graph_true_train.shape[1])]
    graph_true_test = data_true.iloc[ind_train:ind_test,:]
    graph_true_test.columns = [x for x in xrange(graph_true_test.shape[1])]
    graph_true_valid = data_true.iloc[ind_test:,:]
    graph_true_valid.columns = [x for x in xrange(graph_true_valid.shape[1])]

    print('graph_true_train({0[0]},{0[1]})'.format(graph_true_train.shape))
    print('graph_true_test({0[0]},{0[1]})'.format(graph_true_test.shape))
    print('graph_true_valid({0[0]},{0[1]})'.format(graph_true_valid.shape))

    graph_false_train = data_false.iloc[:ind_train,:]
    graph_false_train.columns = [x for x in xrange(graph_false_train.shape[1])]
    graph_false_test = data_false.iloc[ind_train:ind_test,:]
    graph_false_test.columns = [x for x in xrange(graph_false_test.shape[1])]
    graph_false_valid = data_false.iloc[ind_test:,:]
    graph_false_valid.columns = [x for x in xrange(graph_false_valid.shape[1])]

    print('graph_false_train({0[0]},{0[1]})'.format(graph_false_train.shape))
    print('graph_false_test({0[0]},{0[1]})'.format(graph_false_test.shape))
    print('graph_false_valid({0[0]},{0[1]})'.format(graph_false_valid.shape))


    graph_train = pd.concat([graph_true_train,graph_false_train],ignore_index=True)
    graph_test = pd.concat([graph_true_test,graph_false_test])
    graph_valid = pd.concat([graph_true_valid,graph_false_valid])

    print('graph_train({0[0]},{0[1]})'.format(graph_train.shape))
    print('graph_test({0[0]},{0[1]})'.format(graph_test.shape))
    print('graph_valid({0[0]},{0[1]})'.format(graph_valid.shape))
    # print(graph_train.head())

    graph_train.to_csv('Datarows/'+graphtype+'/data_train.csv')
    print('Datarows/'+graphtype+'/data_train.csv write!')

    graph_test.to_csv('Datarows/'+graphtype+'/data_test.csv')
    print('Datarows/'+graphtype+'/data_test.csv write!')

    graph_valid.to_csv('Datarows/'+graphtype+'/data_valid.csv')
    print('Datarows/'+graphtype+'/data_valid.csv write!')

def edgeForDensity(n,density):
    return (n*(n-1)*density)/2


# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# print(mnist.train.images[0],mnist.train.labels[0])
# for img in mnist.train.images[:1]:
#     exportImg(img,'img_'+randomword(2))
#     exportImgDown(img,'img_'+randomword(2))
#     exportImgTop(img,'img_'+randomword(2))
#     exportGraph(img,'img_'+randomword(2))
#     exportRandomGraph('img_'+randomword(2))

# graphFactoryPlanar(100,28,'Test')

datarowsFactory('Test')
