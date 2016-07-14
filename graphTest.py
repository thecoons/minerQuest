import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import networkx as nx
from tensorflow.examples.tutorials.mnist import input_data
from pylab import *
import random, string

def randomword(length):
   return ''.join(random.choice(string.lowercase) for i in range(length))

def exportImgTop(img,name,path=''):
    # (784) => (28,28)
    one_image_entry = img.reshape(28,28)
    one_image_half = np.triu(one_image_entry,1)
    one_image = np.maximum(one_image_half,one_image_half.transpose())
    plt.matshow(one_image,cmap=plt.cm.gray)
    savefig(path+name+'.png')

def exportImgDown(img,name,path=''):
    # (784) => (28,28)
    one_image_entry = img.reshape(28,28)
    one_image_half = np.tril(one_image_entry,1)
    one_image = np.maximum(one_image_half,one_image_half.transpose())
    plt.matshow(one_image,cmap=plt.cm.gray)
    savefig(path+name+'.png')

def exportImg(img,name,path=''):
    # (784) => (28,28)
    one_image = img.reshape(28,28)
    plt.matshow(one_image,cmap=plt.cm.gray)
    savefig(path+name+'.png')

# def exportGraph(graph):


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(mnist.train.images[0],mnist.train.labels[0])
for img in mnist.train.images[:1]:
    exportImg(img,'img_'+randomword(2))
    exportImgDown(img,'img_'+randomword(2))
    exportImgTop(img,'img_'+randomword(2))
