import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import itertools as it
import numpy as np
import networkx as nx
import planarity as pl
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from pylab import *
import csv
import os
import random, string

LEARNING_RATE=1e-4
TRAINING_ITERATIONS=20000
DROPOUT = 0.5
BATCH_SIZE = 50
VALIDATION_SIZE = 2000
IMAGE_TO_DISPLAY = 33
SESSION = 'RDM'



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

def display(img,img_w,img_h):
    one_graph = img.reshape(img_w,img_h)
    G = nx.from_numpy_matrix(one_graph)
    nx.draw_circular(G)
    plt.savefig('img_graph_'+randomword(2)+'.png')
    plt.clf()

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    ind_lab = index_offset + labels_dense.ravel()
    for n in ind_lab:
        labels_one_hot.flat[int(n)] = 1
    return labels_one_hot

# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# serve data by batches
def next_batch(batch_size):

    global graphs_train
    global labels_train
    global index_in_epoch
    global epochs_completed
    global num_examples

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        graphs_train = graphs_train[perm]
        labels_train = labels_train[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return graphs_train[start:end], labels_train[start:end]

###START###

data = pd.read_csv('Datarows/'+SESSION+'/data_train.csv')
print('data({0[0]},{0[1]})'.format(data.shape))
# print (data.head())
graphs = data.iloc[:, 1:-1].values
graphs = graphs.astype(np.float)
print('graphs({0[0]},{0[1]})'.format(graphs.shape))

graph_size = graphs.shape[1]
print('image size => {0}'.format(graph_size))

graph_width = graph_height = np.ceil(np.sqrt(graph_size)).astype(np.uint8)
print('graph_width => {0}\ngraph_height => {0}'.format(graph_width, graph_height))

display(graphs[IMAGE_TO_DISPLAY], graph_width, graph_height)

labels_flat = data[[-1]].values.ravel()

print('labels_flat({0})'.format(len(labels_flat)))
print ('labels_flat[{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels_flat[IMAGE_TO_DISPLAY]))

labels_count = np.unique(labels_flat).shape[0]

print('labels_count => {0}'.format(labels_count))

labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

print('labels({0[0]},{0[1]})'.format(labels.shape))
print ('labels[{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels[IMAGE_TO_DISPLAY]))

graphs_validation = graphs[:VALIDATION_SIZE]
labels_validation = labels[:VALIDATION_SIZE]

graphs_train = graphs[VALIDATION_SIZE:]
labels_train = labels[VALIDATION_SIZE:]

print('graphs_train({0[0]},{0[1]})'.format(graphs_train.shape))
print('graphs_validation({0[0]},{0[1]})'.format(graphs_validation.shape))
# input & output of NN

# images
x = tf.placeholder('float', shape=[None, graph_size])
# labels
y_ = tf.placeholder('float', shape=[None, labels_count])

# first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# (40000,784) => (40000,28,28,1)
graph = tf.reshape(x, [-1,graph_width , graph_height,1])
#print (image.get_shape())
# =>(40000,28,28,1)


h_conv1 = tf.nn.relu(conv2d(graph, W_conv1) + b_conv1)
#print (h_conv1.get_shape()) # => (40000, 28, 28, 32)
h_pool1 = max_pool_2x2(h_conv1)
#print (h_pool1.get_shape()) # => (40000, 14, 14, 32)


# Prepare for visualization
# display 32 fetures in 4 by 8 grid
layer1 = tf.reshape(h_conv1, (-1, graph_height, graph_width, 4 ,8))

# reorder so the channels are in the first dimension, x and y follow.
layer1 = tf.transpose(layer1, (0, 3, 1, 4,2))

layer1 = tf.reshape(layer1, (-1, graph_height*4, graph_width*8))

# second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#print (h_conv2.get_shape()) # => (40000, 14,14, 64)
h_pool2 = max_pool_2x2(h_conv2)
#print (h_pool2.get_shape()) # => (40000, 7, 7, 64)

# Prepare for visualization
# display 64 fetures in 4 by 16 grid
layer2 = tf.reshape(h_conv2, (-1, 14, 14, 4 ,16))

# reorder so the channels are in the first dimension, x and y follow.
layer2 = tf.transpose(layer2, (0, 3, 1, 4,2))

layer2 = tf.reshape(layer2, (-1, 14*4, 14*16))

# densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# (40000, 7, 7, 64) => (40000, 3136)
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#print (h_fc1.get_shape()) # => (40000, 1024)

# dropout
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer for deep net
W_fc2 = weight_variable([1024, labels_count])
b_fc2 = bias_variable([labels_count])

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#print (y.get_shape()) # => (40000, 10)

# cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y))


# optimisation function
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

# evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# prediction function
#[0.1, 0.9, 0.2, 0.1, 0.1 0.3, 0.5, 0.1, 0.2, 0.3] => 1
predict = tf.argmax(y,1)

epochs_completed = 0
index_in_epoch = 0
num_examples = graphs_train.shape[0]


# start TensorFlow session
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

sess.run(init)

# visualisation variables
train_accuracies = []
validation_accuracies = []
x_range = []

display_step=1

for i in range(TRAINING_ITERATIONS):

    #get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE)


    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:

        train_accuracy = accuracy.eval(feed_dict={x:batch_xs,
                                                  y_: batch_ys,
                                                  keep_prob: 1.0})
        if(VALIDATION_SIZE):
            validation_accuracy = accuracy.eval(feed_dict={ x: graphs_validation[0:BATCH_SIZE],
                                                            y_: labels_validation[0:BATCH_SIZE],
                                                            keep_prob: 1.0})
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))

            validation_accuracies.append(validation_accuracy)

        else:
             print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
        train_accuracies.append(train_accuracy)
        x_range.append(i)

        # increase display_step
        if i%(display_step*10) == 0 and i:
            display_step *= 10
    # train on batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})

    # check final accuracy on validation set
if(VALIDATION_SIZE):
    validation_accuracy = accuracy.eval(feed_dict={x: graphs_validation,
                                                   y_: labels_validation,
                                                   keep_prob: 1.0})
    print('validation_accuracy => %.4f'%validation_accuracy)
    plt.plot(x_range, train_accuracies,'-b', label='Training')
    plt.plot(x_range, validation_accuracies,'-g', label='Validation')
    plt.legend(loc='lower right', frameon=False)
    plt.ylim(ymax = 1.1, ymin = 0.7)
    plt.ylabel('accuracy')
    plt.xlabel('step')
    savefig('img_accuracy.png')
    plt.clf()

# read test data from CSV file
test_data = pd.read_csv('Datarows/'+SESSION+'/data_test.csv')
test_graphs = test_data.iloc[:,1:-1].values
test_labels = test_data.iloc[:,-1:].values
print('test_graphs({0[0]},{0[1]})'.format(test_graphs.shape))
print('test_labels({0[0]},{0[1]})'.format(test_labels.shape))


# predict test set
#predicted_lables = predict.eval(feed_dict={x: test_graphs, keep_prob: 1.0})

# using batches is more resource efficient
predicted_lables = np.zeros(test_graphs.shape[0])
for i in range(0,test_graphs.shape[0]//BATCH_SIZE):
    predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict={x: test_graphs[i*BATCH_SIZE : (i+1)*BATCH_SIZE],
                                                                                keep_prob: 1.0})


print('predicted_lables({0})'.format(len(predicted_lables)))

# output test image and prediction
display(test_graphs[IMAGE_TO_DISPLAY],graph_width,graph_height)
print ('predicted_lables[{0}] => {1}'.format(IMAGE_TO_DISPLAY,predicted_lables[IMAGE_TO_DISPLAY]))
print ('right_lables[{0}] => {1}'.format(IMAGE_TO_DISPLAY,test_labels.item((IMAGE_TO_DISPLAY))))

# save results
np.savetxt('submission_softmax.csv',
           np.c_[range(1,len(test_graphs)+1),predicted_lables],
           delimiter=',',
           header = 'ImageId,Label',
           comments = '',
           fmt='%d')

layer1_grid = layer1.eval(feed_dict={x: test_graphs[IMAGE_TO_DISPLAY:IMAGE_TO_DISPLAY+1], keep_prob: 1.0})
plt.axis('off')
plt.imshow(layer1_grid[0], cmap=cm.seismic )
savefig('img_convLayer.png')
plt.clf()
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# print(mnist.train.images[0],mnist.train.labels[0])
# for img in mnist.train.images[:1]:
#     exportImg(img,'img_'+randomword(2))
#     exportImgDown(img,'img_'+randomword(2))
#     exportImgTop(img,'img_'+randomword(2))
#     exportGraph(img,'img_'+randomword(2))
#     exportRandomGraph('img_'+randomword(2))

# graphFactoryPlanar(100000,28,'RDM')

# datarowsFactory('RDM')
