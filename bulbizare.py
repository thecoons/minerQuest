import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf

#Graph type
GRAPH_TYPE = "NormalDistrib"

# settings
LEARNING_RATE = 1e-4
# set to 20000 on local environment to get 0.99 accuracy
TRAINING_ITERATIONS = 2500

DROPOUT = 0.5
BATCH_SIZE = 50

# set to 0 to train on all available data
VALIDATION_SIZE = 2000

# image number to output
IMAGE_TO_DISPLAY = 10

# read training data from CSV file
data = pd.read_csv('Datarows/'+GRAPH_TYPE+'/datatrain.csv')

print('data({0[0]},{0[1]})'.format(data.shape))
print (data.head())
