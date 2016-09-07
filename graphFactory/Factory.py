###
#Class : Factory
#Attributs:
#   sess_name : str
#   data_engine : DataEngine
#Methodes:
#   generate_data_by_classes()
#   datarows_analyses()
###
import os
import csv
import networkx as nx
import numpy as np

class Factory:
    def __init__(self,data_engine):
        self._sess_name = ''
        self._data_engine = data_engine

    @property
    def sess_name(self):
        return self._sess_name

    @sess_name.setter
    def sess_name(self,name):
        self._sess_name = name

    @property
    def data_engine(self):
        return self._data_engine

    @data_engine.setter
    def data_engine(self,data_engine):
        self._data_engine = data_engine

    def generate_data_by_classes(self,nb_element_by_class,params):
        directory = "Datarows/"+self.sess_name+"/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        arr_csv_pts = {}
        for label in self.data_engine.arr_labels:
            arr_csv_pts[label] = csv.writer(open(directory+self.sess_name+"_"+label+".csv","a"))
        for step in range(nb_element_by_class):
            for label in self.data_engine.arr_labels:
                G = self.data_engine.generate_element(params,label)
                A = nx.to_numpy_matrix(G)
                A = np.reshape(A,-1)
                arr_A = np.squeeze(np.asarray(A))
                arr_csv_pts[label].writerow(np.append(arr_A, label))


class GraphFactory(Factory):
    def __init__(self,data_engine,sess_name='DefaultGraph'):
        self._sess_name = sess_name
        self._data_engine = data_engine


    def datarows_analyses(self):
        pass
