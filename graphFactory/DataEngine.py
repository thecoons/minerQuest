###
#Class : DataEngine
#Attributs:
#   engine_type : str
#   arr_labels : str[]
#Methodes:
#   elementGenerator()
###
import networkx as nx
import numpy as np

class DataEngine:
    def __init__(self):
        self._engine_type = ''
        self._arr_labels = []
    @property
    def engine_type(self):
        return self._engine_type

    @engine_type.setter
    def engine_type(self,name):
        self._engine_type = name

    @property
    def _arr_labels(self):
        return self.__arr_labels

    @_arr_labels.setter
    def _arr_labels(self,arr_labels):
        self.__arr_labels = arr_labels


class GraphConnexBinEngine(DataEngine):
    def __init__(self):
        self._engine_type = 'GraphConnexBin'
        self._arr_labels = ['True','False']

    def generate_element(self,n,p,value):
        if(value):
            G = nx.gnp_random_graph(n,p)
            ##Test
            print("nbr_comp_1:"+str(nx.number_connected_components(G)))
            if(nx.number_connected_components(G)!=1):
                return self.generate_element(n,p,value)
            else:
                return (G, value)
        else:
            p1 = np.random.random_integers((n/2)-(n*0.1),(n/2)+(n*0.1))
            g_1 = nx.gnp_random_graph(p1,p)
            g_2 = nx.gnp_random_graph(n-p1,p)
            G = nx.disjoint_union(g_1,g_2)
            ##Test
            print("nbr_comp_2:"+str(nx.number_connected_components(G)))
            if(nx.number_connected_components(G)<2):
                return self.generate_element(n,p,value)
            else:
                return (G, value)
