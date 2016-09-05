###
#Class : DataEngine
#Attributs:
#   ~
#Methodes:
#   elementGenerator()
###
import networkx as nx

class DataEngine:
    def __init__(self):
        self._engine_type = ''

    @property
    def engine_type(self):
        return self._engine_type

    @engine_type.setter
    def engine_type(self,name):
        self._engine_type = name

class GraphConnexBinEngine(DataEngine):
    def __init__(self):
        self._engine_type = 'GraphConnexBin'

    def generate_element(self,n,p,value):
        if(value):
            G = nx.gnp_random_graph(n,p)
            if(nx.number_connected_components(G)>1):
                return self.generate_element(n,p,value)
            else:
                return (G, value)
            
