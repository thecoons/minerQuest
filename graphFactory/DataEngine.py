###
#Class : DataEngine
#Attributs:
#   ~
#Methodes:
#   elementGenerator()
###

class DataEngine:
    def __init__(self):
        self._sess_name = ''

    @property
    def sess_name(self):
        return self._sess_name

    @sess_name.setter
    def sess_name(self,name):
        self._sess_name = name

class GraphConnexBinEngine(DataEngine):
    def __init__(self):
        self._sess_name = 'GraphConnexBin'
