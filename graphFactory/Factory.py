###
#Class : Factory
#Attributs:
#   sess_name : str
#   data_engine : DataEngine
#Methodes:
#   generate_data_by_classes()
#   datarows_analyses()
###
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

    def generate_data_by_classes(self):
        

class graphFactory(Factory):
    def __init__(self,data_engine,sess_name='DefaultGraph'):
        self._sess_name = sess_name
        self._data_engine = data_engine


    def datarows_analyses(self):
        pass
