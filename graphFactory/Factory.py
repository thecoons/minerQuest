###
#Class : Factory
#Attributs:
#   sess_name : str
#Methodes:
#   ClassExportation()
#   DatarowFactory()
###
class Factory:
    def __init__(self):
        self._sess_name = ''

    @property
    def sess_name(self):
        return self._sess_name

    @sess_name.setter
    def sess_name(self,name):
        self._sess_name = name
