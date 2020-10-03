from ....core import *
from abc import ABC, abstractmethod

class DatasetBase(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def get_dataset_and_metric(self):
        pass

    @abstractmethod
    def get_dataset_name(self):
        pass

    




    
