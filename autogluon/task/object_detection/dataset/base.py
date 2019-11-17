from ....core import *
from abc import ABC, abstractmethod

class DatasetBase(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def get_train_val_metric(self):
        pass

    @abstractmethod
    def get_dataset_name(self):
        pass

    




    
