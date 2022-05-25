import abc
import os 
import pandas as pd
from autogluon.text.automm.constants import (
    BINARY,
    MULTICLASS,
    REGRESSION,
    ACC,
    RMSE,
    CATEGORICAL,
    NUMERICAL,
)
from utils import download


class BaseTabularDataset(abc.ABC):
    @property
    @abc.abstractmethod
    def data(self):
        pass
    
    @property
    @abc.abstractmethod
    def label_column(self):
        pass
    
    @property
    @abc.abstractmethod
    def label_type(self):
        pass
    
    @property
    @abc.abstractmethod
    def metric(self):
        pass

    @property
    @abc.abstractmethod
    def problem_type(self):
        pass

  
class AdultTabularDataset(BaseTabularDataset):
    _INFO = {
        'train': {
            'url': 's3://autogluon/datasets/tabular/adult/train.csv',
            'sha1sum': '0a797588f36e05b740cd4fca518e12afa2aa7650'
        },
        'val': {
            'url': 's3://autogluon/datasets/tabular/adult/val.csv',
            'sha1sum': '26b01ed3806bebe2004a2564fd2081b6888ac56c'
        },
        'test': {
            'url': 's3://autogluon/datasets/tabular/adult/test.csv',
            'sha1sum': 'c8842fc31699c582746926ab274d13451c4415fd'
        },
    }
    def __init__(self, split='train', path='./dataset/'):
        self._split = split
        self._path = os.path.join(path,'adult',f'{split}.csv')
        download(self._INFO[split]['url'],
                 path=self._path,
                 sha1_hash=self._INFO[split]['sha1sum'])
        self._data = pd.read_csv(self._path)
    
    @property  
    def data(self):
        return self._data
    
    @property
    def label_column(self):
        return 'target'
    
    @property
    def label_type(self):
        return NUMERICAL
    
    @property
    def metric(self):
        return ACC
    
    @property
    def problem_type(self):
        return BINARY
