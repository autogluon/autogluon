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


# TODO: release the auto_mm_bench package or reuse the huggingface datasets API
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
        assert split in ['train', 'val', 'test'], f'Unsupported split {split}. Split must be one of train, val, or test.'
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
        return CATEGORICAL
    
    @property
    def metric(self):
        return ACC
    
    @property
    def problem_type(self):
        return BINARY


class AloiTabularDataset(BaseTabularDataset):
    _INFO = {
        'train': {
            'url': 's3://autogluon/datasets/tabular/aloi/train.csv',
            'sha1sum': '48fec570223b865f7392aa5476040335e06e10d8'
        },
        'val': {
            'url': 's3://autogluon/datasets/tabular/aloi/val.csv',
            'sha1sum': 'e2edeafd00c56591153b47ef0d1ef52d4adb63ad'
        },
        'test': {
            'url': 's3://autogluon/datasets/tabular/aloi/test.csv',
            'sha1sum': 'a57a2dd03839949e10a12179862cdd99157a8beb'
        },
    }
    def __init__(self, split='train', path='./dataset/'):
        assert split in ['train', 'val', 'test'], f'Unsupported split {split}. Split must be one of train, val, or test.'
        self._split = split
        self._path = os.path.join(path,'aloi',f'{split}.csv')
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
        return CATEGORICAL
    
    @property
    def metric(self):
        return ACC
    
    @property
    def problem_type(self):
        return MULTICLASS


class CaliforniaHousingTabularDataset(BaseTabularDataset):
    _INFO = {
        'train': {
            'url': 's3://autogluon/datasets/tabular/california_housing/train.csv',
            'sha1sum': '0044f5d10336b17376e95b7935cc6047e20105d1'
        },
        'val': {
            'url': 's3://autogluon/datasets/tabular/california_housing/val.csv',
            'sha1sum': 'c2d79c5f041418396b45dd54d79c997851e4e168'
        },
        'test': {
            'url': 's3://autogluon/datasets/tabular/california_housing/test.csv',
            'sha1sum': '10c9594779023486d8ad136f8bbffe8dd5016d2b'
        },
    }
    def __init__(self, split='train', path='./dataset/'):
        assert split in ['train', 'val', 'test'], f'Unsupported split {split}. Split must be one of train, val, or test.'
        self._split = split
        self._path = os.path.join(path,'california_housing',f'{split}.csv')
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
        return RMSE
    
    @property
    def problem_type(self):
        return REGRESSION


class CovtypeTabularDataset(BaseTabularDataset):
    _INFO = {
        'train': {
            'url': 's3://autogluon/datasets/tabular/covtype/train.csv',
            'sha1sum': 'b73899f72eacb4e7895fd0232e503235ba7eb5b5'
        },
        'val': {
            'url': 's3://autogluon/datasets/tabular/covtype/val.csv',
            'sha1sum': '33ccd97b6aa741612ae11888adc1a6b2dc48e4e8'
        },
        'test': {
            'url': 's3://autogluon/datasets/tabular/covtype/test.csv',
            'sha1sum': 'e88918ff9878dff80a1b062874db320b259899f1'
        },
    }
    def __init__(self, split='train', path='./dataset/'):
        assert split in ['train', 'val', 'test'], f'Unsupported split {split}. Split must be one of train, val, or test.'
        self._split = split
        self._path = os.path.join(path,'covtype',f'{split}.csv')
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
        return CATEGORICAL
    
    @property
    def metric(self):
        return ACC
    
    @property
    def problem_type(self):
        return MULTICLASS


class EpsilonTabularDataset(BaseTabularDataset):
    _INFO = {
        'train': {
            'url': 's3://autogluon/datasets/tabular/epsilon/train.csv',
            'sha1sum': '8444901bdb20d42359b85ca076eff7f16a34b94c'
        },
        'val': {
            'url': 's3://autogluon/datasets/tabular/epsilon/val.csv',
            'sha1sum': '9d607e0db43979d3d9a6034dc7603ef09934b5c8'
        },
        'test': {
            'url': 's3://autogluon/datasets/tabular/epsilon/test.csv',
            'sha1sum': '0a33633875a87a8c9f316e78d225ddfb41c54718'
        },
    }
    def __init__(self, split='train', path='./dataset/'):
        assert split in ['train', 'val', 'test'], f'Unsupported split {split}. Split must be one of train, val, or test.'
        self._split = split
        self._path = os.path.join(path,'epsilon',f'{split}.csv')
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
        return CATEGORICAL
    
    @property
    def metric(self):
        return ACC
    
    @property
    def problem_type(self):
        return BINARY


class HelenaTabularDataset(BaseTabularDataset):
    _INFO = {
        'train': {
            'url': 's3://autogluon/datasets/tabular/helena/train.csv',
            'sha1sum': '8af8b8fc01535446c189f60d13dcd026c0743bc4'
        },
        'val': {
            'url': 's3://autogluon/datasets/tabular/helena/val.csv',
            'sha1sum': '8146b59c01f78ad9831fde388b7235d42b9cf7fc'
        },
        'test': {
            'url': 's3://autogluon/datasets/tabular/helena/test.csv',
            'sha1sum': 'd098ce6b36ed52cd5be6e3abb80695ae3a682acc'
        },
    }
    def __init__(self, split='train', path='./dataset/'):
        assert split in ['train', 'val', 'test'], f'Unsupported split {split}. Split must be one of train, val, or test.'
        self._split = split
        self._path = os.path.join(path,'helena',f'{split}.csv')
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
        return CATEGORICAL
    
    @property
    def metric(self):
        return ACC
    
    @property
    def problem_type(self):
        return MULTICLASS


class HiggsSmallTabularDataset(BaseTabularDataset):
    _INFO = {
        'train': {
            'url': 's3://autogluon/datasets/tabular/higgs_small/train.csv',
            'sha1sum': '90fe90bb5313f0718e98c58b0f36a013a46680c5'
        },
        'val': {
            'url': 's3://autogluon/datasets/tabular/higgs_small/val.csv',
            'sha1sum': '83a5e93b2b64fc0fcef60ecc9089431e6b316cc7'
        },
        'test': {
            'url': 's3://autogluon/datasets/tabular/higgs_small/test.csv',
            'sha1sum': '68fc67a3546be69553dff61ae07ed5d9e5c6c45a'
        },
    }
    def __init__(self, split='train', path='./dataset/'):
        assert split in ['train', 'val', 'test'], f'Unsupported split {split}. Split must be one of train, val, or test.'
        self._split = split
        self._path = os.path.join(path,'higgs_small',f'{split}.csv')
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
        return CATEGORICAL
    
    @property
    def metric(self):
        return ACC
    
    @property
    def problem_type(self):
        return BINARY


class JannisTabularDataset(BaseTabularDataset):
    _INFO = {
        'train': {
            'url': 's3://autogluon/datasets/tabular/jannis/train.csv',
            'sha1sum': '1044879f8a16bd5c18e471bb587619758e52267b'
        },
        'val': {
            'url': 's3://autogluon/datasets/tabular/jannis/val.csv',
            'sha1sum': 'a4a69d823237a2a52f30caf3ffea35bb7423420f'
        },
        'test': {
            'url': 's3://autogluon/datasets/tabular/jannis/test.csv',
            'sha1sum': '3bf2c2827b607ecab2ecd7a25083c81edecc9c52'
        },
    }
    def __init__(self, split='train', path='./dataset/'):
        assert split in ['train', 'val', 'test'], f'Unsupported split {split}. Split must be one of train, val, or test.'
        self._split = split
        self._path = os.path.join(path,'jannis',f'{split}.csv')
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
        return CATEGORICAL
    
    @property
    def metric(self):
        return ACC
    
    @property
    def problem_type(self):
        return MULTICLASS


class MicrosoftTabularDataset(BaseTabularDataset):
    _INFO = {
        'train': {
            'url': 's3://autogluon/datasets/tabular/microsoft/train.csv',
            'sha1sum': '00c6ffd175e03859359ec41c7ff9e3742055954e'
        },
        'val': {
            'url': 's3://autogluon/datasets/tabular/microsoft/val.csv',
            'sha1sum': '0a29767b36d9ffe5b9c63dc9a64eaabc6028f557'
        },
        'test': {
            'url': 's3://autogluon/datasets/tabular/microsoft/test.csv',
            'sha1sum': '3fbfc60b4c17f1009df180dde6aac742ff6f5aca'
        },
    }
    def __init__(self, split='train', path='./dataset/'):
        assert split in ['train', 'val', 'test'], f'Unsupported split {split}. Split must be one of train, val, or test.'
        self._split = split
        self._path = os.path.join(path,'microsoft',f'{split}.csv')
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
        return RMSE
    
    @property
    def problem_type(self):
        return REGRESSION


class YahooTabularDataset(BaseTabularDataset):
    _INFO = {
        'train': {
            'url': 's3://autogluon/datasets/tabular/yahoo/train.csv',
            'sha1sum': '4a6ca419807da6991f755d0a349bfbac71cb6289'
        },
        'val': {
            'url': 's3://autogluon/datasets/tabular/yahoo/val.csv',
            'sha1sum': '50b5017d7aa01b0c07f0bc2e6b8c2ee298b2a1e2'
        },
        'test': {
            'url': 's3://autogluon/datasets/tabular/yahoo/test.csv',
            'sha1sum': '8b550c1dec3d1d6f980f7e51f3358336cd9d1973'
        },
    }
    def __init__(self, split='train', path='./dataset/'):
        assert split in ['train', 'val', 'test'], f'Unsupported split {split}. Split must be one of train, val, or test.'
        self._split = split
        self._path = os.path.join(path,'yahoo',f'{split}.csv')
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
        return RMSE
    
    @property
    def problem_type(self):
        return REGRESSION


class YearTabularDataset(BaseTabularDataset):
    _INFO = {
        'train': {
            'url': 's3://autogluon/datasets/tabular/year/train.csv',
            'sha1sum': '1f609cc6fb7d3cb71098041a5082eda190538de8'
        },
        'val': {
            'url': 's3://autogluon/datasets/tabular/year/val.csv',
            'sha1sum': '4afdc7452aa3ef643d1c24719c91970791c31ecd'
        },
        'test': {
            'url': 's3://autogluon/datasets/tabular/year/test.csv',
            'sha1sum': 'b412b511519e04444337f82e54e292d6d4beeb5e'
        },
    }
    def __init__(self, split='train', path='./dataset/'):
        assert split in ['train', 'val', 'test'], f'Unsupported split {split}. Split must be one of train, val, or test.'
        self._split = split
        self._path = os.path.join(path,'year',f'{split}.csv')
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
        return RMSE
    
    @property
    def problem_type(self):
        return REGRESSION
