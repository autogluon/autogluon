#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:56:33 2020

@author: yiran
"""


import logging
import mxnet as mx
from gluonnlp.data import SQuAD 
from mxnet.metric import EM, F1,CompositeEvalMetric

log = logging.getLogger('autogluon')
__all__ = ['SQuADBERTTask_V1', 'SQuADBERTTask_V2']



@func()
def get_dataset(path=None, name=None, version=None, *args, **kwargs):
    """Load different SQuAD version dataset from Autogluon.data API
       in order to train AutoGluon models on.
        
        Parameters
        ----------
        path : str
            local directory containing text dataset. This dataset should be in Stanford QA format.
        name : str
            Name describing which built-in popular QA dataset to use.
            Options include: 'SQuADBERTTask'. (now it has only one dataset)
            Detailed descriptions can be found in the file: 
                                       `autogluon/task/question_answering/dataset.py`
        version : float
            version corresponding to Stanford QA dataset
    """
    if path is not None:
        raise NotImplementedError
        
    if name is not None and (name.lower() in built_in_tasks) and \
            version is not None and (version == 1.1):
        return built_in_tasks[name.lower()+str(version)](*args, **kwargs)
    
    elif name is not None and name.lower() in built_in_tasks and \
            version is not None and (version == 2.0):
        return built_in_tasks[name.lower()+str(version)](*args, **kwargs)
    
    else:
        raise NotImplementedError
        
class AbstractBERTTask:
    """Abstract task class for datasets with SQuAD format.

    Parameters
    ----------
    version : float
        SQuAD dataset version{1.1, 2.0}
    metrics : str
        choose which metrics used in train, default is F1&EM
    calib : str
        whether use calibration mode or not during finetuning
    debug: bool
        choose debug mode or not
    
    """
    def __init__(self, version, metrics,mode='train',debug=False):
        self.version = version
        self.metrics = metrics
        self.mode = mode
        self.debug = debug
        

    def get_dataset(self, mode='train'):
        """Get the corresponding dataset for the task.

        Parameters
        ----------
        mode : str, default 'train'
            Dataset mode.

        Returns
        -------
        TSVDataset : the dataset of target mode.
        """
        raise NotImplementedError

    def dataset_train(self):
        """Get the training mode of the dataset for the task.

        Returns
        -------
        tuple of str: the mode name, and the dataset.
        """
        return 'train', self.get_dataset(mode='train')

    def dataset_dev(self):
        """Get the development (i.e. validation) mode of the dataset for this task.

        Returns
        -------
        tuple of (str), or list of tuple : the mode name, and the dataset.
        """
        return 'dev', self.get_dataset(mode='dev')

    def dataset_calib(self):
        """Get the test mode of the dataset for the task.

        Returns
        -------
        tuple of (str), or list of tuple : the mode name, and the dataset.
        """
        return 'test', self.get_dataset(mode='calib')
    

class SQuADBERTTask_V1(AbstractBERTTask):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.version = 1.1
        self.metric = CompositeEvalMetric()
        self.metric.add(F1())
        self.metric.add(EM())
#        self.mode = 'train'
        if 'dev' in args:
            self.mode = 'dev'
        if 'calib' in args:
            self.mode = 'calib'
        super(SQuADBERTTask_V1, self).__init__(self.version, self.metric, self.mode)
        
    def get_dataset(self):
        """Get the corresponding dataset for SQuAD

        Parameters
        ----------
        mode : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'calib'.
            default return dataset is 'train'
        """

        log.info('Loading %s data...', self.mode)
        # if self.version == 1.1:
        data_set = SQuAD('train', version=str(self.version))
        if self.debug:
            sampled_data = [data_set[i] for i in range(0, 1000)]
            data_set = mx.gluon.data.SimpleDataset(sampled_data)
        log.info('Number of records in Train data:{}'.format(len(data_set)))
        return data_set
        
            

    def dataset_train(self):
        """
        Get the training mode of the dataset for the task.
        """
        self.mode = 'train'  #if not args.debug else 'dev'
        log.info('Loading %s data...', self.mode)
        # if self.version == 2.0:
        train_data = SQuAD(self.mode, version=str(self.version))

        if self.debug:
            sampled_data = [train_data[i] for i in range(0, 1000)]
            train_data = mx.gluon.data.SimpleDataset(sampled_data)
        log.info('Number of records in Train data:{}'.format(len(train_data)))
        return 'train', train_data
    

    def dataset_dev(self):
        """
        Get the development (i.e. validation) mode of the dataset for QA task.
        """
        self.mode = 'dev'
        log.info('Loading dev data...')
        dev_data = SQuAD(self.mode, version=str(self.version))
        if self.debug:
            sampled_data = [dev_data[0], dev_data[1], dev_data[2]]
            dev_data = mx.gluon.data.SimpleDataset(sampled_data)
        log.info('Number of records in dev data:{}'.format(len(dev_data)))
        return 'dev', dev_data

    
    def dataset_calib(self):
        """
        Get the calibration mode of the dataset for QA task.
        """
        pass
    

class SQuADBERTTask_V2(AbstractBERTTask):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.version = 2.0
        self.metric = CompositeEvalMetric()
        self.metric.add(F1())
        self.metric.add(EM())
#        self.mode = 'train'
        if 'dev' in args:
            self.mode = 'dev'
        if 'calib' in args:
            self.mode = 'calib'
        super(SQuADBERTTask_V1, self).__init__(self.version, self.metric, self.mode)
        
    def get_dataset(self):
        """Get the corresponding dataset for SQuAD

        Parameters
        ----------
        mode : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'calib'.
            default return dataset is 'train'
        """
        log.info('Loading %s data...', self.mode)
        data_set = SQuAD('train', version=str(self.version))
        if self.debug:
            sampled_data = [data_set[i] for i in range(0, 1000)]
            data_set = mx.gluon.data.SimpleDataset(sampled_data)
        log.info('Number of records in Train data:{}'.format(len(data_set)))
        return data_set
        
        

    def dataset_train(self):
        """
        Get the training mode of the dataset for the task.
        """
        self.mode = 'train'  #if not args.debug else 'dev'
        log.info('Loading %s data...', self.mode)
        # if self.version == 2.0:
        train_data = SQuAD(self.mode, version=str(self.version))

        if self.debug:
            sampled_data = [train_data[i] for i in range(0, 1000)]
            train_data = mx.gluon.data.SimpleDataset(sampled_data)
        log.info('Number of records in Train data:{}'.format(len(train_data)))
        return 'train', train_data
    

    def dataset_dev(self):
        """
        Get the development (i.e. validation) mode of the dataset for QA task.
        """
        self.mode = 'dev'
        log.info('Loading dev data...')
        dev_data = SQuAD(self.mode, version=str(self.version))
        if self.debug:
            sampled_data = [dev_data[0], dev_data[1], dev_data[2]]
            dev_data = mx.gluon.data.SimpleDataset(sampled_data)
        log.info('Number of records in dev data:{}'.format(len(dev_data)))
        return 'dev', dev_data

    
    def dataset_calib(self):
        """
        Get the calibration mode of the dataset for QA task.
        """
        pass

built_in_tasks = {
    'squad' : AbstractBERTTask,
    'squad1.1':SQuADBERTTask_V1,
    'squad2.0':SQuADBERTTask_V2,
}